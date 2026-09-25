"""Typed judgments through fenic's shared request scheduler."""

import json
import logging
from typing import TYPE_CHECKING, Optional, Sequence, Union

from fenic._inference.model_client import (
    FatalException,
    ModelClient,
    TransientException,
)
from fenic._inference.rate_limit_strategy import RateLimitStrategy, TokenEstimate
from fenic._inference.token_counter import TiktokenTokenCounter
from fenic._inference.types import (
    FenicCompletionsRequest,
    FenicCompletionsResponse,
    ResponseUsage,
)
from fenic._inference.typesafe.judge_requests import partition_questions
from fenic._inference.typesafe.typesafe_provider import TypeSafeModelProvider
from fenic.core._inference.model_catalog import ModelProvider, model_catalog
from fenic.core.metrics import LMMetrics
from fenic.core.types.judge import JudgeQuestion, flatten_answers

if TYPE_CHECKING:
    from fenic._inference.cache.protocol import LLMResponseCache
    from fenic.core._resolved_session_config import (
        ResolvedAdaptiveTokenEstimationConfig,
    )

logger = logging.getLogger(__name__)


class TypeSafeSystemOneClient(
    ModelClient[FenicCompletionsRequest, FenicCompletionsResponse]
):
    """Serve explicit typed questions, never recover questions from text prompts."""

    def __init__(
        self,
        rate_limit_strategy: RateLimitStrategy,
        model: str,
        queue_size: int = 100,
        max_backoffs: int = 2,
        cache: Optional["LLMResponseCache"] = None,
        base_url: Optional[str] = None,
        adaptive_estimation: Optional["ResolvedAdaptiveTokenEstimationConfig"] = None,
    ):
        token_counter = TiktokenTokenCounter(
            model_name="gpt-4o", fallback_encoding="o200k_base"
        )
        super().__init__(
            model=model,
            model_provider=ModelProvider.TYPESAFE,
            model_provider_class=TypeSafeModelProvider(base_url=base_url),
            rate_limit_strategy=rate_limit_strategy,
            queue_size=queue_size,
            max_backoffs=max_backoffs,
            token_counter=token_counter,
            cache=cache,
            adaptive_estimation=adaptive_estimation,
        )
        self._model_parameters = model_catalog.get_completion_model_parameters(
            ModelProvider.TYPESAFE, model
        )
        self._client = self.model_provider_class.create_aio_client()
        self._metrics = LMMetrics()

    def judge_partitions(
        self, state: str, questions: Sequence[JudgeQuestion]
    ) -> list[tuple[JudgeQuestion, ...]]:
        """Pack intact questions using estimated, not authoritative, token counts."""
        return partition_questions(state, questions, self.token_counter.count_tokens)

    async def make_single_request(
        self, request: FenicCompletionsRequest
    ) -> Union[None, FenicCompletionsResponse, TransientException, FatalException]:
        """Evaluate a typed request and account for returned usage before decoding."""
        from typesafe_sdk import (
            TypeSafeAPIConnectionError,
            TypeSafeAPIResponseValidationError,
            TypeSafeBadRequestError,
            TypeSafeError,
            TypeSafeInternalServerError,
            TypeSafeRateLimitError,
            TypeSafeUnprocessableEntityError,
        )

        questions = request.judge_questions
        if not questions:
            return FatalException(
                ValueError(
                    "The TypeSafe provider requires typed judgment questions. "
                    "Open-ended map, extract, summarize, and reduce are unsupported."
                )
            )
        state = request.messages.user
        try:
            if len(self.judge_partitions(state, questions)) != 1:
                raise ValueError("Judge request requires question partitioning")
        except ValueError:
            return None
        try:
            result = await self._client.system_one(
                request.judge_state if request.judge_state is not None else state,
                {q.name: q.body() for q in questions},
                model=self.model,
            )
        except (
            TypeSafeRateLimitError,
            TypeSafeInternalServerError,
            TypeSafeAPIConnectionError,
        ) as error:
            # SDK error bodies can contain row text. The scheduler owns retry policy.
            return TransientException(RuntimeError(type(error).__name__))
        except (
            TypeSafeBadRequestError,
            TypeSafeUnprocessableEntityError,
            TypeSafeAPIResponseValidationError,
        ):
            # Request admission and validation failures are row-local. Do not cache
            # them or manufacture usage for a request the provider rejected.
            return None
        except TypeSafeError as error:
            # Fatal errors must fail the query without exposing SDK response bodies.
            return FatalException(RuntimeError(type(error).__name__))

        provider_usage = result.usage
        input_tokens = (
            provider_usage.input_tokens if provider_usage is not None else None
        )
        output_tokens = (
            provider_usage.output_tokens if provider_usage is not None else None
        )
        if any(
            value is not None and (type(value) is not int or value < 0)
            for value in (input_tokens, output_tokens)
        ):
            return None
        self._metrics.num_requests += 1
        usage = None
        if input_tokens is not None and output_tokens is not None:
            usage = ResponseUsage(
                input_tokens, output_tokens, input_tokens + output_tokens
            )
            self._metrics.num_uncached_input_tokens += input_tokens
            self._metrics.num_output_tokens += output_tokens
            self._metrics.cost += model_catalog.calculate_completion_model_cost(
                model_provider=ModelProvider.TYPESAFE,
                model_name=self.model,
                uncached_input_tokens=input_tokens,
                cached_input_tokens_read=0,
                output_tokens=output_tokens,
            )
        else:
            logger.warning(
                "TypeSafe response reported incomplete usage; excluding this request "
                "from displayed token and cost totals."
            )
        try:
            decoded = flatten_answers(
                questions,
                {
                    name: answer.model_dump(mode="json")
                    for name, answer in result.answers.items()
                },
            )
        except (ValueError, TypeError, AttributeError):
            decoded = None
        return FenicCompletionsResponse(
            json.dumps(decoded),
            None,
            usage,
            cacheable=decoded is not None,
        )

    def estimate_tokens_for_request(
        self, request: FenicCompletionsRequest
    ) -> TokenEstimate:
        """Estimate pacing costs without claiming access to the provider tokenizer."""
        questions = request.judge_questions or ()
        return TokenEstimate(
            input_tokens=self.token_counter.count_tokens(request.messages.user or "")
            + sum(
                self.token_counter.count_tokens(json.dumps(q.body(), sort_keys=True))
                for q in questions
            ),
            output_tokens=sum(
                16 + 8 * max(len(q.options), len(q.levels), 1) for q in questions
            ),
        )

    def reset_metrics(self):
        self._metrics = LMMetrics()

    def get_metrics(self) -> LMMetrics:
        return self._metrics

    def _get_max_output_token_request_limit(
        self, request: FenicCompletionsRequest
    ) -> int:
        return self._model_parameters.max_output_tokens

    async def _close_provider(self) -> None:
        """Close the SDK after the shared scheduler stops all in-flight work."""
        await self._client.aclose()
