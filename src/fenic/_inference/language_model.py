import logging
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

from fenic._inference.model_client import (
    ModelClient,
)
from fenic._inference.token_counter import Tokenizable
from fenic._inference.types import (
    FenicCompletionsRequest,
    FenicCompletionsResponse,
    LMRequestMessages,
)
from fenic.core._inference.model_catalog import (
    model_catalog,
)
from fenic.core._logical_plan.resolved_types import ResolvedResponseFormat
from fenic.core.error import ConfigurationError
from fenic.core.metrics import LMMetrics

logger = logging.getLogger(__name__)

@dataclass
class InferenceConfiguration:
    # If max_output_tokens is not provided, model_client will add a guardrail based on the estimated output tokens.
    max_output_tokens: Optional[int]
    temperature: float
    top_logprobs: Optional[int] = None
    response_format: Optional[ResolvedResponseFormat] = None  # Resolved JSON schema
    model_profile: Optional[str] = None
    request_timeout: Optional[float] = None  # Timeout in seconds for a single LLM request

class LanguageModel:
    def __init__(self, client: ModelClient[FenicCompletionsRequest, FenicCompletionsResponse]):
        self.provider = client.model_provider
        self.model = client.model
        self.model_parameters = model_catalog.get_completion_model_parameters(self.provider, self.model)
        if self.model_parameters is None:
            raise ConfigurationError(model_catalog.generate_unsupported_completion_model_error_message(self.provider, self.model))
        # TPM might limit us before being limited by the actual context window length.
        self.max_context_window_length =  min(client.context_tokens_per_minute, self.model_parameters.context_window_length)
        self.client = client

    def get_completions(
        self,
        messages: list[LMRequestMessages],
        max_tokens: int,
        temperature: float = 0,
        response_format: Optional[ResolvedResponseFormat] = None,
        top_logprobs: Optional[int] = None,
        model_profile: Optional[str] = None,
        operation_name: Optional[str] = None,
        request_timeout: Optional[float] = None,
    ) -> list[Optional[FenicCompletionsResponse]]:
        # Create batch requests
        requests = []
        # Check model specific requirements for request params.
        temperature_param = temperature if self.model_parameters.supports_custom_temperature else None
        if temperature and not temperature_param:
            logger.warning(f"Model {self.model} does not support custom temperature.  Ignoring temperature parameter.")

        for message_list in messages:
            # if there are no messages, set the request as None, so it can be skipped.
            if not message_list:
                requests.append(None)
                continue
            request = FenicCompletionsRequest(
                messages=message_list,
                max_completion_tokens=max_tokens,
                top_logprobs=top_logprobs,
                structured_output=response_format,
                temperature=temperature_param,
                model_profile=model_profile,
            )
            requests.append(request)

        # Process batch requests
        return self.client.make_batch_requests(
            requests,
            operation_name=operation_name,
            request_timeout=request_timeout,
        )

    def iter_completions(
        self,
        messages: Iterable[Optional[LMRequestMessages]],
        max_tokens: int,
        temperature: float = 0,
        response_format: Optional[ResolvedResponseFormat] = None,
        top_logprobs: Optional[int] = None,
        model_profile: Optional[str] = None,
        operation_name: Optional[str] = None,
        request_timeout: Optional[float] = None,
        batch_size: int = 100,
    ) -> Iterator[Optional[FenicCompletionsResponse]]:
        """Build and submit completion requests from an ordered message stream.

        This is the row-local streaming counterpart to ``get_completions``. It
        deliberately leaves the list-shaped API intact for aggregation operators
        such as ``semantic.reduce`` while callers that can consume a stream avoid
        retaining all rendered messages and requests at once. ``batch_size`` is a
        minimum look-ahead, not a hard memory cap: the client admits an effective
        live window of ``max(batch_size, rate_limit_strategy.rpm)`` so streaming
        preserves the rate limiter's configured burst concurrency.
        """
        temperature_param = (
            temperature if self.model_parameters.supports_custom_temperature else None
        )
        if temperature and not temperature_param:
            logger.warning(
                f"Model {self.model} does not support custom temperature.  Ignoring temperature parameter."
            )

        def build_requests() -> Iterator[Optional[FenicCompletionsRequest]]:
            for message_list in messages:
                if not message_list:
                    yield None
                    continue
                yield FenicCompletionsRequest(
                    messages=message_list,
                    max_completion_tokens=max_tokens,
                    top_logprobs=top_logprobs,
                    structured_output=response_format,
                    temperature=temperature_param,
                    model_profile=model_profile,
                )

        return self.client.iter_batch_requests(
            build_requests(),
            operation_name=operation_name,
            request_timeout=request_timeout,
            batch_size=batch_size,
        )

    def count_tokens(self, messages: Tokenizable) -> int:
        return self.client.count_tokens(messages)


    def reset_metrics(self):
        self.client.reset_metrics()

    def get_metrics(self) -> LMMetrics:
        return self.client.get_metrics()
