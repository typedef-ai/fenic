"""Client for making batch requests to OpenRouter's chat completions API."""
from typing import Optional, Union

from fenic._inference.model_client import (
    FatalException,
    ModelClient,
    TransientException,
)
from fenic._inference.openrouter.openrouter_profile_manager import (
    OpenRouterCompletionsProfileManager,
)
from fenic._inference.rate_limit_strategy import (
    NoopRateLimitStrategy,
    TokenEstimate,
)
from fenic._inference.token_counter import TiktokenTokenCounter
from fenic._inference.types import FenicCompletionsRequest, FenicCompletionsResponse
from fenic.core._inference.model_catalog import ModelProvider, model_catalog
from fenic.core.metrics import LMMetrics

from ...core.error import ConfigurationError
from .openrouter_provider import OpenRouterModelProvider


class OpenRouterBatchChatCompletionsClient(ModelClient[FenicCompletionsRequest, FenicCompletionsResponse]):
    """Client for making batch requests to OpenRouter's chat completions API.

    Notes:
        - Uses the OpenAI SDK pointed at OpenRouter via base_url.
        - Default rate limiting uses NoopRateLimitStrategy; provider backoffs still apply.
    """

    def __init__(
        self,
        model: str,
        rate_limit_strategy: Optional[NoopRateLimitStrategy] = None,
        queue_size: int = 100,
        max_backoffs: int = 10,
        profiles: Optional[dict[str, object]] = None,
        default_profile_name: Optional[str] = None,
    ):
        if rate_limit_strategy is None:
            rate_limit_strategy = NoopRateLimitStrategy()
        super().__init__(
            model=model,
            model_provider=ModelProvider.OPENROUTER,
            model_provider_class=OpenRouterModelProvider(),
            rate_limit_strategy=rate_limit_strategy,
            queue_size=queue_size,
            max_backoffs=max_backoffs,
            token_counter=TiktokenTokenCounter(model_name=model, fallback_encoding="o200k_base"),
        )
        self._model_parameters = model_catalog.get_completion_model_parameters(ModelProvider.OPENROUTER, model)
        self._profile_manager = OpenRouterCompletionsProfileManager(
            model_parameters=self._model_parameters,
            profile_configurations=profiles,
            default_profile_name=default_profile_name,
        )
        self._aio_client = OpenRouterModelProvider().aio_client
        self._metrics = LMMetrics()
        self._metrics_lock = None

    async def make_single_request(
        self, request: FenicCompletionsRequest
    ) -> Union[None, FenicCompletionsResponse, TransientException, FatalException]:
        profile = self._profile_manager.get_profile_by_name(request.model_profile)
        try:
            additional_reasoning_tokens = 0
            if profile.reasoning_max_tokens:
                additional_reasoning_tokens = profile.reasoning_max_tokens
            if profile.reasoning_effort == "low":
                additional_reasoning_tokens = 1024
            if profile.reasoning_effort == "medium":
                additional_reasoning_tokens = 4096
            if profile.reasoning_effort == "high":
                additional_reasoning_tokens = 8192
            common_params = {
                "model": self.model,
                "messages": request.messages.to_message_list(),
                "max_completion_tokens": request.max_completion_tokens + additional_reasoning_tokens,
                "n": 1,
            }


            if request.top_logprobs:
                common_params.update({"logprobs": True, "top_logprobs": request.top_logprobs})
            if request.temperature:
                common_params.update({"temperature": request.temperature})
            extra_body = profile.extra_body
            # Enable native usage accounting so we don't need a follow-up generation fetch
            extra_body["usage"] = {"include": True}
            if request.structured_output:
                if "structured_outputs" not in self._model_parameters.supported_parameters:
                   return FatalException(ConfigurationError(f"Model {self.model} does not support structured output."))
                common_params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "fenic_response",
                        "schema": request.structured_output.strict_schema,
                        "strict": True,
                    },
                }
                response = await self._aio_client.beta.chat.completions.parse(**common_params, extra_body=extra_body)
                if response.choices[0].message.refusal:
                    return None
            else:

                response = await self._aio_client.chat.completions.create(**common_params, extra_body=extra_body)

            usage = response.usage
            cached_input_tokens = usage.prompt_tokens_details.cached_tokens if usage.prompt_tokens_details else 0
            uncached_input_tokens = usage.prompt_tokens - cached_input_tokens
            total_prompt_tokens = usage.prompt_tokens
            reasoning_tokens = (
                usage.completion_tokens_details.reasoning_tokens if usage.completion_tokens_details else 0
            )
            total_output_tokens = usage.completion_tokens
            completion_tokens = total_output_tokens - reasoning_tokens

            from fenic._inference.types import FenicCompletionsResponse, ResponseUsage
            fenic_usage = ResponseUsage(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_prompt_tokens + total_output_tokens,
                cached_tokens=cached_input_tokens,
                thinking_tokens=reasoning_tokens,
            )
            # Update metrics synchronously using native counts and cost when available
            self._metrics.num_cached_input_tokens += cached_input_tokens
            self._metrics.num_uncached_input_tokens += uncached_input_tokens
            self._metrics.num_output_tokens += total_output_tokens
            self._metrics.num_requests += 1

            # Cost from OpenRouter usage, fallback to catalog if not provided
            model_extra = usage.model_extra
            cost_value = model_extra.get("cost")
            if isinstance(cost_value, (int, float)):
                self._metrics.cost += float(cost_value)
            else:
                try:
                    self._metrics.cost += model_catalog.calculate_completion_model_cost(
                        model_provider=self.model_provider,
                        model_name=self.model,
                        uncached_input_tokens=uncached_input_tokens,
                        cached_input_tokens_read=cached_input_tokens,
                        output_tokens=total_output_tokens,
                    )
                except Exception:
                    pass

            return FenicCompletionsResponse(
                completion=response.choices[0].message.content,
                logprobs=response.choices[0].logprobs,
                usage=fenic_usage,
            )
        except Exception as e:
            try:
                from openai import (
                    APIConnectionError,
                    APITimeoutError,
                    OpenAIError,
                    RateLimitError,
                )
                if isinstance(e, (RateLimitError, APITimeoutError, APIConnectionError)):
                    return TransientException(e)
                if isinstance(e, OpenAIError):
                    return FatalException(e)
            except Exception:
                pass
            return FatalException(e)

    def get_request_key(self, request: FenicCompletionsRequest) -> str:
        from fenic._inference.request_utils import generate_completion_request_key
        return generate_completion_request_key(request)

    def estimate_tokens_for_request(self, request: FenicCompletionsRequest) -> TokenEstimate:
        return TokenEstimate(
            input_tokens=self.token_counter.count_tokens(request.messages),
            output_tokens=self._get_max_output_tokens(request),
        )

    def reset_metrics(self):
        self._metrics = LMMetrics()

    def get_metrics(self) -> LMMetrics:
        return self._metrics

    def _get_max_output_tokens(self, request: FenicCompletionsRequest) -> int:
        base_tokens = request.max_completion_tokens
        profile_config = self._profile_manager.get_profile_by_name(request.model_profile)
        if profile_config.reasoning_max_tokens:
            base_tokens += profile_config.reasoning_max_tokens
        return base_tokens


