"""Client for making batch requests to OpenRouter's chat completions API."""
import importlib.util
import logging
import math
from json.decoder import JSONDecodeError
from typing import Optional, Union

from openai import APIConnectionError, APITimeoutError, OpenAIError, RateLimitError
from pydantic import ValidationError as PydanticValidationError

from fenic._inference.cache.protocol import LLMResponseCache
from fenic._inference.common_openai.openai_utils import convert_messages
from fenic._inference.common_openai.utils import handle_openai_compatible_response
from fenic._inference.model_client import (
    FatalException,
    ModelClient,
    TransientException,
)
from fenic._inference.openrouter.openrouter_profile_manager import (
    OpenRouterCompletionProfileConfiguration,
    OpenRouterCompletionsProfileManager,
)
from fenic._inference.openrouter.openrouter_provider import OpenRouterModelProvider
from fenic._inference.profile_hash_mixin import ProfileHashMixin
from fenic._inference.rate_limit_strategy import (
    AdaptiveBackoffRateLimitStrategy,
    RateLimitStrategy,
    TokenEstimate,
)
from fenic._inference.request_utils import parse_openrouter_rate_limit_headers
from fenic._inference.token_counter import TiktokenTokenCounter
from fenic._inference.types import (
    FenicCompletionsRequest,
    FenicCompletionsResponse,
    ResponseUsage,
)
from fenic.core._inference.model_catalog import (
    CompletionModelParameters,
    ModelProvider,
    model_catalog,
)
from fenic.core._inference.output_token_limits import (
    OPENROUTER_REASONING_EFFORT_RATIOS,
    validate_effective_output_token_limit,
)
from fenic.core._resolved_session_config import ResolvedAdaptiveTokenEstimationConfig
from fenic.core.error import ConfigurationError, ExecutionError, ValidationError
from fenic.core.metrics import LMMetrics

TOOLS = "tools"

TOOL_CHOICE = "tool_choice"

STRUCTURED_OUTPUTS = "structured_outputs"

RESPONSE_FORMAT = "response_format"

OUTPUT_FORMATTER_TOOL_NAME = "output_formatter"

ANTHROPIC_STRUCTURED_OUTPUTS_BETA_HEADER = "structured-outputs-2025-11-13"
logger = logging.getLogger(__name__)


class OpenRouterBatchChatCompletionsClient(
    ProfileHashMixin,
    ModelClient[FenicCompletionsRequest, FenicCompletionsResponse]
):
    """Client for making batch requests to OpenRouter's chat completions API.

    Notes:
        - Uses the OpenAI SDK pointed at OpenRouter via base_url.
        - Default rate limiting uses AdaptiveBackoffRateLimitStrategy; provider backoffs still apply.
    """

    def __init__(
        self,
        model: str,
        rate_limit_strategy: RateLimitStrategy = None,
        queue_size: int = 100,
        max_backoffs: int = 10,
        profiles: Optional[dict[str, object]] = None,
        default_profile_name: Optional[str] = None,
        cache: Optional[LLMResponseCache] = None,
        adaptive_estimation: Optional[ResolvedAdaptiveTokenEstimationConfig] = None,
    ):
        # Choose token counter based on the model's provider
        token_counter = None
        provider_and_model = model.split("/")
        if provider_and_model[0] == "google" and importlib.util.find_spec("google.genai") is not None:
            # If fenic is built with google module, use the GeminiLocalTokenCounter.
            # Otherwise, fall back to the TiktokenTokenCounter.
            from fenic._inference.google.gemini_token_counter import (
                GeminiLocalTokenCounter,
            )
            token_counter = GeminiLocalTokenCounter(model_name=provider_and_model[1])
        else:
            token_counter = TiktokenTokenCounter(
                model_name=provider_and_model[1], fallback_encoding="o200k_base"
            )
        super().__init__(
            model=model,
            model_provider=ModelProvider.OPENROUTER,
            model_provider_class=OpenRouterModelProvider(),
            rate_limit_strategy=rate_limit_strategy,
            queue_size=queue_size,
            max_backoffs=max_backoffs,
            token_counter=token_counter,
            cache=cache,
            adaptive_estimation=adaptive_estimation,
        )
        self._model_parameters = model_catalog.get_completion_model_parameters(
            ModelProvider.OPENROUTER, model
        )
        self._profile_manager = OpenRouterCompletionsProfileManager(
            model_parameters=self._model_parameters,
            profile_configurations=profiles,
            default_profile_name=default_profile_name,
        )
        self._aio_client = OpenRouterModelProvider().aio_client
        self._metrics = LMMetrics()



    async def make_single_request(
        self, request: FenicCompletionsRequest
    ) -> Union[None, FenicCompletionsResponse, TransientException, FatalException]:
        profile = self._profile_manager.get_profile_by_name(request.model_profile)
        common_params = {
                "model": self.model,
                "messages": convert_messages(request.messages),
                "n": 1,
            }

        try:
            max_completion_tokens = self._get_max_output_token_request_limit(request)
        except ValidationError as e:
            # Deterministic request-construction failure: retrying cannot help.
            return FatalException(e)
        if max_completion_tokens is not None:
            common_params["max_completion_tokens"] = max_completion_tokens

        if request.top_logprobs:
            common_params.update(
                {"logprobs": True, "top_logprobs": request.top_logprobs}
            )

        if request.temperature and self._model_parameters.supports_custom_temperature:
            common_params.update({"temperature": request.temperature})

        used_tools = False
        try:
            if request.structured_output:
                strategy = profile.structured_output_strategy or "prefer_response_format"
                supports_structured = STRUCTURED_OUTPUTS in self._model_parameters.supported_parameters
                supports_tools = TOOLS in self._model_parameters.supported_parameters
                supports_tool_choice = (
                    TOOL_CHOICE in self._model_parameters.supported_parameters
                )
                can_force_tools = supports_tools and supports_tool_choice
                uses_incompatible_manual_thinking = (
                    self._uses_incompatible_anthropic_manual_thinking(profile)
                )
                can_use_tools = (
                    can_force_tools and not uses_incompatible_manual_thinking
                )

                if supports_structured and can_use_tools:
                    use_tools = strategy == "prefer_tools"
                else:
                    use_tools = can_use_tools and not supports_structured

                if (
                    strategy == "prefer_tools"
                    and supports_structured
                    and supports_tools
                    and not can_use_tools
                ):
                    logger.debug(
                        "Model %s cannot guarantee structured output with forced "
                        "tool calling for the selected profile. Falling back to "
                        "native structured outputs.",
                        self.model,
                    )

                if supports_structured and not use_tools:
                    common_params[RESPONSE_FORMAT] = request.structured_output.pydantic_model
                    response = await self._aio_client.chat.completions.parse(
                        **common_params, extra_body=profile.extra_body
                    )
                elif can_use_tools:
                    used_tools = True
                    response_schema = dict(request.structured_output.json_schema)
                    response_schema["additionalProperties"] = False
                    common_params[TOOLS] = [
                        {
                            "type": "function",
                            "function": {
                                "name": OUTPUT_FORMATTER_TOOL_NAME,
                                "description": "Format the output of the model to correspond strictly to the provided schema.",
                                "parameters": response_schema,
                                "strict": True,
                            },
                        }
                    ]
                    common_params[TOOL_CHOICE] = {
                        "type": "function",
                        "function": {"name": OUTPUT_FORMATTER_TOOL_NAME},
                    }
                    request_options = {}
                    if self._can_enable_anthropic_strict_tools(profile):
                        request_options["extra_headers"] = {
                            "x-anthropic-beta": (
                                ANTHROPIC_STRUCTURED_OUTPUTS_BETA_HEADER
                            )
                        }
                    response = await self._aio_client.chat.completions.create(
                        **common_params,
                        extra_body=profile.extra_body,
                        **request_options,
                    )
                else:
                    return FatalException(
                        self._structured_output_configuration_error(
                            supports_tools=supports_tools,
                            supports_tool_choice=supports_tool_choice,
                            uses_incompatible_manual_thinking=(
                                uses_incompatible_manual_thinking
                            ),
                        )
                    )
            else:
                response = await self._aio_client.chat.completions.create(
                    **common_params, extra_body=profile.extra_body
                )

            completion_choice, maybe_exception = handle_openai_compatible_response(
                model_provider=ModelProvider.OPENROUTER,
                model_name=self.model,
                request=request,
                response=response,
                request_key_generator=self.get_request_key,
            )
            if maybe_exception:
                return maybe_exception

            usage = response.usage
            cached_input_tokens = (
                (usage.prompt_tokens_details.cached_tokens or 0)
                if usage.prompt_tokens_details
                else 0
            )
            uncached_input_tokens = usage.prompt_tokens - cached_input_tokens
            total_prompt_tokens = usage.prompt_tokens
            reasoning_tokens = (
                (usage.completion_tokens_details.reasoning_tokens or 0)
                if usage.completion_tokens_details
                else 0
            )
            total_output_tokens = usage.completion_tokens
            completion_tokens = total_output_tokens - reasoning_tokens

            fenic_usage = ResponseUsage(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_prompt_tokens + total_output_tokens,
                cached_tokens=cached_input_tokens,
                thinking_tokens=reasoning_tokens,
            )
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
                self._metrics.cost += model_catalog.calculate_completion_model_cost(
                    model_provider=self.model_provider,
                    model_name=self.model,
                    uncached_input_tokens=uncached_input_tokens,
                    cached_input_tokens_read=cached_input_tokens,
                    output_tokens=total_output_tokens,
                )

            if used_tools:
                tool_calls = completion_choice.message.tool_calls or []
                if len(tool_calls) != 1:
                    return TransientException(
                        ExecutionError(
                            f"The completion model {self.model_provider}/{self.model} "
                            f"returned {len(tool_calls)} tool calls for a structured "
                            "output request; expected exactly one call to "
                            f"{OUTPUT_FORMATTER_TOOL_NAME}."
                        )
                    )
                tool_call = tool_calls[0]
                tool_function = getattr(tool_call, "function", None)
                if (
                    getattr(tool_call, "type", None) != "function"
                    or tool_function is None
                ):
                    return TransientException(
                        ExecutionError(
                            f"The completion model {self.model_provider}/{self.model} "
                            f"returned a {getattr(tool_call, 'type', None)!r} tool "
                            "call for a structured output request; expected a "
                            f"function call to {OUTPUT_FORMATTER_TOOL_NAME!r}."
                        )
                    )
                tool_name = getattr(tool_function, "name", None)
                if tool_name != OUTPUT_FORMATTER_TOOL_NAME:
                    return TransientException(
                        ExecutionError(
                            f"The completion model {self.model_provider}/{self.model} "
                            f"called {tool_name!r} for a structured output "
                            f"request; expected {OUTPUT_FORMATTER_TOOL_NAME!r}."
                        )
                    )
                validated_output = (
                    request.structured_output.pydantic_model.model_validate_json(
                        getattr(tool_function, "arguments", None),
                        strict=True,
                    )
                )
                completion = validated_output.model_dump_json(by_alias=True)
            else:
                completion = completion_choice.message.content

            return FenicCompletionsResponse(
                completion=completion,
                logprobs=completion_choice.logprobs,
                usage=fenic_usage,
            )
        except RateLimitError as e:
            if isinstance(self.rate_limit_strategy, AdaptiveBackoffRateLimitStrategy):
                rpm_hint, retry_at_s = parse_openrouter_rate_limit_headers(
                    e.response.headers
                )
                self.rate_limit_strategy.register_rate_limit_hint(rpm_hint, retry_at_s)
            return TransientException(e)
        except (APITimeoutError, APIConnectionError) as e:
            return TransientException(e)
        # encountered when the response is not valid JSON. can sometimes be fixed with a retry
        # sending the request to a different provider.
        except (JSONDecodeError, PydanticValidationError) as e:
            return TransientException(e)
        except OpenAIError as e:
            return FatalException(e)

    def _uses_incompatible_anthropic_manual_thinking(
        self,
        profile: OpenRouterCompletionProfileConfiguration,
    ) -> bool:
        """Return whether forced tools conflict with Anthropic manual thinking."""
        anthropic_models = [
            model
            for model in [self.model, *(profile.models or [])]
            if self._is_anthropic_model(model)
        ]
        if not anthropic_models:
            return False
        if profile.reasoning_max_tokens is not None:
            return True
        thinking_enabled = profile.reasoning_effort not in (None, "none")
        if not thinking_enabled:
            return False
        for model in anthropic_models:
            parameters = self._get_model_parameters(model)
            if parameters is None or not parameters.uses_adaptive_thinking:
                return True
        return False

    def _can_enable_anthropic_strict_tools(
        self,
        profile: OpenRouterCompletionProfileConfiguration,
    ) -> bool:
        """Return whether strict tools are supported by all Anthropic candidates."""
        anthropic_models = [
            model
            for model in [self.model, *(profile.models or [])]
            if self._is_anthropic_model(model)
        ]
        if not anthropic_models:
            return False
        for model in anthropic_models:
            parameters = self._get_model_parameters(model)
            if (
                parameters is None
                or STRUCTURED_OUTPUTS not in parameters.supported_parameters
            ):
                return False
        return True

    def _get_model_parameters(
        self,
        model: str,
    ) -> Optional[CompletionModelParameters]:
        """Return cached OpenRouter parameters for a primary or fallback model."""
        if model == self.model:
            return self._model_parameters
        return model_catalog.get_completion_model_parameters(
            ModelProvider.OPENROUTER,
            model,
        )

    @staticmethod
    def _is_anthropic_model(model: str) -> bool:
        """Return whether an OpenRouter model ID targets Anthropic."""
        return model.removeprefix("~").startswith("anthropic/")

    def _structured_output_configuration_error(
        self,
        *,
        supports_tools: bool,
        supports_tool_choice: bool,
        uses_incompatible_manual_thinking: bool,
    ) -> ConfigurationError:
        """Build an actionable error for an unavailable structured-output strategy."""
        if supports_tools and not supports_tool_choice:
            return ConfigurationError(
                f"Model {self.model} supports tools but not `tool_choice`, so "
                "fenic cannot force the formatter required for structured output. "
                "Select a model that supports `structured_outputs`, or both `tools` "
                "and `tool_choice`."
            )
        if supports_tools and uses_incompatible_manual_thinking:
            return ConfigurationError(
                f"Model {self.model} cannot combine forced tool calling with "
                "Anthropic manual thinking for structured output. Disable reasoning "
                "for this profile, use adaptive thinking, or select a model that "
                "supports native `structured_outputs`."
            )
        return ConfigurationError(
            f"Model {self.model} does not support a guaranteed structured-output "
            "strategy. Select a model that supports `structured_outputs`, or both "
            "`tools` and `tool_choice`."
        )

    def estimate_tokens_for_request(
        self, request: FenicCompletionsRequest
    ) -> TokenEstimate:
        return TokenEstimate(
            input_tokens=self._estimate_input_tokens(request),
            output_tokens=self._estimate_output_tokens(request),
        )

    def reset_metrics(self):
        self._metrics = LMMetrics()

    def get_metrics(self) -> LMMetrics:
        return self._metrics

    def _estimate_output_tokens(self, request: FenicCompletionsRequest) -> int:
        """Estimate the number of output tokens for a request."""
        base_tokens = request.max_completion_tokens or 0
        if request.max_completion_tokens is None and request.messages.user_file:
            # TODO(DY): the semantic operator should dictate how the file affects the token estimate
            base_tokens += self.token_counter.count_file_output_tokens(messages=request.messages)
        reasoning_tokens = self._get_expected_additional_reasoning_tokens(request)
        static_ceiling = base_tokens + reasoning_tokens
        return self._adaptive_output_reservation(
            request, static_ceiling=static_ceiling, reasoning=reasoning_tokens > 0
        )

    def _get_max_output_token_request_limit(self, request: FenicCompletionsRequest) -> Optional[int]:
        """Return the maximum output token limit for a request.

        Returns None if max_completion_tokens is not provided (no limit should be set).
        If max_completion_tokens is provided, includes the thinking token budget with a safety margin."""
        if request.max_completion_tokens is None:
            return None
        return validate_effective_output_token_limit(
            model_provider=self.model_provider,
            model_name=self.model,
            model_max_output_tokens=self._model_parameters.max_output_tokens,
            requested_completion_tokens=request.max_completion_tokens,
            estimated_reasoning_tokens=self._get_expected_additional_reasoning_tokens(request),
        )

    def _estimate_input_tokens(self, request: FenicCompletionsRequest) -> int:
        """Estimate the number of input tokens for a request."""
        input_tokens = self.token_counter.count_tokens(request.messages, ignore_file=True)
        if request.messages.user_file:
            input_tokens += self._estimate_file_input_tokens(request)
        return input_tokens

    def _estimate_file_input_tokens(self, request: FenicCompletionsRequest) -> int:
        """Estimate the number of input tokens from a file in a request."""
        profile_config = self._profile_manager.get_profile_by_name(request.model_profile)
        if profile_config.parsing_engine and profile_config.parsing_engine == "native":
            return self.token_counter.count_file_input_tokens(messages=request.messages)
        # OpenRouter's engine tool processes the file first and passes annotated text to the model.
        # We can estimate by extracting the text and tokenizing it (which is what count_file_output_tokens does)
        return self.token_counter.count_file_output_tokens(messages=request.messages)

    # OpenRouter documents these effort ratios for models that only support reasoning.max_tokens.
    # https://openrouter.ai/docs/guides/best-practices/reasoning-tokens#reasoning-effort-level
    def _get_expected_additional_reasoning_tokens(self, request: FenicCompletionsRequest) -> int:
        profile_config = self._profile_manager.get_profile_by_name(request.model_profile)
        additional_reasoning_tokens = 0
        if profile_config.reasoning_max_tokens:
            additional_reasoning_tokens = profile_config.reasoning_max_tokens
        elif profile_config.reasoning_effort:
            additional_reasoning_tokens = math.ceil(
                OPENROUTER_REASONING_EFFORT_RATIOS[profile_config.reasoning_effort]
                * self._model_parameters.max_output_tokens
            )
        return additional_reasoning_tokens

    def _resolve_profile_for_hash(self, profile_name: Optional[str]) -> OpenRouterCompletionProfileConfiguration:
        return self._profile_manager.get_profile_by_name(profile_name)
