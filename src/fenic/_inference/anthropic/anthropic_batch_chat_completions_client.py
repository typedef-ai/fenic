import functools
import logging
import math
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

if TYPE_CHECKING:
    from fenic._inference.cache.protocol import LLMResponseCache
    from fenic.core._resolved_session_config import (
        ResolvedAdaptiveTokenEstimationConfig,
    )

import anthropic
from anthropic import (
    AnthropicError,
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    RateLimitError,
)
from anthropic.types import (
    MessageParam,
    ToolChoiceToolParam,
    ToolParam,
)

from fenic._inference.anthropic.anthropic_profile_manager import (
    AnthropicCompletionsProfileManager,
    AnthropicProfileConfiguration,
)
from fenic._inference.anthropic.anthropic_provider import AnthropicModelProvider
from fenic._inference.anthropic.anthropic_utils import (
    CONTENT_BLOCK_DELTA,
    EPHEMERAL_CACHE_CONTROL,
    INPUT_JSON_DELTA,
    MESSAGE_STOP,
    TEXT_DELTA,
    convert_messages,
)
from fenic._inference.model_client import (
    FatalException,
    ModelClient,
    TransientException,
)
from fenic._inference.profile_hash_mixin import ProfileHashMixin
from fenic._inference.rate_limit_strategy import (
    SeparatedTokenRateLimitStrategy,
    TokenEstimate,
)
from fenic._inference.token_counter import TiktokenTokenCounter, Tokenizable
from fenic._inference.types import (
    FenicCompletionsRequest,
    FenicCompletionsResponse,
    ResponseUsage,
)
from fenic.core._inference.model_catalog import (
    AnthropicTokenizer,
    ModelProvider,
    model_catalog,
)
from fenic.core._inference.output_token_limits import (
    ANTHROPIC_ADAPTIVE_THINKING_EFFORT_RATIOS,
    validate_effective_output_token_limit,
)
from fenic.core._logical_plan.resolved_types import ResolvedResponseFormat
from fenic.core._resolved_session_config import (
    ResolvedAnthropicModelProfile,
)
from fenic.core.error import ValidationError
from fenic.core.metrics import LMMetrics

logger = logging.getLogger(__name__)

StructuredOutputShape = Literal["output_format", "forced_tool", "tool"]

# Anthropic count_tokens / tiktoken cl100k_base ratios for each Claude tokenizer, set near the
# maximum measured on English, French and tool-call JSON with tools/predictive_token_accuracy.py.
CL100K_TO_ANTHROPIC_TOKENIZER_RATIOS: dict[AnthropicTokenizer, float] = {
    "claude-3": 1.05,
    "claude-opus-4-7": 1.40,
}


class AnthropicBatchCompletionsClient(
    ProfileHashMixin, ModelClient[FenicCompletionsRequest, FenicCompletionsResponse]
):
    """Anthropic batch chat completions client.

    This client handles communication with Anthropic's Claude models for batch
    chat completions. It supports streaming responses, structured output,
    thinking/reasoning capabilities, and token counting with Anthropic-specific
    adjustments.

    """

    def __init__(
        self,
        rate_limit_strategy: SeparatedTokenRateLimitStrategy,
        model: str,
        queue_size: int = 100,
        max_backoffs: int = 10,
        profiles: Optional[dict[str, ResolvedAnthropicModelProfile]] = None,
        default_profile_name: Optional[str] = None,
        cache: Optional["LLMResponseCache"] = None,
        base_url: Optional[str] = None,
        adaptive_estimation: Optional["ResolvedAdaptiveTokenEstimationConfig"] = None,
    ):
        """Initialize the Anthropic batch completions client.

        Args:
            rate_limit_strategy: Strategy for rate limiting requests
            queue_size: Maximum size of the request queue
            model: Anthropic model name to use
            max_backoffs: Maximum number of retry backoffs
            profiles: Dictionary of profile configurations
            default_profile_name: Name of the default profile to use
            cache: Optional LLM response cache
            base_url: Custom base URL for the Anthropic API
            adaptive_estimation: Optional config for adaptive output-token estimation
        """
        model_provider_class = AnthropicModelProvider(base_url=base_url)
        super().__init__(
            model=model,
            model_provider=ModelProvider.ANTHROPIC,
            model_provider_class=model_provider_class,
            rate_limit_strategy=rate_limit_strategy,
            queue_size=queue_size,
            max_backoffs=max_backoffs,
            token_counter=TiktokenTokenCounter(
                model_name=model, fallback_encoding="cl100k_base"
            ),
            cache=cache,
            adaptive_estimation=adaptive_estimation,
        )
        self._sync_client = self.model_provider_class.create_client()
        self._client = self.model_provider_class.create_aio_client()
        self._metrics = LMMetrics()
        self._output_formatter_tool_name = "output_formatter"
        self._output_formatter_tool_description = "Format the output of the model to correspond strictly to the provided schema."
        self._model_parameters = model_catalog.get_completion_model_parameters(
            ModelProvider.ANTHROPIC, model
        )
        # Apply this factor to the estimated token count to approximate Anthropic's encoding.
        self._tokenizer_adjustment_ratio = CL100K_TO_ANTHROPIC_TOKENIZER_RATIOS[
            self._model_parameters.anthropic_tokenizer
        ]

        # Use the profile configuration manager
        self._profile_manager = AnthropicCompletionsProfileManager(
            model_parameters=self._model_parameters,
            profile_configurations=profiles or {},
            default_profile_name=default_profile_name,
        )

    def _resolve_profile_for_hash(self, profile_name: Optional[str]) -> Any:
        return self._profile_manager.get_profile_by_name(profile_name)

    async def make_single_request(
        self, request: FenicCompletionsRequest
    ) -> Union[None, FenicCompletionsResponse, TransientException, FatalException]:
        """Make a single completion request to Anthropic.

        Handles both text and structured output requests, with support for
        thinking/reasoning when enabled. Processes streaming responses and
        extracts usage metrics.

        Args:
            request: The completion request to process

        Returns:
            Completion response, transient exception, or fatal exception
        """
        system_prompt, message_params = convert_messages(request.messages)
        profile_configuration = self._profile_manager.get_profile_by_name(
            request.model_profile
        )
        try:
            request_max_tokens = self._get_max_output_token_request_limit(request)
        except ValidationError as e:
            # Deterministic request-construction failure: retrying cannot help.
            return FatalException(e)
        messages_creation_payload: dict[str, Any] = {
            "model": self.model,
            "system": [system_prompt],
            "messages": message_params,
            "max_tokens": request_max_tokens,
            "thinking": profile_configuration.thinking_config,
        }
        output_config = dict(profile_configuration.output_config or {})
        if request.structured_output:
            structured_output_params = self._structured_output_params(
                request.structured_output,
                self._structured_output_shape(profile_configuration),
            )
            output_config.update(structured_output_params.pop("output_config", {}))
            messages_creation_payload.update(structured_output_params)
        if output_config:
            messages_creation_payload["output_config"] = output_config

        if (
            not profile_configuration.thinking_enabled
            and self._model_parameters.supports_custom_temperature
            and request.temperature is not None
        ):
            # Anthropic does not allow configuring temperature if thinking is enabled.
            messages_creation_payload.update({"temperature": request.temperature})

        try:
            if request.structured_output:
                (
                    content,
                    usage_data,
                ) = await self._handle_structured_output_streaming_response(
                    messages_creation_payload
                )
            else:
                content, usage_data = await self._handle_text_streaming_response(
                    messages_creation_payload
                )
            if content is None:
                return FenicCompletionsResponse(completion="", logprobs=None)
            if usage_data:
                # Extract usage metrics
                # The cache token counts are Optional in the Anthropic SDK: they are
                # absent whenever prompt caching is not in play, so treat them as zero.
                num_cache_tokens_written = usage_data.cache_creation_input_tokens or 0
                num_pre_cached_tokens = usage_data.cache_read_input_tokens or 0
                num_uncached_input_tokens = usage_data.input_tokens
                prompt_tokens = (
                    num_pre_cached_tokens
                    + num_uncached_input_tokens
                    + num_cache_tokens_written
                )
                output_tokens = usage_data.output_tokens

                # Create ResponseUsage object
                usage = ResponseUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=output_tokens,  # For Anthropic, all output tokens are completion tokens
                    total_tokens=prompt_tokens + output_tokens,
                    cached_tokens=num_pre_cached_tokens,
                    thinking_tokens=0,  # Anthropic doesn't separate thinking tokens yet
                )

                # Update metrics (existing logic)
                self._metrics.num_cached_input_tokens += num_pre_cached_tokens
                self._metrics.num_uncached_input_tokens += num_uncached_input_tokens
                self._metrics.num_output_tokens += output_tokens
                self._metrics.num_requests += 1
                self._metrics.cost += model_catalog.calculate_completion_model_cost(
                    model_provider=ModelProvider.ANTHROPIC,
                    model_name=self.model,
                    uncached_input_tokens=num_uncached_input_tokens,
                    cached_input_tokens_read=num_pre_cached_tokens,
                    cached_input_tokens_written=num_cache_tokens_written,
                    output_tokens=output_tokens,
                )
        except RateLimitError as e:
            # Anthropic marks non-retryable 429s (e.g. a hard org spend-cap breach)
            # with `x-should-retry: false`; genuine per-minute rate limits are
            # retryable. Failing fast here avoids burning the full exponential
            # backoff budget on an error retries cannot fix.
            non_retryable = (
                e.response is not None
                and e.response.headers.get("x-should-retry", "").lower() == "false"
            )
            if non_retryable:
                logger.error(
                    f"Non-retryable rate limit error on anthropic provider: {e}"
                )
                return FatalException(e)
            return TransientException(e)
        except (APITimeoutError, APIConnectionError) as e:
            return TransientException(e)
        except APIStatusError as e:
            # 529 overloaded_error is transient by definition. Match on the status
            # code rather than the OverloadedError class: the anthropic SDK has
            # mapped 529 differently across releases (InternalServerError vs.
            # OverloadedError) and does not export OverloadedError from its top-level
            # namespace, so status-code matching stays robust across versions.
            if e.status_code == 529:
                return TransientException(e)
            return FatalException(e)
        except AnthropicError as e:
            return FatalException(e)

        return FenicCompletionsResponse(completion=content, logprobs=None, usage=usage)

    async def _handle_text_streaming_response(
        self, payload: dict[str, Any]
    ) -> tuple[str, Optional[anthropic.types.Usage]]:
        """Handle streaming text response from Anthropic.

        Processes streaming chunks to extract text content and usage data.

        Args:
            payload: The request payload sent to Anthropic

        Returns:
            Tuple of (content, usage_data)
        """
        content = ""
        usage_data: anthropic.types.Usage | None = None
        async with self._client.messages.stream(**payload) as stream:
            async for chunk in stream:
                if chunk.type == CONTENT_BLOCK_DELTA:
                    if chunk.delta.type == TEXT_DELTA:
                        content += chunk.delta.text
                elif chunk.type == MESSAGE_STOP:
                    usage_data = (
                        chunk.message.usage if hasattr(chunk.message, "usage") else None
                    )
        return content, usage_data

    async def _handle_structured_output_streaming_response(
        self, payload: dict[str, Any]
    ) -> tuple[str, Optional[anthropic.types.Usage]]:
        """Handle streaming structured output response from Anthropic.

        Processes streaming chunks to extract the JSON content and usage data. The
        JSON arrives as formatter tool input, or as text when the request uses
        structured outputs (``output_config.format``) instead of a tool.

        Args:
            payload: The request payload sent to Anthropic

        Returns:
            Tuple of (json_content, usage_data)
        """
        uses_formatter_tool = "tools" in payload
        json_content: str = ""
        usage_data: anthropic.types.Usage | None = None
        async with self._client.messages.stream(**payload) as stream:
            async for chunk in stream:
                if chunk.type == CONTENT_BLOCK_DELTA:
                    if uses_formatter_tool and chunk.delta.type == INPUT_JSON_DELTA:
                        json_content += chunk.delta.partial_json
                    elif not uses_formatter_tool and chunk.delta.type == TEXT_DELTA:
                        json_content += chunk.delta.text
                elif chunk.type == MESSAGE_STOP:
                    usage_data = (
                        chunk.message.usage if hasattr(chunk.message, "usage") else None
                    )
            return json_content, usage_data

    @staticmethod
    def _structured_output_shape(
        profile_configuration: AnthropicProfileConfiguration,
    ) -> StructuredOutputShape:
        """How a structured request carries its schema under this profile.

        Several adaptive-thinking models reject a forced tool_choice, and with
        tool_choice "auto" the model can answer in text instead of calling the
        formatter tool, so adaptive profiles use structured outputs. Without
        thinking, the formatter tool is forced. Manual (budget) thinking does not
        support a forced tool_choice, so the tool is offered without one.
        """
        if profile_configuration.uses_adaptive_thinking:
            return "output_format"
        if not profile_configuration.thinking_enabled:
            return "forced_tool"
        return "tool"

    def _structured_output_params(
        self, response_format: ResolvedResponseFormat, shape: StructuredOutputShape
    ) -> dict[str, Any]:
        """Request parameters that carry the schema; shared by requests and estimates."""
        if shape == "output_format":
            return {
                "output_config": {
                    "format": self.create_response_format_output(response_format)
                }
            }
        params: dict[str, Any] = {
            "tools": [self.create_response_format_tool(response_format)]
        }
        if shape == "forced_tool":
            params["tool_choice"] = ToolChoiceToolParam(
                name=self._output_formatter_tool_name, type="tool"
            )
        return params

    # lightweight caching to allow us to approximate the tokens in a given tool param
    # will replace with something more sophisticated later.
    @functools.cache  # noqa: B019
    def estimate_response_format_tokens(
        self, response_format: ResolvedResponseFormat, shape: StructuredOutputShape
    ) -> int:
        """Estimate token count for a response format schema.

        Uses Anthropic's API to count tokens for the response format schema, sent
        with the same parameters the request uses. Results are cached per schema
        and shape.

        Args:
            response_format: Pydantic model class defining the response format
            shape: How the request carries the schema

        Returns:
            Estimated token count for the response format
        """
        approx_tool_tokens = self._sync_client.messages.count_tokens(
            model=self.model,
            messages=[
                MessageParam(content="user prompt", role="user"),
            ],
            system="empty",
            **self._structured_output_params(response_format, shape),
        )
        return approx_tool_tokens.input_tokens

    def _count_auxiliary_input_tokens(self, request: FenicCompletionsRequest) -> int:
        """Count structured-output schema tokens in the shape this request's profile sends."""
        if not request.structured_output:
            return 0
        profile_configuration = self._profile_manager.get_profile_by_name(
            request.model_profile
        )
        return self.estimate_response_format_tokens(
            request.structured_output,
            self._structured_output_shape(profile_configuration),
        )

    def _get_max_output_token_request_limit(
        self, request: FenicCompletionsRequest
    ) -> int:
        """Get maximum output tokens including thinking budget.

        Args:
            request: The completion request

        Returns:
            Maximum output tokens (completion + thinking budget)
        """
        profile_configuration = self._profile_manager.get_profile_by_name(
            request.model_profile
        )
        return validate_effective_output_token_limit(
            model_provider=self.model_provider,
            model_name=self.model,
            model_max_output_tokens=self._model_parameters.max_output_tokens,
            requested_completion_tokens=request.max_completion_tokens,
            estimated_reasoning_tokens=profile_configuration.thinking_token_budget,
            reasoning_shares_output_window=profile_configuration.uses_adaptive_thinking,
        )

    # Override default behavior to account for the fact that Anthropic's encoding differs from OpenAI's
    # (by ~5% for the legacy tokenizer and ~40% for the Opus 4.7+ tokenizer on Latin-script text).
    # This is a rough estimate, but it's good enough for our purposes.
    def count_tokens(self, messages: Tokenizable) -> int:
        """Count tokens with Anthropic encoding adjustment.

        Applies a tokenizer adjustment ratio to account for differences
        between Anthropic's and OpenAI's tokenization.

        Args:
            messages: Messages to count tokens for

        Returns:
            Adjusted token count
        """
        return math.ceil(
            super().count_tokens(messages) * self._tokenizer_adjustment_ratio
        )

    def estimate_tokens_for_request(
        self, request: FenicCompletionsRequest
    ) -> TokenEstimate:
        """Estimate the number of tokens for a request."""
        input_tokens = self.count_tokens(request.messages)
        input_tokens += self._count_auxiliary_input_tokens(request)

        # Estimate output tokens for rate limiting. This is intentionally
        # separate from the provider-side max_tokens budget: adaptive thinking
        # can have a very large maximum window, but reserving that maximum for
        # throttling would make small-output requests impossible under normal
        # output TPM limits. Route the decoupled estimate through the adaptive
        # estimator (which clamps to this ceiling and learns from actuals).
        static_ceiling = self._estimate_output_tokens(request)
        thinking_budget = self._profile_manager.get_profile_by_name(
            request.model_profile
        ).thinking_token_budget
        output_tokens = self._adaptive_output_reservation(
            request, static_ceiling=static_ceiling, reasoning=thinking_budget > 0
        )
        return TokenEstimate(input_tokens=input_tokens, output_tokens=output_tokens)

    def _estimate_output_tokens(self, request: FenicCompletionsRequest) -> int:
        """Estimate output tokens for rate limiting."""
        completion_tokens = request.max_completion_tokens or 0
        profile_config = self._profile_manager.get_profile_by_name(
            request.model_profile
        )
        return completion_tokens + self._estimate_thinking_tokens_for_rate_limit(
            profile_config, completion_tokens
        )

    def _estimate_thinking_tokens_for_rate_limit(
        self, profile_config: AnthropicProfileConfiguration, completion_tokens: int
    ) -> int:
        """Estimate thinking tokens for throttling without using adaptive maxima."""
        if not profile_config.thinking_enabled:
            return 0
        if profile_config.uses_adaptive_thinking and profile_config.effort:
            return min(
                profile_config.thinking_token_budget,
                math.ceil(
                    ANTHROPIC_ADAPTIVE_THINKING_EFFORT_RATIOS[profile_config.effort]
                    * completion_tokens
                ),
            )
        return profile_config.thinking_token_budget

    def get_metrics(self) -> LMMetrics:
        """Get current metrics.

        Returns:
            Current language model metrics
        """
        return self._metrics

    def reset_metrics(self):
        """Reset metrics to initial state."""
        self._metrics = LMMetrics()

    def create_response_format_tool(
        self, response_format: ResolvedResponseFormat
    ) -> ToolParam:
        """Create a tool parameter for structured output.

        Converts a JSON schema to an Anthropic tool parameter for
        structured output formatting.

        Args:
            response_format: Resolved JSON schema defining the response format

        Returns:
            Anthropic tool parameter
        """
        tool_param = ToolParam(
            name=self._output_formatter_tool_name,
            input_schema=response_format.json_schema,
            description=self._output_formatter_tool_description,
            cache_control=EPHEMERAL_CACHE_CONTROL,
        )
        return tool_param

    def create_response_format_output(
        self, response_format: ResolvedResponseFormat
    ) -> dict[str, Any]:
        """Create an ``output_config.format`` value for structured outputs.

        Structured outputs require ``additionalProperties: false`` on every object
        and reject constraints such as ``minimum`` and ``minLength``, so the schema
        is converted with the SDK's ``transform_schema``, which moves unsupported
        constraints into field descriptions.

        Args:
            response_format: Resolved JSON schema defining the response format

        Returns:
            JSON schema output format for ``output_config``
        """
        return {
            "type": "json_schema",
            "schema": anthropic.transform_schema(response_format.json_schema),
        }
