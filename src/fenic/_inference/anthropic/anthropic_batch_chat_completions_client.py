import functools
import math
from typing import TYPE_CHECKING, Any, Optional, Union

if TYPE_CHECKING:
    from fenic._inference.cache.protocol import LLMResponseCache

import anthropic
from anthropic import (
    AnthropicError,
    APIConnectionError,
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
    ModelProvider,
    model_catalog,
)
from fenic.core._logical_plan.resolved_types import ResolvedResponseFormat
from fenic.core._resolved_session_config import (
    ResolvedAnthropicModelProfile,
)
from fenic.core.metrics import LMMetrics


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
        """
        super().__init__(
            model=model,
            model_provider=ModelProvider.ANTHROPIC,
            model_provider_class=AnthropicModelProvider(),
            rate_limit_strategy=rate_limit_strategy,
            queue_size=queue_size,
            max_backoffs=max_backoffs,
            token_counter=TiktokenTokenCounter(
                model_name=model, fallback_encoding="cl100k_base"
            ),
            cache=cache,
        )
        # Apply this factor to the estimated token count to approximate Anthropic's encoding.
        self._tokenizer_adjustment_ratio = 1.05
        self._sync_client = self.model_provider_class.create_client()
        self._client = self.model_provider_class.create_aio_client()
        self._metrics = LMMetrics()
        self._output_formatter_tool_name = "output_formatter"
        self._output_formatter_tool_description = "Format the output of the model to correspond strictly to the provided schema."
        self._model_parameters = model_catalog.get_completion_model_parameters(
            ModelProvider.ANTHROPIC, model
        )

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
        request_max_tokens = (
            request.max_completion_tokens + profile_configuration.thinking_token_budget
        )
        messages_creation_payload: dict[str, Any] = {
            "model": self.model,
            "system": [system_prompt],
            "messages": message_params,
            "max_tokens": request_max_tokens,
            "thinking": profile_configuration.thinking_config,
        }
        if request.structured_output:
            tool_param = self.create_response_format_tool(request.structured_output)
            messages_creation_payload.update({"tools": [tool_param]})
            if not profile_configuration.thinking_enabled:
                # Anthropic does not allow forced tool use if thinking is enabled.
                messages_creation_payload.update(
                    {
                        "tool_choice": ToolChoiceToolParam(
                            name=self._output_formatter_tool_name, type="tool"
                        )
                    }
                )

        if not profile_configuration.thinking_enabled:
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
                num_cache_tokens_written = usage_data.cache_creation_input_tokens
                num_pre_cached_tokens = usage_data.cache_read_input_tokens
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
                    cached_input_tokens_written=usage_data.cache_creation_input_tokens,
                    output_tokens=output_tokens,
                )
        except (
            RateLimitError,
            APITimeoutError,
            APIConnectionError,
        ) as e:  # consider the various APIStatusErrors
            return TransientException(e)
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

        Processes streaming chunks to extract JSON content from tool use and usage data.

        Args:
            payload: The request payload sent to Anthropic

        Returns:
            Tuple of (tool_use_content, usage_data)
        """
        tool_use_content: str = ""
        usage_data: anthropic.types.Usage | None = None
        async with self._client.messages.stream(**payload) as stream:
            async for chunk in stream:
                if chunk.type == CONTENT_BLOCK_DELTA:
                    if chunk.delta.type == INPUT_JSON_DELTA:
                        tool_use_content += chunk.delta.partial_json
                elif chunk.type == MESSAGE_STOP:
                    usage_data = (
                        chunk.message.usage if hasattr(chunk.message, "usage") else None
                    )
            return tool_use_content, usage_data

    # lightweight caching to allow us to approximate the tokens in a given tool param
    # will replace with something more sophisticated later.
    @functools.cache  # noqa: B019
    def estimate_response_format_tokens(
        self, response_format: ResolvedResponseFormat
    ) -> int:
        """Estimate token count for a response format schema.

        Uses Anthropic's API to count tokens in a tool parameter that represents
        the response format schema. Results are cached for performance.

        Args:
            response_format: Pydantic model class defining the response format

        Returns:
            Estimated token count for the response format
        """
        tool_param = self.create_response_format_tool(response_format)
        approx_tool_tokens = self._sync_client.messages.count_tokens(
            model=self.model,
            messages=[
                MessageParam(content="user prompt", role="user"),
            ],
            system="empty",
            tools=[tool_param],
            tool_choice=ToolChoiceToolParam(
                name=self._output_formatter_tool_name, type="tool"
            ),
        )
        return approx_tool_tokens.input_tokens

    def _estimate_structured_output_overhead(self, response_format) -> int:
        """Use Anthropic's API-based token counting for structured output.

        Args:
            response_format: Pydantic model class defining the response format

        Returns:
            Estimated token overhead for structured output
        """
        return self.estimate_response_format_tokens(response_format)

    def _get_max_output_token_request_limit(
        self, request: FenicCompletionsRequest
    ) -> int:
        """Get maximum output tokens including thinking budget.

        Args:
            request: The completion request

        Returns:
            Maximum output tokens (completion + thinking budget)
        """
        return (
            request.max_completion_tokens
            + self._profile_manager.get_profile_by_name(
                request.model_profile
            ).thinking_token_budget
        )

    # Override default behavior to account for the fact that Anthropic's encoding is slightly different from OpenAI's.
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

    def estimate_tokens_for_request(self, request: FenicCompletionsRequest):
        """Estimate the number of tokens for a request.

        Args:
            request: The request to estimate tokens for

        Returns:
            TokenEstimate: The estimated token usage
        """

        # Count input tokens
        input_tokens = self.count_tokens(request.messages)
        input_tokens += self._count_auxiliary_input_tokens(request)

        # Estimate output tokens
        output_tokens = self._get_max_output_token_request_limit(request)

        return TokenEstimate(input_tokens=input_tokens, output_tokens=output_tokens)

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
