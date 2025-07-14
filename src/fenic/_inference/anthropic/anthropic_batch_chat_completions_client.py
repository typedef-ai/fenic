import functools
import math
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import anthropic
from anthropic import (
    AnthropicError,
    APIConnectionError,
    APITimeoutError,
    AsyncAnthropic,
    RateLimitError,
)
from anthropic.types import (
    CacheControlEphemeralParam,
    MessageParam,
    TextBlockParam,
    ToolChoiceToolParam,
    ToolParam,
)
from pydantic import BaseModel

from fenic._inference.model_client import (
    FatalException,
    ModelClient,
    TransientException,
)
from fenic._inference.preset_config_manager import (
    BasePresetConfiguration,
    PresetConfigurationManager,
)
from fenic._inference.rate_limit_strategy import SeparatedTokenRateLimitStrategy
from fenic._inference.request_utils import generate_completion_request_key
from fenic._inference.token_counter import TiktokenTokenCounter, Tokenizable
from fenic._inference.types import (
    FenicCompletionsRequest,
    FenicCompletionsResponse,
    LMRequestMessages,
    ResponseUsage,
)
from fenic.core._inference.model_catalog import (
    CompletionModelParameters,
    ModelProvider,
    model_catalog,
)
from fenic.core._resolved_session_config import (
    ResolvedAnthropicModelPreset,
)
from fenic.core.metrics import LMMetrics

TEXT_DELTA = "text_delta"

MESSAGE_STOP = "message_stop"

INPUT_JSON_DELTA = "input_json_delta"

CONTENT_BLOCK_DELTA = "content_block_delta"

EPHEMERAL_CACHE_CONTROL = CacheControlEphemeralParam(type="ephemeral")

@dataclass
class AnthropicPresetConfiguration(BasePresetConfiguration):
    """Configuration for Anthropic model presets.

    Attributes:
        thinking_enabled: Whether thinking/reasoning is enabled for this preset
        thinking_token_budget: Token budget allocated for thinking/reasoning
        thinking_config: Anthropic-specific thinking configuration
    """
    thinking_enabled: bool = False
    thinking_token_budget: int = 0
    thinking_config: anthropic.types.ThinkingConfigParam = field(default_factory= lambda : anthropic.types.ThinkingConfigDisabledParam(type="disabled"))


class AnthropicPresetConfigurationManager(PresetConfigurationManager[ResolvedAnthropicModelPreset, AnthropicPresetConfiguration]):
    """Manages Anthropic-specific preset configurations.

    This class handles the conversion of Fenic preset configurations to
    Anthropic-specific configurations, including thinking/reasoning settings.
    """

    def __init__(self,
                 model_parameters : CompletionModelParameters,
                 preset_configurations: Optional[dict[str, ResolvedAnthropicModelPreset]] = None,
                 default_preset_name: Optional[str] = None):
        """Initialize the Anthropic preset configuration manager.

        Args:
            model_parameters: Parameters for the completion model
            preset_configurations: Dictionary of preset configurations
            default_preset_name: Name of the default preset to use
        """
        self.model_parameters = model_parameters
        super().__init__(preset_configurations, default_preset_name)

    def _process_preset(self, preset: ResolvedAnthropicModelPreset) -> AnthropicPresetConfiguration:
        """Process Anthropic preset configuration.

        Converts a Fenic preset configuration to an Anthropic-specific configuration,
        handling thinking/reasoning settings based on model capabilities.

        Args:
            preset: The Fenic preset configuration to process

        Returns:
            Anthropic-specific preset configuration
        """
        if preset.thinking_token_budget and self.model_parameters.supports_reasoning:
            return AnthropicPresetConfiguration(
                thinking_enabled=True,
                thinking_token_budget=preset.thinking_token_budget,
                thinking_config=anthropic.types.ThinkingConfigEnabledParam(
                    type="enabled",
                    budget_tokens=preset.thinking_token_budget
                )
            )
        else:
            return AnthropicPresetConfiguration()

    def _get_default_configuration(self) -> AnthropicPresetConfiguration:
        """Get default Anthropic configuration.

        Returns:
            Default configuration with thinking disabled
        """
        return AnthropicPresetConfiguration()

class AnthropicBatchCompletionsClient(
    ModelClient[FenicCompletionsRequest, FenicCompletionsResponse]
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
        queue_size: int = 100,
        model: str = "claude-3-5-haiku-latest",
        max_backoffs: int = 10,
        preset_configurations: Optional[dict[str, ResolvedAnthropicModelPreset]] = None,
        default_preset_name: Optional[str] = None,
    ):
        """Initialize the Anthropic batch completions client.

        Args:
            rate_limit_strategy: Strategy for rate limiting requests
            queue_size: Maximum size of the request queue
            model: Anthropic model name to use
            max_backoffs: Maximum number of retry backoffs
            preset_configurations: Dictionary of preset configurations
            default_preset_name: Name of the default preset to use
        """
        super().__init__(
            model=model,
            model_provider=ModelProvider.ANTHROPIC,
            rate_limit_strategy=rate_limit_strategy,
            queue_size=queue_size,
            max_backoffs=max_backoffs,
            token_counter=TiktokenTokenCounter(model_name=model, fallback_encoding="cl100k_base")
        )
        # Apply this factor to the estimated token count to approximate Anthropic's encoding.
        self.tokenizer_adjustment_ratio = 1.05
        self.model = model
        self.sync_client = anthropic.Client()
        self.client = AsyncAnthropic()
        self.metrics = LMMetrics()
        self.output_formatter_tool_name = "output_formatter"
        self.output_formatter_tool_description = "Format the output of the model to correspond strictly to the provided schema."
        self.model_parameters = model_catalog.get_completion_model_parameters(ModelProvider.ANTHROPIC, model)

        # Use the preset configuration manager
        self.preset_manager = AnthropicPresetConfigurationManager(
            model_parameters=self.model_parameters,
            preset_configurations=preset_configurations or {},
            default_preset_name=default_preset_name
        )

    async def make_single_request(self, request: FenicCompletionsRequest) -> Union[
        None, FenicCompletionsResponse, TransientException, FatalException]:
        """Make a single completion request to Anthropic.

        Handles both text and structured output requests, with support for
        thinking/reasoning when enabled. Processes streaming responses and
        extracts usage metrics.

        Args:
            request: The completion request to process

        Returns:
            Completion response, transient exception, or fatal exception
        """
        system_prompt, message_params = self.convert_messages(request.messages)
        preset_configuration = self.preset_manager.get_preset_configuration(request.model_preset)
        request_max_tokens = request.max_completion_tokens + preset_configuration.thinking_token_budget
        messages_creation_payload: dict[str, Any] = {
            "model": self.model,
            "system": [system_prompt],
            "messages": message_params,
            "max_tokens": request_max_tokens,
            "thinking": preset_configuration.thinking_config,
        }
        if request.structured_output:
            tool_param = self.create_response_format_tool(request.structured_output)
            messages_creation_payload.update({"tools": [tool_param]})
            if not preset_configuration.thinking_enabled:
                # Anthropic does not allow forced tool use if thinking is enabled.
                messages_creation_payload.update({"tool_choice": ToolChoiceToolParam(
                    name=self.output_formatter_tool_name,
                    type="tool"
                )})

        if not preset_configuration.thinking_enabled:
            # Anthropic does not allow configuring temperature if thinking is enabled.
            messages_creation_payload.update({"temperature": request.temperature})

        try:
            if request.structured_output:
                content, usage_data = await self._handle_structured_output_streaming_response(messages_creation_payload)
            else:
                content, usage_data = await self._handle_text_streaming_response(messages_creation_payload)
            if content is None:
                return FenicCompletionsResponse(completion="", logprobs=None)
            if usage_data:
                # Extract usage metrics
                num_pre_cached_tokens = usage_data.cache_read_input_tokens
                total_input_tokens = usage_data.input_tokens + usage_data.cache_read_input_tokens
                output_tokens = usage_data.output_tokens

                # Create ResponseUsage object
                usage = ResponseUsage(
                    prompt_tokens=total_input_tokens,
                    completion_tokens=output_tokens,  # For Anthropic, all output tokens are completion tokens
                    total_tokens=total_input_tokens + output_tokens,
                    cached_tokens=num_pre_cached_tokens,
                    thinking_tokens=0  # Anthropic doesn't separate thinking tokens yet
                )

                # Update metrics (existing logic)
                self.metrics.num_cached_input_tokens += num_pre_cached_tokens
                self.metrics.num_uncached_input_tokens += total_input_tokens
                self.metrics.num_output_tokens += output_tokens
                self.metrics.num_requests += 1
                self.metrics.cost += model_catalog.calculate_completion_model_cost(
                    model_provider=ModelProvider.ANTHROPIC,
                    model_name=self.model,
                    uncached_input_tokens=total_input_tokens,
                    cached_input_tokens_read=num_pre_cached_tokens,
                    cached_input_tokens_written=usage_data.cache_creation_input_tokens,
                    output_tokens=output_tokens,
                )
        except (RateLimitError, APITimeoutError, APIConnectionError) as e:  # consider the various APIStatusErrors
            return TransientException(e)
        except AnthropicError as e:
            return FatalException(e)

        return FenicCompletionsResponse(completion=content, logprobs=None, usage=usage)

    async def _handle_text_streaming_response(self, payload: dict[str, Any]) -> tuple[str, Optional[anthropic.types.Usage]]:
        """Handle streaming text response from Anthropic.

        Processes streaming chunks to extract text content and usage data.

        Args:
            payload: The request payload sent to Anthropic

        Returns:
            Tuple of (content, usage_data)
        """
        content = ""
        usage_data: anthropic.types.Usage | None = None
        async with self.client.messages.stream(**payload) as stream:
            async for chunk in stream:
                if chunk.type == CONTENT_BLOCK_DELTA:
                    if chunk.delta.type == TEXT_DELTA:
                        content += chunk.delta.text
                elif chunk.type == MESSAGE_STOP:
                    usage_data = chunk.message.usage if hasattr(chunk.message, 'usage') else None
        return content, usage_data

    async def _handle_structured_output_streaming_response(self, payload: dict[str, Any]) -> tuple[str, Optional[anthropic.types.Usage]]:
        """Handle streaming structured output response from Anthropic.

        Processes streaming chunks to extract JSON content from tool use and usage data.

        Args:
            payload: The request payload sent to Anthropic

        Returns:
            Tuple of (tool_use_content, usage_data)
        """
        tool_use_content: str = ""
        usage_data: anthropic.types.Usage | None = None
        async with self.client.messages.stream(**payload) as stream:
            async for chunk in stream:
                if chunk.type == CONTENT_BLOCK_DELTA:
                    if chunk.delta.type == INPUT_JSON_DELTA:
                        tool_use_content += chunk.delta.partial_json
                elif chunk.type == MESSAGE_STOP:
                    usage_data = chunk.message.usage if hasattr(chunk.message, 'usage') else None
            return tool_use_content, usage_data

    # lightweight caching to allow us to approximate the tokens in a given tool param
    # will replace with something more sophisticated later.
    @functools.cache # noqa: B019
    def estimate_response_format_tokens(self, response_format: type[BaseModel]) -> int:
        """Estimate token count for a response format schema.

        Uses Anthropic's API to count tokens in a tool parameter that represents
        the response format schema. Results are cached for performance.

        Args:
            response_format: Pydantic model class defining the response format

        Returns:
            Estimated token count for the response format
        """
        tool_param = self.create_response_format_tool(response_format)
        approx_tool_tokens = self.sync_client.messages.count_tokens(
            model=self.model, messages=[
                MessageParam(content="user prompt", role="user"),
            ], system="empty",
            tools=[tool_param],
            tool_choice=ToolChoiceToolParam(
                name=self.output_formatter_tool_name,
                type="tool"
            )
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

    def _get_max_output_tokens(self, request: FenicCompletionsRequest) -> int:
        """Get maximum output tokens including thinking budget.

        Args:
            request: The completion request

        Returns:
            Maximum output tokens (completion + thinking budget)
        """
        return request.max_completion_tokens + self.preset_manager.get_preset_configuration(request.model_preset).thinking_token_budget


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
        return math.ceil(super().count_tokens(messages) * self.tokenizer_adjustment_ratio)

    def get_request_key(self, request: FenicCompletionsRequest) -> str:
        """Generate a unique key for the request.

        Args:
            request: The completion request

        Returns:
            Unique request key for caching
        """
        return generate_completion_request_key(request)

    def estimate_tokens_for_request(self, request: FenicCompletionsRequest):
        """Estimate the number of tokens for a request.

        Args:
            request: The request to estimate tokens for

        Returns:
            TokenEstimate: The estimated token usage
        """
        from fenic._inference.rate_limit_strategy import TokenEstimate
        
        # Count input tokens
        input_tokens = self.count_tokens(request.messages)
        input_tokens += self._count_auxiliary_input_tokens(request)
        
        # Estimate output tokens
        output_tokens = self._get_max_output_tokens(request)
        
        return TokenEstimate(
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )

    def get_metrics(self) -> LMMetrics:
        """Get current metrics.

        Returns:
            Current language model metrics
        """
        return self.metrics

    def reset_metrics(self):
        """Reset metrics to initial state."""
        self.metrics = LMMetrics()

    def create_response_format_tool(self, response_format: type[BaseModel]) -> ToolParam:
        """Create a tool parameter for structured output.

        Converts a Pydantic model to an Anthropic tool parameter for
        structured output formatting.

        Args:
            response_format: Pydantic model class defining the response format

        Returns:
            Anthropic tool parameter
        """
        # Convert Pydantic model to JSON schema
        json_schema = response_format.model_json_schema()
        tool_param = ToolParam(
            name=self.output_formatter_tool_name,
            input_schema=json_schema,
            description=self.output_formatter_tool_description,
            cache_control=EPHEMERAL_CACHE_CONTROL
        )
        return tool_param

    def convert_messages(self, messages: LMRequestMessages) -> tuple[TextBlockParam, list[MessageParam]]:
        """Convert Fenic messages to Anthropic format.

        Converts Fenic LMRequestMessages to Anthropic's TextBlockParam and
        MessageParam format, including system prompt and conversation history.

        Args:
            messages: Fenic message format

        Returns:
            Tuple of (system_prompt, message_params)
        """
        system_prompt = TextBlockParam(
            text=messages.system,
            type="text",
            cache_control=EPHEMERAL_CACHE_CONTROL
        )
        message_params: list[anthropic.types.MessageParam] = []
        for example in messages.examples:
            message_params.append(MessageParam(content=example.user, role="user"))
            message_params.append(MessageParam(content=example.assistant, role="assistant"))
        message_params.append(MessageParam(content=messages.user, role="user"))
        return system_prompt, message_params
