import functools
import hashlib
import json
import math
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
    TextBlock,
    TextBlockParam,
    ToolChoiceToolParam,
    ToolParam,
    ToolUseBlock,
)
from pydantic import BaseModel

from fenic._inference.model_catalog import ModelProvider, model_catalog
from fenic._inference.model_client import (
    FatalException,
    FenicCompletionsRequest,
    FenicCompletionsResponse,
    ModelClient,
    SeparatedTokenRateLimitStrategy,
    TokenEstimate,
    TransientException,
)
from fenic._inference.token_counter import TiktokenTokenCounter, Tokenizable
from fenic._inference.types import LMRequestMessages
from fenic.core.metrics import LMMetrics

EPHEMERAL_CACHE_CONTROL = CacheControlEphemeralParam(type="ephemeral")

class AnthropicBatchCompletionsClient(
    ModelClient[FenicCompletionsRequest, FenicCompletionsResponse]
):
    def __init__(self,
                 rate_limit_strategy: SeparatedTokenRateLimitStrategy,
                 queue_size: int = 100,
                 model: str = "claude-3-5-haiku-latest",
                 max_backoffs: int = 10,
                 ):
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

    def validate_request(self, request: FenicCompletionsRequest) -> Optional[FatalException]:
        """Validate the request before making it to the Anthropic API."""
        if request.temperature < 0 or request.temperature > self.model_parameters.max_temperature:
            return FatalException(ValueError(f"[{self.model_provider}:{self.model}] temperature must be between 0 and {self.model_parameters.max_temperature}"))
        return None

    async def make_single_request(self, request: FenicCompletionsRequest) -> Union[
        None, FenicCompletionsResponse, TransientException, FatalException]:
        if validation_result := self.validate_request(request):
            return validation_result
        system_prompt, message_params = self.convert_messages(request.messages)
        try:
            if request.structured_output:
                tool_param = self.create_response_format_tool(request.structured_output)
                result = await self.client.messages.create(
                    model=self.model,
                    temperature=request.temperature,
                    system=[system_prompt],
                    messages=message_params,
                    max_tokens=request.max_completion_tokens,
                    tools=[tool_param],
                    tool_choice=ToolChoiceToolParam(
                        name=self.output_formatter_tool_name,
                        type="tool"
                    )
                )
            else:
                result = await self.client.messages.create(
                    model=self.model,
                    system=[system_prompt],
                    messages=message_params,
                    max_tokens=request.max_completion_tokens,
                    temperature=request.temperature,
                )
            content = None
            for block in result.content:
                if isinstance(block, TextBlock):
                    content = block.text
                    break
                if isinstance(block, ToolUseBlock):
                    if block.name == self.output_formatter_tool_name:
                        content = block.input
                        if isinstance(content, dict):
                            content = json.dumps(content)
            if content is None:
                return FenicCompletionsResponse(completion="", logprobs=None)
            num_pre_cached_tokens = result.usage.cache_read_input_tokens
            num_uncached_tokens = result.usage.input_tokens + result.usage.cache_read_input_tokens
            self.metrics.num_cached_input_tokens += num_pre_cached_tokens
            self.metrics.num_uncached_input_tokens += num_uncached_tokens
            self.metrics.num_output_tokens += result.usage.output_tokens
            self.metrics.num_requests += 1
            self.metrics.cost += model_catalog.calculate_completion_model_cost(
                model_provider=ModelProvider.ANTHROPIC,
                model_name=self.model,
                uncached_input_tokens=num_uncached_tokens,
                cached_input_tokens_read=num_pre_cached_tokens,
                cached_input_tokens_written=result.usage.cache_creation_input_tokens,
                output_tokens=result.usage.output_tokens,
            )
        except (RateLimitError, APITimeoutError, APIConnectionError) as e:  # consider the various APIStatusErrors
            return TransientException(e)
        except AnthropicError as e:
            return FatalException(e)

        return FenicCompletionsResponse(completion=content, logprobs=None)

    # lightweight caching to allow us to approximate the tokens in a given tool param
    # will replace with something more sophisticated later.
    @functools.cache # noqa: B019
    def estimate_response_format_tokens(self, response_format: type[BaseModel]) -> int:
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

    def estimate_tokens_for_request(self, request: FenicCompletionsRequest) -> TokenEstimate:
        auxiliary_input_tokens = 0
        if request.structured_output:
            auxiliary_input_tokens += self.estimate_response_format_tokens(request.structured_output)
        message_tokens = self.count_tokens(request.messages.to_message_list())
        return TokenEstimate(input_tokens=message_tokens + auxiliary_input_tokens,
                             output_tokens=request.max_completion_tokens)

    # Override default behavior to account for the fact that Anthropic's encoding is slightly different from OpenAI's.
    # This is a rough estimate, but it's good enough for our purposes.
    def count_tokens(self, messages: Tokenizable) -> int:
        return math.ceil(super().count_tokens(messages) * self.tokenizer_adjustment_ratio)

    def get_request_key(self, request: FenicCompletionsRequest) -> Any:
        contents_json = json.dumps(request.messages.to_message_list(), sort_keys=True)
        return hashlib.sha256(contents_json.encode()).hexdigest()[:10]

    def get_metrics(self) -> LMMetrics:
        return self.metrics

    def reset_metrics(self):
        self.metrics = LMMetrics()

    def create_response_format_tool(self, response_format: type[BaseModel]) -> ToolParam:
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
