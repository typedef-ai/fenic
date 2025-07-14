"""Core functionality for OpenAI chat completions clients."""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Union

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    OpenAIError,
    RateLimitError,
)
from openai.types import CompletionUsage

from fenic._inference.model_client import (
    FatalException,
    TransientException,
)
from fenic._inference.preset_config_manager import (
    BasePresetConfiguration,
    PresetConfigurationManager,
)
from fenic._inference.rate_limit_strategy import TokenEstimate
from fenic._inference.request_utils import generate_completion_request_key
from fenic._inference.token_counter import TokenCounter
from fenic._inference.types import (
    FenicCompletionsRequest,
    FenicCompletionsResponse,
    ResponseUsage,
)
from fenic.core._inference.model_catalog import (
    ModelProvider,
    model_catalog,
)
from fenic.core._resolved_session_config import ResolvedOpenAIModelPreset
from fenic.core.metrics import LMMetrics

logger = logging.getLogger(__name__)


@dataclass
class OpenAIPresetConfiguration(BasePresetConfiguration):
    additional_parameters: dict[str, Any] = field(default_factory=dict)
    reasoning_effort: Optional[str] = None
    expected_additional_reasoning_tokens: int = 0


class OpenAIPresetConfigurationManager(
    PresetConfigurationManager[ResolvedOpenAIModelPreset, OpenAIPresetConfiguration]):
    """Manages OpenAI-specific preset configurations."""

    def __init__(self,
                 model_parameters,
                 preset_configurations: Optional[dict[str, ResolvedOpenAIModelPreset]] = None,
                 default_preset_name: Optional[str] = None):
        self.model_parameters = model_parameters
        super().__init__(preset_configurations, default_preset_name)

    def _process_preset(self, preset: ResolvedOpenAIModelPreset) -> OpenAIPresetConfiguration:
        """Process OpenAI preset configuration."""
        additional_parameters = {}
        additional_reasoning_tokens = 0
        if self.model_parameters.supports_reasoning:
            reasoning_effort = preset.reasoning_effort
            # OpenAI does not support disabling reasoning for o-series models, so we default to low
            if not reasoning_effort:
                reasoning_effort = "low"
            additional_parameters["reasoning_effort"] = reasoning_effort
            if reasoning_effort == "low":
                additional_reasoning_tokens = 4096
            elif reasoning_effort == "medium":
                additional_reasoning_tokens = 8192
            elif reasoning_effort == "high":
                additional_reasoning_tokens = 16384

        return OpenAIPresetConfiguration(
            reasoning_effort=preset.reasoning_effort,
            additional_parameters=additional_parameters,
            expected_additional_reasoning_tokens=additional_reasoning_tokens
        )

    def _get_default_configuration(self) -> OpenAIPresetConfiguration:
        """Get default OpenAI configuration."""
        return OpenAIPresetConfiguration()


class OpenAIChatCompletionsCore:
    """Core functionality for OpenAI chat completions clients."""

    def __init__(
            self,
            model: str,
            model_provider: ModelProvider,
            token_counter: TokenCounter,
            client: AsyncOpenAI,
    ):
        """Initialize the OpenAI chat completions client core.

        Args:
            model: The model to use
            model_provider: The provider of the model
            token_counter: Counter for estimating token usage
            client: The OpenAI client
            additional_params: Additional parameters to pass to the API, e.g. {"reasoning_effort": "none"} for thinking models.
        """
        self._model = model
        self._model_provider = model_provider
        self._token_counter = token_counter
        self._client = client
        self._metrics = LMMetrics()
        self._model_parameters = model_catalog.get_completion_model_parameters(self._model_provider, self._model)
        self._model_identifier = f"{model_provider.value}:{model}"

    def reset_metrics(self) -> None:
        """Reset the metrics."""
        self._metrics = LMMetrics()

    def get_metrics(self) -> LMMetrics:
        """Get the metrics."""
        return self._metrics

    async def make_single_request(
        self,
        request: FenicCompletionsRequest,
        preset_configuration: Optional[OpenAIPresetConfiguration] = None
    ) -> Union[None, FenicCompletionsResponse, TransientException, FatalException]:
        """Make a single request to the OpenAI API.

        Args:
            request: The messages to send
            preset_configuration: The optional preset configuration for the request (for passing reasoning_effort)
        Returns:
            The response text or an exception
        """
        try:
            common_params: dict[str, Any] = {
                "model": self._model,
                "messages": request.messages.to_message_list(),
                "max_completion_tokens": request.max_completion_tokens + preset_configuration.expected_additional_reasoning_tokens,
                "n": 1,
            }
            if not preset_configuration or not preset_configuration.reasoning_effort:
                # OpenAI does not allow temperature to be modified for o-series reasoning models.
                common_params["temperature"] = request.temperature

            # Determine if we need logprobs
            if request.top_logprobs:
                common_params.update(
                    {
                        "logprobs": True,
                        "top_logprobs": request.top_logprobs,
                    }
                )
            if preset_configuration:
                common_params.update(preset_configuration.additional_parameters)

            # Choose between parse and create based on structured_output
            if request.structured_output:
                common_params["response_format"] = request.structured_output
                response = await self._client.beta.chat.completions.parse(
                    **common_params
                )
                if response.choices[0].message.refusal:
                    return None
            else:
                response = await self._client.chat.completions.create(**common_params)

            # Extract usage metrics
            usage: CompletionUsage = response.usage

            cached_input_tokens = (
                usage.prompt_tokens_details.cached_tokens
                if usage.prompt_tokens_details
                else 0
            )
            uncached_input_tokens = usage.prompt_tokens - cached_input_tokens
            total_prompt_tokens = usage.prompt_tokens

            # Extract reasoning (thinking) tokens if available
            reasoning_tokens = (
                usage.completion_tokens_details.reasoning_tokens
                if usage.completion_tokens_details
                else 0
            )

            # Separate completion tokens from reasoning tokens
            total_output_tokens = usage.completion_tokens
            completion_tokens = total_output_tokens - reasoning_tokens

            # Create ResponseUsage object
            response_usage = ResponseUsage(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=completion_tokens,  # Actual completion tokens (excluding reasoning)
                total_tokens=total_prompt_tokens + total_output_tokens,
                cached_tokens=cached_input_tokens,
                thinking_tokens=reasoning_tokens  # OpenAI's reasoning tokens
            )

            # Update metrics (existing logic)
            self._metrics.num_cached_input_tokens += cached_input_tokens
            self._metrics.num_uncached_input_tokens += uncached_input_tokens
            self._metrics.num_output_tokens += total_output_tokens
            self._metrics.num_requests += 1

            self._metrics.cost += model_catalog.calculate_completion_model_cost(
                model_provider=self._model_provider,
                model_name=self._model,
                uncached_input_tokens=uncached_input_tokens,
                cached_input_tokens_read=cached_input_tokens,
                output_tokens=total_output_tokens,
            )
            completion = response.choices[0].message.content
            if completion is None:
                logger.warning(
                    f"[{self._model_provider.value}:{self._model}] returned None for completion for {self.get_request_key(request)}: {response}")
            return FenicCompletionsResponse(
                completion=response.choices[0].message.content,
                logprobs=response.choices[0].logprobs,
                usage=response_usage,
            )

        except (RateLimitError, APITimeoutError, APIConnectionError) as e:
            return TransientException(e)

        except OpenAIError as e:
            return FatalException(e)

    def get_request_key(self, request: FenicCompletionsRequest) -> str:
        """Generate a unique key for request deduplication.

        Args:
            request: The request to generate a key for

        Returns:
            A unique key for the request
        """
        return generate_completion_request_key(request)

    def estimate_tokens_for_request(self, request: FenicCompletionsRequest) -> TokenEstimate:
        """Estimate the number of tokens for a request.

        Args:
            request: The request to estimate tokens for

        Returns:
            TokenEstimate with input token count
        """
        return TokenEstimate(
            input_tokens=self._token_counter.count_tokens(request.messages.to_message_list()),
            output_tokens=request.max_completion_tokens
        )
