"""Client for making batch requests to OpenAI's chat completions API."""
import logging
from typing import Optional, Union

from openai import AsyncOpenAI

from fenic._inference.common_openai.openai_chat_completions_core import (
    OpenAIChatCompletionsCore,
    OpenAIPresetConfigurationManager,
)
from fenic._inference.model_client import (
    FatalException,
    ModelClient,
    TransientException,
)
from fenic._inference.rate_limit_strategy import (
    TokenEstimate,
    UnifiedTokenRateLimitStrategy,
)
from fenic._inference.token_counter import TiktokenTokenCounter
from fenic._inference.types import FenicCompletionsRequest, FenicCompletionsResponse
from fenic.core._inference.model_catalog import ModelProvider, model_catalog
from fenic.core._resolved_session_config import ResolvedOpenAIModelPreset
from fenic.core.metrics import LMMetrics

logger = logging.getLogger(__name__)


class OpenAIBatchChatCompletionsClient(ModelClient[FenicCompletionsRequest, FenicCompletionsResponse]):
    """Client for making batch requests to OpenAI's chat completions API."""

    def __init__(
        self,
        rate_limit_strategy: UnifiedTokenRateLimitStrategy,
        queue_size: int = 100,
        model: str = "gpt-4.1-nano",
        max_backoffs: int = 10,
        preset_configurations: Optional[dict[str, ResolvedOpenAIModelPreset]] = None,
        default_preset_name: Optional[str] = None,
    ):
        """Initialize the OpenAI batch chat completions client.

        Args:
            rate_limit_strategy: Strategy for handling rate limits
            queue_size: Size of the request queue
            model: The model to use
            max_backoffs: Maximum number of backoff attempts
            preset_configurations: Dictionary of preset configurations
            default_preset_name: Default preset to use when none specified
        """
        super().__init__(
            model=model,
            model_provider=ModelProvider.OPENAI,
            rate_limit_strategy=rate_limit_strategy,
            queue_size=queue_size,
            max_backoffs=max_backoffs,
            token_counter=TiktokenTokenCounter(model_name=model, fallback_encoding="o200k_base"),
        )
        self.model_parameters = model_catalog.get_completion_model_parameters(ModelProvider.OPENAI, model)

        # Use the preset configuration manager
        self.preset_manager = OpenAIPresetConfigurationManager(
            model_parameters=self.model_parameters,
            preset_configurations=preset_configurations,
            default_preset_name=default_preset_name
        )

        self._core = OpenAIChatCompletionsCore(
            model=model,
            model_provider=ModelProvider.OPENAI,
            token_counter=TiktokenTokenCounter(model_name=model, fallback_encoding="o200k_base"),
            client=AsyncOpenAI()
        )

    async def make_single_request(
        self, request: FenicCompletionsRequest
    ) -> Union[None, FenicCompletionsResponse, TransientException, FatalException]:
        """Make a single request to the OpenAI API.

        Args:
            request: The request to make

        Returns:
            The response from the API or an exception
        """
        # Get preset-specific parameters
        return await self._core.make_single_request(request, self.preset_manager.get_preset_configuration(request.model_preset))

    def get_request_key(self, request: FenicCompletionsRequest) -> str:
        """Generate a unique key for request deduplication.

        Args:
            request: The request to generate a key for

        Returns:
            A unique key for the request
        """
        return self._core.get_request_key(request)

    def estimate_tokens_for_request(self, request: FenicCompletionsRequest) -> TokenEstimate:
        """Estimate the number of tokens for a request.

        Args:
            request: The request to estimate tokens for

        Returns:
            TokenEstimate: The estimated token usage
        """
        return self._core.estimate_tokens_for_request(request)

    def reset_metrics(self):
        """Reset all metrics to their initial values."""
        self._core.reset_metrics()

    def get_metrics(self) -> LMMetrics:
        """Get the current metrics.

        Returns:
            The current metrics
        """
        return self._core.get_metrics()

    def _get_max_output_tokens(self, request: FenicCompletionsRequest) -> int:
        """Conservative estimate: max_completion_tokens + reasoning effort-based thinking tokens."""
        base_tokens = request.max_completion_tokens

        # Get preset-specific reasoning effort
        preset_config = self.preset_manager.get_preset_configuration(request.model_preset)
        return base_tokens + preset_config.expected_additional_reasoning_tokens