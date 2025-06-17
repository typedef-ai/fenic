"""Client for making requests to Google's Gemini model using OpenAI compatibility layer or Vertex AI."""
import os

# Standard library imports
from typing import Literal, Optional, Union

from openai import AsyncOpenAI

from fenic._inference.common_openai.openai_chat_completions_core import (
    OpenAIChatCompletionsCore,
)

# Local imports
from fenic._inference.model_catalog import (
    GOOGLE_GLA_AVAILABLE_MODELS,
    ModelProvider,
    model_catalog,
)
from fenic._inference.model_client import (
    FatalException,
    FenicCompletionsRequest,
    FenicCompletionsResponse,
    ModelClient,
    TokenEstimate,
    TransientException,
    UnifiedTokenRateLimitStrategy,
)
from fenic._inference.token_counter import TiktokenTokenCounter
from fenic.core.metrics import LMMetrics


class GeminiOAIChatCompletionsClient(ModelClient[FenicCompletionsRequest, FenicCompletionsResponse]):
    """Client for making requests to Google's Gemini model using OpenAI compatibility layer or Vertex AI."""

    def __init__(
        self,
        rate_limit_strategy: UnifiedTokenRateLimitStrategy,
        model: GOOGLE_GLA_AVAILABLE_MODELS = "gemini-2.0-flash-lite",
        queue_size: int = 100,
        max_backoffs: int = 10,
        reasoning_effort: Optional[Literal["none", "low", "medium", "high"]] = None,
    ):
        """Initialize the Gemini client with OpenAI compatibility layer or Vertex AI.

        Args:
            rate_limit_strategy: Strategy for handling rate limits
            model: The Gemini model to use
            queue_size: Size of the request queue
            max_backoffs: Maximum number of backoff attempts
            reasoning_effort: Reasoning effort level for thinking models (Gemini 2.5 only)
        """
        token_counter = TiktokenTokenCounter(model_name=model, fallback_encoding="o200k_base")
        model_provider = ModelProvider.GOOGLE_GLA
        super().__init__(
            model=model,
            model_provider=model_provider,
            rate_limit_strategy=rate_limit_strategy,
            queue_size=queue_size,
            max_backoffs=max_backoffs,
            token_counter=token_counter,
        )
        api_key = os.getenv("GEMINI_API_KEY")
        self._client = AsyncOpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=api_key,
        )
        model_parameters = model_catalog.get_completion_model_parameters(model_provider, model)
        additional_parameters = {}
        if reasoning_effort and model_parameters.requires_reasoning_effort:
            additional_parameters["reasoning_effort"] = reasoning_effort

        self._core = OpenAIChatCompletionsCore(
            model=model,
            model_provider=model_provider,
            token_counter=token_counter,
            client=self._client,
            additional_params=additional_parameters,
        )

        self.model = model


    async def make_single_request(
        self, request: FenicCompletionsRequest
    ) -> Union[None, FenicCompletionsResponse, TransientException, FatalException]:
        """Make a single request to the Gemini API.

        Args:
            request: The request to make

        Returns:
            The response from the API or an exception
        """
        return await self._core.make_single_request(request)

    def estimate_tokens_for_request(self, request: FenicCompletionsRequest) -> TokenEstimate:
        """Estimate the number of tokens for a request.

        Args:
            request: The request to estimate tokens for

        Returns:
            TokenEstimate with input and output token counts
        """
        return self._core.estimate_tokens_for_request(request)

    def get_request_key(self, request: FenicCompletionsRequest) -> str:
        """Generate a unique key for request deduplication.

        Args:
            request: The request to generate a key for

        Returns:
            A unique key for the request
        """
        return self._core.get_request_key(request)

    def reset_metrics(self):
        """Reset all metrics to their initial values."""
        self._core.reset_metrics()

    def get_metrics(self) -> LMMetrics:
        """Get the current metrics.

        Returns:
            The current metrics
        """
        return self._core.get_metrics()

