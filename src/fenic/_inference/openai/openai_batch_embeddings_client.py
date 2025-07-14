"""Client for making batch requests to OpenAI's embeddings API."""

from typing import Union

from openai import AsyncOpenAI

from fenic._inference.common_openai.openai_embeddings_core import (
    OpenAIEmbeddingsCore,
)
from fenic._inference.model_client import (
    FatalException,
    ModelClient,
    RequestT,
    TransientException,
)
from fenic._inference.rate_limit_strategy import (
    TokenEstimate,
    UnifiedTokenRateLimitStrategy,
)
from fenic._inference.token_counter import TiktokenTokenCounter
from fenic.core._inference.model_catalog import ModelProvider
from fenic.core.metrics import RMMetrics


class OpenAIBatchEmbeddingsClient(ModelClient[str, list[float]]):
    """Client for making batch requests to OpenAI's embeddings API."""

    def __init__(
        self,
        rate_limit_strategy: UnifiedTokenRateLimitStrategy,
        queue_size: int = 500,
        model: str = "text-embedding-3-small",
        max_backoffs: int = 10,
    ):
        """Initialize the OpenAI batch embeddings client.

        Args:
            rate_limit_strategy: Strategy for handling rate limits
            queue_size: Size of the request queue
            model: The model to use
            max_backoffs: Maximum number of backoff attempts
        """
        super().__init__(
            model=model,
            model_provider=ModelProvider.OPENAI,
            rate_limit_strategy=rate_limit_strategy,
            queue_size=queue_size,
            max_backoffs=max_backoffs,
            token_counter=TiktokenTokenCounter(model_name=model),
        )
        self._core = OpenAIEmbeddingsCore(
            model=model,
            model_provider=ModelProvider.OPENAI,
            token_counter=TiktokenTokenCounter(model_name=model),
            client=AsyncOpenAI(),
        )

    async def make_single_request(
        self, request: str
    ) -> Union[None, list[float], TransientException, FatalException]:
        """Make a single request to the OpenAI API.

        Args:
            request: The request to make

        Returns:
            The response from the API or an exception
        """
        return await self._core.make_single_request(request)

    def get_request_key(self, request: str) -> str:
        """Generate a unique key for request deduplication.

        Args:
            request: The request to generate a key for

        Returns:
            A unique key for the request
        """
        return self._core.get_request_key(request)

    def estimate_tokens_for_request(self, request: str) -> TokenEstimate:
        """Estimate the number of tokens for a request. Overriding the behavior in the base class
           as Embedding models do not generate any output tokens.

        Args:
            request: The request to estimate tokens for

        Returns:
            TokenEstimate with input token count
        """
        return self._core.estimate_tokens_for_request(request)

    def reset_metrics(self):
        """Reset all metrics to their initial values."""
        self._core.reset_metrics()

    def get_metrics(self) -> RMMetrics:
        """Get the current metrics.

        Returns:
            The current metrics
        """
        return self._core.get_metrics()

    def _get_max_output_tokens(self, request: RequestT) -> int:
        return 0