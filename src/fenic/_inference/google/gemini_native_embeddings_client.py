import hashlib
from typing import Union, Literal

from google import genai
from google.genai.errors import ServerError, ClientError
from google.genai.types import EmbedContentResponse
from numpy.ma.core import outer

from fenic import RMMetrics
from fenic._inference import token_counter
from fenic._inference.model_catalog import ModelProvider, model_catalog
from fenic._inference.model_client import ModelClient, UnifiedTokenRateLimitStrategy, TokenEstimate, TransientException, \
    FatalException
from fenic._inference.token_counter import TokenCounter, TiktokenTokenCounter


class GeminiNativeEmbeddingsClient(ModelClient[str, list[float]]):
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
            token_counter=TiktokenTokenCounter(model_name=model, fallback_encoding="o200k_base"),
        )
        self._model = model
        self._model_provider = ModelProvider.GOOGLE_VERTEX
        self._token_counter = token_counter
        self._base_client = genai.Client(vertexai=True)
        self._client = self._base_client.aio
        self._metrics = RMMetrics()
        self._model_parameters = model_catalog.get_embedding_model_parameters(self._model_provider, self._model)
        self._model_identifier = f"{self._model_provider.value}:{model}"

    def reset_metrics(self) -> None:
        """Reset the metrics."""
        self._metrics = RMMetrics()

    def get_metrics(self) -> RMMetrics:
        """Get the metrics."""
        return self._metrics

    async def make_single_request(
        self, request: str
    ) -> Union[None, list[float], TransientException, FatalException]:
        """Make a single request to the OpenAI API.

        Args:
            request: The text to embed

        Returns:
            The embedding vector or an exception
        """
        try:
            # TODO(rohitrastogi): Embeddings API supports multiple inputs per request.
            # We should use this feature if we're RPM constrained instead of TPM constrained.
            response: EmbedContentResponse = await self._client.models.embed_content(
                model=self._model,
                contents=request,
            )
            usage = response.metadata
            if usage is None:
                total_characters = len(request)
            else:
                total_characters = usage.billable_character_count

            self._metrics.num_input_tokens += total_characters
            self._metrics.num_requests += 1
            self._metrics.cost += model_catalog.calculate_embedding_model_cost(
                model_provider=self._model_provider,
                model_name=self._model,
                billable_inputs=total_characters
            )
            return response.embeddings[0].values

        except ServerError as e:
            # Treat quota/timeouts as transient (retryable)
            return TransientException(e)
        except ClientError as e:
            if e.code == 429:
                return TransientException(e)
            else:
                return FatalException(e)
        except Exception as e:  # noqa: BLE001 – catch-all mapped to Fatal
            return FatalException(e)

    def get_request_key(self, request: str) -> str:
        """Generate a unique key for request deduplication.

        Args:
            request: The request to generate a key for

        Returns:
            A unique key for the request
        """
        return hashlib.sha256(request.encode()).hexdigest()[:10]

    def estimate_tokens_for_request(self, request: str) -> TokenEstimate:
        """Estimate the number of tokens for a request.

        Args:
            request: The request to estimate tokens for

        Returns:
            TokenEstimate with input token count. For Google, embedding costs are character based,
            so we are returning the number of characters in the request.
        """
        return TokenEstimate(input_tokens=len(request), output_tokens=0)