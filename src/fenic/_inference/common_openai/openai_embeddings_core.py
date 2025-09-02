"""Core functionality for OpenAI embeddings clients."""

import hashlib
import logging
from typing import List, Union

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    NotFoundError,
    OpenAIError,
    RateLimitError,
)

from fenic._inference.model_client import (
    FatalException,
    TransientException,
)
from fenic._inference.rate_limit_strategy import TokenEstimate
from fenic._inference.token_counter import TokenCounter
from fenic._inference.types import FenicEmbeddingsRequest
from fenic.core._inference.model_catalog import (
    ModelProvider,
    model_catalog,
)
from fenic.core.metrics import RMMetrics

logger = logging.getLogger(__name__)

class OpenAIEmbeddingsCore:
    """Core functionality for OpenAI embeddings clients."""

    def __init__(
        self,
        model: str,
        model_provider: ModelProvider,
        token_counter: TokenCounter,
        client: AsyncOpenAI,
    ):
        """Initialize the OpenAI embeddings client core.

        Args:
            model: The model to use
            model_provider: The provider of the model
            token_counter: Counter for estimating token usage
            client: The OpenAI client
        """
        self._model = model
        self._model_provider = model_provider
        self._token_counter = token_counter
        self._client = client
        self._metrics = RMMetrics()
        self._model_parameters = model_catalog.get_embedding_model_parameters(self._model_provider, self._model)
        self._model_identifier = f"{model_provider.value}/{model}"

    def reset_metrics(self) -> None:
        """Reset the metrics."""
        self._metrics = RMMetrics()

    def get_metrics(self) -> RMMetrics:
        """Get the metrics."""
        return self._metrics

    async def make_single_request(
        self, request: FenicEmbeddingsRequest
    ) -> Union[None, List[float], TransientException, FatalException]:
        """Make a single request to the OpenAI API.

        Args:
            request: The text to embed

        Returns:
            The embedding vector or an exception
        """
        if request.model_profile:
            logger.warning(
                "OpenAI embeddings client does not support model presets."
            )
        try:
            # TODO(rohitrastogi): Embeddings API supports multiple inputs per request.
            # We should use this feature if we're RPM constrained instead of TPM constrained.
            response = await self._client.embeddings.create(
                input=request.doc,
                model=self._model,
            )
            usage = response.usage
            if usage is None:
                # TODO(bcallender): This is a temporary fix to handle the case where the usage is not returned. 
                # We should probably instead require the TokenEstimate to be passed into the client, as this is already 
                # calculated in the ModelClient.
                total_tokens = self.estimate_tokens_for_request(request).input_tokens
            else:
                total_tokens = usage.total_tokens

            self._metrics.num_input_tokens += total_tokens
            self._metrics.num_requests += 1
            self._metrics.cost += model_catalog.calculate_embedding_model_cost(
                model_provider=self._model_provider,
                model_name=self._model,
                billable_inputs=total_tokens
            )
            return response.data[0].embedding

        except (RateLimitError, APITimeoutError, APIConnectionError) as e:
            return TransientException(e)
        
        except NotFoundError as e:
            # During our CI tests, where we run a larger set of tests, we've seen an intermittent 404 error
            # that is not coming from OpenAI.
            # Usually 404 errors should be considered as fatal, as there is no use in retrying them,
            # in this case some of the requests have succeeded, so we're marking this as transient and as such
            # the request will be retried.
            if e.response.headers.get("openai-processing-ms") == "0":
                logger.error("404 with zero processing time → likely routing glitch - Marking this as transient.")
                return TransientException(e)
            else:
                return FatalException(e)

        except OpenAIError as e:
            return FatalException(e)

    def get_request_key(self, request: FenicEmbeddingsRequest) -> str:
        """Generate a unique key for request deduplication.

        Args:
            request: The request to generate a key for

        Returns:
            A unique key for the request
        """
        return hashlib.sha256(request.doc.encode()).hexdigest()[:10]

    def estimate_tokens_for_request(self, request: FenicEmbeddingsRequest) -> TokenEstimate:
        """Estimate the number of tokens for a request.

        Args:
            request: The request to estimate tokens for

        Returns:
            TokenEstimate with input token count
        """
        return TokenEstimate(input_tokens=self._token_counter.count_tokens(request.doc), output_tokens=0)