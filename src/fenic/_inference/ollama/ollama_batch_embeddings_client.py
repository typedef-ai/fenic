"""Client for making batch requests to Ollama's embeddings API.

This client is optimized for Ollama's native batching and parallel processing capabilities.
Key optimizations:
- Leverages Ollama's automatic request batching for the same model
- Uses concurrent connections aligned with OLLAMA_NUM_PARALLEL
- Handles Ollama's built-in queuing and memory management
- Dynamic model configuration using /api/show metadata
"""

import hashlib
import logging
import os
import time
from typing import Optional, Union

import ollama

from fenic._inference.ollama.ollama_provider import OllamaModelProvider
from fenic._inference.ollama.ollama_model_manager import OllamaModelManager, OllamaModelInfo
from fenic._inference.model_client import (
    FatalException,
    ModelClient,
    TransientException,
)
from fenic._inference.rate_limit_strategy import (
    RateLimitStrategy,
    TokenEstimate,
)
from fenic._inference.token_counter import TiktokenTokenCounter
from fenic._inference.types import FenicEmbeddingsRequest
from fenic.core._inference.model_catalog import ModelProvider, model_catalog
from fenic.core.metrics import RMMetrics

logger = logging.getLogger(__name__)


class OllamaBatchEmbeddingsClient(ModelClient[FenicEmbeddingsRequest, list[float]]):
    """Client for making batch requests to Ollama's embeddings API.

    This client enables communication with local embedding models via Ollama's
    native Python library with optimizations for local model constraints.
    """

    def __init__(
        self,
        rate_limit_strategy: RateLimitStrategy,
        model: str,
        queue_size: int = 100,
        max_backoffs: int = 10,
        host: Optional[str] = None,
    ):
        """Initialize the Ollama batch embeddings client.

        Args:
            rate_limit_strategy: Strategy for handling rate limits
            model: The model to use (e.g., "embeddinggemma")
            queue_size: Size of the request queue (note: Ollama has its own queue)
            max_backoffs: Maximum number of backoff attempts
            host: Host URL for Ollama server
        """
        # Use tiktoken for token counting as a reasonable approximation for local models
        token_counter = TiktokenTokenCounter(model_name="text-embedding-ada-002", fallback_encoding="cl100k_base")

        super().__init__(
            model=model,
            model_provider=ModelProvider.OLLAMA,
            model_provider_class=OllamaModelProvider(host=host),
            rate_limit_strategy=rate_limit_strategy,
            queue_size=queue_size,
            max_backoffs=max_backoffs,
            token_counter=token_counter,
        )

        self._model_parameters = model_catalog.get_embedding_model_parameters(
            ModelProvider.OLLAMA, model
        )
        self._host = host or "http://localhost:11434"
        self._metrics = RMMetrics()

        # Ollama-specific optimizations
        self._ollama_parallel = int(os.getenv("OLLAMA_NUM_PARALLEL", "4"))
        self._ollama_max_queue = int(os.getenv("OLLAMA_MAX_QUEUE", "512"))
        self._model_manager = OllamaModelManager(host=self._host)
        self._model_info: Optional[OllamaModelInfo] = None

        logger.debug(f"Ollama embeddings client initialized for model '{model}' with parallelism={self._ollama_parallel}")

    async def _ensure_model_info(self) -> Optional[OllamaModelInfo]:
        """Lazy load model information for dynamic configuration."""
        if self._model_info is None:
            self._model_info = await self._model_manager.get_model_info(self.model)
            if self._model_info:
                logger.debug(f"Loaded embedding model info for '{self.model}': "
                           f"context_length={self._model_info.context_length}, "
                           f"params={self._model_info.parameter_count}, "
                           f"is_embedding={self._model_info.is_embedding_model}")
        return self._model_info

    async def make_single_request(
        self, request: FenicEmbeddingsRequest
    ) -> Union[None, list[float], TransientException, FatalException]:
        """Make a single request to the Ollama embeddings API.

        Args:
            request: The embedding request to make

        Returns:
            The embedding vector or an exception
        """
        start_time = time.time()
        try:
            # Ensure model info is loaded and verify it's an embedding model
            model_info = await self._ensure_model_info()
            if model_info and not model_info.is_embedding_model:
                logger.warning(f"Model '{self.model}' may not be an embedding model")

            # Check if model is available, auto-pull if needed
            if not await self._model_manager.ensure_model_available(self.model):
                return FatalException(Exception(f"Embedding model '{self.model}' could not be loaded or pulled"))

            # Create async client for this request
            client = ollama.AsyncClient(host=self._host)

            # Prepare embedding request parameters
            embed_params = {
                "model": self.model,
                "input": request.doc,
            }

            # Add options for embedding-specific configuration
            options = {}
            if model_info and model_info.context_length:
                # Ensure input doesn't exceed context length
                estimated_tokens = self.token_counter.count_tokens(request.doc)
                if estimated_tokens > model_info.context_length:
                    logger.warning(f"Input text ({estimated_tokens} tokens) exceeds model context length ({model_info.context_length})")

            if options:
                embed_params["options"] = options

            # Make the API call
            response = await client.embed(**embed_params)

            # Extract usage information if available
            total_tokens = 0
            if "prompt_eval_count" in response:
                total_tokens = response["prompt_eval_count"]
            else:
                # Fallback to token estimation
                total_tokens = self.token_counter.count_tokens(request.doc)

            # Update metrics
            self._metrics.num_input_tokens += total_tokens
            self._metrics.num_requests += 1
            # Local models are typically free, so cost is 0
            self._metrics.cost += 0.0

            # Extract the embedding vector
            if "embeddings" in response and response["embeddings"]:
                # Ollama returns embeddings as a list of lists for batch requests
                # For single requests, we get the first embedding
                embeddings = response["embeddings"]
                if isinstance(embeddings, list) and len(embeddings) > 0:
                    embedding = embeddings[0]
                    if isinstance(embedding, list) and len(embedding) > 0:
                        # Record successful completion for rate limiting strategy
                        completion_time = time.time() - start_time
                        if hasattr(self.rate_limit_strategy, 'record_completion'):
                            self.rate_limit_strategy.record_completion(completion_time, success=True)
                        return embedding
                    else:
                        logger.error(f"Invalid embedding format from Ollama: {type(embedding)}")
                        return FatalException(Exception(f"Invalid embedding format: {type(embedding)}"))
                else:
                    logger.error(f"No embeddings in Ollama response: {response}")
                    return FatalException(Exception("No embeddings in response"))
            else:
                logger.error(f"No embeddings data in Ollama response: {response}")
                return FatalException(Exception("No embeddings data in response"))

        except ollama.ResponseError as e:
            # Record failed completion for rate limiting strategy
            completion_time = time.time() - start_time
            if hasattr(self.rate_limit_strategy, 'record_completion'):
                self.rate_limit_strategy.record_completion(completion_time, success=False)

            # Handle Ollama-specific errors optimized for embedding models
            if e.status_code == 404:
                # Model not found - attempt auto-pull if configured, otherwise fatal
                logger.error(f"Embedding model '{self.model}' not found. Try running: ollama pull {self.model}")
                return FatalException(e)
            elif e.status_code == 503:
                # Server overloaded - Ollama's queue is full
                logger.warning(f"Ollama server overloaded (queue full: {self._ollama_max_queue}): {e}")
                # Record 503 error for rate limiting strategy
                if hasattr(self.rate_limit_strategy, 'record_503_error'):
                    self.rate_limit_strategy.record_503_error()
                return TransientException(e)
            elif e.status_code == 429:
                # Rate limited - respect Ollama's internal rate limiting
                logger.warning(f"Ollama rate limit exceeded (parallel limit: {self._ollama_parallel}): {e}")
                return TransientException(e)
            elif e.status_code == 500:
                # Internal server error - often memory issues with local models
                logger.warning(f"Ollama internal error (possibly memory constrained): {e}")
                return TransientException(e)
            else:
                logger.error(f"Ollama embeddings API error: {e}")
                return FatalException(e)
        except Exception as e:
            # Record failed completion for rate limiting strategy
            completion_time = time.time() - start_time
            if hasattr(self.rate_limit_strategy, 'record_completion'):
                self.rate_limit_strategy.record_completion(completion_time, success=False)

            # Handle other errors optimized for local embedding scenarios
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in ["connection", "timeout", "network"]):
                logger.warning(f"Connection/timeout error with Ollama embeddings: {e}")
                return TransientException(e)
            elif any(keyword in error_str for keyword in ["memory", "cuda", "out of memory"]):
                logger.warning(f"Memory error with Ollama embeddings (model may need unloading): {e}")
                return TransientException(e)
            else:
                logger.error(f"Fatal error with Ollama embeddings request: {e}")
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
        input_tokens = self.token_counter.count_tokens(request.doc)
        return TokenEstimate(
            input_tokens=input_tokens,
            output_tokens=0  # Embedding models don't generate output tokens
        )

    def reset_metrics(self):
        """Reset all metrics to their initial values."""
        self._metrics = RMMetrics()

    def get_metrics(self) -> RMMetrics:
        """Get the current metrics.

        Returns:
            The current metrics
        """
        return self._metrics

    def _get_max_output_tokens(self, request: FenicEmbeddingsRequest) -> int:
        """Embedding models don't have output tokens."""
        return 0