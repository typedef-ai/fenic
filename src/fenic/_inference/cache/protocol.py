"""Protocol and types for LLM response caching."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Protocol, Union

from fenic._inference.types import (
    FenicCompletionsRequest,
    FenicCompletionsResponse,
    FenicEmbeddingsRequest,
    FenicEmbeddingsResponse,
    ResponseUsage,
)


class ResponseType(str, Enum):
    """Type of cached response.

    Attributes:
        COMPLETION: A completion response from a language model.
        EMBEDDING: An embedding response from an embedding model.
    """

    COMPLETION = "completion"
    EMBEDDING = "embedding"


@dataclass
class CachedResponse:
    """Cached LLM response with metadata.

    Supports both completion and embedding responses. Either `completion` or
    `embedding` must be set, determined by `response_type`.

    Attributes:
        completion: The completion text from the LLM (for completion responses).
        embedding: The embedding vector (for embedding responses).
        response_type: Type of response (ResponseType enum).
        model: The model that generated this response.
        cached_at: Timestamp when this response was cached.
        prompt_tokens: Number of prompt tokens (if available).
        completion_tokens: Number of completion tokens (if available).
        total_tokens: Total number of tokens (if available).
        cached_tokens: Number of cached tokens (default: 0).
        thinking_tokens: Number of thinking tokens (default: 0).
        logprobs: Token log probabilities (if available, completion only).
        access_count: Number of times this cached response has been accessed.

    Example:
        Creating a cached completion response:

        ```python
        from fenic._inference.cache.protocol import ResponseType

        cached = CachedResponse(
            completion="Hello, world!",
            response_type=ResponseType.COMPLETION,
            model="gpt-4o-mini",
            cached_at=datetime.now(),
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        )
        ```

        Creating a cached embedding response:

        ```python
        from fenic._inference.cache.protocol import ResponseType

        cached = CachedResponse(
            embedding=[0.1, 0.2, 0.3],
            response_type=ResponseType.EMBEDDING,
            model="text-embedding-3-small",
            cached_at=datetime.now(),
            prompt_tokens=10,
            total_tokens=10,
        )
        ```
    """

    completion: Optional[str] = None
    embedding: Optional[List[float]] = None
    response_type: ResponseType = ResponseType.COMPLETION
    model: str = ""
    cached_at: datetime = field(default_factory=datetime.now)
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cached_tokens: int = 0
    thinking_tokens: int = 0
    logprobs: Optional[list] = None
    access_count: int = 0

    def to_fenic_completion_response(self) -> FenicCompletionsResponse:
        """Convert cached response to FenicCompletionsResponse.

        Returns:
            FenicCompletionsResponse with cached data and usage information.

        Raises:
            ValueError: If this is not a completion response.

        Example:
            ```python
            from fenic._inference.cache.protocol import ResponseType

            cached = CachedResponse(
                completion="Hello!",
                response_type=ResponseType.COMPLETION,
                model="gpt-4o-mini",
                cached_at=datetime.now(),
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            )
            response = cached.to_fenic_completion_response()
            ```
        """
        if self.response_type != ResponseType.COMPLETION or self.completion is None:
            raise ValueError("This cached response is not a completion response")
        usage = None
        if self.prompt_tokens is not None:
            usage = ResponseUsage(
                prompt_tokens=self.prompt_tokens,
                completion_tokens=self.completion_tokens or 0,
                total_tokens=self.total_tokens or 0,
                cached_tokens=self.cached_tokens,
                thinking_tokens=self.thinking_tokens,
            )

        return FenicCompletionsResponse(
            completion=self.completion,
            logprobs=self.logprobs,
            usage=usage,
        )

    def to_fenic_embedding_response(self) -> List[float]:
        """Convert cached response to embedding list.

        Returns:
            List of floats representing the embedding vector.

        Raises:
            ValueError: If this is not an embedding response.

        Example:
            ```python
            from fenic._inference.cache.protocol import ResponseType

            cached = CachedResponse(
                embedding=[0.1, 0.2, 0.3],
                response_type=ResponseType.EMBEDDING,
                model="text-embedding-3-small",
                cached_at=datetime.now(),
                prompt_tokens=10,
                total_tokens=10,
            )
            embedding = cached.to_fenic_embedding_response()
            ```
        """
        if self.response_type != ResponseType.EMBEDDING or self.embedding is None:
            raise ValueError("This cached response is not an embedding response")
        return self.embedding

    def to_fenic_response(self) -> Union[FenicCompletionsResponse, List[float]]:
        """Convert cached response to appropriate Fenic response type.

        Returns:
            FenicCompletionsResponse for completion responses, or List[float] for embedding responses.

        Example:
            ```python
            cached = CachedResponse(...)
            response = cached.to_fenic_response()
            ```
        """
        if self.response_type == ResponseType.EMBEDDING:
            return self.to_fenic_embedding_response()
        return self.to_fenic_completion_response()


@dataclass
class CacheStats:
    """Cache performance statistics.

    Attributes:
        hits: Number of cache hits.
        misses: Number of cache misses.
        stores: Number of successful cache stores.
        errors: Number of cache errors.
        hit_rate: Cache hit rate (hits / (hits + misses)).
        total_entries: Total number of entries in cache.
        size_bytes: Total size of cache in bytes.

    Example:
        ```python
        stats = cache.stats()
        print(f"Hit rate: {stats.hit_rate:.1%}")
        print(f"Total entries: {stats.total_entries}")
        ```
    """

    hits: int
    misses: int
    stores: int
    errors: int
    hit_rate: float
    total_entries: int = 0
    size_bytes: int = 0


class LLMResponseCache(Protocol):
    """Protocol for LLM response caching.

    All implementations must be thread-safe and handle errors gracefully
    without raising exceptions that could break the LLM pipeline.

    Example:
        Implementing a custom cache:

        ```python
        class MyCache:
            def get(self, cache_key: str) -> Optional[CachedResponse]:
                # Implementation
                pass

            def set(
                self,
                cache_key: str,
                response: FenicCompletionsResponse,
                model: str,
            ) -> bool:
                # Implementation
                pass

            # ... implement other methods
        ```
    """

    def compute_key(
        self,
        request: Union[FenicCompletionsRequest, FenicEmbeddingsRequest],
        model: str,
        profile_hash: Optional[str] = None,
    ) -> str:
        """Compute a deterministic cache key for a request.

        Args:
            request: The request object (e.g. FenicCompletionsRequest or FenicEmbeddingsRequest).
            model: The model name.
            profile_hash: Optional hash of the resolved model profile configuration.

        Returns:
            A unique cache key string.
        """
        ...

    def get(self, cache_key: str) -> Optional[CachedResponse]:
        """Retrieve a cached response.
        Args:
            cache_key: Unique key for the cached response.

        Returns:
            CachedResponse if found and not expired, None otherwise.

        Note:
            This method should never raise exceptions. All errors should be
            logged and None returned.
        """
        ...

    def get_batch(self, cache_keys: List[str]) -> Dict[str, Optional[CachedResponse]]:
        """Retrieve multiple cached responses.

        Args:
            cache_keys: List of cache keys to retrieve.

        Returns:
            Dictionary mapping cache keys to responses. Keys with no cached
            response should map to None.

        Note:
            This method should never raise exceptions. All errors should be
            logged and an empty dict or partial results returned.
        """
        ...

    def set(
        self,
        cache_key: str,
        response: Union[FenicCompletionsResponse, FenicEmbeddingsResponse],
        model: str,
    ) -> bool:
        """Store response in cache.

        Args:
            cache_key: Unique key for the response.
            response: The response to cache (completion or embedding).
            model: The model that generated this response.

        Returns:
            True if stored successfully, False otherwise.

        Note:
            This method should never raise exceptions. All errors should be
            logged and False returned.
        """
        ...

    def set_batch(
        self, entries: List[tuple[str, Union[FenicCompletionsResponse, FenicEmbeddingsResponse], str]]
    ) -> int:
        """Store multiple responses in cache.

        Args:
            entries: List of (cache_key, response, model) tuples. Responses can be
                either FenicCompletionsResponse or FenicEmbeddingsResponse.

        Returns:
            Count of successfully stored entries.

        Note:
            This method should never raise exceptions. All errors should be
            logged and partial success count returned.
        """
        ...

    def delete(self, cache_key: str) -> bool:
        """Delete cached entry.

        Args:
            cache_key: Key of entry to delete.

        Returns:
            True if found and deleted, False otherwise.
        """
        ...

    def clear(self) -> int:
        """Clear all entries in cache namespace.

        Returns:
            Number of entries cleared.
        """
        ...

    def stats(self) -> CacheStats:
        """Get cache performance statistics.

        Returns:
            CacheStats with current metrics.
        """
        ...

    def close(self) -> None:
        """Release cache resources.

        Should be called when the cache is no longer needed.
        """
        ...
