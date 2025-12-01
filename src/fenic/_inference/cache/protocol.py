"""Protocol and types for LLM response caching."""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Protocol

from fenic._inference.types import FenicCompletionsResponse, ResponseUsage


@dataclass
class CachedResponse:
    """Cached LLM response with metadata.

    Attributes:
        completion: The completion text from the LLM.
        model: The model that generated this response.
        cached_at: Timestamp when this response was cached.
        prompt_tokens: Number of prompt tokens (if available).
        completion_tokens: Number of completion tokens (if available).
        total_tokens: Total number of tokens (if available).
        cached_tokens: Number of cached tokens (default: 0).
        thinking_tokens: Number of thinking tokens (default: 0).
        logprobs: Token log probabilities (if available).
        access_count: Number of times this cached response has been accessed.

    Example:
        Creating a cached response:

        ```python
        cached = CachedResponse(
            completion="Hello, world!",
            model="gpt-4o-mini",
            cached_at=datetime.now(),
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        )
        ```
    """

    completion: str
    model: str
    cached_at: datetime
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    cached_tokens: int = 0
    thinking_tokens: int = 0
    logprobs: Optional[list] = None
    access_count: int = 0

    def to_fenic_response(self) -> FenicCompletionsResponse:
        """Convert cached response to FenicCompletionsResponse.

        Returns:
            FenicCompletionsResponse with cached data and usage information.

        Example:
            ```python
            cached = CachedResponse(
                completion="Hello!",
                model="gpt-4o-mini",
                cached_at=datetime.now(),
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            )
            response = cached.to_fenic_response()
            ```
        """
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

    def get_batch(self, cache_keys: List[str]) -> Dict[str, CachedResponse]:
        """Retrieve multiple cached responses.

        Args:
            cache_keys: List of cache keys to retrieve.

        Returns:
            Dictionary mapping cache keys to responses (only includes hits).

        Note:
            This method should never raise exceptions. All errors should be
            logged and an empty dict or partial results returned.
        """
        ...

    def set(
        self,
        cache_key: str,
        response: FenicCompletionsResponse,
        model: str,
    ) -> bool:
        """Store response in cache.

        Args:
            cache_key: Unique key for the response.
            response: The response to cache.
            model: The model that generated this response.

        Returns:
            True if stored successfully, False otherwise.

        Note:
            This method should never raise exceptions. All errors should be
            logged and False returned.
        """
        ...

    def set_batch(
        self, entries: List[tuple[str, FenicCompletionsResponse, str]]
    ) -> int:
        """Store multiple responses in cache.

        Args:
            entries: List of (cache_key, response, model) tuples.

        Returns:
            Count of successfully stored entries.

        Note:
            This method should never raise exceptions. All errors should be
            logged and partial success count returned.
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
