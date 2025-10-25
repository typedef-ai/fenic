"""Integration tests for LLM response cache."""

import tempfile
from pathlib import Path

import pytest

from fenic._inference.cache.key_generator import CacheKeyGenerator
from fenic._inference.cache.sqlite_cache import SQLiteLLMCache
from fenic._inference.types import (
    FenicCompletionsRequest,
    FenicCompletionsResponse,
    LMRequestMessages,
    ResponseUsage,
)


class TestCacheIntegration:
    """Integration tests for the complete cache workflow."""

    @pytest.fixture
    def cache(self):
        """Create a temporary cache for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        cache = SQLiteLLMCache(
            db_path=db_path,
            ttl_seconds=3600,
            max_size_mb=100,
            namespace="integration_test",
        )

        yield cache

        cache.close()
        Path(db_path).unlink(missing_ok=True)

    def test_end_to_end_workflow(self, cache):
        """Test complete cache workflow from request to response."""
        # Create a request
        messages = LMRequestMessages(
            system="You are a helpful assistant",
            examples=[],
            user="What is the capital of France?",
        )

        request = FenicCompletionsRequest(
            messages=messages,
            max_completion_tokens=100,
            top_logprobs=None,
            structured_output=None,
            temperature=0.7,
        )

        # Generate cache key
        cache_key = CacheKeyGenerator.compute_key(request, "gpt-4o-mini")

        # Verify cache miss
        cached = cache.get(cache_key)
        assert cached is None

        # Create a mock response
        response = FenicCompletionsResponse(
            completion="The capital of France is Paris.",
            logprobs=None,
            usage=ResponseUsage(
                prompt_tokens=20,
                completion_tokens=10,
                total_tokens=30,
            ),
        )

        # Store in cache
        success = cache.set(cache_key, response, "gpt-4o-mini")
        assert success

        # Verify cache hit
        cached = cache.get(cache_key)
        assert cached is not None
        assert cached.completion == "The capital of France is Paris."
        assert cached.model == "gpt-4o-mini"

        # Convert back to Fenic response
        restored = cached.to_fenic_response()
        assert restored.completion == response.completion
        assert restored.usage.prompt_tokens == response.usage.prompt_tokens

    def test_multiple_requests_caching(self, cache):
        """Test caching multiple different requests."""
        requests = []
        responses = []

        # Create 10 different requests
        for i in range(10):
            messages = LMRequestMessages(
                system="You are a helpful assistant",
                examples=[],
                user=f"Question {i}",
            )
            request = FenicCompletionsRequest(
                messages=messages,
                max_completion_tokens=100,
                top_logprobs=None,
                structured_output=None,
                temperature=0.7,
            )
            requests.append(request)

            response = FenicCompletionsResponse(
                completion=f"Answer {i}",
                logprobs=None,
                usage=ResponseUsage(
                    prompt_tokens=10 + i,
                    completion_tokens=5 + i,
                    total_tokens=15 + 2 * i,
                ),
            )
            responses.append(response)

        # Generate cache keys and store all responses
        cache_keys = [
            CacheKeyGenerator.compute_key(req, "gpt-4o-mini") for req in requests
        ]

        for key, response in zip(cache_keys, responses, strict=False):
            cache.set(key, response, "gpt-4o-mini")

        # Batch retrieve all
        cached_responses = cache.get_batch(cache_keys)

        # Verify all are cached
        assert len(cached_responses) == 10
        for i, key in enumerate(cache_keys):
            cached = cached_responses[key]
            assert cached is not None
            assert cached.completion == f"Answer {i}"
            assert cached.prompt_tokens == 10 + i

    def test_cache_hit_rate_tracking(self, cache):
        """Test that cache statistics correctly track hits and misses."""
        messages = LMRequestMessages(
            system="You are helpful",
            examples=[],
            user="Test question",
        )

        request = FenicCompletionsRequest(
            messages=messages,
            max_completion_tokens=100,
            top_logprobs=None,
            structured_output=None,
            temperature=0.7,
        )

        cache_key = CacheKeyGenerator.compute_key(request, "gpt-4o-mini")

        # Initial miss
        cache.get(cache_key)

        # Store
        response = FenicCompletionsResponse(
            completion="Test answer", logprobs=None, usage=None
        )
        cache.set(cache_key, response, "gpt-4o-mini")

        # Two hits
        cache.get(cache_key)
        cache.get(cache_key)

        # One more miss (different key)
        cache.get("nonexistent_key")

        # Check stats
        stats = cache.stats()
        assert stats.hits == 2
        assert stats.misses == 2
        assert stats.stores == 1
        assert stats.hit_rate == 0.5

    def test_deterministic_key_generation(self):
        """Test that identical requests always generate the same cache key."""
        messages = LMRequestMessages(
            system="You are helpful",
            examples=[],
            user="Same question",
        )

        # Create same request twice
        request1 = FenicCompletionsRequest(
            messages=messages,
            max_completion_tokens=100,
            top_logprobs=None,
            structured_output=None,
            temperature=0.7,
        )

        request2 = FenicCompletionsRequest(
            messages=messages,
            max_completion_tokens=100,
            top_logprobs=None,
            structured_output=None,
            temperature=0.7,
        )

        key1 = CacheKeyGenerator.compute_key(request1, "gpt-4o-mini")
        key2 = CacheKeyGenerator.compute_key(request2, "gpt-4o-mini")

        # Should be identical
        assert key1 == key2

    def test_cache_with_different_models(self, cache):
        """Test that same request to different models caches separately."""
        messages = LMRequestMessages(
            system="You are helpful",
            examples=[],
            user="What is 2+2?",
        )

        request = FenicCompletionsRequest(
            messages=messages,
            max_completion_tokens=100,
            top_logprobs=None,
            structured_output=None,
            temperature=0.7,
        )

        # Generate keys for different models
        key_gpt4 = CacheKeyGenerator.compute_key(request, "gpt-4o")
        key_gpt4_mini = CacheKeyGenerator.compute_key(request, "gpt-4o-mini")

        # Keys should be different
        assert key_gpt4 != key_gpt4_mini

        # Store responses for both models
        response_gpt4 = FenicCompletionsResponse(
            completion="Four (GPT-4)", logprobs=None, usage=None
        )
        response_gpt4_mini = FenicCompletionsResponse(
            completion="Four (GPT-4 Mini)", logprobs=None, usage=None
        )

        cache.set(key_gpt4, response_gpt4, "gpt-4o")
        cache.set(key_gpt4_mini, response_gpt4_mini, "gpt-4o-mini")

        # Retrieve and verify
        cached_gpt4 = cache.get(key_gpt4)
        cached_gpt4_mini = cache.get(key_gpt4_mini)

        assert cached_gpt4.completion == "Four (GPT-4)"
        assert cached_gpt4.model == "gpt-4o"
        assert cached_gpt4_mini.completion == "Four (GPT-4 Mini)"
        assert cached_gpt4_mini.model == "gpt-4o-mini"
