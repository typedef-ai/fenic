"""End-to-end integration tests for LLM response cache with real semantic operations."""

import uuid

import pytest

from fenic import col, semantic
from fenic._backends.local.session_state import LocalSessionState
from fenic.api.session import SemanticConfig, Session, SessionConfig
from fenic.api.session.config import LLMResponseCacheConfig
from tests.conftest import (
    LANGUAGE_MODEL_NAME_ARG,
    LANGUAGE_MODEL_PROVIDER_ARG,
    ModelProvider,
    configure_language_model,
)


class TestCacheEndToEndSemantic:
    """Integration tests with real semantic operations to verify cache behavior."""

    @pytest.fixture
    def session_with_cache(self, tmp_path, request):
        """Create a Session with caching enabled for testing."""
        language_model_provider = ModelProvider(
            request.config.getoption(LANGUAGE_MODEL_PROVIDER_ARG)
        )
        model_name = request.config.getoption(LANGUAGE_MODEL_NAME_ARG)
        language_model = configure_language_model(language_model_provider, model_name)
        
        app_name = f"cache_test_{uuid.uuid4().hex[:8]}"
        
        config = SessionConfig(
            app_name=app_name,
            db_path=tmp_path,
            semantic=SemanticConfig(
                language_models={"test_model": language_model},
                default_language_model="test_model",
                llm_response_cache=LLMResponseCacheConfig(
                    enabled=True,
                    ttl="1h",
                    max_size_mb=100,
                    namespace="test",
                ),
            ),
        )
        
        session = Session.get_or_create(config)
        yield session
        session.stop()

    def test_semantic_map_cache_hits(self, session_with_cache):
        """Test that repeated semantic.map operations result in expected cache hits."""
        # Create initial dataframe with names
        df = session_with_cache.create_dataframe({
            "name": ["Alice", "Bob", "Charlie"],
            "city": ["New York", "Los Angeles", "Chicago"],
        })
        
        # Access the cache from session state
        session_state: LocalSessionState = session_with_cache._session_state
        cache = session_state._llm_cache
        assert cache is not None, "Cache should be enabled"
        
        # First pass: Execute semantic map operation (should be cache misses)
        df_result_1 = df.select(
            col("name"),
            semantic.map(
                "What state is {{city}} in? Answer in one word.",
                city=col("city"),
            ).alias("state"),
        )
        result_1 = df_result_1.collect("polars")
        
        # Verify we got results (collect returns QueryResult with data)
        assert len(result_1.data) == 3
        assert "state" in result_1.data.columns
        
        # Check initial cache stats
        stats_1 = cache.stats()
        initial_misses = stats_1.misses
        initial_stores = stats_1.stores
        initial_hits = stats_1.hits
        
        # Should have made requests (misses) and stored them
        assert initial_misses > 0, "Should have cache misses on first pass"
        assert initial_stores > 0, "Should have stored responses"
        assert initial_hits == 0, "Should have no hits on first pass"
        
        # Second pass: Execute EXACT same operation (should be cache hits)
        df_result_2 = df.select(
            col("name"),
            semantic.map(
                "What state is {{city}} in? Answer in one word.",
                city=col("city"),
                request_timeout=120,
            ).alias("state"),
        )
        result_2 = df_result_2.collect("polars")
        
        # Verify we got the same results (should be from cache)
        assert len(result_2.data) == 3
        assert "state" in result_2.data.columns
        
        # Check cache stats after second pass
        stats_2 = cache.stats()
        
        # Should have more hits now (one for each row that was cached)
        assert stats_2.hits == df.count(), "Should have cache hits on second pass"
        assert stats_2.misses == initial_misses, "Misses should not increase"
        assert stats_2.stores == initial_stores, "Stores should not increase"
        assert stats_2.hit_rate == 0.5, "Hit rate should be 50% since we have 6 total stats entries (3 hits, 3 misses)"
        
        # Verify results are the same (cache is working correctly)
        # We can't guarantee exact string match due to LLM non-determinism,
        # but the structure should be the same
        assert len(result_1.data) == len(result_2.data)
        
        # Third pass: Execute with same data again to verify cache is persistent
        df_result_3 = df.select(
            col("name"),
            semantic.map(
                "What state is {{city}} in? Answer in one word.",
                city=col("city"),
                request_timeout=30,
            ).alias("state"),
        )
        df_result_3.collect("polars")
        
        stats_3 = cache.stats()
        
        # Should have even more hits now
        assert stats_3.hits == df.count() * 2, "Should have cache hits on third pass"
        assert stats_3.misses == initial_misses, "Misses should remain the same"
        assert stats_3.hit_rate == (2.0 / 3.0), "Hit rate should be 66.66% since we have 9 total stats entries (6 hits, 3 misses)"

    def test_semantic_map_different_prompts(self, session_with_cache):
        """Test that different prompts generate different cache keys."""
        df = session_with_cache.create_dataframe({
            "name": ["Alice", "Bob"],
        })
        
        cache = session_with_cache._session_state._llm_cache
        assert cache is not None
        
        # First operation with prompt 1
        df.select(
            semantic.map(
                "What is a nickname for {{name}}?",
                name=col("name"),
            ).alias("nickname"),
        ).collect("polars")
        
        stats_1 = cache.stats()
        initial_misses_1 = stats_1.misses
        
        # Second operation with DIFFERENT prompt (should be new misses)
        df.select(
            semantic.map(
                "What is the capital of the state where {{name}} lives?",
                name=col("name"),
            ).alias("capital"),
        ).collect("polars")
        
        stats_2 = cache.stats()
        
        # Should have more misses because different prompt = different cache key
        assert stats_2.misses > initial_misses_1, "Different prompts should generate different cache keys"
        
        # Third operation: Repeat first prompt (should hit cache)
        df.select(
            semantic.map(
                "What is a nickname for {{name}}?",
                name=col("name"),
            ).alias("nickname2"),
        ).collect("polars")
        
        stats_3 = cache.stats()
        
        # Should have more hits now (reused first prompt's cache)
        assert stats_3.hits > stats_2.hits, "Repeating same prompt should hit cache"
        assert stats_3.misses == stats_2.misses, "Misses should not increase for same prompt"
