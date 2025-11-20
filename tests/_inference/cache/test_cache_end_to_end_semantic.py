"""End-to-end integration tests for LLM response cache with real semantic operations."""

import uuid

import pytest

from fenic import col, semantic
from fenic._backends.local.session_state import LocalSessionState
from fenic.api.session import SemanticConfig, Session, SessionConfig
from fenic.api.session.config import LLMResponseCacheConfig
from fenic.core._inference.model_catalog import model_catalog
from fenic.core.types.semantic import ModelAlias
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
            ).alias("state"),
        )
        result_2 = df_result_2.collect("polars")
        
        # Verify we got the same results (should be from cache)
        assert len(result_2.data) == 3
        assert "state" in result_2.data.columns
        
        # Check cache stats after second pass
        stats_2 = cache.stats()
        
        # Should have more hits now (one for each row that was cached)
        assert stats_2.hits > initial_hits, "Should have cache hits on second pass"
        assert stats_2.misses == initial_misses, "Misses should not increase"
        assert stats_2.stores == initial_stores, "Stores should not increase"
        assert stats_2.hit_rate > 0, "Hit rate should be positive"
        
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
            ).alias("state"),
        )
        df_result_3.collect("polars")
        
        stats_3 = cache.stats()
        
        # Should have even more hits now
        assert stats_3.hits > stats_2.hits, "Should have more hits on third pass"
        assert stats_3.misses == initial_misses, "Misses should remain the same"
        assert stats_3.hit_rate > stats_2.hit_rate, "Hit rate should improve"

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

    def test_semantic_map_profile_content_change_cache_miss(self, tmp_path, request):
        """Test that changing profile contents with same profile name results in cache misses."""
        language_model_provider = ModelProvider(
            request.config.getoption(LANGUAGE_MODEL_PROVIDER_ARG)
        )
        model_name = request.config.getoption(LANGUAGE_MODEL_NAME_ARG)
        model_parameters = model_catalog.get_completion_model_parameters(
            language_model_provider, model_name
        )

        # Skip test if model doesn't support profiles
        if not model_parameters.supports_profiles:
            pytest.skip(f"Model {model_name} does not support profiles")

        # Determine profile configurations based on provider capabilities
        from fenic.api.session.config import (
            AnthropicLanguageModel,
            GoogleDeveloperLanguageModel,
            GoogleVertexLanguageModel,
            OpenAILanguageModel,
        )

        # Use same app_name and tmp_path to ensure cache sharing
        app_name = "cache_test_profile_change"
        
        # Create session 1 with profile "test_profile" using first config
        if language_model_provider == ModelProvider.OPENAI:
            if not model_parameters.supports_reasoning:
                pytest.skip(f"Model {model_name} does not support reasoning profiles")
            
            if model_parameters.supports_minimal_reasoning:
                # Use reasoning_effort differences for gpt5 models
                profile_config_1 = OpenAILanguageModel.Profile(reasoning_effort="low")
                profile_config_2 = OpenAILanguageModel.Profile(reasoning_effort="medium")
            else:
                # Use reasoning_effort differences for o-series models
                profile_config_1 = OpenAILanguageModel.Profile(reasoning_effort="low")
                profile_config_2 = OpenAILanguageModel.Profile(reasoning_effort="medium")
            
            language_model_1 = OpenAILanguageModel(
                model_name=model_name,
                rpm=500,
                tpm=100_000,
                profiles={"test_profile": profile_config_1},
                default_profile="test_profile",
            )
            language_model_2 = OpenAILanguageModel(
                model_name=model_name,
                rpm=500,
                tpm=100_000,
                profiles={"test_profile": profile_config_2},
                default_profile="test_profile",
            )
        elif language_model_provider == ModelProvider.ANTHROPIC:
            if not model_parameters.supports_reasoning:
                pytest.skip(f"Model {model_name} does not support reasoning profiles")
            
            profile_config_1 = AnthropicLanguageModel.Profile(thinking_token_budget=1024)
            profile_config_2 = AnthropicLanguageModel.Profile(thinking_token_budget=4096)
            
            language_model_1 = AnthropicLanguageModel(
                model_name=model_name,
                rpm=500,
                input_tpm=100_000,
                output_tpm=75_000,
                profiles={"test_profile": profile_config_1},
                default_profile="test_profile",
            )
            language_model_2 = AnthropicLanguageModel(
                model_name=model_name,
                rpm=500,
                input_tpm=100_000,
                output_tpm=75_000,
                profiles={"test_profile": profile_config_2},
                default_profile="test_profile",
            )
        elif language_model_provider == ModelProvider.GOOGLE_DEVELOPER:
            if not model_parameters.supports_reasoning:
                pytest.skip(f"Model {model_name} does not support reasoning profiles")
            
            profile_config_1 = GoogleDeveloperLanguageModel.Profile(thinking_token_budget=1024)
            profile_config_2 = GoogleDeveloperLanguageModel.Profile(thinking_token_budget=4096)
            
            language_model_1 = GoogleDeveloperLanguageModel(
                model_name=model_name,
                rpm=1000,
                tpm=500_000,
                profiles={"test_profile": profile_config_1},
                default_profile="test_profile",
            )
            language_model_2 = GoogleDeveloperLanguageModel(
                model_name=model_name,
                rpm=1000,
                tpm=500_000,
                profiles={"test_profile": profile_config_2},
                default_profile="test_profile",
            )
        elif language_model_provider == ModelProvider.GOOGLE_VERTEX:
            if not model_parameters.supports_reasoning:
                pytest.skip(f"Model {model_name} does not support reasoning profiles")
            
            profile_config_1 = GoogleVertexLanguageModel.Profile(thinking_token_budget=1024)
            profile_config_2 = GoogleVertexLanguageModel.Profile(thinking_token_budget=4096)
            
            language_model_1 = GoogleVertexLanguageModel(
                model_name=model_name,
                rpm=1000,
                tpm=500_000,
                profiles={"test_profile": profile_config_1},
                default_profile="test_profile",
            )
            language_model_2 = GoogleVertexLanguageModel(
                model_name=model_name,
                rpm=1000,
                tpm=500_000,
                profiles={"test_profile": profile_config_2},
                default_profile="test_profile",
            )
        else:
            pytest.skip(f"Provider {language_model_provider} not yet supported in this test")

        # Session 1: Create with first profile configuration
        config_1 = SessionConfig(
            app_name=app_name,
            db_path=tmp_path,
            semantic=SemanticConfig(
                language_models={"test_model": language_model_1},
                default_language_model="test_model",
                llm_response_cache=LLMResponseCacheConfig(
                    enabled=True,
                    ttl="1h",
                    max_size_mb=100,
                    namespace="test",
                ),
            ),
        )
        
        session_1 = Session.get_or_create(config_1)
        cache_1 = session_1._session_state._llm_cache
        assert cache_1 is not None
        
        # Make semantic calls using the profile
        df_1 = session_1.create_dataframe({
            "name": ["Alice", "Bob"],
        })
        
        df_1.select(
            semantic.map(
                "What is a nickname for {{name}}?",
                name=col("name"),
                model_alias=ModelAlias(name="test_model", profile="test_profile"),
            ).alias("nickname"),
        ).collect("polars")
        
        stats_1 = cache_1.stats()
        initial_total_entries = stats_1.total_entries
        
        # Stop session 1
        session_1.stop()
        
        # Session 2: Create with same app_name (same cache) but different profile contents
        config_2 = SessionConfig(
            app_name=app_name,  # Same app_name = same cache DB
            db_path=tmp_path,
            semantic=SemanticConfig(
                language_models={"test_model": language_model_2},  # Different profile config
                default_language_model="test_model",
                llm_response_cache=LLMResponseCacheConfig(
                    enabled=True,
                    ttl="1h",
                    max_size_mb=100,
                    namespace="test",
                ),
            ),
        )
        
        session_2 = Session.get_or_create(config_2)
        cache_2 = session_2._session_state._llm_cache
        assert cache_2 is not None
        
        # Make the SAME semantic calls - should result in cache misses due to different profile hash
        df_2 = session_2.create_dataframe({
            "name": ["Alice", "Bob"],
        })
        
        df_2.select(
            semantic.map(
                "What is a nickname for {{name}}?",
                name=col("name"),
                model_alias=ModelAlias(name="test_model", profile="test_profile"),
            ).alias("nickname"),
        ).collect("polars")
        
        stats_2 = cache_2.stats()
        session_2_misses = stats_2.misses
        
        # Since cache stats are instance-local, check that:
        # 1. Session 2 had misses (didn't hit cache from session 1)
        # 2. Total entries increased (new entries were stored due to different profile hash)
        assert session_2_misses > 0, (
            "Session 2 should have cache misses because profile contents changed "
            "(different profile hash despite same profile name)"
        )
        assert stats_2.total_entries > initial_total_entries, (
            "Total cache entries should increase because profile hash changed, "
            "preventing cache hits even though profile name is the same"
        )
        
        # Verify the new entries are different from the old ones
        # (we stored new responses due to cache misses)
        assert stats_2.total_entries == initial_total_entries + session_2_misses, (
            "New entries should equal the number of misses in session 2"
        )
        
        session_2.stop()
