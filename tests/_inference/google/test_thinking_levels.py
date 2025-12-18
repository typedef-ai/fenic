import pytest

pytest.importorskip("google.genai")

from fenic.api.session import SessionConfig
from fenic.api.session.config import (
    GoogleDeveloperLanguageModel,
    GoogleVertexLanguageModel,
    SemanticConfig,
)
from fenic.core._inference.model_catalog import (
    GEMINI_3_FLASH_THINKING_LEVELS,
    GEMINI_3_PRO_THINKING_LEVELS,
    ModelProvider,
    model_catalog,
)
from fenic.core.error import ConfigurationError


class TestThinkingLevelConstants:
    """Test that thinking level constants are correctly defined."""

    def test_gemini_3_pro_thinking_levels(self):
        """Gemini 3 Pro supports only high and low thinking levels."""
        assert GEMINI_3_PRO_THINKING_LEVELS == {"high", "low"}

    def test_gemini_3_flash_thinking_levels(self):
        """Gemini 3 Flash supports all four thinking levels."""
        assert GEMINI_3_FLASH_THINKING_LEVELS == {"high", "medium", "low", "minimal"}


class TestModelCatalogThinkingLevels:
    """Test that model catalog correctly reports supported thinking levels."""

    def test_gemini_3_pro_preview_thinking_levels(self):
        """Gemini 3 Pro Preview should support high and low."""
        params = model_catalog.get_completion_model_parameters(
            ModelProvider.GOOGLE_DEVELOPER, "gemini-3-pro-preview"
        )
        assert params is not None
        assert params.supported_thinking_levels == GEMINI_3_PRO_THINKING_LEVELS

    def test_gemini_3_flash_preview_thinking_levels(self):
        """Gemini 3 Flash Preview should support all four levels."""
        params = model_catalog.get_completion_model_parameters(
            ModelProvider.GOOGLE_DEVELOPER, "gemini-3-flash-preview"
        )
        assert params is not None
        assert params.supported_thinking_levels == GEMINI_3_FLASH_THINKING_LEVELS

    def test_gemini_25_pro_no_thinking_levels(self):
        """Gemini 2.5 Pro should use thinking_budget, not thinking_level."""
        params = model_catalog.get_completion_model_parameters(
            ModelProvider.GOOGLE_DEVELOPER, "gemini-2.5-pro"
        )
        assert params is not None
        assert params.supported_thinking_levels is None


class TestAutoProfileCreation:
    """Test that auto-profile creation works for models with thinking levels."""

    def test_gemini_3_flash_auto_profiles(self):
        """Gemini 3 Flash should auto-create profiles for all 4 thinking levels."""
        config = SessionConfig(
            app_name="test_auto_profiles",
            semantic=SemanticConfig(
                language_models={
                    "flash": GoogleDeveloperLanguageModel(
                        model_name="gemini-3-flash-preview",
                        rpm=100,
                        tpm=1000,
                    )
                }
            ),
        )
        model = config.semantic.language_models["flash"]
        assert model.profiles is not None
        assert set(model.profiles.keys()) == {"high", "medium", "low", "minimal"}
        assert model.default_profile == "low"

    def test_gemini_3_pro_auto_profiles(self):
        """Gemini 3 Pro should auto-create profiles for high and low only."""
        config = SessionConfig(
            app_name="test_auto_profiles",
            semantic=SemanticConfig(
                language_models={
                    "pro": GoogleDeveloperLanguageModel(
                        model_name="gemini-3-pro-preview",
                        rpm=100,
                        tpm=1000,
                    )
                }
            ),
        )
        model = config.semantic.language_models["pro"]
        assert model.profiles is not None
        assert set(model.profiles.keys()) == {"high", "low"}
        assert model.default_profile == "low"


class TestThinkingLevelValidation:
    """Test validation of thinking levels for different models."""

    def test_invalid_thinking_level_on_gemini_3_pro(self):
        """Gemini 3 Pro should reject 'medium' thinking level."""
        with pytest.raises(
            ConfigurationError,
            match="does not support thinking_level='medium'",
        ):
            SessionConfig(
                app_name="test_validation",
                semantic=SemanticConfig(
                    language_models={
                        "pro": GoogleDeveloperLanguageModel(
                            model_name="gemini-3-pro-preview",
                            rpm=100,
                            tpm=1000,
                            profiles={
                                "invalid": GoogleDeveloperLanguageModel.Profile(
                                    thinking_level="medium"
                                )
                            },
                        )
                    }
                ),
            )

    def test_invalid_thinking_level_minimal_on_gemini_3_pro(self):
        """Gemini 3 Pro should reject 'minimal' thinking level."""
        with pytest.raises(
            ConfigurationError,
            match="does not support thinking_level='minimal'",
        ):
            SessionConfig(
                app_name="test_validation",
                semantic=SemanticConfig(
                    language_models={
                        "pro": GoogleDeveloperLanguageModel(
                            model_name="gemini-3-pro-preview",
                            rpm=100,
                            tpm=1000,
                            profiles={
                                "invalid": GoogleDeveloperLanguageModel.Profile(
                                    thinking_level="minimal"
                                )
                            },
                        )
                    }
                ),
            )

    def test_valid_thinking_levels_on_gemini_3_flash(self):
        """Gemini 3 Flash should accept all four thinking levels."""
        # Should not raise
        config = SessionConfig(
            app_name="test_validation",
            semantic=SemanticConfig(
                language_models={
                    "flash": GoogleDeveloperLanguageModel(
                        model_name="gemini-3-flash-preview",
                        rpm=100,
                        tpm=1000,
                        default_profile="low",
                        profiles={
                            "high": GoogleDeveloperLanguageModel.Profile(thinking_level="high"),
                            "medium": GoogleDeveloperLanguageModel.Profile(thinking_level="medium"),
                            "low": GoogleDeveloperLanguageModel.Profile(thinking_level="low"),
                            "minimal": GoogleDeveloperLanguageModel.Profile(thinking_level="minimal"),
                        },
                    )
                }
            ),
        )
        assert config.semantic.language_models["flash"].profiles is not None
        assert len(config.semantic.language_models["flash"].profiles) == 4

    def test_thinking_level_on_model_without_support(self):
        """Gemini 2.5 Pro should reject thinking_level (uses thinking_budget instead)."""
        with pytest.raises(
            ConfigurationError,
            match="does not support thinking_level",
        ):
            SessionConfig(
                app_name="test_validation",
                semantic=SemanticConfig(
                    language_models={
                        "pro": GoogleDeveloperLanguageModel(
                            model_name="gemini-2.5-pro",
                            rpm=100,
                            tpm=1000,
                            profiles={
                                "invalid": GoogleDeveloperLanguageModel.Profile(
                                    thinking_level="high"
                                )
                            },
                        )
                    }
                ),
            )


class TestGoogleVertexThinkingLevels:
    """Test thinking levels work the same for Google Vertex provider."""

    def test_vertex_gemini_3_flash_auto_profiles(self):
        """Google Vertex Gemini 3 Flash should auto-create all 4 profiles."""
        config = SessionConfig(
            app_name="test_vertex",
            semantic=SemanticConfig(
                language_models={
                    "flash": GoogleVertexLanguageModel(
                        model_name="gemini-3-flash-preview",
                        rpm=100,
                        tpm=1000,
                    )
                }
            ),
        )
        model = config.semantic.language_models["flash"]
        assert model.profiles is not None
        assert set(model.profiles.keys()) == {"high", "medium", "low", "minimal"}

    def test_vertex_gemini_3_pro_auto_profiles(self):
        """Google Vertex Gemini 3 Pro should auto-create high and low profiles."""
        config = SessionConfig(
            app_name="test_vertex",
            semantic=SemanticConfig(
                language_models={
                    "pro": GoogleVertexLanguageModel(
                        model_name="gemini-3-pro-preview",
                        rpm=100,
                        tpm=1000,
                    )
                }
            ),
        )
        model = config.semantic.language_models["pro"]
        assert model.profiles is not None
        assert set(model.profiles.keys()) == {"high", "low"}
