from typing import get_args

import pytest

from fenic._inference.common_openai.openai_profile_manager import (
    OpenAICompletionsProfileManager,
)
from fenic.api.session.config import (
    GoogleDeveloperLanguageModel,
    GoogleVertexLanguageModel,
    OpenAILanguageModel,
    SemanticConfig,
    SessionConfig,
)
from fenic.core._inference.model_catalog import (
    GEMINI_3_8_FLASH_THINKING_LEVELS,
    GoogleDeveloperLanguageModelName,
    GoogleVertexLanguageModelName,
    ModelProvider,
    OpenAILanguageModelName,
    model_catalog,
)
from fenic.core.error import ConfigurationError


def test_gpt_61_sol_parameters():
    """Match OpenAI's standard prices, limits, and Chat Completions capabilities."""
    params = model_catalog.get_completion_model_parameters(
        ModelProvider.OPENAI, "gpt-6.1-sol"
    )
    assert "gpt-6.1-sol" in get_args(OpenAILanguageModelName)
    assert params is not None
    assert params.input_token_cost == 2 / 1_000_000
    assert params.cached_input_token_read_cost == 0.10 / 1_000_000
    assert params.cached_input_token_write_cost == 2.50 / 1_000_000
    assert params.output_token_cost == 10 / 1_000_000
    assert params.context_window_length == 1_050_000
    assert params.max_output_tokens == 128_000
    assert params.supports_reasoning
    assert not params.supports_disabled_reasoning
    assert not params.supports_minimal_reasoning
    assert params.supports_xhigh_reasoning
    assert params.supports_max_reasoning
    assert not params.supports_custom_temperature
    assert params.supports_pdf_parsing
    assert "response_format" in params.supported_parameters
    assert "tools" not in params.supported_parameters
    assert (
        OpenAICompletionsProfileManager(params).get_default_profile().reasoning_effort
        == "medium"
    )


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
def test_gpt_61_sol_accepts_documented_efforts(effort):
    config = SessionConfig(
        app_name="gpt_61_sol_profiles",
        semantic=SemanticConfig(
            language_models={
                "sol": OpenAILanguageModel(
                    model_name="gpt-6.1-sol",
                    rpm=100,
                    tpm=1000,
                    profiles={
                        "test": OpenAILanguageModel.Profile(reasoning_effort=effort)
                    },
                )
            }
        ),
    )
    assert (
        config.semantic.language_models["sol"].profiles["test"].reasoning_effort
        == effort
    )


@pytest.mark.parametrize("effort", ["none", "minimal"])
def test_gpt_61_sol_rejects_unsupported_efforts(effort):
    with pytest.raises(ConfigurationError, match=f"does not support '{effort}'"):
        SessionConfig(
            app_name="gpt_61_sol_profiles",
            semantic=SemanticConfig(
                language_models={
                    "sol": OpenAILanguageModel(
                        model_name="gpt-6.1-sol",
                        rpm=100,
                        tpm=1000,
                        profiles={
                            "test": OpenAILanguageModel.Profile(reasoning_effort=effort)
                        },
                    )
                }
            ),
        )


def test_unreleased_gpt_61_luna_is_not_registered():
    assert "gpt-6.1-luna" not in get_args(OpenAILanguageModelName)
    assert (
        model_catalog.get_completion_model_parameters(
            ModelProvider.OPENAI, "gpt-6.1-luna"
        )
        is None
    )
    assert (
        model_catalog.get_completion_model_parameters(
            ModelProvider.OPENAI, "gpt-6-luna"
        )
        is not None
    )


@pytest.mark.parametrize(
    "provider, names, config_class",
    [
        (
            ModelProvider.GOOGLE_DEVELOPER,
            GoogleDeveloperLanguageModelName,
            GoogleDeveloperLanguageModel,
        ),
        (
            ModelProvider.GOOGLE_VERTEX,
            GoogleVertexLanguageModelName,
            GoogleVertexLanguageModel,
        ),
    ],
)
def test_gemini_38_flash_parameters_and_profiles(provider, names, config_class):
    """Use standard promotional rates, not the discounted Batch/Flex rates."""
    pytest.importorskip("google.genai")
    params = model_catalog.get_completion_model_parameters(provider, "gemini-3.8-flash")
    assert "gemini-3.8-flash" in get_args(names)
    assert params is not None
    assert params.input_token_cost == 0.75 / 1_000_000
    assert params.cached_input_token_read_cost == 0.075 / 1_000_000
    assert params.output_token_cost == 3.75 / 1_000_000
    assert params.context_window_length == 1_048_576
    assert params.max_output_tokens == 65_536
    assert params.max_temperature == 2.0
    assert params.supports_reasoning
    assert not params.supports_disabled_reasoning
    assert not params.supports_custom_temperature
    assert (
        params.supported_thinking_levels
        == GEMINI_3_8_FLASH_THINKING_LEVELS
        == {"low", "medium", "high"}
    )
    assert params.supports_pdf_parsing
    assert params.supports_media_resolution
    assert {
        "response_mime_type",
        "response_schema",
        "tools",
    } <= params.supported_parameters
    assert model_catalog.calculate_completion_model_cost(
        provider, "gemini-3.8-flash", 2, 3, 4
    ) == pytest.approx((2 * 0.75 + 3 * 0.075 + 4 * 3.75) / 1_000_000)
    config = SessionConfig(
        app_name="gemini_38_flash_profiles",
        semantic=SemanticConfig(
            language_models={
                "flash": config_class(model_name="gemini-3.8-flash", rpm=100, tpm=1000)
            }
        ),
    )
    model = config.semantic.language_models["flash"]
    assert set(model.profiles) == {"low", "medium", "high"}
    # Fenic intentionally defaults to low rather than the provider's medium.
    assert model.default_profile == "low"


@pytest.mark.parametrize(
    "config_class", [GoogleDeveloperLanguageModel, GoogleVertexLanguageModel]
)
def test_gemini_38_flash_rejects_minimal(config_class):
    pytest.importorskip("google.genai")
    with pytest.raises(ConfigurationError, match="minimal"):
        SessionConfig(
            app_name="gemini_38_flash_profiles",
            semantic=SemanticConfig(
                language_models={
                    "flash": config_class(
                        model_name="gemini-3.8-flash",
                        rpm=100,
                        tpm=1000,
                        profiles={
                            "test": config_class.Profile(thinking_level="minimal")
                        },
                    )
                }
            ),
        )
