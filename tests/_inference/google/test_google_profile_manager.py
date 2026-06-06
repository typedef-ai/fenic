import pytest

pytest.importorskip("google.genai")

from google.genai.types import EmbedContentConfig, ThinkingConfig, ThinkingLevel

from fenic._inference.google.google_profile_manager import (
    GoogleCompletionsProfileManager,
    GoogleEmbeddingsProfileManager,
)
from fenic.core._inference.model_catalog import ModelProvider, model_catalog
from fenic.core._resolved_session_config import ResolvedGoogleModelProfile


def test_gemini_3_thinking_level_profile_uses_typed_config():
    params = model_catalog.get_completion_model_parameters(
        ModelProvider.GOOGLE_DEVELOPER, "gemini-3-flash-preview"
    )
    profile = GoogleCompletionsProfileManager(
        model_parameters=params,
        profile_configurations={
            "fast": ResolvedGoogleModelProfile(
                thinking_level="minimal",
                media_resolution="low",
            )
        },
        default_profile_name="fast",
    ).get_profile_by_name(None)

    assert isinstance(profile.thinking_config, ThinkingConfig)
    assert profile.thinking_config.thinking_level == ThinkingLevel.MINIMAL
    assert profile.thinking_token_budget == 2048
    assert profile.media_resolution == "low"


def test_gemini_25_budget_profile_uses_typed_config():
    params = model_catalog.get_completion_model_parameters(
        ModelProvider.GOOGLE_DEVELOPER, "gemini-2.5-flash"
    )
    profile = GoogleCompletionsProfileManager(
        model_parameters=params,
        profile_configurations={
            "budget": ResolvedGoogleModelProfile(thinking_token_budget=4096)
        },
        default_profile_name="budget",
    ).get_profile_by_name(None)

    assert isinstance(profile.thinking_config, ThinkingConfig)
    assert profile.thinking_config.include_thoughts is False
    assert profile.thinking_config.thinking_budget == 4096
    assert profile.thinking_token_budget == 4096


def test_google_embedding_profile_uses_typed_config():
    params = model_catalog.get_embedding_model_parameters(
        ModelProvider.GOOGLE_DEVELOPER, "gemini-embedding-001"
    )
    profile = GoogleEmbeddingsProfileManager(
        model_parameters=params,
        profiles={
            "search": ResolvedGoogleModelProfile(
                embedding_dimensionality=1536,
                embedding_task_type="RETRIEVAL_DOCUMENT",
            )
        },
        default_profile_name="search",
    ).get_profile_by_name(None)

    assert isinstance(profile.embedding_config, EmbedContentConfig)
    assert profile.embedding_config.output_dimensionality == 1536
    assert profile.embedding_config.task_type == "RETRIEVAL_DOCUMENT"
