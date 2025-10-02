"""Tests for Ollama multi-model validation."""

import os
import pytest

from fenic.api.session import (
    OllamaEmbeddingModel,
    OllamaLanguageModel,
    OpenAILanguageModel,
)
from fenic.api.session.config import SemanticConfig
from fenic.core.error import ConfigurationError


def test_single_ollama_language_model_allowed():
    """Single Ollama language model should be allowed."""
    config = SemanticConfig(
        language_models={"gemma": OllamaLanguageModel(model_name="gemma3:4b")}
    )
    assert config is not None
    assert len(config.language_models) == 1


def test_single_ollama_embedding_model_allowed():
    """Single Ollama embedding model should be allowed."""
    config = SemanticConfig(
        embedding_models={"embed": OllamaEmbeddingModel(model_name="embeddinggemma:latest")}
    )
    assert config is not None
    assert len(config.embedding_models) == 1


def test_multiple_ollama_language_models_blocked():
    """Multiple Ollama language models should raise ConfigurationError."""
    with pytest.raises(ConfigurationError, match="Multiple Ollama models are not currently supported"):
        SemanticConfig(
            language_models={
                "gemma": OllamaLanguageModel(model_name="gemma3:4b"),
                "qwen": OllamaLanguageModel(model_name="qwen3:4b"),
            }
        )


def test_ollama_language_and_embedding_blocked():
    """Ollama language model + embedding model should raise ConfigurationError."""
    with pytest.raises(ConfigurationError, match="Multiple Ollama models are not currently supported"):
        SemanticConfig(
            language_models={"gemma": OllamaLanguageModel(model_name="gemma3:4b")},
            embedding_models={"embed": OllamaEmbeddingModel(model_name="embeddinggemma:latest")},
        )


def test_multiple_ollama_embedding_models_blocked():
    """Multiple Ollama embedding models should raise ConfigurationError."""
    with pytest.raises(ConfigurationError, match="Multiple Ollama models are not currently supported"):
        SemanticConfig(
            embedding_models={
                "embed1": OllamaEmbeddingModel(model_name="embeddinggemma:latest"),
                "embed2": OllamaEmbeddingModel(model_name="nomic-embed-text"),
            }
        )


def test_ollama_with_cloud_provider_allowed():
    """Ollama + cloud provider should be allowed (mixed providers OK)."""
    config = SemanticConfig(
        language_models={
            "gemma": OllamaLanguageModel(model_name="gemma3:4b"),
            "gpt": OpenAILanguageModel(model_name="gpt-4o-mini", rpm=10000, tpm=1000000),
        },
        default_language_model="gemma"
    )
    assert config is not None
    assert len(config.language_models) == 2


def test_multiple_ollama_models_with_env_override():
    """Multiple Ollama models should be allowed with env var override."""
    # Save original env var state
    original_value = os.environ.get("FENIC_ALLOW_MULTIPLE_OLLAMA_MODELS")

    try:
        os.environ["FENIC_ALLOW_MULTIPLE_OLLAMA_MODELS"] = "true"

        config = SemanticConfig(
            language_models={"gemma": OllamaLanguageModel(model_name="gemma3:4b")},
            embedding_models={"embed": OllamaEmbeddingModel(model_name="embeddinggemma:latest")},
        )

        assert config is not None
        assert len(config.language_models) == 1
        assert len(config.embedding_models) == 1

    finally:
        # Restore original env var state
        if original_value is None:
            os.environ.pop("FENIC_ALLOW_MULTIPLE_OLLAMA_MODELS", None)
        else:
            os.environ["FENIC_ALLOW_MULTIPLE_OLLAMA_MODELS"] = original_value


def test_env_var_case_insensitive():
    """Env var should work with different cases."""
    original_value = os.environ.get("FENIC_ALLOW_MULTIPLE_OLLAMA_MODELS")

    try:
        # Test with uppercase TRUE
        os.environ["FENIC_ALLOW_MULTIPLE_OLLAMA_MODELS"] = "TRUE"

        config = SemanticConfig(
            language_models={
                "gemma": OllamaLanguageModel(model_name="gemma3:4b"),
                "qwen": OllamaLanguageModel(model_name="qwen3:4b"),
            },
            default_language_model="gemma"
        )
        assert config is not None

    finally:
        if original_value is None:
            os.environ.pop("FENIC_ALLOW_MULTIPLE_OLLAMA_MODELS", None)
        else:
            os.environ["FENIC_ALLOW_MULTIPLE_OLLAMA_MODELS"] = original_value


def test_env_var_false_still_blocks():
    """Env var set to 'false' should still block multiple models."""
    original_value = os.environ.get("FENIC_ALLOW_MULTIPLE_OLLAMA_MODELS")

    try:
        os.environ["FENIC_ALLOW_MULTIPLE_OLLAMA_MODELS"] = "false"

        with pytest.raises(ConfigurationError, match="Multiple Ollama models are not currently supported"):
            SemanticConfig(
                language_models={
                    "gemma": OllamaLanguageModel(model_name="gemma3:4b"),
                    "qwen": OllamaLanguageModel(model_name="qwen3:4b"),
                }
            )

    finally:
        if original_value is None:
            os.environ.pop("FENIC_ALLOW_MULTIPLE_OLLAMA_MODELS", None)
        else:
            os.environ["FENIC_ALLOW_MULTIPLE_OLLAMA_MODELS"] = original_value
