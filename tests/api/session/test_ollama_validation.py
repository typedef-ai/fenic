"""Tests for Ollama multi-model warnings."""

import pytest

from fenic.api.session import (
    OllamaEmbeddingModel,
    OllamaLanguageModel,
    OpenAILanguageModel,
)
from fenic.api.session.config import SemanticConfig


def test_single_ollama_language_model_allowed():
    """Single Ollama language model should be allowed without warning."""
    config = SemanticConfig(
        language_models={"gemma": OllamaLanguageModel(model_name="gemma3:4b")}
    )
    assert config is not None
    assert len(config.language_models) == 1


def test_single_ollama_embedding_model_allowed():
    """Single Ollama embedding model should be allowed without warning."""
    config = SemanticConfig(
        embedding_models={"embed": OllamaEmbeddingModel(model_name="embeddinggemma:latest")}
    )
    assert config is not None
    assert len(config.embedding_models) == 1


def test_multiple_ollama_language_models_warns(caplog):
    """Multiple Ollama language models should warn but still be allowed."""
    config = SemanticConfig(
        language_models={
            "gemma": OllamaLanguageModel(model_name="gemma3:4b"),
            "qwen": OllamaLanguageModel(model_name="qwen3:4b"),
        },
        default_language_model="gemma"
    )
    assert config is not None
    assert len(config.language_models) == 2

    # Check that warning was logged
    assert any("Multiple Ollama models detected" in record.message for record in caplog.records)
    assert any("VRAM/RAM" in record.message for record in caplog.records)


def test_ollama_language_and_embedding_warns(caplog):
    """Ollama language model + embedding model should warn but still be allowed."""
    config = SemanticConfig(
        language_models={"gemma": OllamaLanguageModel(model_name="gemma3:4b")},
        embedding_models={"embed": OllamaEmbeddingModel(model_name="embeddinggemma:latest")},
    )
    assert config is not None
    assert len(config.language_models) == 1
    assert len(config.embedding_models) == 1

    # Check that warning was logged
    assert any("Multiple Ollama models detected" in record.message for record in caplog.records)
    assert any("VRAM/RAM" in record.message for record in caplog.records)


def test_multiple_ollama_embedding_models_warns(caplog):
    """Multiple Ollama embedding models should warn but still be allowed."""
    config = SemanticConfig(
        embedding_models={
            "embed1": OllamaEmbeddingModel(model_name="embeddinggemma:latest"),
            "embed2": OllamaEmbeddingModel(model_name="nomic-embed-text:latest"),
        },
        default_embedding_model="embed1"
    )
    assert config is not None
    assert len(config.embedding_models) == 2

    # Check that warning was logged
    assert any("Multiple Ollama models detected" in record.message for record in caplog.records)
    assert any("VRAM/RAM" in record.message for record in caplog.records)


def test_ollama_with_cloud_provider_no_warning(caplog):
    """Ollama + cloud provider should be allowed without warning (mixed providers OK)."""
    config = SemanticConfig(
        language_models={
            "gemma": OllamaLanguageModel(model_name="gemma3:4b"),
            "gpt": OpenAILanguageModel(model_name="gpt-4o-mini", rpm=10000, tpm=1000000),
        },
        default_language_model="gemma"
    )
    assert config is not None
    assert len(config.language_models) == 2

    # Check that NO warning was logged (only 1 Ollama model)
    assert not any("Multiple Ollama models detected" in record.message for record in caplog.records)
