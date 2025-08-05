from enum import Enum
from typing import get_args

import pytest

from fenic.core._inference.model_catalog import (
    AnthropicLanguageModelName,
    CohereEmbeddingModelName,
    CompletionModelParameters,
    EmbeddingModelParameters,
    GoogleDeveloperEmbeddingModelName,
    GoogleDeveloperLanguageModelName,
    GoogleVertexEmbeddingModelName,
    GoogleVertexLanguageModelName,
    ModelCatalog,
    ModelProvider,
    OpenAIEmbeddingModelName,
    OpenAILanguageModelName,
)


@pytest.mark.parametrize("models,provider", [
    (OpenAILanguageModelName, ModelProvider.OPENAI),
    (AnthropicLanguageModelName, ModelProvider.ANTHROPIC),
    (GoogleDeveloperLanguageModelName, ModelProvider.GOOGLE_DEVELOPER),
    (GoogleVertexLanguageModelName, ModelProvider.GOOGLE_VERTEX),
])
def test_all_language_models_have_valid_parameters(models: Enum, provider: ModelProvider):
    """Test that all fetched model parameters have the required fields."""
    catalog = ModelCatalog()
    
    # Test Language models
    model_names = get_args(models)
    for model_name in model_names:
        params = catalog.get_completion_model_parameters(provider, model_name)
        assert params is not None and isinstance(params, CompletionModelParameters), (
            f"Could not fetch parameters for {provider} model: {model_name}"
        )
        assert hasattr(params, "input_token_cost"), f"Missing input_token_cost for {provider} model: {model_name}"
        assert hasattr(params, "output_token_cost"), f"Missing output_token_cost for {provider} model: {model_name}"
        assert hasattr(params, "context_window_length"), f"Missing context_window_length for {provider} model: {model_name}"
        assert hasattr(params, "max_output_tokens"), f"Missing max_output_tokens for {provider} model: {model_name}"


@pytest.mark.parametrize("models,provider", [
    (OpenAIEmbeddingModelName, ModelProvider.OPENAI),
    (GoogleDeveloperEmbeddingModelName, ModelProvider.GOOGLE_DEVELOPER),
    (GoogleVertexEmbeddingModelName, ModelProvider.GOOGLE_VERTEX),
    (CohereEmbeddingModelName, ModelProvider.COHERE),
])
def test_all_embedding_models_have_valid_parameters(models: Enum, provider: ModelProvider):
    """Test that all fetched embedding model parameters have the required fields."""
    catalog = ModelCatalog()

    model_names = get_args(models)
    for model_name in model_names:
        params = catalog.get_embedding_model_parameters(provider, model_name)
        assert params is not None and isinstance(params, EmbeddingModelParameters), (
            f"Could not fetch parameters for {provider} embedding model: {model_name}"
        )
        assert params.input_token_cost, (
            f"Missing input_token_cost for {provider} embedding model: {model_name}"
        )
        assert params.output_dimensions, f"Missing output_dimensions for {provider} embedding model: {model_name}"