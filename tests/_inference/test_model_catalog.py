import math
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
    model_catalog,
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
    catalog = model_catalog
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

def test_openrouter_provider_loads_models():
    """Test that the OpenRouter provider can fetch the models from the OpenRouter API."""
    catalog = model_catalog
    assert catalog._get_supported_completions_models_by_provider(ModelProvider.OPENROUTER)

def test_openrouter_provider_loads_openai_models_correctly():
    """Test that the OpenRouter provider fetches models when they are first requested, and that their parameters match what is configured in the catalog for the base model providers."""
    catalog = model_catalog

    # OpenAI models
    openrouter_gpt_4o_parameters = catalog.get_completion_model_parameters(ModelProvider.OPENROUTER, "openai/gpt-4o")
    standard_gpt_4o_parameters = catalog.get_completion_model_parameters(ModelProvider.OPENAI, "gpt-4o")

    assert math.isclose(openrouter_gpt_4o_parameters.input_token_cost, standard_gpt_4o_parameters.input_token_cost)
    assert math.isclose(openrouter_gpt_4o_parameters.output_token_cost, standard_gpt_4o_parameters.output_token_cost)
    assert openrouter_gpt_4o_parameters.context_window_length == standard_gpt_4o_parameters.context_window_length
    assert openrouter_gpt_4o_parameters.max_output_tokens == standard_gpt_4o_parameters.max_output_tokens
    assert openrouter_gpt_4o_parameters.supports_reasoning == standard_gpt_4o_parameters.supports_reasoning
    assert openrouter_gpt_4o_parameters.supports_custom_temperature == standard_gpt_4o_parameters.supports_custom_temperature

    openrouter_gpt_5_parameters = catalog.get_completion_model_parameters(ModelProvider.OPENROUTER, "openai/gpt-5")
    standard_gpt_5_parameters = catalog.get_completion_model_parameters(ModelProvider.OPENAI, "gpt-5")

    assert math.isclose(openrouter_gpt_4o_parameters.input_token_cost, standard_gpt_4o_parameters.input_token_cost)
    assert math.isclose(openrouter_gpt_5_parameters.output_token_cost, standard_gpt_5_parameters.output_token_cost)
    assert openrouter_gpt_5_parameters.context_window_length == standard_gpt_5_parameters.context_window_length
    assert openrouter_gpt_5_parameters.max_output_tokens == standard_gpt_5_parameters.max_output_tokens
    assert openrouter_gpt_5_parameters.supports_reasoning == standard_gpt_5_parameters.supports_reasoning
    assert openrouter_gpt_5_parameters.supports_custom_temperature == standard_gpt_5_parameters.supports_custom_temperature

def test_openrouter_provider_loads_anthropic_models_correctly():
    """Test that the OpenRouter provider fetches models when they are first requested, and that their parameters match what is configured in the catalog for the base model providers."""
    catalog = model_catalog

    # Anthropic models
    openrouter_sonnet_4_parameters = catalog.get_completion_model_parameters(ModelProvider.OPENROUTER, "anthropic/claude-sonnet-4")
    standard_sonnet_4_parameters = catalog.get_completion_model_parameters(ModelProvider.ANTHROPIC, "claude-sonnet-4-0")
    assert math.isclose(openrouter_sonnet_4_parameters.input_token_cost, standard_sonnet_4_parameters.input_token_cost)
    assert math.isclose(openrouter_sonnet_4_parameters.output_token_cost, standard_sonnet_4_parameters.output_token_cost)
    # assert openrouter_sonnet_4_parameters.context_window_length == standard_sonnet_4_parameters.context_window_length # TODO: add 1m context window support for sonnet in standard anthropic client
    assert openrouter_sonnet_4_parameters.max_output_tokens == standard_sonnet_4_parameters.max_output_tokens
    assert openrouter_sonnet_4_parameters.supports_reasoning == standard_sonnet_4_parameters.supports_reasoning
    assert openrouter_sonnet_4_parameters.supports_custom_temperature == standard_sonnet_4_parameters.supports_custom_temperature

def test_openrouter_provider_loads_google_models_correctly():
    """Test that the OpenRouter provider fetches models when they are first requested, and that their parameters match what is configured in the catalog for the base model providers."""
    catalog = model_catalog

    # Google models
    openrouter_flash_parameters = catalog.get_completion_model_parameters(ModelProvider.OPENROUTER, "google/gemini-2.0-flash-001")
    standard_flash_parameters = catalog.get_completion_model_parameters(ModelProvider.GOOGLE_DEVELOPER, "gemini-2.0-flash-001")
    assert math.isclose(openrouter_flash_parameters.input_token_cost, standard_flash_parameters.input_token_cost)
    assert math.isclose(openrouter_flash_parameters.output_token_cost, standard_flash_parameters.output_token_cost)
    assert openrouter_flash_parameters.context_window_length == standard_flash_parameters.context_window_length
    assert openrouter_flash_parameters.max_output_tokens == standard_flash_parameters.max_output_tokens
    assert openrouter_flash_parameters.supports_reasoning == standard_flash_parameters.supports_reasoning
    assert openrouter_flash_parameters.supports_custom_temperature == standard_flash_parameters.supports_custom_temperature

    openrouter_pro_parameters = catalog.get_completion_model_parameters(ModelProvider.OPENROUTER, "google/gemini-2.5-pro")
    standard_pro_parameters = catalog.get_completion_model_parameters(ModelProvider.GOOGLE_DEVELOPER, "gemini-2.5-pro")
    assert math.isclose(openrouter_pro_parameters.input_token_cost, standard_pro_parameters.input_token_cost)
    assert math.isclose(openrouter_pro_parameters.output_token_cost, standard_pro_parameters.output_token_cost)
    assert openrouter_pro_parameters.context_window_length == standard_pro_parameters.context_window_length
    assert openrouter_pro_parameters.max_output_tokens == standard_pro_parameters.max_output_tokens
    assert openrouter_pro_parameters.supports_reasoning == standard_pro_parameters.supports_reasoning
    assert openrouter_pro_parameters.supports_custom_temperature == standard_pro_parameters.supports_custom_temperature
