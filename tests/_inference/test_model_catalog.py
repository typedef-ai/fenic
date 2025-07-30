from typing import get_args

from fenic.core._inference.model_catalog import (
    AnthropicLanguageModelName,
    CompletionModelParameters,
    EmbeddingModelParameters,
    ModelCatalog,
    ModelProvider,
    OpenAIEmbeddingModelName,
    OpenAILanguageModelName,
)


def test_all_language_models_have_valid_parameters():
    """Test that all fetched model parameters have the required fields."""
    catalog = ModelCatalog()
    
    # Test OpenAI models
    openai_models = get_args(OpenAILanguageModelName)
    for model_name in openai_models:
        params = catalog.get_completion_model_parameters(ModelProvider.OPENAI, model_name)
        assert params is not None and isinstance(params, CompletionModelParameters), f"Could not fetch parameters for OpenAI model: {model_name}"
        assert hasattr(params, "input_token_cost"), f"Missing input_token_cost for OpenAI model: {model_name}"
        assert hasattr(params, "output_token_cost"), f"Missing output_token_cost for OpenAI model: {model_name}"
        assert hasattr(params, "context_window_length"), f"Missing context_window_length for OpenAI model: {model_name}"
        assert hasattr(params, "max_output_tokens"), f"Missing max_output_tokens for OpenAI model: {model_name}"
    
    # Test Anthropic models
    anthropic_models = get_args(AnthropicLanguageModelName)
    for model_name in anthropic_models:
        params = catalog.get_completion_model_parameters(ModelProvider.ANTHROPIC, model_name)
        assert params is not None and isinstance(params, CompletionModelParameters), f"Could not fetch parameters for Anthropic model: {model_name}"
        assert hasattr(params, "input_token_cost"), f"Missing input_token_cost for Anthropic model: {model_name}"
        assert hasattr(params, "output_token_cost"), f"Missing output_token_cost for Anthropic model: {model_name}"
        assert hasattr(params, "context_window_length"), f"Missing context_window_length for Anthropic model: {model_name}"
        assert hasattr(params, "max_output_tokens"), f"Missing max_output_tokens for Anthropic model: {model_name}"

def test_all_embedding_models_have_valid_parameters():
    """Test that all fetched embedding model parameters have the required fields."""
    catalog = ModelCatalog()

    # Test OpenAI embedding models
    openai_embedding_models = get_args(OpenAIEmbeddingModelName)
    for model_name in openai_embedding_models:
        params = catalog.get_embedding_model_parameters(ModelProvider.OPENAI, model_name)
        assert params is not None and isinstance(params, EmbeddingModelParameters), f"Could not fetch parameters for OpenAI embedding model: {model_name}"
        assert hasattr(params, "input_token_cost"), f"Missing input_token_cost for OpenAI embedding model: {model_name}"