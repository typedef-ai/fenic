import math
from enum import Enum
from typing import get_args

import pytest

from fenic._inference.common_openai.openai_profile_manager import (
    OpenAICompletionsProfileManager,
)
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


class _DummyResponse:
    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json = json_data
        self.text = text

    def json(self):
        return self._json


@pytest.fixture
def mock_openrouter_models(monkeypatch):
    """Mock OpenRouter models endpoint and reset dynamic loader state.

    Ensures deterministic unit tests without live HTTP calls.
    """
    from fenic._inference.openrouter.openrouter_provider import (
        OpenRouterModelProvider,
    )

    # Snapshot and reset state so dynamic load happens fresh for this test
    provider = OpenRouterModelProvider()
    openrouter_collection = model_catalog.provider_model_collections[ModelProvider.OPENROUTER]
    snapshot_completion_models = dict(openrouter_collection.completion_models)
    snapshot_dynamic_loaded = set(getattr(model_catalog, "_dynamic_loaded", set()))
    snapshot_models_loaded = getattr(provider, "_models_loaded", None)

    provider._models_loaded = False
    openrouter_collection.completion_models.clear()
    if hasattr(model_catalog, "_dynamic_loaded"):
        model_catalog._dynamic_loaded.discard(ModelProvider.OPENROUTER)

    # Values mirror those configured in the base provider catalog
    data = [
        {
            "id": "openai/gpt-4o",
            "pricing": {
                "prompt": 2.50 / 1_000_000,
                "completion": 10.00 / 1_000_000,
                "input_cache_read": 1.25 / 1_000_000,
            },
            "top_provider": {
                "context_length": 128_000,
                "max_completion_tokens": 16_384,
            },
            "supported_parameters": ["tools", "tool_choice", "max_tokens", "temperature", "top_p", "frequency_penalty", "presence_penalty", "stop", "logit_bias", "seed", "logprobs", "top_logprobs", "response_format", "structured_output"],
        },
        {
            "id": "openai/gpt-5",
            "pricing": {
                "prompt": 1.25 / 1_000_000,
                "completion": 10.00 / 1_000_000,
            },
            "top_provider": {
                "context_length": 400_000,
                "max_completion_tokens": 128_000,
            },
            # Include reasoning/verbosity but omit temperature to match base flags
            "supported_parameters": ["tools", "tool_choice", "max_tokens", "response_format", "structured_output", "reasoning", "verbosity"],
        },
        {
            "id": "anthropic/claude-sonnet-4",
            "pricing": {
                "prompt": 3.00 / 1_000_000,
                "completion": 15.00 / 1_000_000,
                "input_cache_read": 0.30 / 1_000_000,
            },
            "top_provider": {
                "context_length": 200_000,
                "max_completion_tokens": 64_000,
            },
            "supported_parameters": ["reasoning", "temperature", "max_tokens", "tool_choice", "tools"],
        },
        {
            "id": "google/gemini-2.5-flash",
            "pricing": {
                "prompt": 0.15 / 1_000_000,
                "completion": 2.50 / 1_000_000,
            },
            "top_provider": {
                "context_length": 1_048_576,
                "max_completion_tokens": 65_536,
            },
            "supported_parameters": ["tools", "tool_choice", "max_tokens", "temperature", "top_p", "stop", "seed", "logprobs", "top_logprobs", "response_format", "structured_output", "reasoning"],
        },
        {
            "id": "google/gemini-2.5-pro",
            "pricing": {
                "prompt": 1.25 / 1_000_000,
                "completion": 10.00 / 1_000_000,
            },
            "top_provider": {
                "context_length": 1_048_576,
                "max_completion_tokens": 65_536,
            },
            "supported_parameters": ["tools", "tool_choice", "max_tokens", "temperature", "top_p", "stop", "seed", "logprobs", "top_logprobs", "response_format", "structured_output", "reasoning"],
        },
    ]

    def _fake_get(url, headers=None, timeout=None):
        if url.endswith("/key"):
            return _DummyResponse(200, json_data={})
        if "/models" in url:
            return _DummyResponse(200, json_data={"data": data})
        return _DummyResponse(404, text="not found")

    import requests  # noqa: WPS433 (import inside function in tests)

    monkeypatch.setattr(requests, "get", _fake_get)

    # Hand control to the test
    yield None

    # Teardown: restore global singleton state to prevent cross-test pollution
    openrouter_collection.completion_models.clear()
    openrouter_collection.completion_models.update(snapshot_completion_models)
    if hasattr(model_catalog, "_dynamic_loaded"):
        model_catalog._dynamic_loaded.clear()
        model_catalog._dynamic_loaded.update(snapshot_dynamic_loaded)
    if snapshot_models_loaded is not None:
        provider._models_loaded = snapshot_models_loaded

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

def test_latest_frontier_models_are_registered():
    """Sanity-check newly released frontier model IDs and snapshots."""
    catalog = model_catalog

    openai_gpt_55 = catalog.get_completion_model_parameters(ModelProvider.OPENAI, "gpt-5.5")
    openai_gpt_55_snapshot = catalog.get_completion_model_parameters(ModelProvider.OPENAI, "gpt-5.5-2026-04-23")
    assert openai_gpt_55 is openai_gpt_55_snapshot
    assert openai_gpt_55.supports_xhigh_reasoning
    assert openai_gpt_55.context_window_length == 1_050_000

    anthropic_opus_48 = catalog.get_completion_model_parameters(ModelProvider.ANTHROPIC, "claude-opus-4-8")
    assert anthropic_opus_48.context_window_length == 1_000_000
    assert anthropic_opus_48.max_output_tokens == 128_000
    assert not anthropic_opus_48.supports_custom_temperature

    google_flash = catalog.get_completion_model_parameters(ModelProvider.GOOGLE_DEVELOPER, "gemini-3.5-flash")
    assert google_flash.context_window_length == 1_048_576
    assert google_flash.max_output_tokens == 65_536

    google_flash_lite = catalog.get_completion_model_parameters(ModelProvider.GOOGLE_DEVELOPER, "gemini-3.1-flash-lite")
    assert google_flash_lite.context_window_length == 1_048_576

    google_pro = catalog.get_completion_model_parameters(ModelProvider.GOOGLE_DEVELOPER, "gemini-3.1-pro-preview")
    google_pro_customtools = catalog.get_completion_model_parameters(ModelProvider.GOOGLE_DEVELOPER, "gemini-3.1-pro-preview-customtools")
    assert google_pro is google_pro_customtools

    google_embedding = catalog.get_embedding_model_parameters(ModelProvider.GOOGLE_DEVELOPER, "gemini-embedding-2")
    assert google_embedding.supports_dimensions(3072)
    assert google_embedding.supports_dimensions(128)

    assert catalog.get_completion_model_parameters(ModelProvider.OPENAI, "gpt-4-0314") is None
    assert catalog.get_completion_model_parameters(ModelProvider.ANTHROPIC, "claude-3-7-sonnet-20250219") is None
    assert catalog.get_completion_model_parameters(ModelProvider.ANTHROPIC, "claude-3-5-sonnet-20241022") is None
    assert catalog.get_completion_model_parameters(ModelProvider.ANTHROPIC, "claude-3-5-sonnet-20240620") is None
    assert catalog.get_completion_model_parameters(ModelProvider.ANTHROPIC, "claude-3-5-haiku-20241022") is None
    assert catalog.get_completion_model_parameters(ModelProvider.ANTHROPIC, "claude-3-opus-20240229") is None
    assert catalog.get_completion_model_parameters(ModelProvider.ANTHROPIC, "claude-3-haiku-20240307") is None
    assert catalog.get_completion_model_parameters(ModelProvider.ANTHROPIC, "claude-sonnet-4-20250514") is None
    assert catalog.get_completion_model_parameters(ModelProvider.ANTHROPIC, "claude-4-sonnet-20250514") is None
    assert catalog.get_completion_model_parameters(ModelProvider.ANTHROPIC, "claude-sonnet-4-0") is None
    assert catalog.get_completion_model_parameters(ModelProvider.ANTHROPIC, "claude-opus-4-20250514") is None
    assert catalog.get_completion_model_parameters(ModelProvider.ANTHROPIC, "claude-4-opus-20250514") is None
    assert catalog.get_completion_model_parameters(ModelProvider.ANTHROPIC, "claude-opus-4-0") is None
    assert catalog.get_completion_model_parameters(ModelProvider.GOOGLE_DEVELOPER, "gemini-2.0-flash-lite") is None
    assert catalog.get_completion_model_parameters(ModelProvider.GOOGLE_DEVELOPER, "gemini-2.0-flash-lite-001") is None
    assert catalog.get_completion_model_parameters(ModelProvider.GOOGLE_DEVELOPER, "gemini-2.0-flash") is None
    assert catalog.get_completion_model_parameters(ModelProvider.GOOGLE_DEVELOPER, "gemini-2.0-flash-001") is None
    assert catalog.get_completion_model_parameters(ModelProvider.GOOGLE_VERTEX, "gemini-2.0-flash-lite") is None
    assert catalog.get_completion_model_parameters(ModelProvider.GOOGLE_VERTEX, "gemini-2.0-flash") is None
    assert catalog.get_completion_model_parameters(ModelProvider.GOOGLE_DEVELOPER, "gemini-3.1-flash-lite-preview") is None
    assert catalog.get_completion_model_parameters(ModelProvider.GOOGLE_DEVELOPER, "gemini-2.5-pro-preview-06-05") is None
    assert catalog.get_embedding_model_parameters(ModelProvider.GOOGLE_DEVELOPER, "gemini-embedding-exp-03-07") is None
    assert catalog.get_embedding_model_parameters(ModelProvider.GOOGLE_DEVELOPER, "text-embedding-004") is None

def test_gpt_55_default_profile_uses_provider_default_reasoning():
    """GPT-5.5 defaults to medium reasoning unless the user configures a profile."""
    params = model_catalog.get_completion_model_parameters(ModelProvider.OPENAI, "gpt-5.5")
    profile = OpenAICompletionsProfileManager(params).get_default_profile()

    assert profile.reasoning_effort == "medium"
    assert profile.verbosity is None
    assert profile.expected_additional_reasoning_tokens == 8192

def test_openrouter_provider_loads_models(mock_openrouter_models):
    """Test that the OpenRouter provider can fetch the models from the OpenRouter API."""
    catalog = model_catalog
    assert len(catalog._get_supported_completions_models_by_provider(ModelProvider.OPENROUTER)) == 5

def test_openrouter_provider_loads_openai_models_correctly(mock_openrouter_models):
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

def test_openrouter_provider_loads_anthropic_models_correctly(mock_openrouter_models):
    """Test that the OpenRouter provider fetches models when they are first requested, and that their parameters match what is configured in the catalog for the base model providers."""
    catalog = model_catalog

    # Anthropic models
    openrouter_sonnet_4_parameters = catalog.get_completion_model_parameters(ModelProvider.OPENROUTER, "anthropic/claude-sonnet-4")
    standard_sonnet_4_parameters = catalog.get_completion_model_parameters(ModelProvider.ANTHROPIC, "claude-sonnet-4-5")
    assert math.isclose(openrouter_sonnet_4_parameters.input_token_cost, standard_sonnet_4_parameters.input_token_cost)
    assert math.isclose(openrouter_sonnet_4_parameters.output_token_cost, standard_sonnet_4_parameters.output_token_cost)
    # assert openrouter_sonnet_4_parameters.context_window_length == standard_sonnet_4_parameters.context_window_length # TODO: add 1m context window support for sonnet in standard anthropic client
    assert openrouter_sonnet_4_parameters.max_output_tokens == standard_sonnet_4_parameters.max_output_tokens
    assert openrouter_sonnet_4_parameters.supports_reasoning == standard_sonnet_4_parameters.supports_reasoning
    assert openrouter_sonnet_4_parameters.supports_custom_temperature == standard_sonnet_4_parameters.supports_custom_temperature

def test_openrouter_provider_loads_google_models_correctly(mock_openrouter_models):
    """Test that the OpenRouter provider fetches models when they are first requested, and that their parameters match what is configured in the catalog for the base model providers."""
    catalog = model_catalog

    # Google models
    openrouter_flash_parameters = catalog.get_completion_model_parameters(ModelProvider.OPENROUTER, "google/gemini-2.5-flash")
    standard_flash_parameters = catalog.get_completion_model_parameters(ModelProvider.GOOGLE_DEVELOPER, "gemini-2.5-flash")
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
