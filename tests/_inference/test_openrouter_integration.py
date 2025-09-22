import pytest

from fenic.core._inference.model_catalog import (
    CompletionModelParameters,
    ModelProvider,
    model_catalog,
)


@pytest.mark.integration
def test_openrouter_provider_live_fetch_models():
    """Live integration: ensure OpenRouter models load and basic fields exist.

    Skipped by default; run with -m integration.
    """
    models = model_catalog._get_supported_completions_models_by_provider(
        ModelProvider.OPENROUTER
    )
    assert models, "Expected at least one OpenRouter model from live API"

    # Sample a few models for basic sanity checks
    for params in models.values():
        assert isinstance(params, CompletionModelParameters)
        assert params.context_window_length > 0
        assert params.max_output_tokens > 0
        assert params.supports_profiles
