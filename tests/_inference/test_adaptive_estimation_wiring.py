from fenic._backends.local.model_registry import SessionModelRegistry
from fenic.api.session.config import (
    AdaptiveTokenEstimationConfig,
    OpenAILanguageModel,
    SemanticConfig,
    SessionConfig,
)


async def _noop_validate(providers):
    """Async no-op replacement for _validate_provider_api_keys."""
    return


def _registry(adaptive, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    # SessionModelRegistry.__init__ validates provider API keys with a LIVE network
    # call. Stub it out so the test is hermetic.
    monkeypatch.setattr(
        "fenic._backends.local.model_registry._validate_provider_api_keys",
        _noop_validate,
    )
    semantic = SemanticConfig(
        language_models={"m": OpenAILanguageModel(model_name="gpt-4o-mini", rpm=100, tpm=100000)},
        default_language_model="m",
        adaptive_token_estimation=adaptive,
    )
    resolved = SessionConfig(app_name="t", semantic=semantic)._to_resolved_config()
    return SessionModelRegistry(resolved.semantic)


def test_client_estimator_reflects_config(monkeypatch):
    reg = _registry(AdaptiveTokenEstimationConfig(safety_margin=1.5), monkeypatch)
    try:
        client = reg.get_language_model().client
        assert client._output_estimator._enabled is True
        assert client._output_estimator._safety_margin == 1.5
    finally:
        reg.shutdown_models()


def test_client_estimator_disabled(monkeypatch):
    reg = _registry(AdaptiveTokenEstimationConfig(enabled=False), monkeypatch)
    try:
        client = reg.get_language_model().client
        assert client._output_estimator._enabled is False
    finally:
        reg.shutdown_models()
