import pytest
from pydantic import ValidationError

from fenic.api.session.config import (
    AdaptiveTokenEstimationConfig,
    OpenAILanguageModel,
    SemanticConfig,
    SessionConfig,
)


def test_defaults_enabled_with_margin():
    cfg = AdaptiveTokenEstimationConfig()
    assert cfg.enabled is True
    assert cfg.safety_margin == 1.15


def test_safety_margin_bounds():
    with pytest.raises(ValidationError):
        AdaptiveTokenEstimationConfig(safety_margin=0.5)
    with pytest.raises(ValidationError):
        AdaptiveTokenEstimationConfig(safety_margin=10.0)


def _session(adaptive=None):
    return SessionConfig(
        app_name="t",
        semantic=SemanticConfig(
            language_models={"m": OpenAILanguageModel(model_name="gpt-4o-mini", rpm=100, tpm=1000)},
            default_language_model="m",
            adaptive_token_estimation=adaptive,
        ),
    )


def test_resolution_defaults_to_enabled_when_absent():
    resolved = _session(None)._to_resolved_config()
    ate = resolved.semantic.adaptive_token_estimation
    assert ate.enabled is True
    assert ate.safety_margin == 1.15


def test_resolution_respects_disabled():
    resolved = _session(AdaptiveTokenEstimationConfig(enabled=False))._to_resolved_config()
    assert resolved.semantic.adaptive_token_estimation.enabled is False
