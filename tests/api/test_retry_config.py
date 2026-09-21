"""Public retry configuration reaches the local language clients."""

import pytest
from pydantic import ValidationError as PydanticValidationError

from fenic._backends.local.model_registry import SessionModelRegistry
from fenic.api.session.config import (
    CloudConfig,
    OpenAILanguageModel,
    SemanticConfig,
    SessionConfig,
    TypeSafeLanguageModel,
)
from fenic.core._resolved_session_config import (
    ResolvedOpenAIModelConfig,
    ResolvedTypeSafeModelConfig,
)
from fenic.core.error import ConfigurationError


def _session_config(model):
    return SessionConfig(
        semantic=SemanticConfig(language_models={"model": model})
    )


@pytest.mark.parametrize(
    ("model", "default"),
    [
        (OpenAILanguageModel(model_name="gpt-4.1-nano", rpm=1, tpm=1), 10),
        (TypeSafeLanguageModel(model_name="jev-1.13.0", rpm=1, tpm=1), 2),
    ],
)
def test_retry_defaults_are_preserved(model, default):
    assert model.max_backoffs == default


@pytest.mark.parametrize(
    "model_factory",
    [
        lambda value: OpenAILanguageModel(
            model_name="gpt-4.1-nano", rpm=1, tpm=1, max_backoffs=value
        ),
        lambda value: TypeSafeLanguageModel(
            model_name="jev-1.13.0", rpm=1, tpm=1, max_backoffs=value
        ),
    ],
)
@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_retry_limit_requires_a_nonnegative_integer(model_factory, value):
    with pytest.raises(PydanticValidationError):
        model_factory(value)


@pytest.mark.parametrize(
    ("model", "resolved_type"),
    [
        (
            OpenAILanguageModel(
                model_name="gpt-4.1-nano", rpm=1, tpm=1, max_backoffs=0
            ),
            ResolvedOpenAIModelConfig,
        ),
        (
            TypeSafeLanguageModel(
                model_name="jev-1.13.0", rpm=1, tpm=1, max_backoffs=0
            ),
            ResolvedTypeSafeModelConfig,
        ),
    ],
)
def test_zero_round_trips_through_json_and_resolved_config(model, resolved_type):
    original = _session_config(model)
    parsed = SessionConfig.model_validate_json(original.to_json())
    resolved = parsed._to_resolved_config()
    config = resolved.semantic.language_models.model_configs["model"]

    assert parsed.semantic.language_models["model"].max_backoffs == 0
    assert isinstance(config, resolved_type)
    assert config.max_backoffs == 0


def test_registry_passes_the_resolved_limit_to_language_clients(monkeypatch):
    constructed = []

    class FakeOpenAI:
        def __init__(self, **kwargs):
            constructed.append(("openai", kwargs))
            self.model_provider = "openai"
            self.model = kwargs["model"]

    class FakeTypeSafe:
        def __init__(self, **kwargs):
            constructed.append(("typesafe", kwargs))
            self.model_provider = "typesafe"
            self.model = kwargs["model"]

    monkeypatch.setattr(
        "fenic._backends.local.model_registry.OpenAIBatchChatCompletionsClient",
        FakeOpenAI,
    )
    monkeypatch.setattr(
        "fenic._inference.typesafe.typesafe_system_one_client.TypeSafeSystemOneClient",
        FakeTypeSafe,
    )
    monkeypatch.setattr(
        "fenic._backends.local.model_registry.LanguageModel", lambda client: client
    )
    registry = object.__new__(SessionModelRegistry)

    openai = _session_config(
        OpenAILanguageModel(
            model_name="gpt-4.1-nano", rpm=1, tpm=1, max_backoffs=0
        )
    )._to_resolved_config().semantic.language_models.model_configs["model"]
    typesafe = _session_config(
        TypeSafeLanguageModel(
            model_name="jev-1.13.0", rpm=1, tpm=1, max_backoffs=0
        )
    )._to_resolved_config().semantic.language_models.model_configs["model"]

    SessionModelRegistry._initialize_language_model(registry, openai)
    SessionModelRegistry._initialize_language_model(registry, typesafe)

    assert [kind for kind, _ in constructed] == ["openai", "typesafe"]
    assert [kwargs["max_backoffs"] for _, kwargs in constructed] == [0, 0]


@pytest.mark.parametrize(
    "model",
    [
        OpenAILanguageModel(
            model_name="gpt-4.1-nano", rpm=1, tpm=1, max_backoffs=0
        ),
        TypeSafeLanguageModel(
            model_name="jev-1.13.0", rpm=1, tpm=1, max_backoffs=0
        ),
    ],
)
def test_cloud_rejects_a_nondefault_local_retry_override(model):
    with pytest.raises(ConfigurationError, match="max_backoffs"):
        SessionConfig(
            semantic=SemanticConfig(
                language_models={"model": model}
            ),
            cloud=CloudConfig(),
        )


@pytest.mark.parametrize(
    "model",
    [
        OpenAILanguageModel(model_name="gpt-4.1-nano", rpm=1, tpm=1),
        TypeSafeLanguageModel(model_name="jev-1.13.0", rpm=1, tpm=1),
    ],
)
def test_cloud_allows_default_retry_configuration(model):
    config = SessionConfig(
        semantic=SemanticConfig(language_models={"model": model}),
        cloud=CloudConfig(),
    )
    assert config.cloud is not None
