"""Regression coverage for user-declared OpenAI-compatible models."""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import httpx
import pytest
from openai import AsyncOpenAI
from pydantic import ValidationError as PydanticValidationError

from fenic import (
    OpenAICompatibleEmbeddingModel,
    OpenAICompatibleLanguageModel,
    OpenAIEmbeddingModel,
    OpenAILanguageModel,
    SemanticConfig,
    SessionConfig,
)
from fenic._backends.local.model_registry import SessionModelRegistry
from fenic._inference.openai.openai_batch_chat_completions_client import (
    OpenAIBatchChatCompletionsClient,
)
from fenic._inference.openai.openai_batch_embeddings_client import (
    OpenAIBatchEmbeddingsClient,
)
from fenic._inference.types import (
    FenicCompletionsRequest,
    FenicEmbeddingsRequest,
    LMRequestMessages,
)
from fenic.core._inference.model_catalog import (
    ModelProvider,
    ProviderModelCollection,
    model_catalog,
)
from fenic.core.error import ConfigurationError


@pytest.fixture(autouse=True)
def isolated_catalog(monkeypatch):
    collections = deepcopy(model_catalog.provider_model_collections)
    collections[ModelProvider.OPENAI_COMPATIBLE] = ProviderModelCollection(
        ModelProvider.OPENAI_COMPATIBLE
    )
    monkeypatch.setattr(
        model_catalog,
        "provider_model_collections",
        collections,
    )


def test_openai_typo_keeps_catalog_literal_error():
    with pytest.raises(PydanticValidationError) as exc:
        OpenAILanguageModel(model_name="gpt-4.1-nanoo", rpm=100, tpm=100)
    assert exc.value.errors()[0]["type"] == "literal_error"
    assert "gpt-4.1-nano" in str(exc.value)
    assert "base_url" not in str(exc.value)
    assert "model_parameters" not in str(exc.value)


@pytest.mark.parametrize("custom_first", [False, True])
@pytest.mark.parametrize("embedding", [False, True])
def test_openai_catalog_collision_preserves_both_models(custom_first, embedding):
    cls = OpenAIEmbeddingModel if embedding else OpenAILanguageModel
    name = "text-embedding-3-small" if embedding else "gpt-4.1-nano"
    lookup = (
        model_catalog.get_embedding_model_parameters
        if embedding
        else model_catalog.get_completion_model_parameters
    )
    original = lookup(ModelProvider.OPENAI, name)
    original_fields = vars(original).copy()
    fields = (
        dict(output_dimensions=768, max_input_size=512, input_token_cost=2e-7)
        if embedding
        else dict(
            context_window_length=8192,
            max_output_tokens=512,
            input_token_cost=2e-7,
            output_token_cost=8e-7,
        )
    )

    def custom():
        compatible_cls = (
            OpenAICompatibleEmbeddingModel
            if embedding
            else OpenAICompatibleLanguageModel
        )
        return compatible_cls(
            model_name=name,
            rpm=100,
            tpm=100,
            base_url="https://local.example.com/v1",
            model_parameters=compatible_cls.ModelParameters(**fields),
        )

    def catalog():
        return cls(model_name=name, rpm=100, tpm=100)

    if custom_first:
        local, direct = custom(), catalog()
    else:
        direct, local = catalog(), custom()
    configs = {"local": local, "direct": direct}
    semantic = (
        SemanticConfig(embedding_models=configs, default_embedding_model="local")
        if embedding
        else SemanticConfig(language_models=configs, default_language_model="local")
    )
    resolved = SessionConfig(semantic=semantic)._to_resolved_config().semantic
    models = resolved.embedding_models if embedding else resolved.language_models
    local_config = models.model_configs["local"]
    declared = lookup(local_config.model_provider, name)
    for key, value in fields.items():
        catalog_key = "default_dimensions" if key == "output_dimensions" else key
        assert getattr(declared, catalog_key) == value
    assert local_config.model_provider != ModelProvider.OPENAI
    assert models.model_configs["direct"].model_provider == ModelProvider.OPENAI
    assert lookup(ModelProvider.OPENAI, name) is original
    assert vars(original) == original_fields


@pytest.mark.parametrize("embedding", [False, True])
def test_conflicting_redeclaration_is_rejected(embedding):
    cls = OpenAICompatibleEmbeddingModel if embedding else OpenAICompatibleLanguageModel
    fields = (
        dict(output_dimensions=768, max_input_size=512)
        if embedding
        else dict(context_window_length=8192, max_output_tokens=512)
    )
    kwargs = dict(
        model_name=f"test-redeclaration-{embedding}",
        rpm=100,
        tpm=100,
        base_url="https://local.example.com/v1",
    )
    first = cls(**kwargs, model_parameters=cls.ModelParameters(**fields))
    identical = cls(**kwargs, model_parameters=cls.ModelParameters(**fields))
    assert first == identical
    with pytest.raises(ConfigurationError, match="Conflicting.*test-redeclaration"):
        cls(
            **kwargs,
            model_parameters=cls.ModelParameters(**fields, input_token_cost=1e-6),
        )


@pytest.mark.parametrize(
    "cls", [OpenAICompatibleLanguageModel, OpenAICompatibleEmbeddingModel]
)
@pytest.mark.parametrize("field", ["model_name", "base_url", "model_parameters"])
def test_compatible_fields_are_required(cls, field):
    assert cls.model_fields[field].is_required()
    with pytest.raises(PydanticValidationError) as exc:
        cls(rpm=100, tpm=100)
    assert field in [error["loc"][0] for error in exc.value.errors()]
    assert cls.model_fields["model_name"].annotation is str
    assert "model_parameters" not in OpenAILanguageModel.model_fields
    assert "model_parameters" not in OpenAIEmbeddingModel.model_fields


def test_identical_concurrent_registration():
    def construct(_):
        return OpenAICompatibleLanguageModel(
            model_name="test-compatible-concurrent",
            rpm=100,
            tpm=100,
            base_url="https://local.example.com/v1",
            model_parameters=dict(context_window_length=8192, max_output_tokens=512),
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        models = list(pool.map(construct, range(24)))
    assert all(model == models[0] for model in models)


@pytest.mark.parametrize("priced", [False, True])
def test_existing_clients_route_and_account_declared_models(monkeypatch, priced):
    """Exercise the real SDK and clients with an in-memory HTTP transport."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    requests = []
    base_url = "https://local.example.com/custom/v1"

    def handle(request):
        requests.append(request)
        body = json.loads(request.content)
        if request.url.path.endswith("/chat/completions"):
            assert body["model"] == "gpt-4.1-nano"
            assert body["max_completion_tokens"] == 512
            return httpx.Response(
                200,
                json={
                    "id": "test",
                    "object": "chat.completion",
                    "created": 0,
                    "model": body["model"],
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": "hello"},
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1000,
                        "completion_tokens": 1000,
                        "total_tokens": 2000,
                    },
                },
            )
        assert request.url.path.endswith("/embeddings")
        assert body["model"] == "text-embedding-3-small"
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": body["model"],
                "data": [{"object": "embedding", "index": 0, "embedding": [0.25, 0.5]}],
                "usage": {"prompt_tokens": 1000, "total_tokens": 1000},
            },
        )

    def sdk_client(**kwargs):
        kwargs["http_client"] = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        return AsyncOpenAI(**kwargs)

    async def reject_validation(_):
        pytest.fail("Custom endpoint construction must not validate API keys")

    monkeypatch.setattr(
        "fenic._inference.openai.openai_provider.AsyncOpenAI", sdk_client
    )
    monkeypatch.setattr(
        "fenic._backends.local.model_registry._validate_provider_api_keys",
        reject_validation,
    )
    cost_fields = dict(input_token_cost=2e-7, output_token_cost=8e-7) if priced else {}
    config = SessionConfig(
        semantic=SemanticConfig(
            language_models={
                "local": OpenAICompatibleLanguageModel(
                    model_name="gpt-4.1-nano",
                    base_url=base_url,
                    rpm=100,
                    tpm=100_000,
                    model_parameters=dict(
                        context_window_length=8192,
                        max_output_tokens=512,
                        **cost_fields,
                    ),
                )
            },
            embedding_models={
                "local": OpenAICompatibleEmbeddingModel(
                    model_name="text-embedding-3-small",
                    base_url=base_url,
                    rpm=100,
                    tpm=100_000,
                    model_parameters=dict(
                        output_dimensions=2,
                        max_input_size=512,
                        input_token_cost=2e-7 if priced else 0.0,
                    ),
                )
            },
        )
    )
    restored = SessionConfig.model_validate_json(config.to_json())
    assert isinstance(
        restored.semantic.language_models["local"], OpenAICompatibleLanguageModel
    )
    assert isinstance(
        restored.semantic.embedding_models["local"], OpenAICompatibleEmbeddingModel
    )
    registry = SessionModelRegistry(restored._to_resolved_config().semantic)
    try:
        assert not requests
        language_client = registry.get_language_model().client
        embedding_client = registry.get_embedding_model().client
        assert isinstance(language_client, OpenAIBatchChatCompletionsClient)
        assert isinstance(embedding_client, OpenAIBatchEmbeddingsClient)
        for client in [language_client, embedding_client]:
            assert client.model_provider == ModelProvider.OPENAI_COMPATIBLE
        assert language_client._model_parameters.context_window_length == 8192
        assert embedding_client._core._model_parameters.max_input_size == 512
        completion = asyncio.run(
            language_client.make_single_request(
                FenicCompletionsRequest(
                    messages=LMRequestMessages(
                        system="hello", examples=[], user="hello"
                    ),
                    max_completion_tokens=512,
                    top_logprobs=None,
                    structured_output=None,
                    temperature=None,
                )
            )
        )
        vector = asyncio.run(
            embedding_client.make_single_request(FenicEmbeddingsRequest(doc="hello"))
        )
        assert completion.completion == "hello"
        assert vector == [0.25, 0.5]
        assert [str(request.url) for request in requests] == [
            f"{base_url}/chat/completions",
            f"{base_url}/embeddings",
        ]
        assert language_client.get_metrics().cost == pytest.approx(
            1e-3 if priced else 0
        )
        assert embedding_client.get_metrics().cost == pytest.approx(
            2e-4 if priced else 0
        )
    finally:
        registry.shutdown_models()
