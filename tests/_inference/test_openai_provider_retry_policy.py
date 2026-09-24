"""OpenAI SDK retry policy is disabled for each scheduler invocation."""

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from openai import AsyncOpenAI, InternalServerError

from fenic._inference.openai.openai_batch_chat_completions_client import (
    OpenAIBatchChatCompletionsClient,
)
from fenic._inference.openai.openai_batch_embeddings_client import (
    OpenAIBatchEmbeddingsClient,
)
from fenic._inference.openai.openai_provider import OpenAIModelProvider
from fenic._inference.rate_limit_strategy import AdaptiveBackoffRateLimitStrategy
from fenic._inference.types import (
    FenicCompletionsRequest,
    FenicEmbeddingsRequest,
    LMRequestMessages,
)
from fenic.core.error import ExecutionError


def test_async_factory_passes_zero_retries_and_preserves_transport_options(monkeypatch):
    factory = Mock(return_value=object())
    transport = object()
    monkeypatch.setattr(
        "fenic._inference.openai.openai_provider.httpx.AsyncClient",
        lambda **kwargs: transport,
    )
    monkeypatch.setattr(
        "fenic._inference.openai.openai_provider.AsyncOpenAI", factory
    )

    result = OpenAIModelProvider(base_url="http://test").create_aio_client()

    assert result is factory.return_value
    assert factory.call_args.kwargs == {
        "base_url": "http://test",
        "http_client": transport,
        "max_retries": 0,
    }


def test_provider_async_client_does_not_retry_a_retryable_response(monkeypatch):
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503, request=request)

    actual_async_client = httpx.AsyncClient
    transport = httpx.MockTransport(handler)

    def make_http_client(**kwargs):
        return actual_async_client(transport=transport, **kwargs)

    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(
        "fenic._inference.openai.openai_provider.httpx",
        SimpleNamespace(AsyncClient=make_http_client),
    )

    async def invoke():
        client = OpenAIModelProvider(base_url="https://example.test/v1").create_aio_client()
        try:
            await client.chat.completions.create(
                model="test", messages=[{"role": "user", "content": "x"}]
            )
        finally:
            await client.close()

    with pytest.raises(InternalServerError) as error:
        asyncio.run(invoke())
    assert error.value.status_code == 503
    assert calls == 1


def _chat_request() -> FenicCompletionsRequest:
    return FenicCompletionsRequest(
        messages=LMRequestMessages(system="", examples=[], user="retry me"),
        max_completion_tokens=None,
        top_logprobs=None,
        structured_output=None,
        temperature=None,
    )


def _success_response(kind: str, request: httpx.Request) -> httpx.Response:
    if kind == "chat":
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-4.1-nano",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
            request=request,
        )
    return httpx.Response(
        200,
        json={
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1]}],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
        },
        request=request,
    )


def _scheduler_client(monkeypatch, kind: str, statuses: list[int], max_backoffs: int):
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        status = statuses[calls - 1] if calls <= len(statuses) else 503
        if status != 200:
            return httpx.Response(status, request=request)
        return _success_response(kind, request)

    sdk_client = AsyncOpenAI(
        api_key="test",
        base_url="https://example.test/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )
    monkeypatch.setattr(
        OpenAIModelProvider, "create_aio_client", lambda _self: sdk_client
    )
    strategy = AdaptiveBackoffRateLimitStrategy(rpm=1_000, min_rpm=1)
    if kind == "chat":
        client = OpenAIBatchChatCompletionsClient(
            strategy, "gpt-4.1-nano", max_backoffs=max_backoffs
        )
        request = _chat_request()
    else:
        client = OpenAIBatchEmbeddingsClient(
            strategy, "text-embedding-3-small", max_backoffs=max_backoffs
        )
        request = FenicEmbeddingsRequest("retry me")
    client.initial_backoff_seconds = 0
    return client, request, sdk_client, lambda: calls


@pytest.mark.parametrize("kind", ["chat", "embedding"])
@pytest.mark.parametrize(
    ("statuses", "max_backoffs", "succeeds"),
    [
        ([503, 200], 1, True),
        ([503, 503, 503], 2, False),
        ([503], 0, False),
    ],
)
def test_scheduler_retries_openai_5xx_with_request_local_attempt_cap(
    monkeypatch, kind, statuses, max_backoffs, succeeds
):
    client, request, sdk_client, calls = _scheduler_client(
        monkeypatch, kind, statuses, max_backoffs
    )
    try:
        if succeeds:
            assert client.make_batch_requests([request], "retry-policy")[0] is not None
        else:
            with pytest.raises(ExecutionError, match="maximum number of retries"):
                client.make_batch_requests([request], "retry-policy")
        assert calls() == max_backoffs + 1
    finally:
        client.shutdown()
        asyncio.run(sdk_client.close())


@pytest.mark.parametrize("status", [408, 409])
def test_scheduler_retries_other_sdk_retryable_statuses(monkeypatch, status):
    client, request, sdk_client, calls = _scheduler_client(
        monkeypatch, "chat", [status, 200], max_backoffs=1
    )
    try:
        assert client.make_batch_requests([request], "retry-policy")[0].completion == "ok"
        assert calls() == 2
    finally:
        client.shutdown()
        asyncio.run(sdk_client.close())


@pytest.mark.parametrize("status", [400, 401, 403])
def test_scheduler_does_not_retry_fatal_openai_request_or_access_errors(
    monkeypatch, status
):
    client, request, sdk_client, calls = _scheduler_client(
        monkeypatch, "chat", [status], max_backoffs=2
    )
    try:
        with pytest.raises(ExecutionError):
            client.make_batch_requests([request], "retry-policy")
        assert calls() == 1
    finally:
        client.shutdown()
        asyncio.run(sdk_client.close())
