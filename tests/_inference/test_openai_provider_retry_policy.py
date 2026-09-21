"""OpenAI SDK retry policy is disabled for each scheduler invocation."""

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from openai import InternalServerError

from fenic._inference.openai.openai_provider import OpenAIModelProvider


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
