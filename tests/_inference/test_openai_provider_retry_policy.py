"""OpenAI SDK retry policy is disabled for each scheduler invocation."""

import asyncio
from unittest.mock import Mock

import httpx
from openai import AsyncOpenAI

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


def test_actual_async_openai_does_not_retry_a_retryable_response():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503, request=request)

    async def invoke():
        client = AsyncOpenAI(
            api_key="test",
            base_url="https://example.test/v1",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        try:
            await client.chat.completions.create(
                model="test", messages=[{"role": "user", "content": "x"}]
            )
        finally:
            await client.close()

    try:
        asyncio.run(invoke())
    except Exception:
        pass
    assert calls == 1
