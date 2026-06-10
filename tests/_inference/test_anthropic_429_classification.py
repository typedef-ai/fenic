"""429/529 classification for the Anthropic completions client.

- A 429 carrying `x-should-retry: false` (e.g. a hard org spend-cap breach) is not
  resolvable by retrying and must be Fatal; a plain per-minute 429 stays Transient.
- A 529 `overloaded_error` is transient by definition, but the async anthropic SDK
  (0.54.0) maps it to InternalServerError, which previously fell into the
  `AnthropicError -> Fatal` catch-all and permanently failed the request.
"""

import asyncio

import httpx
import pytest

pytest.importorskip("anthropic")

from anthropic import APIStatusError, RateLimitError  # noqa: E402

from fenic._inference.anthropic.anthropic_batch_chat_completions_client import (  # noqa: E402
    AnthropicBatchCompletionsClient,
)
from fenic._inference.model_client import (  # noqa: E402
    FatalException,
    TransientException,
)
from fenic._inference.rate_limit_strategy import (
    SeparatedTokenRateLimitStrategy,  # noqa: E402
)
from fenic._inference.types import (  # noqa: E402
    FenicCompletionsRequest,
    LMRequestMessages,
)


def _status_error(status_code: int, headers: dict | None = None):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    body = {"type": "error", "error": {"type": "rate_limit_error", "message": "x"}}
    response = httpx.Response(status_code, json=body, headers=headers or {}, request=request)
    cls = RateLimitError if status_code == 429 else APIStatusError
    return cls("err", response=response, body=body)


def _request() -> FenicCompletionsRequest:
    return FenicCompletionsRequest(
        messages=LMRequestMessages(system="s", examples=[], user="u"),
        max_completion_tokens=128,
        top_logprobs=None,
        structured_output=None,
        temperature=0.0,
    )


def _client(monkeypatch, exc: Exception) -> AnthropicBatchCompletionsClient:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    client = AnthropicBatchCompletionsClient(
        model="claude-sonnet-4-6",
        rate_limit_strategy=SeparatedTokenRateLimitStrategy(
            rpm=100, input_tpm=10_000, output_tpm=10_000
        ),
    )

    async def _raise(_payload):
        raise exc

    # The classification try-block wraps the streaming handlers; raising from the
    # handler exercises the real except clauses without any network call.
    monkeypatch.setattr(client, "_handle_text_streaming_response", _raise)
    return client


def _classify(monkeypatch, exc: Exception):
    client = _client(monkeypatch, exc)
    try:
        return asyncio.run(client.make_single_request(_request()))
    finally:
        client.shutdown()


def test_plain_429_is_transient(monkeypatch):
    result = _classify(monkeypatch, _status_error(429))
    assert isinstance(result, TransientException)


def test_non_retryable_429_is_fatal(monkeypatch):
    result = _classify(monkeypatch, _status_error(429, headers={"x-should-retry": "false"}))
    assert isinstance(result, FatalException)


def test_529_overloaded_is_transient(monkeypatch):
    result = _classify(monkeypatch, _status_error(529))
    assert isinstance(result, TransientException)


def test_500_is_fatal(monkeypatch):
    result = _classify(monkeypatch, _status_error(500))
    assert isinstance(result, FatalException)
