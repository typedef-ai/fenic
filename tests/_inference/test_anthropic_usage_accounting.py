"""Usage accounting for the Anthropic completions client.

`cache_creation_input_tokens` and `cache_read_input_tokens` are Optional in the
Anthropic SDK and are absent whenever prompt caching is not in play, so the
accounting block must treat them as zero instead of adding None.
"""

import asyncio

import pytest

pytest.importorskip("anthropic")

from anthropic.types import Usage  # noqa: E402

from fenic._inference.anthropic.anthropic_batch_chat_completions_client import (  # noqa: E402
    AnthropicBatchCompletionsClient,
)
from fenic._inference.rate_limit_strategy import (
    SeparatedTokenRateLimitStrategy,  # noqa: E402
)
from fenic._inference.types import (  # noqa: E402
    FenicCompletionsRequest,
    LMRequestMessages,
)
from fenic.core._inference.model_catalog import (  # noqa: E402
    ModelProvider,
    model_catalog,
)

MODEL = "claude-sonnet-4-6"


def _request() -> FenicCompletionsRequest:
    return FenicCompletionsRequest(
        messages=LMRequestMessages(system="s", examples=[], user="u"),
        max_completion_tokens=128,
        top_logprobs=None,
        structured_output=None,
        temperature=0.0,
    )


def _respond_with(monkeypatch, usage: Usage) -> AnthropicBatchCompletionsClient:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    client = AnthropicBatchCompletionsClient(
        model=MODEL,
        rate_limit_strategy=SeparatedTokenRateLimitStrategy(
            rpm=100, input_tpm=10_000, output_tpm=10_000
        ),
    )

    async def _stream(_payload):
        return "hello", usage

    # Returning from the streaming handler exercises the real accounting block
    # without any network call.
    monkeypatch.setattr(client, "_handle_text_streaming_response", _stream)
    return client


def _make_request(monkeypatch, usage: Usage):
    client = _respond_with(monkeypatch, usage)
    try:
        return asyncio.run(client.make_single_request(_request())), client
    finally:
        client.shutdown()


def test_absent_cache_token_fields_are_treated_as_zero(monkeypatch):
    # A Usage without any cache fields is what Anthropic returns when prompt
    # caching is not in play; both fields default to None.
    usage = Usage(input_tokens=7, output_tokens=3)
    assert usage.cache_creation_input_tokens is None
    assert usage.cache_read_input_tokens is None

    result, client = _make_request(monkeypatch, usage)

    assert result.usage.prompt_tokens == 7
    assert result.usage.cached_tokens == 0
    assert result.usage.completion_tokens == 3
    assert result.usage.total_tokens == 10
    assert client.get_metrics().num_cached_input_tokens == 0
    assert client.get_metrics().num_uncached_input_tokens == 7


@pytest.mark.parametrize(
    ("cache_read", "cache_written"),
    [(None, None), (0, 0), (None, 4), (4, None)],
)
def test_nullable_cache_token_fields_do_not_raise(monkeypatch, cache_read, cache_written):
    usage = Usage(
        input_tokens=5,
        output_tokens=2,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_written,
    )

    result, _ = _make_request(monkeypatch, usage)

    assert result.usage.prompt_tokens == 5 + (cache_read or 0) + (cache_written or 0)
    assert result.usage.cached_tokens == (cache_read or 0)


def test_absent_cache_token_fields_contribute_no_cost(monkeypatch):
    # Covers the second unguarded read: the raw cache_creation_input_tokens was
    # also passed to calculate_completion_model_cost, where None * float raises.
    usage = Usage(input_tokens=1000, output_tokens=500)

    _, client = _make_request(monkeypatch, usage)

    # Derive the expectation from the catalog so catalog price updates do not
    # break this test; the cache components must contribute exactly nothing.
    parameters = model_catalog.get_completion_model_parameters(
        ModelProvider.ANTHROPIC, MODEL
    )
    expected_cost = (
        1000 * parameters.input_token_cost + 500 * parameters.output_token_cost
    )
    assert client.get_metrics().cost == pytest.approx(expected_cost)


def test_populated_cache_token_fields_are_still_counted(monkeypatch):
    # Guards against the None-coalescing accidentally discarding real counts.
    usage = Usage(
        input_tokens=10,
        output_tokens=4,
        cache_read_input_tokens=6,
        cache_creation_input_tokens=2,
    )

    result, client = _make_request(monkeypatch, usage)

    assert result.usage.prompt_tokens == 18
    assert result.usage.cached_tokens == 6
    assert result.usage.total_tokens == 22
    assert client.get_metrics().num_cached_input_tokens == 6
    assert client.get_metrics().num_uncached_input_tokens == 10
