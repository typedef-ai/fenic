"""Usage accounting for the Cohere embeddings client.

`meta`, `meta.billed_units` and `billed_units.input_tokens` are all Optional in
the Cohere SDK, so the billed token count must be read defensively and fall back
to the local token counter when it is not populated.
"""

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("cohere")

from fenic._inference.cohere.cohere_batch_embeddings_client import (  # noqa: E402
    CohereBatchEmbeddingsClient,
)
from fenic._inference.rate_limit_strategy import (
    UnifiedTokenRateLimitStrategy,  # noqa: E402
)
from fenic._inference.types import FenicEmbeddingsRequest  # noqa: E402

DOC = "a document to embed"


def _request() -> FenicEmbeddingsRequest:
    return FenicEmbeddingsRequest(doc=DOC)


def _respond_with(monkeypatch, meta) -> CohereBatchEmbeddingsClient:
    monkeypatch.setenv("COHERE_API_KEY", "test-key")
    client = CohereBatchEmbeddingsClient(
        model="embed-v4.0",
        rate_limit_strategy=UnifiedTokenRateLimitStrategy(rpm=100, tpm=10_000),
    )

    async def _embed(**_kwargs):
        return SimpleNamespace(
            embeddings=SimpleNamespace(float=[[0.1, 0.2, 0.3]]),
            meta=meta,
        )

    # Replacing the SDK call exercises the real accounting block with no network.
    monkeypatch.setattr(client._client, "embed", _embed)
    return client


def _make_request(monkeypatch, meta):
    client = _respond_with(monkeypatch, meta)
    try:
        return asyncio.run(client.make_single_request(_request())), client
    finally:
        client.shutdown()


@pytest.mark.parametrize(
    "meta",
    [
        None,
        SimpleNamespace(billed_units=None),
        SimpleNamespace(billed_units=SimpleNamespace(input_tokens=None)),
    ],
    ids=["no_meta", "no_billed_units", "no_input_tokens"],
)
def test_missing_billed_units_falls_back_to_token_counter(monkeypatch, meta):
    result, client = _make_request(monkeypatch, meta)

    assert result == [0.1, 0.2, 0.3]
    # Falls back to the local counter rather than raising or counting None.
    assert client.get_metrics().num_input_tokens == client.token_counter.count_tokens(DOC)
    assert client.get_metrics().num_requests == 1


def test_billed_input_tokens_are_used_when_present(monkeypatch):
    meta = SimpleNamespace(billed_units=SimpleNamespace(input_tokens=42))

    result, client = _make_request(monkeypatch, meta)

    assert result == [0.1, 0.2, 0.3]
    assert client.get_metrics().num_input_tokens == 42
