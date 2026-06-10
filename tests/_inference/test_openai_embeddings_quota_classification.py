"""Quota-429 classification for the OpenAI embeddings client.

An `insufficient_quota` 429 means the account is out of credits — retrying cannot
succeed, so it must be classified Fatal (like the chat path) rather than Transient.
A Transient classification sends the request through up to `max_backoffs` exponential
backoff sleeps (~5 minutes per batch), which is how a drained account turned a
~16-minute CI suite into a 2-hour crawl.
"""

import asyncio

import httpx
from openai import RateLimitError

from fenic._inference.common_openai.openai_embeddings_core import OpenAIEmbeddingsCore
from fenic._inference.model_client import FatalException, TransientException
from fenic._inference.token_counter import TiktokenTokenCounter
from fenic._inference.types import FenicEmbeddingsRequest
from fenic.core._inference.model_catalog import ModelProvider


def _rate_limit_error(body: dict) -> RateLimitError:
    request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
    response = httpx.Response(429, json=body, request=request)
    return RateLimitError("429", response=response, body=body)


class _RaisingEmbeddings:
    def __init__(self, exc: Exception):
        self._exc = exc

    async def create(self, **_kwargs):
        raise self._exc


class _StubClient:
    def __init__(self, exc: Exception):
        self.embeddings = _RaisingEmbeddings(exc)


def _core(exc: Exception) -> OpenAIEmbeddingsCore:
    return OpenAIEmbeddingsCore(
        model="text-embedding-3-small",
        model_provider=ModelProvider.OPENAI,
        token_counter=TiktokenTokenCounter(model_name="text-embedding-3-small"),
        client=_StubClient(exc),
    )


def test_insufficient_quota_429_is_fatal():
    exc = _rate_limit_error(
        {"error": {"message": "You exceeded your current quota", "type": "insufficient_quota"}}
    )
    result = asyncio.run(_core(exc).make_single_request(FenicEmbeddingsRequest(doc="x")))
    assert isinstance(result, FatalException)


def test_generic_429_is_transient():
    exc = _rate_limit_error(
        {"error": {"message": "Rate limit reached, slow down", "type": "tokens"}}
    )
    result = asyncio.run(_core(exc).make_single_request(FenicEmbeddingsRequest(doc="x")))
    assert isinstance(result, TransientException)
