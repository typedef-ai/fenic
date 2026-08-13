from typing import Dict, List, Optional, Union

import polars as pl
import pytest

from fenic._backends.local.semantic_operators.classify import Classify
from fenic._backends.local.semantic_operators.extract import Extract
from fenic._backends.local.semantic_operators.map import Map
from fenic._backends.local.semantic_operators.predicate import Predicate
from fenic._inference.cache.protocol import CachedResponse, CacheStats, LLMResponseCache
from fenic._inference.language_model import LanguageModel
from fenic._inference.model_client import (
    FatalException,
    ModelClient,
    TransientException,
)
from fenic._inference.rate_limit_strategy import RateLimitStrategy, TokenEstimate
from fenic._inference.types import (
    FenicCompletionsRequest,
    FenicCompletionsResponse,
    FenicEmbeddingsRequest,
    LMRequestMessages,
)
from fenic.core._inference.model_catalog import ModelProvider
from fenic.core._inference.model_provider import ModelProviderClass
from fenic.core.error import ExecutionError
from fenic.core.metrics import LMMetrics, RMMetrics


class DummyProvider(ModelProviderClass):
    @property
    def name(self) -> str:
        return "dummy"

    def create_client(self):
        return object()

    def create_aio_client(self):
        return object()

    async def validate_api_key(self) -> None:
        return


class DummyRateLimitStrategy(RateLimitStrategy):
    def __init__(self):
        super().__init__(rpm=100)

    def backoff(self, curr_time: float) -> int:
        return 0

    def check_and_consume_rate_limit(self, token_estimate: TokenEstimate) -> bool:
        return True

    def context_tokens_per_minute(self) -> int:
        return 60_000


class DummyTokenCounter:
    def count_tokens(self, messages, ignore_file: bool = False) -> int:
        return 0

    def count_file_input_tokens(self, messages) -> int:
        return 0

    def count_file_output_tokens(self, messages) -> int:
        return 0


class FakeCache(LLMResponseCache):
    def __init__(self):
        self.get_batch_called = False
        self.set_called = False
        self.store: Dict[str, FenicCompletionsResponse] = {}

    def compute_key(self, request, model: str, profile_hash: Optional[str] = None) -> str:
        return "unused"

    def get(self, cache_key: str) -> Optional[CachedResponse]:
        value = self.store.get(cache_key)
        if value is None:
            return None
        return CachedResponse(
            completion=value.completion,
            model="test",
            cached_at=value.usage.cached_tokens if value.usage else None,  # type: ignore[arg-type]
            prompt_tokens=value.usage.prompt_tokens if value.usage else None,
            completion_tokens=value.usage.completion_tokens if value.usage else None,
            total_tokens=value.usage.total_tokens if value.usage else None,
            cached_tokens=value.usage.cached_tokens if value.usage else 0,
            thinking_tokens=value.usage.thinking_tokens if value.usage else 0,
            logprobs=value.logprobs,
            access_count=0,
        )

    def get_batch(self, cache_keys: List[str]) -> Dict[str, CachedResponse]:
        self.get_batch_called = True
        result = {}
        for key in cache_keys:
            value = self.store.get(key)
            if value is None:
                continue
            result[key] = CachedResponse(
                completion=value.completion,
                model="test",
                cached_at=value.usage.cached_tokens if value.usage else None,  # type: ignore[arg-type]
                prompt_tokens=value.usage.prompt_tokens if value.usage else None,
                completion_tokens=value.usage.completion_tokens if value.usage else None,
                total_tokens=value.usage.total_tokens if value.usage else None,
                cached_tokens=value.usage.cached_tokens if value.usage else 0,
                thinking_tokens=value.usage.thinking_tokens if value.usage else 0,
                logprobs=value.logprobs,
                access_count=0,
            )
        return result

    def set(self, cache_key: str, response, model: str) -> bool:
        self.set_called = True
        self.store[cache_key] = response
        return True

    def set_batch(self, entries):
        return 0

    def delete(self, cache_key: str) -> bool:
        return self.store.pop(cache_key, None) is not None

    def clear(self) -> int:
        count = len(self.store)
        self.store.clear()
        return count

    def stats(self) -> CacheStats:
        return CacheStats(
            hits=0,
            misses=0,
            stores=len(self.store),
            errors=0,
            hit_rate=0.0,
            total_entries=len(self.store),
            size_bytes=0,
        )

    def close(self) -> None:
        pass


class DummyEmbeddingClient(ModelClient[FenicEmbeddingsRequest, List[float]]):
    def __init__(self, cache: Optional[LLMResponseCache] = None):
        super().__init__(
            model="dummy-embedding",
            model_provider=ModelProvider.OPENAI,
            model_provider_class=DummyProvider(),
            rate_limit_strategy=DummyRateLimitStrategy(),
            token_counter=DummyTokenCounter(),
            cache=cache,
        )
        self._metrics = RMMetrics()
        self.call_count = 0

    async def make_single_request(
        self, request: FenicEmbeddingsRequest
    ) -> Union[None, List[float], TransientException, FatalException]:
        self.call_count += 1
        return [0.42]

    def estimate_tokens_for_request(self, request: FenicEmbeddingsRequest) -> TokenEstimate:
        return TokenEstimate(input_tokens=1, output_tokens=0)

    def get_metrics(self) -> RMMetrics:
        return self._metrics

    def reset_metrics(self):
        self._metrics = RMMetrics()

    def _get_max_output_token_request_limit(self, request: FenicEmbeddingsRequest) -> int:
        return 0


class DummyCompletionClient(ModelClient[FenicCompletionsRequest, FenicCompletionsResponse]):
    def __init__(self, cache: Optional[LLMResponseCache] = None):
        super().__init__(
            model="dummy-completion",
            model_provider=ModelProvider.OPENAI,
            model_provider_class=DummyProvider(),
            rate_limit_strategy=DummyRateLimitStrategy(),
            token_counter=DummyTokenCounter(),
            cache=cache,
        )
        self._metrics = LMMetrics()
        self.call_count = 0

    async def make_single_request(
        self, request: FenicCompletionsRequest
    ) -> Union[None, FenicCompletionsResponse, TransientException, FatalException]:
        self.call_count += 1
        return FenicCompletionsResponse(
            completion=f"response-for-{request.messages.user}",
            logprobs=None,
            usage=None,
        )

    def estimate_tokens_for_request(self, request: FenicCompletionsRequest) -> TokenEstimate:
        return TokenEstimate(input_tokens=1, output_tokens=1)

    def get_metrics(self) -> LMMetrics:
        return self._metrics

    def reset_metrics(self):
        self._metrics = LMMetrics()

    def _get_max_output_token_request_limit(self, request: FenicCompletionsRequest) -> int:
        return request.max_completion_tokens or 0


class ProfileAwareCompletionClient(DummyCompletionClient):
    def __init__(self, cache: Optional[LLMResponseCache] = None):
        super().__init__(cache=cache)
        self._profile_hash_value = "initial"

    def get_profile_hash(self, profile_name: Optional[str]) -> Optional[str]:
        return self._profile_hash_value


class ProviderStatusError(Exception):
    """Models SDK exceptions that require context beyond their message."""

    def __init__(self, message: str, *, response: object, body: object) -> None:
        super().__init__(message)
        self.response = response
        self.body = body


class FailingCompletionClient(DummyCompletionClient):
    async def make_single_request(
        self, request: FenicCompletionsRequest
    ) -> Union[None, FenicCompletionsResponse, TransientException, FatalException]:
        return FatalException(ProviderStatusError("Error code: 400", response=object(), body={}))


def _make_completion_request(prompt: str) -> FenicCompletionsRequest:
    messages = LMRequestMessages(system="system", examples=[], user=prompt)
    return FenicCompletionsRequest(
        messages=messages,
        max_completion_tokens=50,
        top_logprobs=None,
        structured_output=None,
        temperature=0.7,
        model_profile="default",
    )


def test_embedding_requests_skip_cache():
    fake_cache = FakeCache()
    client = DummyEmbeddingClient(cache=fake_cache)

    requests = [
        FenicEmbeddingsRequest(doc="hello world", model_profile=None),
        FenicEmbeddingsRequest(doc="hello world", model_profile=None),
    ]
    responses = client.make_batch_requests(requests, "embedding-test")
    assert responses == [[0.42], [0.42]]

    client.shutdown()

    assert fake_cache.get_batch_called is False
    assert fake_cache.set_called is False
    assert client.call_count == 1


def test_completion_requests_use_cache_and_dedup():
    fake_cache = FakeCache()
    client = DummyCompletionClient(cache=fake_cache)

    requests = [
        _make_completion_request("Hi Alice"),
        _make_completion_request("Hi Bob"),
    ]

    first = client.make_batch_requests(requests, "completion-test")
    second = client.make_batch_requests(requests, "completion-test")
    client.shutdown()

    assert first == second
    assert fake_cache.get_batch_called is True
    assert fake_cache.set_called is True
    assert client.call_count == len(requests)


def test_iter_batch_requests_is_bounded_ordered_and_deduplicates_each_block():
    client = DummyCompletionClient()
    yielded_prompts = []

    def requests():
        for prompt in ("first", "first", "second", "third"):
            yielded_prompts.append(prompt)
            yield _make_completion_request(prompt)

    try:
        responses = client.iter_batch_requests(
            requests(),
            "stream-test",
            batch_size=2,
        )

        first = next(responses)
        assert first is not None
        assert first.completion == "response-for-first"
        assert yielded_prompts == ["first", "first"]

        remaining = list(responses)
        assert [response.completion for response in remaining if response] == [
            "response-for-first",
            "response-for-second",
            "response-for-third",
        ]
        assert client.call_count == 3
    finally:
        client.shutdown()


def test_iter_batch_requests_uses_cache_across_blocks():
    fake_cache = FakeCache()
    client = DummyCompletionClient(cache=fake_cache)
    requests = [
        _make_completion_request("first"),
        _make_completion_request("second"),
        _make_completion_request("first"),
    ]

    try:
        responses = list(
            client.iter_batch_requests(
                requests,
                "stream-cache-test",
                batch_size=2,
            )
        )
    finally:
        client.shutdown()

    assert [response.completion for response in responses if response] == [
        "response-for-first",
        "response-for-second",
        "response-for-first",
    ]
    assert fake_cache.get_batch_called is True
    assert fake_cache.set_called is True
    assert client.call_count == 2


def test_iter_batch_requests_normalizes_provider_errors():
    client = FailingCompletionClient()

    try:
        with pytest.raises(ExecutionError, match="Error code: 400") as exc_info:
            list(
                client.iter_batch_requests(
                    [_make_completion_request("Hi Alice")],
                    "stream-error-test",
                    batch_size=1,
                )
            )

        assert isinstance(exc_info.value.__cause__, ProviderStatusError)
    finally:
        client.shutdown()


def test_iter_batch_requests_rejects_non_positive_batch_size():
    client = DummyCompletionClient()
    try:
        with pytest.raises(ValueError, match="batch_size must be positive"):
            list(client.iter_batch_requests([], "stream-test", batch_size=0))
    finally:
        client.shutdown()


def test_row_local_operators_keep_streaming_opt_in_by_default():
    assert all(
        operator.stream_requests is False
        for operator in (Map, Extract, Classify, Predicate)
    )


def test_map_can_opt_into_ordered_bounded_model_client_batches(monkeypatch):
    client = DummyCompletionClient()
    client.model = "gpt-4.1-nano"
    model = LanguageModel(client)
    operator = Map(
        input=pl.Series("input", ["first", "second", "third"]),
        jinja_template="{{ input }}",
        model=model,
        max_tokens=50,
        temperature=0,
    )
    monkeypatch.setattr(Map, "stream_requests", True)
    operator.request_batch_size = 2

    try:
        result = operator.execute()
    finally:
        client.shutdown()

    assert result.to_list() == [
        "response-for-first",
        "response-for-second",
        "response-for-third",
    ]
    assert client.call_count == 3


def test_profile_hash_changes_cache_key():
    fake_cache = FakeCache()
    client = ProfileAwareCompletionClient(cache=fake_cache)
    request = _make_completion_request("Hi Alice")

    client._profile_hash_value = "hash-A"
    client.make_batch_requests([request], "profile-test")
    first_calls = client.call_count
    first_store_size = len(fake_cache.store)

    client._profile_hash_value = "hash-B"
    client.make_batch_requests([request], "profile-test")
    client.shutdown()

    assert client.call_count == first_calls + 1
    assert len(fake_cache.store) == first_store_size + 1


def test_provider_errors_are_normalized_before_crossing_polars_boundary():
    client = FailingCompletionClient()

    try:
        with pytest.raises(ExecutionError, match="Error code: 400") as exc_info:
            client.make_batch_requests([_make_completion_request("Hi Alice")], "completion-test")

        assert isinstance(exc_info.value.__cause__, ProviderStatusError)
    finally:
        client.shutdown()
