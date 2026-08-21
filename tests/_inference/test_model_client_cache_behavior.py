import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
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
    def __init__(self, rpm: int = 100):
        super().__init__(rpm=rpm)

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
        self.get_batch_hit_count = 0
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
        self.get_batch_hit_count += len(result)
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
    def __init__(
        self,
        cache: Optional[LLMResponseCache] = None,
        *,
        rate_limit_rpm: int = 100,
    ):
        super().__init__(
            model="dummy-completion",
            model_provider=ModelProvider.OPENAI,
            model_provider_class=DummyProvider(),
            rate_limit_strategy=DummyRateLimitStrategy(rate_limit_rpm),
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


class SlidingWindowCompletionClient(DummyCompletionClient):
    def __init__(
        self,
        *,
        cache: Optional[LLMResponseCache] = None,
        fail_second: bool = False,
        rate_limit_rpm: int = 100,
        block_after_first: bool = False,
        block_first: bool = False,
        block_second: bool = True,
        blocked_prompts: Optional[set[str]] = None,
    ):
        super().__init__(cache=cache, rate_limit_rpm=rate_limit_rpm)
        self.fail_second = fail_second
        self.block_after_first = block_after_first
        self.block_first = block_first
        self.block_second = block_second
        self.blocked_prompts = blocked_prompts or set()
        self.first_started = threading.Event()
        self.second_started = threading.Event()
        self.third_started = threading.Event()
        self.fourth_started = threading.Event()
        self.third_physical_request_started = threading.Event()
        self.release_second = threading.Event()
        self._active_requests = 0
        self._active_requests_lock = threading.Lock()
        self.max_active_requests = 0
        self.physical_request_count = 0

    async def make_single_request(
        self, request: FenicCompletionsRequest
    ) -> Union[None, FenicCompletionsResponse, TransientException, FatalException]:
        with self._active_requests_lock:
            self._active_requests += 1
            self.max_active_requests = max(
                self.max_active_requests, self._active_requests
            )
            self.physical_request_count += 1
            if self.physical_request_count >= 3:
                self.third_physical_request_started.set()

        try:
            prompt = request.messages.user
            if prompt == "first":
                self.first_started.set()
            elif prompt == "second":
                self.second_started.set()
            elif prompt == "third":
                self.third_started.set()
            elif prompt == "fourth":
                self.fourth_started.set()

            if (
                (self.block_first and prompt == "first")
                or (self.block_second and prompt == "second")
                or (self.block_after_first and prompt != "first")
                or prompt in self.blocked_prompts
            ):
                await asyncio.to_thread(self.release_second.wait)

            self.call_count += 1
            if prompt == "second" and self.fail_second:
                return FatalException(
                    ProviderStatusError("Error code: 400", response=object(), body={})
                )
            return FenicCompletionsResponse(
                completion=f"response-for-{prompt}",
                logprobs=None,
                usage=None,
            )
        finally:
            with self._active_requests_lock:
                self._active_requests -= 1


class DedupTrackingCompletionClient(SlidingWindowCompletionClient):
    def __init__(self, *, dedup_ceiling: int, **kwargs):
        super().__init__(**kwargs)
        self.dedup_ceiling = dedup_ceiling
        self.dedup_at_capacity = threading.Event()
        self.dedup_overflow = threading.Event()
        self.max_live_dedup_entries = 0
        self.live_dedup_entries = None

    def _get_or_create_request_future(
        self,
        unique_futures,
        request,
        request_key=None,
    ):
        result = super()._get_or_create_request_future(
            unique_futures,
            request,
            request_key,
        )
        self.live_dedup_entries = unique_futures
        live_entries = len(unique_futures)
        self.max_live_dedup_entries = max(
            self.max_live_dedup_entries,
            live_entries,
        )
        if live_entries == self.dedup_ceiling:
            self.dedup_at_capacity.set()
        elif live_entries > self.dedup_ceiling:
            self.dedup_overflow.set()
        return result


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


def _counting_completion_requests(
    prompts,
    *,
    admission_watermark: int,
    resume_at_capacity: Optional[threading.Event] = None,
):
    admitted_prompts = []
    admission_at_capacity = threading.Event()
    admission_overflow = threading.Event()

    def requests():
        for prompt in prompts:
            admitted_prompts.append(prompt)
            if len(admitted_prompts) == admission_watermark:
                admission_at_capacity.set()
                if resume_at_capacity is not None:
                    resume_at_capacity.wait()
            elif len(admitted_prompts) > admission_watermark:
                admission_overflow.set()
            yield _make_completion_request(prompt)

    return (
        requests(),
        admitted_prompts,
        admission_at_capacity,
        admission_overflow,
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


def test_iter_batch_requests_is_bounded_ordered_and_deduplicates_within_live_window():
    client = DummyCompletionClient(rate_limit_rpm=2)
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
        assert yielded_prompts == ["first", "first", "second", "third"]

        remaining = list(responses)
        assert [response.completion for response in remaining if response] == [
            "response-for-first",
            "response-for-second",
            "response-for-third",
        ]
        assert client.call_count == 3
    finally:
        client.shutdown()


def test_iter_batch_requests_admits_successor_before_a_slow_window_peer_settles():
    client = SlidingWindowCompletionClient(rate_limit_rpm=2)

    try:
        responses = client.iter_batch_requests(
            [
                _make_completion_request(prompt)
                for prompt in ("first", "second", "third")
            ],
            "sliding-window-test",
            batch_size=2,
        )

        first = next(responses)
        assert first is not None
        assert first.completion == "response-for-first"
        assert client.second_started.wait(timeout=1)
        assert client.third_started.wait(timeout=1)
        assert client.max_active_requests <= max(2, client.rate_limit_strategy.rpm)

        client.release_second.set()
        assert [response.completion for response in responses if response] == [
            "response-for-second",
            "response-for-third",
        ]
    finally:
        client.release_second.set()
        client.shutdown()


def test_iter_batch_requests_refills_pending_slots_behind_a_blocked_first_response():
    look_ahead_basis = 2
    pending_admission_cap = 3 * look_ahead_basis
    client = SlidingWindowCompletionClient(
        rate_limit_rpm=look_ahead_basis,
        block_first=True,
    )
    executor = ThreadPoolExecutor(max_workers=1)
    admitted_beyond_pending_cap = threading.Event()
    prompts = ("first", "second") + tuple(
        f"request-{index}" for index in range(2, pending_admission_cap + 4)
    )

    def requests():
        for index, prompt in enumerate(prompts, start=1):
            if index > pending_admission_cap:
                admitted_beyond_pending_cap.set()
            yield _make_completion_request(prompt)

    try:
        collected = executor.submit(
            list,
            client.iter_batch_requests(requests(), "decoupled-admission-test", batch_size=2),
        )
        assert client.first_started.wait(timeout=1)
        assert admitted_beyond_pending_cap.wait(timeout=1)

        client.release_second.set()
        results = collected.result(timeout=2)
        assert [response.completion for response in results if response] == [
            f"response-for-{prompt}" for prompt in prompts
        ]
    finally:
        client.release_second.set()
        executor.shutdown(wait=True, cancel_futures=True)
        client.shutdown()


def test_iter_batch_requests_keeps_admission_within_combined_slot_bound():
    look_ahead_basis = 2
    completed_result_cap = 10 * look_ahead_basis
    pending_admission_cap = 3 * look_ahead_basis
    combined_slot_bound = completed_result_cap + pending_admission_cap
    client = SlidingWindowCompletionClient(
        rate_limit_rpm=look_ahead_basis,
        blocked_prompts={"first", "second"}
        | {f"request-{index}" for index in range(22, 26)},
    )
    executor = ThreadPoolExecutor(max_workers=1)
    prompts = ("first", "second") + tuple(
        f"request-{index}" for index in range(2, 27)
    )
    admitted_prompts = []
    admitted_beyond_pending_cap = threading.Event()
    admitted_beyond_combined_cap = threading.Event()

    def requests():
        for prompt in prompts:
            admitted_prompts.append(prompt)
            if len(admitted_prompts) > pending_admission_cap:
                admitted_beyond_pending_cap.set()
            if len(admitted_prompts) > combined_slot_bound:
                admitted_beyond_combined_cap.set()
            yield _make_completion_request(prompt)

    try:
        collected = executor.submit(
            list,
            client.iter_batch_requests(requests(), "two-cap-bound-test", batch_size=2),
        )
        assert client.first_started.wait(timeout=1)
        assert admitted_beyond_pending_cap.wait(timeout=1)
        assert len(admitted_prompts) <= combined_slot_bound
        assert not admitted_beyond_combined_cap.wait(timeout=1)

        client.release_second.set()
        results = collected.result(timeout=2)
        assert [response.completion for response in results if response] == [
            f"response-for-{prompt}" for prompt in prompts
        ]
    finally:
        client.release_second.set()
        executor.shutdown(wait=True, cancel_futures=True)
        client.shutdown()


def test_iter_batch_requests_normalizes_later_window_failure_after_successor_admission():
    client = SlidingWindowCompletionClient(fail_second=True, rate_limit_rpm=2)

    try:
        responses = client.iter_batch_requests(
            [
                _make_completion_request(prompt)
                for prompt in ("first", "second", "third")
            ],
            "sliding-window-error-test",
            batch_size=2,
        )

        first = next(responses)
        assert first is not None
        assert first.completion == "response-for-first"
        assert client.third_started.wait(timeout=1)

        client.release_second.set()
        with pytest.raises(ExecutionError, match="Error code: 400") as exc_info:
            next(responses)

        assert isinstance(exc_info.value.__cause__, ProviderStatusError)
        assert client.call_count == 3
    finally:
        client.release_second.set()
        client.shutdown()


def test_iter_batch_requests_buffers_a_later_failure_until_its_ordered_turn():
    client = SlidingWindowCompletionClient(
        fail_second=True,
        rate_limit_rpm=2,
        block_first=True,
        block_second=False,
    )
    executor = ThreadPoolExecutor(max_workers=1)

    try:
        responses = client.iter_batch_requests(
            [_make_completion_request(prompt) for prompt in ("first", "second", "third")],
            "ordered-failure-buffer-test",
            batch_size=2,
        )
        first_result = executor.submit(next, responses)
        assert client.third_started.wait(timeout=1)

        client.release_second.set()
        first = first_result.result(timeout=2)
        assert first is not None
        assert first.completion == "response-for-first"

        with pytest.raises(ExecutionError, match="Error code: 400") as exc_info:
            next(responses)
        assert isinstance(exc_info.value.__cause__, ProviderStatusError)
        with client.thread_exceptions_lock:
            assert client.thread_exceptions == {}
    finally:
        client.release_second.set()
        executor.shutdown(wait=True, cancel_futures=True)
        client.shutdown()


def test_iter_batch_requests_defers_later_failure_while_refilling_after_a_blocked_slot():
    client = SlidingWindowCompletionClient(
        fail_second=True,
        rate_limit_rpm=1,
        block_first=True,
        block_second=False,
    )
    executor = ThreadPoolExecutor(max_workers=1)

    try:
        responses = client.iter_batch_requests(
            [
                _make_completion_request(prompt)
                for prompt in ("first", "second", "third", "fourth")
            ],
            "refill-after-failure-test",
            batch_size=1,
        )
        first_result = executor.submit(next, responses)

        assert client.fourth_started.wait(timeout=1)
        client.release_second.set()

        first = first_result.result(timeout=2)
        assert first is not None
        assert first.completion == "response-for-first"

        with pytest.raises(ExecutionError, match="Error code: 400") as exc_info:
            next(responses)
        assert isinstance(exc_info.value.__cause__, ProviderStatusError)
    finally:
        client.release_second.set()
        executor.shutdown(wait=True, cancel_futures=True)
        client.shutdown()


def test_iter_batch_requests_keeps_dedup_owner_until_final_duplicate_emits():
    client = SlidingWindowCompletionClient(
        rate_limit_rpm=1,
        blocked_prompts={"slow"},
        block_second=False,
    )
    executor = ThreadPoolExecutor(max_workers=1)

    try:
        responses = client.iter_batch_requests(
            [
                _make_completion_request(prompt)
                for prompt in ("original", "slow", "original", "original", "original")
            ],
            "dedup-owner-lifetime-test",
            batch_size=1,
        )

        first = next(responses)
        assert first is not None
        assert first.completion == "response-for-original"

        blocked_next = executor.submit(next, responses)
        assert not client.third_physical_request_started.wait(timeout=0.2)

        client.release_second.set()
        second = blocked_next.result(timeout=2)
        assert second is not None
        assert second.completion == "response-for-slow"

        remaining = list(responses)
        assert [response.completion for response in remaining if response] == [
            "response-for-original",
            "response-for-original",
            "response-for-original",
        ]
        assert client.physical_request_count == 2
    finally:
        client.release_second.set()
        executor.shutdown(wait=True, cancel_futures=True)
        client.shutdown()


def test_streaming_slot_caps_use_look_ahead_and_hard_limits():
    assert ModelClient._streaming_slot_caps(5) == (15, 50)
    assert ModelClient._streaming_slot_caps(10_000) == (1_000, 50_000)


def test_iter_batch_requests_keeps_lifecycle_events_in_one_ordered_window():
    client = DummyCompletionClient()
    events = []
    client.set_request_lifecycle_collector(events.append, execution_id="sliding-window")

    try:
        list(
            client.iter_batch_requests(
                [
                    _make_completion_request(prompt)
                    for prompt in ("first", "second", "third")
                ],
                "semantic.map",
                batch_size=2,
            )
        )
    finally:
        client.shutdown()

    assert [event.request_index for event in events if event.event == "queued"] == [
        0,
        1,
        2,
    ]
    assert {event.batch_id for event in events} == {events[0].batch_id}
    assert {event.operation_name for event in events} == {"semantic.map"}
    assert {event.execution_id for event in events} == {"sliding-window"}
    assert sorted(
        event.request_index for event in events if event.event == "settled"
    ) == [0, 1, 2]


def test_iter_batch_requests_admits_to_rate_limit_watermark_when_it_exceeds_batch_size():
    look_ahead_basis = 3
    pending_admission_cap = 3 * look_ahead_basis
    client = SlidingWindowCompletionClient(
        rate_limit_rpm=look_ahead_basis,
        block_first=True,
        block_after_first=True,
    )
    executor = ThreadPoolExecutor(max_workers=1)
    prompts = ("first",) + tuple(
        f"request-{index}" for index in range(1, pending_admission_cap + 2)
    )
    (
        requests,
        admitted_prompts,
        admission_at_capacity,
        admission_overflow,
    ) = _counting_completion_requests(
        prompts,
        admission_watermark=pending_admission_cap,
    )

    try:
        responses = client.iter_batch_requests(
            requests,
            "admission-watermark-test",
            batch_size=2,
        )

        collected = executor.submit(list, responses)
        assert client.first_started.wait(timeout=1)
        assert admission_at_capacity.wait(timeout=1)
        assert admitted_prompts == list(prompts[:pending_admission_cap])
        assert not admission_overflow.wait(timeout=1)

        client.release_second.set()
        results = collected.result(timeout=2)

        assert [response.completion for response in results if response] == [
            f"response-for-{prompt}" for prompt in prompts
        ]
        assert client.max_active_requests <= pending_admission_cap
    finally:
        client.release_second.set()
        executor.shutdown(wait=True, cancel_futures=True)
        client.shutdown()


def test_iter_batch_requests_captures_admission_watermark_before_rpm_increases():
    look_ahead_basis = 3
    pending_admission_cap = 3 * look_ahead_basis
    raised_rpm = 6
    client = SlidingWindowCompletionClient(
        rate_limit_rpm=look_ahead_basis,
        block_first=True,
        block_after_first=True,
    )
    executor = ThreadPoolExecutor(max_workers=1)
    prompts = ("first",) + tuple(
        f"request-{index}" for index in range(1, pending_admission_cap + 2)
    )
    (
        requests,
        admitted_prompts,
        admission_at_capacity,
        admission_overflow,
    ) = _counting_completion_requests(
        prompts,
        admission_watermark=pending_admission_cap,
    )

    try:
        responses = client.iter_batch_requests(
            requests,
            "captured-admission-watermark-test",
            batch_size=2,
        )

        collected = executor.submit(list, responses)
        assert client.first_started.wait(timeout=1)
        assert admission_at_capacity.wait(timeout=1)

        client.rate_limit_strategy.rpm = raised_rpm
        assert not admission_overflow.wait(timeout=1)
        assert admitted_prompts == list(prompts[:pending_admission_cap])

        client.release_second.set()
        results = collected.result(timeout=2)

        assert [response.completion for response in results if response] == [
            f"response-for-{prompt}" for prompt in prompts
        ]
    finally:
        client.release_second.set()
        executor.shutdown(wait=True, cancel_futures=True)
        client.shutdown()


def test_iter_batch_requests_bounds_live_dedup_map_at_admission_watermark():
    look_ahead_basis = 3
    pending_admission_cap = 3 * look_ahead_basis
    completed_result_cap = 10 * look_ahead_basis
    client = DedupTrackingCompletionClient(
        dedup_ceiling=pending_admission_cap,
        rate_limit_rpm=look_ahead_basis,
        block_first=True,
        block_after_first=True,
    )
    executor = ThreadPoolExecutor(max_workers=1)
    prompts = ("first",) + tuple(
        f"request-{index}"
        for index in range(1, pending_admission_cap + completed_result_cap + 2)
    )

    try:
        responses = client.iter_batch_requests(
            (_make_completion_request(prompt) for prompt in prompts),
            "dedup-watermark-test",
            batch_size=2,
        )

        collected = executor.submit(list, responses)
        assert client.first_started.wait(timeout=1)
        assert client.dedup_at_capacity.wait(timeout=1)
        assert client.max_live_dedup_entries == pending_admission_cap
        assert not client.dedup_overflow.wait(timeout=1)

        client.release_second.set()
        results = collected.result(timeout=2)

        assert [response.completion for response in results if response] == [
            f"response-for-{prompt}" for prompt in prompts
        ]
        assert (
            client.max_live_dedup_entries
            <= pending_admission_cap + completed_result_cap
        )
        assert client.live_dedup_entries == {}
    finally:
        client.release_second.set()
        executor.shutdown(wait=True, cancel_futures=True)
        client.shutdown()


def test_iter_batch_requests_default_rpm_uses_three_times_pending_cap():
    client = SlidingWindowCompletionClient(block_first=True, block_after_first=True)
    executor = ThreadPoolExecutor(max_workers=1)
    pending_admission_cap = 3 * client.rate_limit_strategy.rpm
    prompts = ("first",) + tuple(
        f"request-{index}" for index in range(1, pending_admission_cap + 2)
    )
    (
        requests,
        admitted_prompts,
        admission_at_capacity,
        admission_overflow,
    ) = _counting_completion_requests(
        prompts,
        admission_watermark=pending_admission_cap,
    )

    try:
        responses = client.iter_batch_requests(
            requests,
            "default-rpm-watermark-test",
            batch_size=2,
        )

        collected = executor.submit(list, responses)
        assert client.first_started.wait(timeout=1)
        assert admission_at_capacity.wait(timeout=1)
        assert len(admitted_prompts) == pending_admission_cap
        assert not admission_overflow.wait(timeout=1)

        client.release_second.set()
        results = collected.result(timeout=3)

        assert [response.completion for response in results if response] == [
            f"response-for-{prompt}" for prompt in prompts
        ]
    finally:
        client.release_second.set()
        executor.shutdown(wait=True, cancel_futures=True)
        client.shutdown()


def test_iter_batch_requests_preserves_order_for_cached_live_requests():
    fake_cache = FakeCache()
    client = SlidingWindowCompletionClient(
        cache=fake_cache,
        rate_limit_rpm=3,
        block_first=True,
    )
    executor = ThreadPoolExecutor(max_workers=1)
    prompts = ("first", "second", "third", "first")
    requests = (_make_completion_request(prompt) for prompt in prompts)

    try:
        responses = client.iter_batch_requests(
            requests,
            "stream-cache-test",
            batch_size=2,
        )

        collected = executor.submit(list, responses)
        assert client.first_started.wait(timeout=1)
        client.release_second.set()
        results = collected.result(timeout=2)
    finally:
        client.release_second.set()
        executor.shutdown(wait=True, cancel_futures=True)
        client.shutdown()

    assert [response.completion for response in results if response] == [
        "response-for-first",
        "response-for-second",
        "response-for-third",
        "response-for-first",
    ]
    assert fake_cache.get_batch_called is True
    assert fake_cache.get_batch_hit_count == 0
    assert fake_cache.set_called is True
    assert client.call_count == 3


def test_iter_batch_requests_preserves_none_request_positions():
    client = DummyCompletionClient(rate_limit_rpm=2)

    try:
        results = list(
            client.iter_batch_requests(
                [_make_completion_request("first"), None, _make_completion_request("second")],
                "none-position-test",
                batch_size=2,
            )
        )
    finally:
        client.shutdown()

    assert [response.completion if response else None for response in results] == [
        "response-for-first",
        None,
        "response-for-second",
    ]
    assert client.call_count == 2


def test_iter_batch_requests_accepts_an_empty_iterable():
    client = DummyCompletionClient()

    try:
        assert list(client.iter_batch_requests([], "empty-stream-test", batch_size=2)) == []
    finally:
        client.shutdown()

    assert client.call_count == 0


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
