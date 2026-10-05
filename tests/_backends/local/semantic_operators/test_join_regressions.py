"""Provider-free regressions for bounded semantic join tiles."""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock

import polars as pl
import pytest

from fenic import col
from fenic._backends.local.semantic_operators import join as join_module
from fenic._backends.local.semantic_operators.join import (
    DEFAULT_PAIR_BLOCK_SIZE,
    LEFT_ID_KEY,
    RENDERED_INSTRUCTION_KEY,
    RIGHT_ID_KEY,
    Join,
)
from fenic._backends.local.semantic_operators.predicate import Predicate
from fenic._inference.language_model import LanguageModel
from fenic._inference.model_client import ModelClient
from fenic._inference.rate_limit_strategy import RateLimitStrategy, TokenEstimate
from fenic._inference.types import (
    FenicCompletionsRequest,
    FenicCompletionsResponse,
    LMRequestMessages,
)
from fenic.core._inference.model_catalog import ModelProvider
from fenic.core._inference.model_provider import ModelProviderClass
from fenic.core.error import ExecutionError
from fenic.core.metrics import LMMetrics


@pytest.mark.parametrize("streaming", [False, True])
def test_public_join_skips_an_empty_rendered_tail(
    local_session, monkeypatch, streaming
):
    monkeypatch.setattr(Predicate, "stream_requests", streaming)
    model = local_session._session_state.get_language_model()
    skipped = []

    def answer(messages, **kwargs):
        for message in messages:
            skipped.append(message is None)
            yield (
                SimpleNamespace(completion='{"output": true}')
                if message is not None
                else None
            )

    monkeypatch.setattr(
        model, "get_completions", lambda **kwargs: list(answer(**kwargs))
    )
    monkeypatch.setattr(model, "iter_completions", answer)
    left = local_session.create_dataframe(
        {"document": ["valid"] * 1024 + [""], "left_id": list(range(1025))}
    )
    right = local_session.create_dataframe({"suffix": [""], "right_id": [7]})

    result = left.semantic.join(
        right,
        "{{ left_on }}{{ right_on }}",
        left_on=col("document"),
        right_on=col("suffix"),
    ).to_polars()

    assert result.sort("left_id")["left_id"].to_list() == list(range(1024))
    assert result["right_id"].to_list() == [7] * 1024
    assert skipped.count(True) == 1


@pytest.mark.parametrize("streaming", [False, True])
def test_token_split_null_singleton_is_a_nonmatch(
    local_session, monkeypatch, streaming
):
    monkeypatch.setattr(Predicate, "stream_requests", streaming)
    model = local_session._session_state.get_language_model()
    monkeypatch.setattr(model, "count_tokens", lambda _: 20_000)

    def answer(messages, **kwargs):
        for message in messages:
            yield SimpleNamespace(
                completion=None if message.user == "skip" else '{"output": true}'
            )

    monkeypatch.setattr(
        model, "get_completions", lambda **kwargs: list(answer(**kwargs))
    )
    monkeypatch.setattr(model, "iter_completions", answer)
    result = Join(
        pl.DataFrame({"left_on": ["keep", "skip"]}),
        pl.DataFrame({"right_on": [""]}),
        "{{ left_on }}{{ right_on }}",
        strict=True,
        model=model,
        temperature=0,
    ).execute()
    assert result.rows() == [("keep", "")]


def test_mixed_tile_rejects_each_oversized_user_prompt(local_session, monkeypatch):
    model = local_session._session_state.get_language_model()
    monkeypatch.setattr(model.model_parameters, "context_window_length", 8192)
    counted = []

    def count(prompt):
        counted.append(prompt)
        return 9000 if prompt == "huge" else 10

    monkeypatch.setattr(model, "count_tokens", count)
    dispatched = []
    monkeypatch.setattr(
        Predicate,
        "execute",
        lambda predicate: (
            dispatched.append(predicate.input.to_list())
            or pl.Series([False] * len(predicate.input))
        ),
    )
    join = Join(
        pl.DataFrame({"left_on": ["tiny", "huge"]}),
        pl.DataFrame({"right_on": [""]}),
        "{{ left_on }}{{ right_on }}",
        strict=True,
        model=model,
        temperature=0,
    )
    with pytest.raises(ExecutionError, match="9000 tokens.*8192 tokens"):
        join.execute()
    assert dispatched == []
    assert counted == ["tiny", "huge"]


class _Provider(ModelProviderClass):
    @property
    def name(self):
        return "fake"

    def create_client(self):
        return object()

    def create_aio_client(self):
        return object()

    async def validate_api_key(self):
        return


class _Limiter(RateLimitStrategy):
    def __init__(self, concurrency):
        super().__init__(rpm=concurrency)

    def check_and_consume_rate_limit(self, token_estimate):
        return True

    def backoff(self, curr_time):
        return 0

    def context_tokens_per_minute(self):
        return 10_000_000


class _Counter:
    def __init__(self, tokens):
        self.tokens = tokens

    def count_tokens(self, messages, ignore_file=False):
        return self.tokens

    def count_file_input_tokens(self, messages):
        return 0

    def count_file_output_tokens(self, messages):
        return 0


class _GatedClient(ModelClient[FenicCompletionsRequest, FenicCompletionsResponse]):
    def __init__(self, tokens, concurrency):
        super().__init__(
            model="gpt-4o-mini",
            model_provider=ModelProvider.OPENAI,
            model_provider_class=_Provider(),
            rate_limit_strategy=_Limiter(concurrency),
            token_counter=_Counter(tokens),
        )
        self.release = threading.Event()
        self.full = threading.Event()
        self.oversubscribed = threading.Event()
        self.concurrency = concurrency
        self.inflight = 0
        self.peak = 0
        self.peak_threads = 0
        self.calls = []

    async def make_single_request(self, request):
        self.calls.append(request.messages.user)
        self.inflight += 1
        self.peak = max(self.peak, self.inflight)
        self.peak_threads = max(self.peak_threads, threading.active_count())
        if self.inflight == self.concurrency:
            self.full.set()
        if self.inflight > self.concurrency:
            self.oversubscribed.set()
        try:
            while not self.release.is_set():
                await asyncio.sleep(0.001)
            return FenicCompletionsResponse(
                completion='{"output": true}', logprobs=None, usage=None
            )
        finally:
            self.inflight -= 1

    def estimate_tokens_for_request(self, request):
        return TokenEstimate(input_tokens=1, output_tokens=1)

    def get_metrics(self):
        return LMMetrics()

    def reset_metrics(self):
        return

    def _get_max_output_token_request_limit(self, request):
        return request.max_completion_tokens or 0


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "tokens,n_pairs", [(20_000, 40), (20_000, 200), (1000, 200), (1000, 1025)]
)
def test_token_subblocks_overlap_up_to_provider_capacity(
    monkeypatch, streaming, tokens, n_pairs
):
    monkeypatch.setattr(Predicate, "stream_requests", streaming)
    client = _GatedClient(tokens, n_pairs)
    expected_capacity = (
        min(n_pairs, 64) if tokens > 16_384 else min(n_pairs, DEFAULT_PAIR_BLOCK_SIZE)
    )
    client.concurrency = expected_capacity
    join = Join(
        pl.DataFrame({"left_on": ["doc"]}),
        pl.DataFrame(
            {"right_on": [f"r{i}" for i in range(n_pairs)], "payload": range(n_pairs)}
        ),
        "{{ left_on }}|{{ right_on }}",
        strict=True,
        model=LanguageModel(client),
        temperature=0,
    )
    resident_pairs = 0
    peak_pairs = 0
    lock = threading.Lock()
    original_build = join._build_join_pair_block
    original_select = join._select_survivors

    def build(left, right):
        nonlocal resident_pairs, peak_pairs
        block = original_build(left, right)
        with lock:
            resident_pairs += len(block)
            peak_pairs = max(peak_pairs, resident_pairs)
            assert resident_pairs <= DEFAULT_PAIR_BLOCK_SIZE
        return block

    def select(block, results):
        nonlocal resident_pairs
        prompts = block[RENDERED_INSTRUCTION_KEY]
        assert len(prompts) == 1 or len(prompts) * tokens <= 32_768
        survivors = original_select(block, results)
        assert survivors.columns == [LEFT_ID_KEY, RIGHT_ID_KEY]
        with lock:
            resident_pairs -= len(block)
        return survivors

    monkeypatch.setattr(join, "_build_join_pair_block", build)
    monkeypatch.setattr(join, "_select_survivors", select)
    try:
        with ThreadPoolExecutor(max_workers=1) as runner:
            future = runner.submit(join.execute)
            reached_capacity = client.full.wait(timeout=10)
            over_capacity = client.oversubscribed.wait(timeout=0.1)
            client.release.set()
            result = future.result(timeout=20)
        assert reached_capacity, (
            f"only {client.peak}/{expected_capacity} provider calls overlapped"
        )
        assert not over_capacity
        assert client.peak == expected_capacity
        assert peak_pairs == min(n_pairs, DEFAULT_PAIR_BLOCK_SIZE)
        assert resident_pairs == 0
        assert result.sort("payload")["payload"].to_list() == list(range(n_pairs))
        assert sorted(client.calls) == sorted(f"doc|r{i}" for i in range(n_pairs))
    finally:
        client.release.set()
        client.shutdown()


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("pair_cap", [16, 128])
def test_join_reuses_one_bounded_pool_across_tiles(monkeypatch, streaming, pair_cap):
    monkeypatch.setattr(Predicate, "stream_requests", streaming)
    pools = []

    def create_pool(*args, **kwargs):
        executor = ThreadPoolExecutor(*args, **kwargs)
        pools.append(executor)
        return executor

    monkeypatch.setattr(join_module, "ThreadPoolExecutor", create_pool)
    client = _GatedClient(20_000, 256)
    worker_cap = min(pair_cap, 64)
    client.concurrency = worker_cap
    baseline_threads = threading.active_count()
    join = Join(
        pl.DataFrame({"left_on": ["doc"]}),
        pl.DataFrame({"right_on": [f"r{i}" for i in range(3 * pair_cap)]}),
        "{{ left_on }}|{{ right_on }}",
        strict=True,
        model=LanguageModel(client),
        temperature=0,
        pair_block_size=pair_cap,
    )
    resident = 0
    peak_resident = 0
    lock = threading.Lock()
    original_build = join._build_join_pair_block
    original_select = join._select_survivors

    def build(left, right):
        nonlocal resident, peak_resident
        with lock:
            assert resident == 0, "previous tile has not drained"
            block = original_build(left, right)
            resident += len(block)
            peak_resident = max(peak_resident, resident)
        return block

    def select(block, results):
        nonlocal resident
        survivors = original_select(block, results)
        with lock:
            resident -= len(block)
        return survivors

    monkeypatch.setattr(join, "_build_join_pair_block", build)
    monkeypatch.setattr(join, "_select_survivors", select)
    try:
        with ThreadPoolExecutor(max_workers=1) as runner:
            future = runner.submit(join.execute)
            reached_capacity = client.full.wait(timeout=10)
            over_capacity = client.oversubscribed.wait(timeout=0.1)
            client.release.set()
            result = future.result(timeout=20)
        assert reached_capacity
        assert not over_capacity
        assert len(pools) == 1
        assert pools[0]._max_workers == worker_cap
        assert client.peak_threads - baseline_threads <= worker_cap + 2
        assert all(not worker.is_alive() for worker in pools[0]._threads)
        assert resident == 0
        assert peak_resident == pair_cap
        assert result.height == 3 * pair_cap
        assert sorted(client.calls) == sorted(f"doc|r{i}" for i in range(3 * pair_cap))
    finally:
        client.release.set()
        client.shutdown()


def test_non_streaming_join_shows_one_progress_bar(monkeypatch):
    bars = []

    def progress(*args, **kwargs):
        bars.append(kwargs)
        return MagicMock()

    monkeypatch.setattr("fenic._inference.model_client.tqdm", progress)
    # The coordinator's join-level bar is separate from client batch bars.
    monkeypatch.setattr(join_module, "tqdm", progress, raising=False)
    client = _GatedClient(20_000, 64)
    client.release.set()
    model = LanguageModel(client)
    try:
        result = Join(
            pl.DataFrame({"left_on": ["doc"]}),
            pl.DataFrame({"right_on": [f"r{i}" for i in range(80)]}),
            "{{ left_on }}|{{ right_on }}",
            strict=True,
            model=model,
            temperature=0,
            pair_block_size=40,
        ).execute()
        enabled = [bar for bar in bars if not bar.get("disable", False)]
        assert result.height == 80
        assert len(enabled) == 1
        assert enabled[0]["desc"] == "semantic.join"
        assert enabled[0]["total"] == 80
        bars.clear()
        model.get_completions(
            [LMRequestMessages(system="control", user="outside join", examples=[])],
            max_tokens=64,
        )
        # The join must not change shared-client progress for other operations.
        assert len([bar for bar in bars if not bar.get("disable", False)]) == 2
    finally:
        client.shutdown()


@pytest.mark.parametrize("streaming", [False, True])
def test_rpm_larger_than_pair_cap_cannot_prefetch_another_tile(monkeypatch, streaming):
    monkeypatch.setattr(Predicate, "stream_requests", streaming)
    client = _GatedClient(20_000, 64)
    client.concurrency = 16
    join = Join(
        pl.DataFrame({"left_on": ["doc"]}),
        pl.DataFrame({"right_on": [f"r{i}" for i in range(80)]}),
        "{{ left_on }}|{{ right_on }}",
        strict=True,
        model=LanguageModel(client),
        temperature=0,
        pair_block_size=16,
    )
    try:
        with ThreadPoolExecutor(max_workers=1) as runner:
            future = runner.submit(join.execute)
            reached_cap = client.full.wait(timeout=10)
            # No next tile is rendered or dispatched while this tile is gated.
            calls_before_release = list(client.calls)
            client.release.set()
            result = future.result(timeout=20)
        assert reached_cap, f"only {client.peak}/16 provider calls overlapped"
        assert sorted(calls_before_release) == sorted(f"doc|r{i}" for i in range(16))
        assert result.height == 80
        assert client.peak == 16
        assert len(client.calls) == 80
    finally:
        client.release.set()
        client.shutdown()
