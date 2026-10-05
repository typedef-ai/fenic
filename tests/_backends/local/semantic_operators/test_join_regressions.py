"""Provider-free regressions for bounded semantic join tiles."""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import polars as pl
import pytest

from fenic import col
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
from fenic._inference.types import FenicCompletionsRequest, FenicCompletionsResponse
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
        self.concurrency = concurrency
        self.inflight = 0
        self.peak = 0
        self.calls = []

    async def make_single_request(self, request):
        self.calls.append(request.messages.user)
        self.inflight += 1
        self.peak = max(self.peak, self.inflight)
        if self.inflight == self.concurrency:
            self.full.set()
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
@pytest.mark.parametrize("tokens,n_pairs", [(20_000, 40), (1000, 200)])
def test_token_subblocks_overlap_up_to_provider_capacity(
    monkeypatch, streaming, tokens, n_pairs
):
    monkeypatch.setattr(Predicate, "stream_requests", streaming)
    client = _GatedClient(tokens, n_pairs)
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
            client.release.set()
            result = future.result(timeout=20)
        assert reached_capacity, (
            f"only {client.peak}/{n_pairs} provider calls overlapped"
        )
        assert client.peak == n_pairs
        assert peak_pairs == n_pairs
        assert resident_pairs == 0
        assert result.sort("payload")["payload"].to_list() == list(range(n_pairs))
        assert sorted(client.calls) == sorted(f"doc|r{i}" for i in range(n_pairs))
    finally:
        client.release.set()
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
