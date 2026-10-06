"""Streamed joins share one decision per distinct prompt in each token block."""

import asyncio
from collections import Counter

import polars as pl
import pytest

from fenic import col
from fenic._backends.local.semantic_operators.base import BaseOperator
from fenic._backends.local.semantic_operators.join import (
    DEFAULT_PAIR_BLOCK_SIZE,
    LEFT_ID_KEY,
    RIGHT_ID_KEY,
    Join,
)
from fenic._backends.local.semantic_operators.predicate import Predicate
from fenic._inference.language_model import LanguageModel
from fenic._inference.types import FenicCompletionsResponse
from fenic.core.error import ExecutionError
from tests._inference.test_model_client_thread_exception import FlakyCompletionClient


class JoinCompletionClient(FlakyCompletionClient):
    def __init__(self, *, varying=False, null=False, failing=False):
        super().__init__(failures=0)
        self.model = "gpt-4o-mini"
        self.calls = Counter()
        self.varying = varying
        self.null = null
        self.failing = failing

    async def make_single_request(self, request):
        prompt = request.messages.user
        self.calls[prompt] += 1
        occurrence = self.calls[prompt]
        # Responses settle in a different order from their input positions.
        await asyncio.sleep(0.001 if prompt.endswith("0") else 0)
        if self.failing and prompt.endswith("1"):
            raise ValueError("duplicate block failure")
        completion = (
            None
            if self.null
            else (
                '{"output": true}'
                if not self.varying or occurrence == 1
                else '{"output": false}'
            )
        )
        return FenicCompletionsResponse(
            completion=completion, logprobs=None, usage=None
        )


@pytest.mark.parametrize("stride", [99, 100, 101])
@pytest.mark.parametrize("varying", [False, True])
def test_public_join_shares_duplicates_across_window_boundary(
    construction_only_local_session, monkeypatch, stride, varying
):
    session = construction_only_local_session
    observations = {}
    windows = []
    original_iter = JoinCompletionClient.iter_batch_requests

    def observe_stream(self, *args, **kwargs):
        windows.append(kwargs["batch_size"])
        yield from original_iter(self, *args, **kwargs)

    monkeypatch.setattr(JoinCompletionClient, "iter_batch_requests", observe_stream)
    for stream in (False, True):
        client = JoinCompletionClient(varying=varying)
        model = LanguageModel(client)
        monkeypatch.setattr(
            session._session_state,
            "get_language_model",
            lambda *args, _model=model, **kwargs: _model,
        )
        monkeypatch.setattr(Join, "stream_requests", stream)
        try:
            left = session.create_dataframe({"word": ["same", "same"], "lid": [0, 1]})
            right = session.create_dataframe(
                {
                    "word_right": [f"r{i % stride}" for i in range(stride * 2)],
                    "rid": list(range(stride * 2)),
                }
            )
            result = left.semantic.join(
                right,
                "L={{ left_on }}|R={{ right_on }}",
                left_on=col("word"),
                right_on=col("word_right"),
            ).to_polars()
            observations[stream] = (
                result.sort(["lid", "rid"]).rows(),
                client.calls.copy(),
                result.schema,
            )
        finally:
            client.shutdown()
    expected_calls = Counter({f"L=same|R=r{i}": 1 for i in range(stride)})
    assert observations[False][1] == expected_calls
    assert observations[True] == observations[False]
    assert len(observations[True][0]) == stride * 4
    assert windows == [100]
    assert Predicate.request_batch_size == BaseOperator.request_batch_size == 100


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("null_output", [False, True])
def test_duplicate_scatter_preserves_null_empty_and_pair_order(
    monkeypatch, strict, null_output
):
    observations = {}
    for stream in (False, True):
        client = JoinCompletionClient(varying=True, null=null_output)
        try:
            operator = Join(
                pl.DataFrame(
                    {"left_on": ["same", "same", None], "lid": [0, 1, 2]},
                    schema={"left_on": pl.String, "lid": pl.Int64},
                ),
                pl.DataFrame(
                    {"right_on": ["r0", "r0", ""], "rid": [0, 1, 2]},
                    schema={"right_on": pl.String, "rid": pl.Int64},
                ),
                "{{ right_on }}",
                strict,
                LanguageModel(client),
                0,
            )
            operator.stream_requests = stream
            pairs_and_results = []
            original_select = operator._select_survivors

            def observe(
                pairs, results, _observations=pairs_and_results, _select=original_select
            ):
                _observations.append(
                    (pairs.select(LEFT_ID_KEY, RIGHT_ID_KEY).rows(), results.to_list())
                )
                assert results.dtype == pl.Boolean
                assert len(pairs) <= DEFAULT_PAIR_BLOCK_SIZE
                return _select(pairs, results)

            monkeypatch.setattr(operator, "_select_survivors", observe)
            result = operator.execute()
            observations[stream] = (
                result.sort(["lid", "rid"]).rows(),
                client.calls.copy(),
                pairs_and_results,
                result.schema,
            )
        finally:
            client.shutdown()
    assert observations[True] == observations[False]
    assert observations[True][1] == Counter({"r0": 1})
    assert pl.Null not in observations[True][3].values()


@pytest.mark.parametrize("join_opt_in", [False, True])
@pytest.mark.parametrize("predicate_opt_in", [None, False, True])
@pytest.mark.parametrize("base_opt_in", [False, True])
def test_join_opt_in_is_additive(
    monkeypatch, join_opt_in, predicate_opt_in, base_opt_in
):
    monkeypatch.setattr(BaseOperator, "stream_requests", base_opt_in)
    if predicate_opt_in is not None:
        monkeypatch.setattr(Predicate, "stream_requests", predicate_opt_in)
    elif "stream_requests" in Predicate.__dict__:
        monkeypatch.delattr(Predicate, "stream_requests")
    monkeypatch.setattr(Join, "stream_requests", join_opt_in)
    client = JoinCompletionClient()
    model = LanguageModel(client)
    used = []
    original_get, original_iter = model.get_completions, model.iter_completions

    def get(**kwargs):
        used.append("list")
        return original_get(**kwargs)

    def stream(**kwargs):
        used.append("stream")
        return original_iter(**kwargs)

    monkeypatch.setattr(model, "get_completions", get)
    monkeypatch.setattr(model, "iter_completions", stream)
    try:
        Join(
            pl.DataFrame({"left_on": ["l"]}, schema={"left_on": pl.String}),
            pl.DataFrame({"right_on": ["r"]}, schema={"right_on": pl.String}),
            "{{ left_on }} {{ right_on }}",
            True,
            model,
            0,
        ).execute()
        inherited = base_opt_in if predicate_opt_in is None else predicate_opt_in
        expected = "stream" if join_opt_in or inherited else "list"
        assert used == [expected]
    finally:
        client.shutdown()


@pytest.mark.parametrize("pair_cap,token_budget", [(4, 2), (2, 100)])
def test_dedup_remains_local_to_token_blocks_and_tiles(pair_cap, token_budget):
    for stream in (False, True):
        client = JoinCompletionClient()
        try:
            model = LanguageModel(client)
            model.count_tokens = lambda prompt: 1
            operator = Join(
                pl.DataFrame(
                    {"left_on": ["same", "same"]}, schema={"left_on": pl.String}
                ),
                pl.DataFrame(
                    {"right_on": ["r0", "r0"]}, schema={"right_on": pl.String}
                ),
                "{{ right_on }}",
                True,
                model,
                0,
                pair_block_size=pair_cap,
                block_token_budget=token_budget,
            )
            operator.stream_requests = stream
            assert operator.execute().height == 4
            assert client.calls == Counter({"r0": 2})
        finally:
            client.shutdown()


def test_streamed_duplicate_block_failure_returns_no_partial_frame():
    client = JoinCompletionClient(failing=True)
    try:
        operator = Join(
            pl.DataFrame({"left_on": ["same", "same"]}, schema={"left_on": pl.String}),
            pl.DataFrame({"right_on": ["r0", "r1"]}, schema={"right_on": pl.String}),
            "{{ right_on }}",
            True,
            LanguageModel(client),
            0,
        )
        operator.stream_requests = True
        with pytest.raises(ExecutionError, match="duplicate block failure"):
            operator.execute()
        assert client.calls == Counter({"r0": 1, "r1": 1})
    finally:
        client.shutdown()
