"""Declared output types and batch isolation at the restacked stream boundary."""

from types import SimpleNamespace

import polars as pl
import pytest
from pydantic import BaseModel

from fenic._backends.local.semantic_operators.classify import Classify
from fenic._backends.local.semantic_operators.extract import Extract
from fenic._backends.local.semantic_operators.map import Map
from fenic._backends.local.semantic_operators.predicate import Predicate
from fenic.core._logical_plan.resolved_types import (
    ResolvedClassDefinition,
    ResolvedResponseFormat,
)
from fenic.core.error import ExecutionError
from tests._inference.test_model_client_thread_exception import (
    FlakyCompletionClient,
    _request,
)


class Document(BaseModel):
    title: str


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("rows", [[], [None, None], ["first", "second"]])
@pytest.mark.parametrize("kind", ["predicate", "classify", "map", "extract"])
def test_row_local_operators_declare_empty_and_all_none_output(kind, rows, stream):
    """Even a text Map receiving None completions must retain String dtype."""
    model = SimpleNamespace(
        get_completions=lambda *, messages, **kwargs: [None for _ in messages],
        iter_completions=lambda *, messages, **kwargs: (None for _ in messages),
    )
    series = pl.Series("input", rows, dtype=pl.String)
    common = {"input": series, "model": model, "temperature": 0}
    if kind == "predicate":
        operator = Predicate(**common, jinja_template="{{ input }}")
        expected_dtype = pl.Boolean
    elif kind == "classify":
        operator = Classify(
            **common,
            classes=[ResolvedClassDefinition(label="yes")],
        )
        expected_dtype = pl.String
    elif kind == "map":
        operator = Map(**common, jinja_template="{{ input }}", max_tokens=8)
        expected_dtype = pl.String
    else:
        operator = Extract(
            **common,
            response_format=ResolvedResponseFormat.from_pydantic_model(Document),
            max_output_tokens=8,
        )
        expected_dtype = pl.Struct({"title": pl.String})
    operator.stream_requests = stream
    result = operator.execute()
    assert result.dtype == expected_dtype
    assert result.to_list() == [None] * len(rows)


@pytest.mark.parametrize("failed_stream", [False, True])
@pytest.mark.parametrize("next_stream", [False, True])
def test_failed_list_or_stream_does_not_poison_next_invocation(
    failed_stream, next_stream
):
    client = FlakyCompletionClient()

    def run(user, stream):
        requests = [_request(user)]
        if stream:
            return list(client.iter_batch_requests(requests, "restack", batch_size=1))
        return client.make_batch_requests(requests, "restack")

    try:
        with pytest.raises(ExecutionError, match="first batch failure"):
            run("first", failed_stream)
        assert client.active_batches == set()
        assert client.thread_exceptions == {}
        responses = run("second", next_stream)
        assert [response.completion for response in responses] == [
            "response-for-second"
        ]
        assert client.active_batches == set()
        assert client.thread_exceptions == {}
    finally:
        client.shutdown()
