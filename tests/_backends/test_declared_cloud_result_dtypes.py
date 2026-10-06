import asyncio
from concurrent.futures import Future
from unittest.mock import MagicMock

import polars as pl
import pyarrow as pa
import pytest

from fenic import ColumnField, IntegerType, Schema, StringType

pytest.importorskip("fenic_cloud")

from fenic._backends.cloud.execution import CloudExecution
from fenic.core._logical_plan.plans import InMemorySource


@pytest.mark.parametrize("height", [0, 2])
def test_cloud_arrow_null_columns_are_cast_to_logical_output_schema(
    height, monkeypatch
):
    schema = Schema(
        [ColumnField("text", StringType), ColumnField("number", IntegerType)]
    )
    table = pa.table({"text": pa.nulls(height), "number": pa.nulls(height)})
    arrow_client = MagicMock()
    arrow_client.do_get.return_value.read_all.return_value = table
    monkeypatch.setattr(pa.flight, "connect", lambda uri: arrow_client)
    state = MagicMock(
        arrow_ipc_uri="localhost:1234",
        arrow_ipc_uri_secure=False,
        session_uuid="session",
    )
    operator = CloudExecution(state, MagicMock())
    operator._get_query_execution_metrics = MagicMock()

    def complete(coroutine, loop):
        coroutine.close()
        future = Future()
        future.set_result("execution")
        return future

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", complete)
    plan = InMemorySource.from_schema(
        pl.DataFrame(schema={"text": pl.String, "number": pl.Int64}), schema
    )

    result, _ = operator.collect(plan)

    assert result.schema == {"text": pl.String, "number": pl.Int64}
    assert result.height == height
    assert result["text"].to_list() == [None] * height
    assert result["number"].to_list() == [None] * height
