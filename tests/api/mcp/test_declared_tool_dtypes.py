from unittest.mock import MagicMock

import polars as pl
import pytest

from fenic import ColumnField, IntegerType, Schema
from fenic.api.mcp._tool_generation_utils import (
    _auto_generate_schema_tool,
    _auto_generate_search_summary_tool,
    _compute_profile_for_dataset,
)


def test_schema_tool_empty_dataset_schema_has_typed_nested_fields(local_session):
    dataset = MagicMock()
    dataset.schema.return_value = Schema([])
    tool = _auto_generate_schema_tool(
        {"empty": dataset}, local_session, "Schema", "Schema"
    )
    result, _ = local_session._session_state.execution.collect(tool.func())
    assert result.schema == {
        "dataset": pl.String,
        "schema": pl.List(pl.Struct({"column": pl.String, "type": pl.String})),
    }
    assert result.to_dicts() == [{"dataset": "empty", "schema": []}]


def test_search_summary_without_string_columns_has_declared_types(local_session):
    dataset = MagicMock()
    dataset.df.return_value = local_session.create_dataframe(
        [], schema=Schema([ColumnField("number", IntegerType)])
    )
    tool = _auto_generate_search_summary_tool(
        {"numbers": dataset}, local_session, "Search", "Search"
    )
    result, _ = local_session._session_state.execution.collect(tool.func("query"))
    assert result.schema == {"dataset": pl.String, "total_matches": pl.Int64}
    assert result.to_dicts() == [{"dataset": "numbers", "total_matches": 0}]


@pytest.mark.parametrize("height", [0, 2])
def test_profile_empty_and_all_null_numeric_columns_have_typed_fields(
    local_session, height
):
    dataset = MagicMock(table_name="numbers")
    dataset.df.return_value = local_session.create_dataframe(
        {"number": [None] * height},
        schema=Schema([ColumnField("number", IntegerType)]),
    )
    result = _compute_profile_for_dataset(local_session, dataset, 10).to_polars()
    assert result.schema["hints"] == pl.List(pl.String)
    assert result.schema["numeric_stats"] == pl.Struct(
        dict.fromkeys(
            ["min", "max", "mean", "std_dev", "median", "quantile_25", "quantile_75"],
            pl.Float64,
        )
    )
    assert result.schema["string_stats"].fields[-1].dtype == pl.List(pl.String)
    assert result.schema["boolean_stats"] == pl.Struct(
        {"true_rows": pl.Int64, "false_rows": pl.Int64}
    )
    assert result["sample_percentage_of_original"].to_list() == [
        100.0 if height else 0.0
    ]
