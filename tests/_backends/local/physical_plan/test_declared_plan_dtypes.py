from unittest.mock import MagicMock

import polars as pl
import pytest

from fenic import ColumnField, IntegerType, Schema, StringType
from fenic._backends.local.physical_plan.base import _with_lineage_uuid
from fenic._backends.local.physical_plan.sink import DuckDBTableSinkExec, FileSinkExec
from fenic._backends.local.physical_plan.transform import ProjectionExec, UnionExec
from fenic.api.session.session import _coerce_to_schema, _normalize_data_like_to_polars


@pytest.mark.parametrize("height", [0, 2])
def test_lineage_uuid_is_string_for_empty_and_all_null_batches(height):
    df = pl.DataFrame({"value": [None] * height}, schema={"value": pl.Int64})
    result = _with_lineage_uuid(df)
    assert result.schema == {"value": pl.Int64, "_uuid": pl.String}
    assert result.height == height
    assert result["_uuid"].null_count() == 0


@pytest.mark.parametrize("left_height,right_height", [(0, 0), (0, 2), (2, 0)])
def test_union_lineage_empty_side_keeps_string_uuids(left_height, right_height):
    children = []
    for height in (left_height, right_height):
        child = MagicMock()
        df = pl.DataFrame(
            {"value": [None] * height, "_uuid": [f"old-{i}" for i in range(height)]},
            schema={"value": pl.Int64, "_uuid": pl.String},
        )
        child.build_node_lineage.return_value = (MagicMock(), df)
        children.append(child)
    operator = UnionExec(children=children, cache_info=None, session_state=MagicMock())
    operator._build_binary_operator_lineage = MagicMock()

    _, result = operator.build_node_lineage([])

    assert result.schema == {"value": pl.Int64, "_uuid": pl.String}
    assert result.height == left_height + right_height
    for key in ("left_child", "right_child"):
        mapping = operator._build_binary_operator_lineage.call_args.kwargs[key][1]
        assert mapping.schema == {"_uuid": pl.String, "_backwards_uuid": pl.String}


@pytest.mark.parametrize(
    "data", [[], [{"text": None}], [{"text": None}, {"text": None}]]
)
def test_session_missing_all_null_field_uses_explicit_schema(data):
    schema = Schema(
        [ColumnField("number", IntegerType), ColumnField("text", StringType)]
    )
    normalized, fields = _normalize_data_like_to_polars(
        data, allow_empty_list=True, validate_all_rows=True
    )
    result = _coerce_to_schema(normalized, schema, row_field_names=fields)
    assert result.schema == {"number": pl.Int64, "text": pl.String}
    assert result.height == len(data)


@pytest.mark.parametrize("height", [0, 2])
@pytest.mark.parametrize("sink_type", ["file", "table"])
@pytest.mark.parametrize("ignore_existing", [False, True])
def test_all_null_projection_and_sink_preserve_input_schema(
    height, sink_type, ignore_existing, tmp_path, monkeypatch
):
    schema = Schema([ColumnField("value", IntegerType)])
    df = pl.DataFrame({"value": [None] * height}, schema={"value": pl.Int64})
    state = MagicMock()
    projection = ProjectionExec(
        child=MagicMock(),
        projections=[pl.col("value")],
        cache_info=None,
        session_state=state,
    )
    batch = projection.execute_node([df])
    assert batch.schema == {"value": pl.Int64}
    if sink_type == "file":
        path = tmp_path / "nulls.parquet"
        if ignore_existing:
            df.write_parquet(path)
        monkeypatch.setattr(
            "fenic._backends.local.physical_plan.sink.does_path_exist",
            lambda *args: ignore_existing,
        )
        writer = MagicMock()
        monkeypatch.setattr(
            "fenic._backends.local.physical_plan.sink.write_file", writer
        )
        sink = FileSinkExec(MagicMock(), str(path), "parquet", "ignore", None, state)
    else:
        state.catalog.does_table_exist.return_value = ignore_existing
        writer = state.catalog.write_df_to_table
        sink = DuckDBTableSinkExec(MagicMock(), "nulls", "ignore", None, state, schema)

    result = sink.execute_node([batch])

    assert result.schema == {}
    assert result.is_empty()
    if ignore_existing:
        writer.assert_not_called()
    elif sink_type == "file":
        assert writer.call_args.kwargs["df"].schema == {"value": pl.Int64}
    else:
        assert writer.call_args.args[0].schema == {"value": pl.Int64}


@pytest.mark.parametrize(
    "data", [{"value": [None, None]}, [{"value": None}, {"value": None}]]
)
def test_user_nulls_are_coerced_before_transform_and_table_sink(local_session, data):
    schema = Schema([ColumnField("value", StringType)])
    df = local_session.create_dataframe(data, schema=schema)
    expected = {"value": pl.String}
    assert df.to_polars().schema == expected
    transformed = df.select("value").union(df.select("value"))
    assert transformed.to_polars().schema == expected
    transformed.write.save_as_table("all_null_batch", mode="overwrite")
    result = local_session.table("all_null_batch").to_polars()
    assert result.schema == expected
    assert result["value"].to_list() == [None] * 4
