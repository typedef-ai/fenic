import polars as pl
import pytest

from fenic import (
    ArrayType,
    ColumnField,
    JsonType,
    StringType,
    StructField,
    StructType,
    col,
    json,
)
from fenic.core.error import ValidationError


def test_invalid_jq_query(local_session):
    data = {
        "struct_col": [
            {"user": {"id": 1, "name": "Alice"}, "events": ["click"], "active": True},
        ]
    }
    df = local_session.create_dataframe(data)
    df = df.select(col("struct_col").cast(JsonType).alias("json_col"))
    with pytest.raises(ValidationError):
        df = df.select(json.jq(col("json_col"), "a?ad").alias("user_name"))

def test_jq_simple(local_session):
    data = {
        "struct_col": [
            {
                "user": {"id": 1, "name": "Alice"},
                "events": [{"type": "click", "ts": "2024-01-01"}, {"type": "scroll", "ts": "2024-01-02"}],
                "active": True
            },
            {
                "user": {"id": 2, "name": "Bob"},
                "events": [{"type": "click", "ts": "2024-02-01"}],
                "active": False
            },
        ]
    }
    df = local_session.create_dataframe(data)
    df = df.select(col("struct_col").cast(JsonType).alias("json_col"))
    df = df.select(json.jq(col("json_col"), ".user.name").alias("user_name"))
    assert df.schema.column_fields == [ColumnField(name="user_name", data_type=ArrayType(element_type=JsonType))]
    result = df.to_polars()
    expected = pl.DataFrame({"user_name": [['"Alice"'], ['"Bob"']]})
    assert result.equals(expected)


def test_jq_extract_event_types(local_session):
    data = {
        "struct_col": [
            {
                "user": {"id": 1, "name": "Alice"},
                "events": [{"type": "click", "ts": "2024-01-01"}, {"type": "scroll", "ts": "2024-01-02"}],
                "active": True
            },
            {
                "user": {"id": 2, "name": "Bob"},
                "events": [{"type": "click", "ts": "2024-02-01"}],
                "active": False
            },
        ]
    }
    df = local_session.create_dataframe(data)
    df = df.select(col("struct_col").cast(JsonType).alias("json_col"))
    df = df.select(json.jq(col("json_col"), ".events | map(.type)").alias("event_types"))
    assert df.schema.column_fields == [ColumnField(name="event_types", data_type=ArrayType(element_type=JsonType))]
    result = df.to_polars()
    expected = pl.DataFrame({
        "event_types": [['["click","scroll"]'], ['["click"]']]
    })
    assert result.equals(expected)


def test_jq_conditional_user_name(local_session):
    data = {
        "struct_col": [
            {"user": {"id": 1, "name": "Alice"}, "events": ["click"], "active": True},
            {"user": {"id": 2, "name": "Bob"}, "events": [], "active": False},
        ]
    }
    df = local_session.create_dataframe(data)
    df = df.select(col("struct_col").cast(JsonType).alias("json_col"))
    df = df.select(json.jq(col("json_col"), 'if .active then .user.name else null end').alias("maybe_name"))
    result = df.to_polars()
    expected = pl.DataFrame({"maybe_name": [['"Alice"'], ["null"]]})
    assert result.equals(expected)

def test_jq_flatten_user_and_event(local_session):
    data = {
        "struct_col": [
            {
                "user": {"id": 1, "name": "Alice"},
                "events": [{"type": "click", "ts": "2024-01-01"}],
                "active": True
            },
            {
                "user": {"id": 2, "name": "Bob"},
                "events": [{"type": "scroll", "ts": "2024-02-01"}],
                "active": True
            },
        ]
    }

    df = local_session.create_dataframe(data)
    df = df.select(col("struct_col").cast(JsonType).alias("json_col"))

    # Use jq to extract fields
    df = df.select(
        json.jq(
            col("json_col"),
            '{name: .user.name, first_event: .events[0].type}'
        ).alias("flattened_json")
    )

    # Cast from JSON string back into a struct
    df = df.with_column(
        "struct_col",
        col("flattened_json")
        .get_item(0)
        .cast(
            StructType([
                StructField(name="name", data_type=StringType),
                StructField(name="first_event", data_type=StringType)
            ])
        )
    )

    result = df.to_polars()

    expected = pl.DataFrame({
        "flattened_json": [
            ['{"name":"Alice","first_event":"click"}'],
            ['{"name":"Bob","first_event":"scroll"}']
        ],
        "struct_col": [
            {"name": "Alice", "first_event": "click"},
            {"name": "Bob", "first_event": "scroll"},
        ]
    })

    assert result.equals(expected)


def test_jq_missing_field(local_session):
    data = {
        "struct_col": [
            {"user": {"id": 1, "name": "Alice"}, "events": ["click"], "active": True},
            {"user": {"id": 2}, "events": ["click"], "active": True}, # missing name
        ]
    }
    df = local_session.create_dataframe(data)
    df = df.select(col("struct_col").cast(JsonType).alias("json_col"))
    df = df.select(json.jq(col("json_col"), ".user.name").alias("user_name"))
    result = df.to_polars()
    expected = pl.DataFrame({"user_name": [['"Alice"'], ["null"]]})
    assert result.equals(expected)
