import datetime
import zoneinfo
from datetime import date

import polars as pl

from fenic._backends.local.physical_plan.utils import apply_ingestion_coercions
from fenic.core.types.datatypes import (
    ArrayType,
    EmbeddingType,
    IntegerType,
    StructField,
    StructType,
)
from fenic.core.types.schema import ColumnField, Schema


def test_array_coercion():
    """Array column should be converted to List."""
    df = pl.DataFrame({
        "array_col": [[1, 2, 3], [4, 5, 6]]
    }, schema={"array_col": pl.Array(pl.Int64, 3)})

    result = apply_ingestion_coercions(df, coerce_array=True)

    assert result.schema["array_col"] == pl.List(pl.Int64)


def test_create_dataframe_without_schema_normalizes_polars_array_to_list(local_session):
    """No-schema in-memory sources should still normalize Array columns to List."""
    pl_df = pl.DataFrame(
        {"array_col": [[1, 2, 3], [4, 5, 6]]},
        schema={"array_col": pl.Array(pl.Int64, 3)},
    )

    result = local_session.create_dataframe(pl_df).to_polars()

    assert result.schema["array_col"] == pl.List(pl.Int64)


def test_array_coercion_preserves_explicit_embedding_schema():
    embedding_type = EmbeddingType(dimensions=3, embedding_model="test")
    schema = Schema([ColumnField("embedding", embedding_type)])
    df = pl.DataFrame(
        {"embedding": [[1.0, 2.0, 3.0]]},
        schema={"embedding": pl.Array(pl.Float32, 3)},
    )

    result = apply_ingestion_coercions(
        df,
        coerce_array=True,
        logical_schema=schema,
    )

    assert result.schema["embedding"] == pl.Array(pl.Float32, 3)


def test_schema_aware_coercion_keeps_unrelated_arrays_as_lists():
    embedding_type = EmbeddingType(dimensions=3, embedding_model="test")
    schema = Schema([
        ColumnField("embedding", embedding_type),
        ColumnField("array_col", ArrayType(IntegerType)),
    ])
    df = pl.DataFrame(
        {
            "embedding": [[1.0, 2.0, 3.0]],
            "array_col": [[1, 2, 3]],
        },
        schema={
            "embedding": pl.Array(pl.Float32, 3),
            "array_col": pl.Array(pl.Int64, 3),
        },
    )

    result = apply_ingestion_coercions(
        df,
        coerce_array=True,
        logical_schema=schema,
    )

    assert result.schema["embedding"] == pl.Array(pl.Float32, 3)
    assert result.schema["array_col"] == pl.List(pl.Int64)


def test_schema_aware_coercion_preserves_nested_embedding_in_struct():
    embedding_type = EmbeddingType(dimensions=2, embedding_model="test")
    schema = Schema([
        ColumnField(
            "payload",
            StructType([
                StructField("embedding", embedding_type),
                StructField("values", ArrayType(IntegerType)),
            ]),
        )
    ])
    df = pl.DataFrame(
        {"payload": [{"embedding": [1.0, 2.0], "values": [1, 2]}]},
        schema={
            "payload": pl.Struct([
                pl.Field("embedding", pl.Array(pl.Float32, 2)),
                pl.Field("values", pl.Array(pl.Int64, 2)),
            ])
        },
    )

    result = apply_ingestion_coercions(
        df,
        coerce_array=True,
        logical_schema=schema,
    )

    assert result.schema["payload"] == pl.Struct([
        pl.Field("embedding", pl.Array(pl.Float32, 2)),
        pl.Field("values", pl.List(pl.Int64)),
    ])

def test_array_doesnt_coerce_if_coerce_array_is_false():
    """Array column should not be converted to List if coerce_array is false."""
    df = pl.DataFrame({
        "array_col": [[1, 2, 3], [4, 5, 6]]
    }, schema={"array_col": pl.Array(pl.Int64, 3)})

    result = apply_ingestion_coercions(df, coerce_array=False)

    assert result.schema["array_col"] == pl.Array(pl.Int64, 3)

def test_array_of_dates_coercion():
    """Array of dates should become List of dates."""
    df = pl.DataFrame({
        "date_array": [
            [date(2023, 12, 25), date(2023, 12, 26)],
            [date(2024, 1, 1), date(2024, 1, 2)]
        ]
    }, schema={"date_array": pl.Array(pl.Date, 2)})

    result = apply_ingestion_coercions(df, coerce_array=True)

    assert result.schema["date_array"] == pl.List(pl.Date)

def test_struct_coercion():
    """Struct with date fields should not have those fields converted."""
    df = pl.DataFrame({
        "struct_col": [
            {"id": 1, "created_at": date(2023, 12, 25)},
            {"id": 2, "created_at": date(2024, 1, 1)}
        ]
    })

    result = apply_ingestion_coercions(df, coerce_array=True)

    expected_struct_type = pl.Struct([
        pl.Field("id", pl.Int64),
        pl.Field("created_at", pl.Date)
    ])
    assert result.schema["struct_col"] == expected_struct_type

def test_no_coercion_needed():
    """DataFrame with no types needing coercion should be unchanged."""
    df = pl.DataFrame({
        "int_col": [1, 2, 3],
        "str_col": ["a", "b", "c"],
        "float_col": [1.0, 2.0, 3.0]
    })

    original_schema = df.schema
    result = apply_ingestion_coercions(df, coerce_array=True)

    assert result.schema == original_schema
    assert result.equals(df)

def test_mixed_columns():
    """DataFrame with mix of columns needing and not needing coercion."""
    df = pl.DataFrame({
        "date_col": [date(2023, 12, 25), date(2024, 1, 1)],
        "int_col": [1, 2],
        "array_col": [[1, 2], [3, 4]]
    }, schema={
        "date_col": pl.Date,
        "int_col": pl.Int64,
        "array_col": pl.Array(pl.Int64, 2)
    })

    result = apply_ingestion_coercions(df, coerce_array=True)

    assert result.schema["date_col"] == pl.Date      # unchanged
    assert result.schema["int_col"] == pl.Int64        # unchanged
    assert result.schema["array_col"] == pl.List(pl.Int64)  # coerced


def test_deeply_nested_structure():
    """Test deeply nested structure: List of Structs with Arrays of Dates."""
    df = pl.DataFrame({
        "events": [
            [
                {"timestamps": [date(2023, 12, 25), date(2023, 12, 26)], "count": 5},
                {"timestamps": [date(2023, 12, 27), date(2023, 12, 28)], "count": 3}
            ],
            [
                {"timestamps": [date(2024, 1, 1), date(2024, 1, 2)], "count": 8}
            ]
        ]
    }, schema={
        "events": pl.List(pl.Struct([
            pl.Field("timestamps", pl.Array(pl.Date, 2)),
            pl.Field("count", pl.Int64)
        ]))
    })

    result = apply_ingestion_coercions(df, coerce_array=True)

    # Check the deeply nested coercion: Array(Date) -> List(String)
    expected_schema = pl.List(pl.Struct([
        pl.Field("timestamps", pl.List(pl.Date)),
        pl.Field("count", pl.Int64)
    ]))

    assert result.schema["events"] == expected_schema

    # Verify actual data transformation
    events_data = result["events"].to_list()
    first_event = events_data[0][0]  # First event in first row
    assert isinstance(first_event["timestamps"], list)
    assert all(isinstance(ts, date) for ts in first_event["timestamps"])


def test_timezone_conversion_during_ingestion(local_session):
    """Test timezone conversion during ingestion."""
    naive = datetime.datetime(2025, 1, 15, 10, 30, 0)
    utc = datetime.datetime(2025, 1, 15, 10, 30, 0, tzinfo=zoneinfo.ZoneInfo("UTC"))
    la = datetime.datetime(2025, 1, 15, 10, 30, 0, tzinfo=zoneinfo.ZoneInfo("America/Los_Angeles"))

    df = local_session.create_dataframe({
        "naive_ts": [naive],
        "utc_ts": [utc],
        "la_ts": [la]
    })
    result = df.to_polars()

    assert result["naive_ts"][0] == datetime.datetime(2025, 1, 15, 10, 30, 0, tzinfo=zoneinfo.ZoneInfo("UTC"))
    assert result["utc_ts"][0] == datetime.datetime(2025, 1, 15, 10, 30, 0, tzinfo=zoneinfo.ZoneInfo("UTC"))
    assert result["la_ts"][0] == datetime.datetime(2025, 1, 15, 18, 30, 0, tzinfo=zoneinfo.ZoneInfo("UTC"))
