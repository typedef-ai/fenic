from datetime import date, datetime

import polars as pl

from fenic._backends.local.physical_plan.utils import apply_ingestion_coercions


def test_simple_date_coercion():
    """Date column should be converted to string."""
    df = pl.DataFrame({
        "date_col": [date(2023, 12, 25), date(2024, 1, 1)],
        "int_col": [1, 2]
    })

    result = apply_ingestion_coercions(df)

    # Check that date_col is now string type
    assert result.schema["date_col"] == pl.String
    assert result.schema["int_col"] == pl.Int64  # unchanged

    # Check actual values are string representations
    date_values = result["date_col"].to_list()
    assert all(isinstance(val, str) for val in date_values)

def test_simple_datetime_coercion():
    """Datetime column should be converted to string."""
    df = pl.DataFrame({
        "datetime_col": [datetime(2023, 12, 25, 14, 30), datetime(2024, 1, 1, 9, 15)],
    })

    result = apply_ingestion_coercions(df)

    assert result.schema["datetime_col"] == pl.String
    datetime_values = result["datetime_col"].to_list()
    assert all(isinstance(val, str) for val in datetime_values)

def test_array_coercion():
    """Array column should be converted to List."""
    df = pl.DataFrame({
        "array_col": [[1, 2, 3], [4, 5, 6]]
    }, schema={"array_col": pl.Array(pl.Int64, 3)})

    result = apply_ingestion_coercions(df)

    assert result.schema["array_col"] == pl.List(pl.Int64)

def test_array_of_dates_coercion():
    """Array of dates should become List of strings."""
    df = pl.DataFrame({
        "date_array": [
            [date(2023, 12, 25), date(2023, 12, 26)],
            [date(2024, 1, 1), date(2024, 1, 2)]
        ]
    }, schema={"date_array": pl.Array(pl.Date, 2)})

    result = apply_ingestion_coercions(df)

    assert result.schema["date_array"] == pl.List(pl.String)

def test_struct_coercion():
    """Struct with date fields should have those fields converted to strings."""
    df = pl.DataFrame({
        "struct_col": [
            {"id": 1, "created_at": date(2023, 12, 25)},
            {"id": 2, "created_at": date(2024, 1, 1)}
        ]
    })

    result = apply_ingestion_coercions(df)

    expected_struct_type = pl.Struct([
        pl.Field("id", pl.Int64),
        pl.Field("created_at", pl.String)
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
    result = apply_ingestion_coercions(df)

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

    result = apply_ingestion_coercions(df)

    assert result.schema["date_col"] == pl.String      # coerced
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

    result = apply_ingestion_coercions(df)

    # Check the deeply nested coercion: Array(Date) -> List(String)
    expected_schema = pl.List(pl.Struct([
        pl.Field("timestamps", pl.List(pl.String)),
        pl.Field("count", pl.Int64)
    ]))

    assert result.schema["events"] == expected_schema

    # Verify actual data transformation
    events_data = result["events"].to_list()
    first_event = events_data[0][0]  # First event in first row
    assert isinstance(first_event["timestamps"], list)
    assert all(isinstance(ts, str) for ts in first_event["timestamps"])
