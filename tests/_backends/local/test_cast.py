import datetime
import json
import zoneinfo

import polars as pl
import pytest

from fenic import (
    ArrayType,
    BooleanType,
    ColumnField,
    DateType,
    EmbeddingType,
    FloatType,
    IntegerType,
    JsonType,
    StringType,
    StructField,
    StructType,
    TimestampType,
    col,
)
from fenic.core.error import PlanError


def test_cast_primitive(local_session):
    data = {
        "integer_col": [8],
        "string_col": ["2"],
        "boolean_col": [True],
        "float_col": [3.0],
        "str_date_col": ["2025-01-01"],
        "str_timestamp_col": ["2025-01-01T10:20:00.000"],
        "str_timestamp_tz_col": ["2025-01-01T10:20:00.000+00:00"],
        "int_date_col": [20000],
        "dt_col": [datetime.date(2025, 1, 1)],
        "ts_col_naive": [datetime.datetime(2025, 1, 1, 10, 20, 0)],
        "ts_col": [datetime.datetime(2025, 1, 1, 10, 20, 0, tzinfo=zoneinfo.ZoneInfo(key="UTC"))],
    }
    df = local_session.create_dataframe(data)

    result = df.select(
        col("integer_col").cast(StringType).alias("cast_col"),
        col("integer_col").cast(BooleanType).alias("cast_col2"),
        col("integer_col").cast(FloatType).alias("cast_col3"),
        col("string_col").cast(IntegerType).alias("cast_col4"),
        col("string_col").cast(FloatType).alias("cast_col5"),
        col("float_col").cast(BooleanType).alias("cast_col7"),
        col("float_col").cast(StringType).alias("cast_col8"),
        col("float_col").cast(IntegerType).alias("cast_col9"),
        col("boolean_col").cast(StringType).alias("cast_col10"),
        col("boolean_col").cast(FloatType).alias("cast_col11"),
        col("boolean_col").cast(IntegerType).alias("cast_col12"),
        col("str_date_col").cast(DateType).alias("cast_col13"),
        col("str_timestamp_col").cast(TimestampType).alias("cast_col14_2"),
        col("str_timestamp_tz_col").cast(TimestampType).alias("cast_col14"),
        col("int_date_col").cast(DateType).alias("cast_col15"),
        col("dt_col").cast(StringType).alias("cast_col16"),
        col("ts_col").cast(StringType).alias("cast_col17"),
        col("ts_col").cast(IntegerType).alias("cast_col18"),
        col("dt_col").cast(TimestampType).alias("cast_col19"),
        col("ts_col").cast(DateType).alias("cast_col20"),
        col("ts_col_naive").cast(TimestampType).alias("cast_col21"),
        col("ts_col_naive").cast(StringType).alias("cast_col22"),
    ).to_polars()

    assert result["cast_col"][0] == "8"
    assert result["cast_col2"][0] is True
    assert result["cast_col3"][0] == 8.0
    assert result["cast_col4"][0] == 2
    assert result["cast_col5"][0] == 2.0
    assert result["cast_col7"][0] is True
    assert result["cast_col8"][0] == "3.0"
    assert result["cast_col9"][0] == 3
    assert result["cast_col10"][0] == "true"
    assert result["cast_col11"][0] == 1.0
    assert result["cast_col12"][0] == 1
    assert result["cast_col13"][0] == datetime.date(2025, 1, 1)
    assert result["cast_col14"][0] == datetime.datetime(2025, 1, 1, 10, 20, 0, tzinfo=zoneinfo.ZoneInfo(key="UTC"))
    # TODO: write a string_to_timestamp function for cast.rs to handle ISO strings without timezones.
    # For now, fenic provides other methods to convert timestmaps to/from strings with any format.
    assert result["cast_col14_2"][0] is None
    assert result["cast_col15"][0] == datetime.date(2024, 10, 4)
    assert result["cast_col16"][0] == "2025-01-01"
    assert result["cast_col17"][0] == "2025-01-01 10:20:00.000000+00:00"
    assert result["cast_col18"][0] == 1_735_726_800_000_000
    assert result["cast_col19"][0] == datetime.datetime(2025, 1, 1, 0, 0, 0, tzinfo=zoneinfo.ZoneInfo(key="UTC"))
    assert result["cast_col20"][0] == datetime.date(2025, 1, 1)
    assert result["cast_col21"][0] == datetime.datetime(2025, 1, 1, 10, 20, 0, tzinfo=zoneinfo.ZoneInfo(key="UTC"))
    assert result["cast_col22"][0] == "2025-01-01 10:20:00.000000+00:00"

    # test that cast to boolean from string fails
    with pytest.raises(PlanError):
        df.select(col("string_col").cast(BooleanType))

    # test that cast to boolean from date fails
    with pytest.raises(PlanError):
        df.select(col("dt_col").cast(BooleanType))

    # test that cast to boolean from timestamp fails
    with pytest.raises(PlanError):
        df.select(col("ts_col").cast(BooleanType))


def test_cast_to_timestamp_with_format(local_session):
    """Test to_timestamp() with format string (docs line 68)."""
    from fenic.api.functions.dt import to_timestamp

    df = local_session.create_dataframe({
        "timestamp_str": ["01-15-2025 10:30:00", "01-16-2025 14:00:00"]
    })
    result = df.select(
        to_timestamp(col("timestamp_str"), "MM-dd-yyyy HH:mm:ss").alias("ts")
    ).to_polars()

    assert result["ts"][0] == datetime.datetime(2025, 1, 15, 10, 30, 0, tzinfo=zoneinfo.ZoneInfo("UTC"))
    assert result["ts"][1] == datetime.datetime(2025, 1, 16, 14, 0, 0, tzinfo=zoneinfo.ZoneInfo("UTC"))


def test_cast_to_timestamp_with_format_and_timezone(local_session):
    """Test to_timestamp() with format and timezone (docs line 69)."""
    from fenic.api.functions.dt import to_timestamp

    df = local_session.create_dataframe({
        "timestamp_str": ["01-15-2025 10:30:00 +08:00", "01-16-2025 14:00:00 +08:00"]
    })
    result = df.select(
        to_timestamp(col("timestamp_str"), "MM-dd-yyyy HH:mm:ss XXX").alias("ts")
    ).to_polars()

    assert result["ts"][0] == datetime.datetime(2025, 1, 15, 2, 30, 0, tzinfo=zoneinfo.ZoneInfo("UTC"))
    assert result["ts"][1] == datetime.datetime(2025, 1, 16, 6, 0, 0, tzinfo=zoneinfo.ZoneInfo("UTC"))


def test_cast_to_date_with_format(local_session):
    """Test to_date() with format string (docs line 74)."""
    from fenic.api.functions.dt import to_date

    df = local_session.create_dataframe({
        "date_str": ["01-27-2025"]
    })
    result = df.select(to_date(col("date_str"), format="MM-dd-yyyy").alias("dt")).to_polars()
    assert result["dt"][0] == datetime.date(2025, 1, 27)


def test_cast_array_basic(local_session):
    data = {"my_col": [[1, 2], [3, 4]]}
    df = local_session.create_dataframe(data)
    result = df.with_column(
        "cast_col", col("my_col").cast(ArrayType(element_type=StringType))
    ).to_polars()
    assert result["cast_col"][0].to_list() == ["1", "2"]
    assert result["cast_col"][1].to_list() == ["3", "4"]


def test_cast_struct_basic(local_session):
    data = {"my_col": [{"a": 1, "b": 2}, {"a": 3, "b": 4}]}
    df = local_session.create_dataframe(data)
    dest_type = StructType([StructField("a", StringType), StructField("b", StringType)])
    result = df.with_column("cast_col", col("my_col").cast(dest_type)).to_polars()
    assert result["cast_col"][0] == {"a": "1", "b": "2"}
    assert result["cast_col"][1] == {"a": "3", "b": "4"}


def test_cast_struct_subfields(local_session):
    data = {"my_col": [{"a": 1, "b": 2}, {"a": 3, "b": 4}]}
    df = local_session.create_dataframe(data)
    dest_type = StructType([StructField("a", StringType)])
    result = df.with_column("cast_col", col("my_col").cast(dest_type)).to_polars()
    assert result["cast_col"][0] == {"a": "1"}
    assert result["cast_col"][1] == {"a": "3"}


def test_cast_array_of_structs(local_session):
    data = {"my_col": [[{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]]}
    df = local_session.create_dataframe(data)
    dest_type = ArrayType(
        element_type=StructType(
            [
                StructField("a", IntegerType),
                StructField("b", IntegerType),
            ]
        )
    )
    result = df.with_column("cast_col", col("my_col").cast(dest_type)).to_polars()
    assert result["cast_col"][0].to_list() == [
        {"a": 1, "b": 2},
        {"a": 3, "b": 4},
    ]


def test_cast_struct_of_structs(local_session):
    data = {"my_col": [{"a": {"b": 1, "c": 2}, "d": {"b": 0, "c": 4}}]}
    df = local_session.create_dataframe(data)
    dest_type = StructType(
        [
            StructField(
                "a",
                StructType(
                    [
                        StructField("b", StringType),
                        StructField("c", StringType),
                    ]
                ),
            ),
            StructField(
                "d",
                StructType(
                    [
                        StructField("b", StringType),
                        StructField("c", StringType),
                    ]
                ),
            ),
        ]
    )
    result = df.with_column("cast_col", col("my_col").cast(dest_type)).to_polars()
    assert result["cast_col"][0] == {
        "a": {"b": "1", "c": "2"},
        "d": {"b": "0", "c": "4"},
    }

def test_cast_string_to_json(local_session):
    data = {
        "string_col": [
            '{"a": 1}',            # valid object
            '[1, 2, 3]',           # valid array
            '"a string"',          # valid JSON string literal
            '42',                  # valid JSON number
            '',                    # invalid (empty string)
            'not json',            # invalid
            '[1, 2,, 3]',          # invalid array (extra comma)
            None,                  # null
            '{"nested": {"x": 10}}', # valid nested object
            '"42"',                # valid JSON string containing digits
            '["mixed", 123, {"key": "value"}, null]', # valid mixed array
            '["unclosed array"',  # invalid JSON (missing closing bracket)
        ]
    }

    df = local_session.create_dataframe(data)
    df = df.with_column("json_col", col("string_col").cast(JsonType))
    assert df.schema.column_fields == [
        ColumnField("string_col", StringType),
        ColumnField("json_col", JsonType),
    ]
    result = df.to_polars()

    expected = pl.DataFrame({
        "string_col": data["string_col"],
        "json_col": [
            '{"a": 1}',
            '[1, 2, 3]',
            '"a string"',
            '42',
            None,
            None,
            None,
            None,
            '{"nested": {"x": 10}}',
            '"42"',
            '["mixed", 123, {"key": "value"}, null]',
            None,
        ]
    })

    assert result.equals(expected)

def test_nested_struct_with_json(local_session):
    data = {
        "my_col": [{"a": {"b": 1, "c": '{"d": 1}'}, "a": {"b": 0, "c": '[1, 2,, 3]'}}] # noqa: F601  # F601 = repeated keys ok
    }
    df = local_session.create_dataframe(data)
    to_type = StructType([
        StructField("a", StructType([
            StructField("b", IntegerType),
            StructField("c", JsonType),
        ]))
    ])
    df = df.select(col("my_col").cast(to_type).alias("cast_col"))
    assert df.schema.column_fields == [
        ColumnField("cast_col", to_type),
    ]
    result = df.to_polars()
    expected = pl.DataFrame({
        "cast_col": [{"a": {"b": 1, "c": '{"d": 1}'}, "a": {"b": 0, "c": None}}] # noqa: F601  # F601 = repeated keys ok
    })
    assert result.equals(expected)

def test_cast_struct_to_json(local_session):
    data = {
        "my_col": [
            {"a": 1, "b": {"c": 2, "d": [1, 2, 3]}},
            {"a": 3, "b": {"c": 4, "d": [4, 5, 6]}},
        ]
    }
    df = local_session.create_dataframe(data)
    df = df.select(col("my_col").cast(JsonType).alias("json_col"))

    # Check schema
    assert df.schema.column_fields == [
        ColumnField("json_col", JsonType),
    ]

    result = df.to_polars()

    expected = pl.DataFrame({
        "json_col": [
            '{"a":1,"b":{"c":2,"d":[1,2,3]}}',
            '{"a":3,"b":{"c":4,"d":[4,5,6]}}',
        ]
    })

    assert result.equals(expected)

def test_cast_array_to_json(local_session):
    data = {
        "my_col": [[1, 2, 3], [4, 5, 6]]
    }
    df = local_session.create_dataframe(data)
    df = df.select(col("my_col").cast(JsonType).alias("json_col"))
    result = df.to_polars()
    expected = pl.DataFrame({
        "json_col": [
            '[1,2,3]',
            '[4,5,6]',
        ]
    })
    assert result.equals(expected)

def test_cast_leaf_struct_to_json(local_session):
    data = {
        "my_col": [
            {
                "id": 1,
                "meta": {
                    "created": "2024-01-01",
                    "details": {"user": "alice", "source": "api"},
                },
            },
            {
                "id": 2,
                "meta": {
                    "created": "2024-01-02",
                    "details": {"user": "bob", "source": "web"},
                },
            },
        ]
    }

    df = local_session.create_dataframe(data)

    to_type = StructType([
        StructField("id", IntegerType),
        StructField("meta", StructType([
            StructField("created", StringType),
            StructField("details", JsonType),
        ])),
    ])

    df = df.select(col("my_col").cast(to_type).alias("cast_col"))
    result = df.to_polars()

    # Check schema
    assert df.schema.column_fields == [
        ColumnField("cast_col", to_type),
    ]

    # Check values by parsing JSON content, since order is not guaranteed
    for i, row in enumerate(result["cast_col"]):
        assert row["id"] == [1, 2][i]
        assert row["meta"]["created"] == ["2024-01-01", "2024-01-02"][i]

        # Parse the JSON string and compare as dict
        details_json = json.loads(row["meta"]["details"])
        expected_details = [
            {"user": "alice", "source": "api"},
            {"user": "bob", "source": "web"}
        ][i]

        assert details_json == expected_details

def test_cast_json_to_struct(local_session):
    # Simple flat struct
    data = {
        "my_col": [
            '{"a": 1, "b": 2}',
            '{"a": 3, "b": 4}',
        ]
    }
    df = local_session.create_dataframe(data)
    df = df.select(
        col("my_col")
        .cast(JsonType)
        .cast(StructType([
            StructField("a", IntegerType),
            StructField("b", IntegerType)
        ]))
        .alias("cast_col")
    )
    result = df.to_polars()
    expected = pl.DataFrame({
        "cast_col": [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    })
    assert result.equals(expected)

    # Nested struct with array
    data = {
        "my_col": [
            '{"a": 1, "b": {"c": 2, "d": [1, 2, 3]}}',
            '{"a": 3, "b": {"c": 4, "d": [4, 5, 6]}}',
        ]
    }
    df = local_session.create_dataframe(data)
    df = df.select(
        col("my_col")
        .cast(JsonType)
        .cast(StructType([
            StructField("a", IntegerType),
            StructField("b", StructType([
                StructField("c", IntegerType),
                StructField("d", ArrayType(element_type=IntegerType))
            ]))
        ]))
        .alias("cast_col")
    )
    result = df.to_polars()
    expected = pl.DataFrame({
        "cast_col": [
            {"a": 1, "b": {"c": 2, "d": [1, 2, 3]}},
            {"a": 3, "b": {"c": 4, "d": [4, 5, 6]}}
        ]
    })
    assert result.equals(expected)

    # Extra and missing fields in nested struct
    data = {
        "my_col": [
            '{"a": 1, "b": {"c": 2, "d": [1, 2, 3], "e": 4}}',
            '{"a": 3, "b": {"c": 4, "d": [4, 5, 6], "f": 7}}',
        ]
    }
    df = local_session.create_dataframe(data)
    df = df.select(
        col("my_col")
        .cast(JsonType)
        .cast(StructType([
            StructField("a", IntegerType),
            StructField("b", StructType([
                StructField("c", IntegerType),
                StructField("d", ArrayType(element_type=IntegerType)),
                StructField("e", IntegerType),
                StructField("g", BooleanType)
            ]))
        ]))
        .alias("cast_col")
    )
    result = df.to_polars()
    expected = pl.DataFrame({
        "cast_col": [
            {"a": 1, "b": {"c": 2, "d": [1, 2, 3], "e": 4, "g": None}},
            {"a": 3, "b": {"c": 4, "d": [4, 5, 6], "e": None, "g": None}}
        ]
    })
    assert result.equals(expected)

    # Mismatched type in nested field (array to int)
    data = {
        "my_col": [
            '{"a": 1, "b": {"c": 2, "d": [1, 2, 3]}}',
            '{"a": 3, "b": {"c": 4, "d": [4, 5, 6]}}',
        ]
    }
    df = local_session.create_dataframe(data)
    df = df.select(
        col("my_col")
        .cast(JsonType)
        .cast(StructType([
            StructField("a", IntegerType),
            StructField("b", StructType([
                StructField("c", IntegerType),
                StructField("d", IntegerType)  # will fail to match array
            ]))
        ]))
        .alias("cast_col")
    )
    result = df.to_polars()
    expected = pl.DataFrame({
        "cast_col": [
            {"a": 1, "b": {"c": 2, "d": None}},
            {"a": 3, "b": {"c": 4, "d": None}}
        ]
    })
    assert result.equals(expected)

def test_cast_json_to_array(local_session):
    data = {
        "my_col": [
            '[1, 2, 3]',
            '[4, 5, 6]',
        ]
    }
    df = local_session.create_dataframe(data)
    df = df.select(col("my_col").cast(JsonType).cast(ArrayType(element_type=IntegerType)).alias("cast_col"))
    result = df.to_polars()
    expected = pl.DataFrame({
        "cast_col": [[1, 2, 3], [4, 5, 6]]
    })
    assert result.equals(expected)

def test_cast_array_to_embeddings(local_session):
    data = {
        "my_col": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    }
    df = local_session.create_dataframe(data)
    df = df.select(col("my_col").cast(EmbeddingType(embedding_model="test", dimensions=3)).alias("cast_col"))
    assert df.schema.column_fields == [
        ColumnField("cast_col", EmbeddingType(embedding_model="test", dimensions=3)),
    ]
    result = df.to_polars()
    assert result.schema["cast_col"] == pl.Array(pl.Float32, 3)

def test_cast_embeddings_to_array(local_session):
    data = {
        "my_col": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    }
    df = local_session.create_dataframe(data)
    df = df.select(col("my_col").cast(EmbeddingType(embedding_model="test", dimensions=3)).cast(ArrayType(element_type=FloatType)).alias("cast_col"))
    assert df.schema.column_fields == [
        ColumnField("cast_col", ArrayType(element_type=FloatType)),
    ]
    result = df.to_polars()
    assert result.schema["cast_col"] == pl.List(pl.Float32)

def test_cast_invalid_types(local_session):
    data = {
        "my_array_col": [[1, 2, 3], [4, 5, 6]],
        "my_struct_col": [{"a": 1, "b": 2}, {"a": 3, "b": 4}],
        "my_string_col": ["1", "2"],
        "my_boolean_col": [True, False],
    }
    df = local_session.create_dataframe(data)
    with pytest.raises(PlanError):
        df.with_column(
            "cast_col",
            col("my_string_col").cast(ArrayType(element_type=StringType)),
        )

    with pytest.raises(PlanError):
        df.with_column(
            "cast_col",
            col("my_string_col").cast(StructType([StructField("a", StringType)])),
        )

    with pytest.raises(PlanError):
        df.with_column("cast_col", col("my_array_col").cast(StringType))

    with pytest.raises(PlanError):
        df.with_column("cast_col", col("my_struct_col").cast(StringType))

    with pytest.raises(PlanError):
        df.with_column(
            "cast_col",
            col("my_struct_col").cast(ArrayType(element_type=StringType)),
        )

    with pytest.raises(PlanError):
        df.with_column(
            "cast_col",
            col("my_array_col").cast(StructType([StructField("a", StringType)])),
        )

    with pytest.raises(PlanError):
        df.with_column(
            "cast_col",
            col("my_boolean_col").cast(DateType),
        )

    with pytest.raises(PlanError):
        df.with_column(
            "cast_col",
            col("my_boolean_col").cast(TimestampType),
        )
