
import zoneinfo
from datetime import date, datetime

import polars as pl
import pytest

from fenic import (
    ArrayType,
    ColumnField,
    DateType,
    EmbeddingType,
    FloatType,
    IntegerType,
    Session,
    StringType,
    TimestampType,
    col,
    dt,
    text,
)
from fenic.core.error import PlanError, ValidationError


def test_sql_simple_join(local_session_config):
    session = Session.get_or_create(local_session_config)
    df1 = session.create_dataframe(pl.DataFrame({"id": [1, 2], "name1": ["Alice", "Bob"]}))
    df2 = session.create_dataframe(pl.DataFrame({"id": [2, 3], "name2": ["Charlie", "David"]}))
    df = session.sql("SELECT * FROM {df1} JOIN {df2} USING (id)", df1=df1, df2=df2)
    assert df.schema.column_fields == [
        ColumnField("id", IntegerType),
        ColumnField("name1", StringType),
        ColumnField("name2", StringType),
    ]
    df = df.with_column("name", text.concat(df["name1"], df["name2"]))
    assert df.schema.column_fields == [
        ColumnField("id", IntegerType),
        ColumnField("name1", StringType),
        ColumnField("name2", StringType),
        ColumnField("name", StringType),
    ]
    df = df.to_polars()
    expected = pl.DataFrame({"id": [2], "name1": ["Bob"], "name2": ["Charlie"], "name": ["BobCharlie"]})
    assert df.equals(expected)

def test_sql_simple_join_loading_from_python_dict(local_session_config):
    session = Session.get_or_create(local_session_config)
    df1 = session.create_dataframe({"id": [1, 2], "name1": ["Alice", "Bob"]})
    df2 = session.create_dataframe({"id": [2, 3], "name2": ["Charlie", "David"]})
    df = session.sql("SELECT * FROM {df1} JOIN {df2} USING (id)", df1=df1, df2=df2)
    assert df.schema.column_fields == [
        ColumnField("id", IntegerType),
        ColumnField("name1", StringType),
        ColumnField("name2", StringType),
    ]
    df = df.with_column("name", text.concat(df["name1"], df["name2"]))
    assert df.schema.column_fields == [
        ColumnField("id", IntegerType),
        ColumnField("name1", StringType),
        ColumnField("name2", StringType),
        ColumnField("name", StringType),
    ]
    df = df.to_polars()
    expected = pl.DataFrame({"id": [2], "name1": ["Bob"], "name2": ["Charlie"], "name": ["BobCharlie"]})
    assert df.equals(expected)

def test_sql_simple_multiple_references_same_table(local_session_config):
    session = Session.get_or_create(local_session_config)
    df1 = session.create_dataframe(pl.DataFrame({"id": [1, 2], "name1": ["Alice", "Bob"]}))
    df2 = session.create_dataframe(pl.DataFrame({"id": [2, 3], "name2": ["Charlie", "David"]}))
    df = session.sql("SELECT {df1}.id, {df1}.name1, {df2}.name2 as name2 FROM {df1} JOIN {df2} on {df1}.id = {df2}.id", df1=df1, df2=df2)
    assert df.schema.column_fields == [
        ColumnField("id", IntegerType),
        ColumnField("name1", StringType),
        ColumnField("name2", StringType),
    ]
    df = df.with_column("name", text.concat(df["name1"], df["name2"]))
    assert df.schema.column_fields == [
        ColumnField("id", IntegerType),
        ColumnField("name1", StringType),
        ColumnField("name2", StringType),
        ColumnField("name", StringType),
    ]
    df = df.to_polars()
    expected = pl.DataFrame({"id": [2], "name1": ["Bob"], "name2": ["Charlie"], "name": ["BobCharlie"]})
    assert df.equals(expected)

def test_sql_three_table_join(local_session_config):
    session = Session.get_or_create(local_session_config)
    df1 = session.create_dataframe(pl.DataFrame({"id": [1, 2], "name1": ["Alice", "Bob"]}))
    df2 = session.create_dataframe(pl.DataFrame({"id": [2, 3], "name2": ["Charlie", "David"]}))
    df3 = session.create_dataframe(pl.DataFrame({"id": [2, 4], "name3": ["Eve", "Frank"]}))

    query = """
    SELECT df1.id, name1, name2, name3
    FROM {df1} as df1
    JOIN {df2} as df2 USING (id)
    JOIN {df3} as df3 USING (id)
    """

    df = session.sql(query, df1=df1, df2=df2, df3=df3)

    assert df.schema.column_fields == [
        ColumnField("id", IntegerType),
        ColumnField("name1", StringType),
        ColumnField("name2", StringType),
        ColumnField("name3", StringType),
    ]

    df_result = df.to_polars()
    expected = pl.DataFrame({
        "id": [2],
        "name1": ["Bob"],
        "name2": ["Charlie"],
        "name3": ["Eve"],
    })
    assert df_result.equals(expected)

def test_sql_project(local_session_config):
    session = Session.get_or_create(local_session_config)
    df1 = session.create_dataframe(pl.DataFrame({"id": [1, 2], "name1": ["Alice", "Bob"], "name2": ["Charlie", "David"]}))
    df = session.sql("SELECT concat(name1, ' ', name2) as name FROM {df1}", df1=df1)
    assert df.schema.column_fields == [
        ColumnField("name", StringType),
    ]
    df = df.to_polars()
    expected = pl.DataFrame({"name": ["Alice Charlie", "Bob David"]})
    assert df.equals(expected)


def test_sql_cte(local_session_config):
    session = Session.get_or_create(local_session_config)
    df1 = session.create_dataframe(pl.DataFrame({"id": [1, 2], "name1": ["Alice", "Bob"]}))
    df = session.sql("WITH cte AS (SELECT * FROM {df1}) SELECT * FROM cte", df1=df1)
    assert df.schema.column_fields == [
        ColumnField("id", IntegerType),
        ColumnField("name1", StringType),
    ]
    df = df.to_polars()
    assert df.shape == (2, 2)
    assert df.columns == ["id", "name1"]
    expected = pl.DataFrame({"id": [1, 2], "name1": ["Alice", "Bob"]})
    assert df.equals(expected)

def test_sql_union(local_session_config):
    session = Session.get_or_create(local_session_config)
    df = session.create_dataframe(pl.DataFrame({"id": [1, 2], "name1": ["Alice", "Bob"]}))
    df = session.sql("SELECT * FROM {df1} UNION ALL SELECT * FROM {df2} ORDER BY id", df1=df, df2=df)
    assert df.schema.column_fields == [
        ColumnField("id", IntegerType),
        ColumnField("name1", StringType),
    ]
    df = df.to_polars()
    expected = pl.DataFrame({"id": [1, 1, 2, 2], "name1": ["Alice", "Alice", "Bob", "Bob"]})
    assert df.equals(expected)

def test_sql_datetype(local_session_config):
    session = Session.get_or_create(local_session_config)
    df = session.create_dataframe(pl.DataFrame({
        "id": [1, 2],
        "date_str": ["2023-12-25", "2024-01-01"],
        "date_array": [[date(2023, 12, 25), date(2024, 1, 1)], [date(2024, 1, 2), date(2024, 1, 3)]],
        "embeddings": [[1.0, 2.0], [3.0, 4.0]]
    }))
    df = df.with_column("embeddings", col("embeddings").cast(EmbeddingType(embedding_model="test", dimensions=2)))

    # Use DuckDB to parse string as date
    df = session.sql("""
        SELECT id, CAST(date_str AS DATE) as parsed_date, date_array, embeddings
        FROM {df1}
        ORDER BY id
    """, df1=df)

    assert df.schema.column_fields == [
        ColumnField("id", IntegerType),
        ColumnField("parsed_date", DateType),  # Date coerced back to string
        ColumnField("date_array", ArrayType(DateType)),
        ColumnField("embeddings", ArrayType(FloatType)),
    ]

    # Collect and verify data
    df = df.to_polars()
    expected = pl.DataFrame({
        "id": [1, 2],
        "parsed_date": [date(2023, 12, 25), date(2024, 1, 1)],
        "date_array": [[date(2023, 12, 25), date(2024, 1, 1)], [date(2024, 1, 2), date(2024, 1, 3)]],
        "embeddings": [[1.0, 2.0], [3.0, 4.0]]
    })
    assert df.equals(expected)

def test_sql_self_join(local_session_config):
    session = Session.get_or_create(local_session_config)
    df = session.create_dataframe(pl.DataFrame({"id": [1, 2], "name1": ["Alice", "Bob"]}))
    df = session.sql(
    "SELECT l.id, l.name1, r.id as id_right, r.name1 as name1_right "
    "FROM {df_left} AS l JOIN {df_right} AS r USING (id)",
        df_left=df,
        df_right=df,
    )
    assert df.schema.column_fields == [
        ColumnField("id", IntegerType),
        ColumnField("name1", StringType),
        ColumnField("id_right", IntegerType),
        ColumnField("name1_right", StringType),
    ]
    df = df.to_polars()
    expected = pl.DataFrame({"id": [1, 2], "name1": ["Alice", "Bob"], "id_right": [1, 2], "name1_right": ["Alice", "Bob"]})
    assert df.equals(expected)

def test_sql_window_function(local_session_config):
    session = Session.get_or_create(local_session_config)
    df = session.create_dataframe(pl.DataFrame({
        "letter": ["a", "a", "b", "b"],
        "value": [10, 20, 30, 40],
    }))
    df = session.sql(
        "SELECT letter, value, ROW_NUMBER() OVER (PARTITION BY letter ORDER BY value DESC) AS rank FROM {df1}",
        df1=df
    )

    assert df.schema.column_fields == [
        ColumnField("letter", StringType),
        ColumnField("value", IntegerType),
        ColumnField("rank", IntegerType),
    ]

    df_result = df.to_polars()
    expected = pl.DataFrame({
        "letter": ["a", "a", "b", "b"],
        "value": [20, 10, 40, 30],
        "rank": [1, 2, 1, 2],
    })

    # Sort result for deterministic comparison
    df_result = df_result.sort(["letter", "rank"])
    expected = expected.sort(["letter", "rank"])

    assert df_result.equals(expected)

def test_sql_invalid_query(local_session_config):
    session = Session.get_or_create(local_session_config)
    df1 = session.create_dataframe(pl.DataFrame({"id": [1, 2], "name1": ["Alice", "Bob"]}))
    with pytest.raises(PlanError, match="SQL parsing failed"):
        session.sql("this is not a valid sql query", df1=df1)

def test_sql_missing_placeholder(local_session_config):
    session = Session.get_or_create(local_session_config)
    df1 = session.create_dataframe(pl.DataFrame({"id": [1, 2], "name1": ["Alice", "Bob"]}))
    with pytest.raises(ValidationError, match="Missing DataFrames for placeholders in SQL query"):
        session.sql("SELECT * FROM {df1} JOIN {df2} USING (id)", df1=df1)

def test_sql_multiple_statements(local_session_config):
    session = Session.get_or_create(local_session_config)
    df1 = session.create_dataframe(pl.DataFrame({"id": [1, 2], "name1": ["Alice", "Bob"]}))
    with pytest.raises(PlanError, match="Expected a single SQL statement in session.sql()"):
        session.sql("SELECT * FROM {df1}; SELECT * FROM {df1}", df1=df1)

def test_sql_insert_into(local_session_config):
    session = Session.get_or_create(local_session_config)
    df1 = session.create_dataframe(pl.DataFrame({"id": [1, 2], "name1": ["Alice", "Bob"]}))
    with pytest.raises(PlanError, match="Only read-only queries are supported"):
        session.sql("INSERT INTO foobar SELECT * FROM {df1}", df1=df1)

def test_sql_create_table(local_session_config):
    session = Session.get_or_create(local_session_config)
    df1 = session.create_dataframe(pl.DataFrame({"id": [1, 2], "name1": ["Alice", "Bob"]}))
    with pytest.raises(PlanError, match="Only read-only queries are supported"):
        session.sql("CREATE TABLE foobar AS SELECT * FROM {df1}", df1=df1)

def test_sql_drop_table(local_session_config):
    session = Session.get_or_create(local_session_config)
    session.create_dataframe(pl.DataFrame({"id": [1, 2], "name1": ["Alice", "Bob"]}))
    with pytest.raises(PlanError, match="Only read-only queries are supported"):
        session.sql("DROP TABLE foobar")

def test_sql_temporal_types(local_session_config):
    session = Session.get_or_create(local_session_config)
    df1 = session.create_dataframe(pl.DataFrame({
        "id": [1, 2],
        "date_col": [date(2023, 12, 25), date(2024, 1, 1)],
        "timestamp_col": [datetime(2023, 12, 25, 10, 20, 0, tzinfo=zoneinfo.ZoneInfo(key="UTC")), datetime(2024, 1, 1, 10, 20, 0, tzinfo=zoneinfo.ZoneInfo(key="UTC"))],
    }))
    df = session.sql("SELECT * FROM {df1}", df1=df1)
    assert df.schema.column_fields == [
        ColumnField("id", IntegerType),
        ColumnField("date_col", DateType),
        ColumnField("timestamp_col", TimestampType),
    ]


    df = df.to_polars()
    expected = pl.DataFrame({
        "id": [1, 2],
        "date_col": [date(2023, 12, 25), date(2024, 1, 1)],
        "timestamp_col": [datetime(2023, 12, 25, 10, 20, 0, tzinfo=zoneinfo.ZoneInfo(key="UTC")), datetime(2024, 1, 1, 10, 20, 0, tzinfo=zoneinfo.ZoneInfo(key="UTC"))],
    })
    assert df.equals(expected)

    df = session.sql("SELECT * FROM {df1} WHERE date_col > '2023-12-25'", df1=df1)
    result = df.to_pydict()
    assert result["id"] == [2]

    df = session.sql("SELECT * FROM {df1} WHERE timestamp_col < '2023-12-25 10:21:00'", df1=df1)
    result = df.to_pydict()
    assert result["id"] == [1]

    df = session.sql("SELECT DATE_TRUNC('year', timestamp_col) as date_trunc FROM {df1}", df1=df1)
    result = df.to_pydict()
    # depends on the timezone duckdb session is running in
    assert result["date_trunc"][0].day == 1 or 31
    assert result["date_trunc"][1].month == 1 or 12

    df = df.with_column("year", dt.year(dt.timestamp_add(col("date_trunc"), 1, "year")))
    result = df.to_pydict()
    assert result["year"] == [2024, 2025]

    df = session.sql("SELECT DATE_PART('day', timestamp_col - INTERVAL '1 day') as day FROM {df1}", df1=df1)
    result = df.to_pydict()
    assert result["day"] == [24, 31]

def test_sql_tz(local_session):
    """Going through DuckDB, Timezones will be converted to the session's local timezone.

    Test that we apply coercions to all entrypoints from duckdb back to UTC.
    """

    ts_la = datetime(2025, 1, 2, 1, 1, 1, tzinfo=zoneinfo.ZoneInfo(key="America/Los_Angeles"))
    ts_utc = datetime(2025, 1, 2, 1, 1, 1, tzinfo=zoneinfo.ZoneInfo(key="UTC"))
    df_with_date_types = local_session.create_dataframe(
        {
            "ts_la": [ts_la],
            "ts_utc": [ts_utc],
        })

    df = local_session.sql(
        "SELECT df1.*, ds.value as tz FROM {df1} AS df1 CROSS JOIN duckdb_settings() AS ds WHERE ds.name = 'TimeZone'",
        df1=df_with_date_types)
    df.select(col("ts_la"), col("ts_utc"), col("tz"))
    result_cached_df = df.select(col("ts_la"), col("ts_utc"), col("tz")).cache()

    # The schema for the the resulting dataframe should have timestamps with all the timezones set to UTC.
    assert df.schema.column_fields == [
        ColumnField("ts_la", TimestampType),
        ColumnField("ts_utc", TimestampType),
        ColumnField("tz", StringType),
    ]
    assert result_cached_df.schema.column_fields == [
        ColumnField("ts_la", TimestampType),
        ColumnField("ts_utc", TimestampType),
        ColumnField("tz", StringType),
    ]

    # This column contained a Timestamp in UTC, that now is in the current tz
    # after going through 'SELECT'.
    polars_df = df.to_polars().with_columns(pl.col("ts_la").dt.convert_time_zone("America/Los_Angeles").alias("ts_la_converted_back"))
    assert polars_df["ts_utc"][0] == ts_utc
    assert polars_df["ts_la_converted_back"][0] == ts_la

    polars_cached_df = result_cached_df.to_polars().with_columns(pl.col("ts_la").dt.convert_time_zone("America/Los_Angeles").alias("ts_la_converted_back"))
    assert polars_cached_df["ts_utc"][0] == ts_utc
    assert polars_cached_df["ts_la_converted_back"][0] == ts_la