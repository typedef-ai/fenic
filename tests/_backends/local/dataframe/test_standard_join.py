import polars as pl
import pytest
from pydantic import ValidationError

from fenic.api.functions import col
from fenic.core.error import PlanError
from fenic.core.types import ColumnField, IntegerType, StringType


def test_inner_join(local_session):
    data1 = {
        "id": [1, 2, 3, 4],
        "valA": ["a1", "a2", "a3", "a4"],
    }
    data2 = {
        "id": [3, 4, 5],
        "valB": ["b3", "b4", "b5"],
    }
    df1 = local_session.create_dataframe(data1)
    df2 = local_session.create_dataframe(data2)

    # INNER JOIN on "id"
    joined = df1.join(df2, on="id", how="inner")
    assert joined.schema.column_fields == [
        ColumnField("id", IntegerType),
        ColumnField("valA", StringType),
        ColumnField("valB", StringType),
    ]
    polars_joined = joined.to_polars()
    expected = pl.DataFrame({
        "id": [3, 4],
        "valA": ["a3", "a4"],
        "valB": ["b3", "b4"],
    })
    assert polars_joined.equals(expected)

def test_inner_join_derived_columns(local_session):
    data1 = {
        "id1": [1, 2, 3, 4],
        "valA": ["a1", "a2", "a3", "a4"],
    }
    data2 = {
        "id2": [3, 4, 5],
        "valB": ["b3", "b4", "b5"],
    }
    df1 = local_session.create_dataframe(data1)
    df2 = local_session.create_dataframe(data2)

    # INNER JOIN on "id"
    joined = df1.join(df2, left_on=col("id1") + 1, right_on=col("id2") - 1, how="inner")
    assert joined.schema.column_fields == [
        ColumnField("id1", IntegerType),
        ColumnField("valA", StringType),
        ColumnField("id2", IntegerType),
        ColumnField("valB", StringType),
    ]
    polars_joined = joined.to_polars()
    expected = pl.DataFrame({
        "id1": [1, 2, 3],
        "valA": ["a1", "a2", "a3"],
        "id2": [3, 4, 5],
        "valB": ["b3", "b4", "b5"],
    })
    assert polars_joined.equals(expected)

def test_multi_col_join(local_session):
    df1 = local_session.create_dataframe(
        {"id": [1, 2, 2], "sub": [10, 20, 30], "valA": ["x1", "x2", "x3"]}
    )
    df2 = local_session.create_dataframe(
        {"id": [2, 2, 3], "sub": [20, 99, 30], "valB": ["y1", "y2", "y3"]}
    )

    # Join on ["id", "sub"]
    joined = df1.join(df2, on=["id", "sub"], how="inner")
    assert joined.schema.column_fields == [
        ColumnField("id", IntegerType),
        ColumnField("sub", IntegerType),
        ColumnField("valA", StringType),
        ColumnField("valB", StringType),
    ]
    polars_joined = joined.to_polars()
    expected = pl.DataFrame({
        "id": [2],
        "sub": [20],
        "valA": ["x2"],
        "valB": ["y1"],
    })
    assert polars_joined.equals(expected)

def test_join_with_derived_duplicate_column(local_session):
    df1 = local_session.create_dataframe(
        {"id": [1, 2, 2], "valA": ["x1", "x2", "x3"]}
    )
    df2 = local_session.create_dataframe(
        {"id": [2, 2, 3], "valB": ["y1", "y2", "y3"]}
    )

    with pytest.raises(PlanError, match="Duplicate column names"):
        df1.join(df2, left_on=[col("id") + 1], right_on=[col("id") - 1], how="inner")

def test_multi_column_join_with_derived_columns(local_session):
    df1 = local_session.create_dataframe(
        {"id1": [1, 2, 3], "sub1": [10, 11, 12], "valA": ["x1", "x2", "x3"]}
    )
    df2 = local_session.create_dataframe(
        {"id2": [3, 4, 5], "sub2": [12, 13, 14], "valB": ["y1", "y2", "y3"]}
    )

    joined = df1.join(df2, left_on=[col("id1") + 1, col("sub1") + 1], right_on=[col("id2") - 1, col("sub2") - 1], how="inner")
    assert joined.schema.column_fields == [
        ColumnField("id1", IntegerType),
        ColumnField("sub1", IntegerType),
        ColumnField("valA", StringType),
        ColumnField("id2", IntegerType),
        ColumnField("sub2", IntegerType),
        ColumnField("valB", StringType),
    ]
    polars_joined = joined.to_polars()
    expected = pl.DataFrame({
        "id1": [1, 2, 3],
        "sub1": [10, 11, 12],
        "valA": ["x1", "x2", "x3"],
        "id2": [3, 4, 5],
        "sub2": [12, 13, 14],
        "valB": ["y1", "y2", "y3"],
    })
    assert polars_joined.equals(expected)

def test_left_join(local_session):
    data1 = {
        "id": [1, 2, 3, 4],
        "valA": ["a1", "a2", "a3", "a4"],
    }
    data2 = {
        "id": [3, 4, 5],
        "valB": ["b3", "b4", "b5"],
    }
    df1 = local_session.create_dataframe(data1)
    df2 = local_session.create_dataframe(data2)

    # LEFT JOIN on "id"
    joined = df1.join(df2, on="id", how="left")
    assert joined.schema.column_fields == [
        ColumnField("id", IntegerType),
        ColumnField("valA", StringType),
        ColumnField("valB", StringType),
    ]
    polars_joined = joined.to_polars()
    expected = pl.DataFrame({
        "id": [1, 2, 3, 4],
        "valA": ["a1", "a2", "a3", "a4"],
        "valB": [None, None, "b3", "b4"],
    })
    assert polars_joined.equals(expected)


def test_right_join(local_session):
    data1 = {
        "id": [1, 2, 3, 4],
        "valA": ["a1", "a2", "a3", "a4"],
    }
    data2 = {
        "id": [3, 4, 5],
        "valB": ["b3", "b4", "b5"],
    }
    df1 = local_session.create_dataframe(data1)
    df2 = local_session.create_dataframe(data2)

    # RIGHT JOIN on "id"
    joined = df1.join(df2, on="id", how="right")
    assert joined.schema.column_fields == [
        ColumnField("valA", StringType),
        ColumnField("id", IntegerType),
        ColumnField("valB", StringType),
    ]
    polars_joined = joined.to_polars()
    expected = pl.DataFrame({
        "valA": ["a3", "a4", None],
        "id": [3, 4, 5],
        "valB": ["b3", "b4", "b5"],
    })
    assert polars_joined.equals(expected)


def test_full_outer_join(local_session):
    data1 = {
        "id1": [1, 2, 3],
        "valA": ["a1", "a2", "a3"],
    }
    data2 = {
        "id2": [2, 4],
        "valB": ["b2", "b4"],
    }
    df1 = local_session.create_dataframe(data1)
    df2 = local_session.create_dataframe(data2)

    # FULL/OUTER join
    result = df1.join(df2, left_on="id1", right_on="id2", how="full")
    assert result.schema.column_fields == [
        ColumnField("id1", IntegerType),
        ColumnField("valA", StringType),
        ColumnField("id2", IntegerType),
        ColumnField("valB", StringType),
    ]
    polars_joined = result.to_polars()
    expected = pl.DataFrame({
        "id1": [1, 2, 3, None],
        "valA": ["a1", "a2", "a3", None],
        "id2": [None, 2, None, 4],
        "valB": [None, "b2", None, "b4"],
    })
    assert polars_joined.equals(expected)

def test_full_outer_join_duplicate_join_keys(local_session):
    df1 = local_session.create_dataframe(
        {"id": [1, 2, 3], "valA": ["a1", "a2", "a3"]}
    )
    df2 = local_session.create_dataframe(
        {"id": [2, 4], "valB": ["b2", "b4"]}
    )

    with pytest.raises(PlanError, match="Duplicate column names"):
        df1.join(df2, on="id", how="full")


def test_cross_join(local_session):
    df1 = local_session.create_dataframe({"valA": ["a1", "a2"]})
    df2 = local_session.create_dataframe({"valB": ["b1", "b2", "b3"]})

    joined = df1.join(df2, how="cross")
    assert joined.schema.column_fields == [
        ColumnField("valA", StringType),
        ColumnField("valB", StringType),
    ]
    polars_joined = joined.to_polars()
    expected = pl.DataFrame({
        "valA": ["a1", "a1", "a1", "a2", "a2", "a2"],
        "valB": ["b1", "b2", "b3", "b1", "b2", "b3"],
    })
    assert polars_joined.equals(expected)


def test_join_duplicate_columns(local_session):
    left = local_session.create_dataframe(
        {"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]}
    )
    right = local_session.create_dataframe({"id": [1, 2], "name": ["Alice", "Bob"]})
    with pytest.raises(PlanError, match="Duplicate column names"):
        left.join(right, on="id", how="inner")

def test_unsupported_join_type(local_session):
    df1 = local_session.create_dataframe(
        {"id": [1, 2, 3], "valA": ["a1", "a2", "a3"]}
    )
    df2 = local_session.create_dataframe(
        {"id": [2, 4], "valB": ["b2", "b4"]}
    )
    with pytest.raises(ValidationError):
        df1.join(df2, on="id", how="anti")

def test_one_derived_column_one_not_derived_column(local_session):
    df1 = local_session.create_dataframe(
        {"id": [1, 2, 3], "valA": ["a1", "a2", "a3"]}
    )
    df2 = local_session.create_dataframe(
        {"id1": [2, 4], "valB": ["b2", "b4"]}
    )
    joined = df1.join(df2, left_on=col("id") + 1, right_on="id1", how="inner")
    assert joined.schema.column_fields == [
        ColumnField("id", IntegerType),
        ColumnField("valA", StringType),
        ColumnField("id1", IntegerType),
        ColumnField("valB", StringType),
    ]
    polars_joined = joined.to_polars()
    expected = pl.DataFrame({
        "id": [1, 3],
        "valA": ["a1", "a3"],
        "id1": [2, 4],
        "valB": ["b2", "b4"],
    })
    assert polars_joined.equals(expected)
