import polars as pl
import pytest
from pydantic import ValidationError

from fenic import (
    ColumnField,
    IntegerType,
    Schema,
    StructField,
    StructType,
    array,
    col,
)
from fenic.core.error import PlanError


def test_explode_array(local_session):
    data = {
        "id": [1, 2],
        "tags": [["a", "b"], ["c", "d", "e"]],
    }
    df = local_session.create_dataframe(data)
    exploded = df.explode("tags").to_polars()

    # For id=1, the array has 2 elements; for id=2, it has 3 elements.
    assert len(exploded) == 5

    # Check that the "id" column is repeated appropriately.
    assert exploded["id"].to_list() == [1, 1, 2, 2, 2]
    # And the exploded "tags" column has the individual elements.
    assert exploded["tags"].to_list() == ["a", "b", "c", "d", "e"]


# Test that exploding a non-array column raises a ValueError.
def test_explode_non_array(local_session):
    data = {
        "id": [1, 2],
        "name": ["Alice", "Bob"],
    }
    df = local_session.create_dataframe(data)
    with pytest.raises(
        ValueError,
        match="Explode operator expected an array column for field name, but received StringType instead.",
    ):
        df.explode("name").to_polars()


# Test behavior when an array is empty.
def test_explode_empty_array(local_session):
    data = {
        "id": [1, 2, 3],
        "tags": [[], ["x"], None],
    }
    df = local_session.create_dataframe(data)
    exploded = df.explode("tags").to_polars()

    # Row with an empty list becomes a row with null after explosion,
    # and our ExplodeExec filters it out.
    # So only the row with ["x"] should appear.
    assert len(exploded) == 1
    assert exploded["id"].to_list() == [2]
    assert exploded["tags"].to_list() == ["x"]


# Test that after explosion the schema of the exploded column is updated
# (i.e. it now holds individual elements, not an array).
def test_explode_schema(local_session):
    data = {
        "id": [1, 2],
        "tags": [["hello", "world"], ["polars"]],
    }
    df = local_session.create_dataframe(data)
    exploded_df = df.explode("tags")
    collected = exploded_df.to_polars()

    # Check that the type of the exploded column is now a string (and not a list).
    # (Depending on your schema conversion, you might check that the value is not a list.)
    assert isinstance(collected["tags"][0], str)
    # Also, verify that non-exploded columns remain intact.
    assert "id" in collected.columns


# Test that passing a Column expression yields the same result as passing a string.
def test_explode_with_logical_expression(local_session):
    data = {
        "id": [1, 2],
        "tags": [["a", "b"], ["c", "d", "e"]],
    }
    df = local_session.create_dataframe(data)

    exploded_using_expr = df.explode(col("tags")).to_polars()
    exploded_using_str = df.explode("tags").to_polars()

    # Convert to pandas DataFrames for comparison.
    assert exploded_using_expr.equals(exploded_using_str)

    data = {
        "id": [1, 2],
        "tags": [["a", "b"], ["c", "d", "e"]],
    }
    df = local_session.create_dataframe(data)

    exploded_using_expr = df.explode(col("tags").alias("tags2")).to_polars()
    # check whether dataframe has both tags and tags2
    assert "tags" in exploded_using_expr.columns
    assert "tags2" in exploded_using_expr.columns


# Test that using a Column expression on a non-array column raises a ValueError.
def test_explode_non_array_logical_expression(local_session):
    data = {
        "id": [1, 2],
        "name": ["Alice", "Bob"],
    }
    df = local_session.create_dataframe(data)

    # Using col("name") should trigger an error because "name" is not an array column.
    with pytest.raises(
        ValueError,
        match="Explode operator expected an array column for field name, but received StringType instead.",
    ):
        df.explode(col("name")).to_polars()


def test_explode_array_expression(local_session):
    data = {"a": [1, 4], "b": [2, 5], "c": [3, 6]}
    old_df = local_session.create_dataframe(data)

    new_df = old_df.explode(array(col("a"), col("b"), col("c")).alias("abc"))
    collected = new_df.to_polars()

    assert len(collected) == 6

    assert collected["abc"].to_list() == [1, 2, 3, 4, 5, 6]


def test_drop_duplicates(local_session):
    df = local_session.create_dataframe(
        {
            "c1": [1, 2, 3, 1],
            "c2": ["a", "a", "a", "a"],
            "c3": ["b", "b", "b", "b"],
        }
    )

    df_unique_collected = df.drop_duplicates().to_polars()
    df_distinct_collected = df.distinct().to_polars()
    assert len(df_unique_collected) == 3
    assert len(df_distinct_collected) == 3

    df_unique_collected_subset = df.drop_duplicates(["c2", "c3"]).to_polars()
    assert len(df_unique_collected_subset) == 1
    assert df_unique_collected_subset.write_json() == '[{"c1":1,"c2":"a","c3":"b"}]'


def test_unnest(local_session):
    df = local_session.create_dataframe(
        {
            "id": [1, 2],
            "tags": [{"red": 1, "blue": 2, "green": None}, {"green": 6}],
            "nested_tags": [
                {"a": {"d": 5, "e": 6}, "b": 1, "c": 2},
                {"a": {"d": 5, "e": 6}, "b": 3, "c": 4},
            ],
        }
    )
    unnested_df = df.unnest("tags", "nested_tags")
    assert unnested_df.schema == Schema(
        [
            ColumnField("id", IntegerType),
            ColumnField("red", IntegerType),
            ColumnField("blue", IntegerType),
            ColumnField("green", IntegerType),
            ColumnField(
                "a",
                StructType(
                    [StructField("d", IntegerType), StructField("e", IntegerType)]
                ),
            ),
            ColumnField("b", IntegerType),
            ColumnField("c", IntegerType),
        ]
    )
    unnested_df = unnested_df.unnest("a")
    assert unnested_df.schema == Schema(
        [
            ColumnField("id", IntegerType),
            ColumnField("red", IntegerType),
            ColumnField("blue", IntegerType),
            ColumnField("green", IntegerType),
            ColumnField("d", IntegerType),
            ColumnField("e", IntegerType),
            ColumnField("b", IntegerType),
            ColumnField("c", IntegerType),
        ]
    )
    collected = unnested_df.to_polars()
    df_expected = pl.DataFrame(
        {
            "id": [1, 2],
            "red": [1, None],
            "blue": [2, None],
            "green": [None, 6],
            "d": [5, 5],
            "e": [6, 6],
            "b": [1, 3],
            "c": [2, 4],
        }
    )
    assert collected.equals(df_expected)


def test_unnest_ambiguous_column_name(local_session):
    df = local_session.create_dataframe(
        {"a": [{"a": 1, "b": 2}, {"a": 3, "b": 4}], "b": [5, 6]}
    )
    with pytest.raises(PlanError, match="Duplicate column name"):
        df.unnest("a")


def test_drop_duplicates_invalid_param(local_session):
    df = local_session.create_dataframe(
        {
            "c1": [1, 2, 3, 1],
        }
    )

    with pytest.raises(ValidationError):
        df.drop_duplicates(col("c1")).to_polars()

    with pytest.raises(ValidationError):
        df.drop_duplicates("c1", col("c1")).to_polars()

    with pytest.raises(ValidationError):
        df.drop_duplicates("c2").to_polars()


def test_show_smoke_test(local_session):
    df = local_session.create_dataframe(
        {
            "c1": [1, 2, 3, 1],
            "c2": ["a", "a", "a", "a"],
            "c3": ["b", "b", "b", "b"],
        }
    )
    df.show()
