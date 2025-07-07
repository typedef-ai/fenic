import pytest

from fenic import array, array_contains, array_size, col, lit, struct
from fenic.core.error import TypeMismatchError


def test_array_size_happy_path(local_session):
    # Test with simple string array
    data = {"text_col": [["hello", "bar"], None, ["hello", "foo", "bar"]]}
    df = local_session.create_dataframe(data)
    result = df.with_column("size_col", array_size(col("text_col"))).to_polars()
    assert result["size_col"][0] == 2
    assert result["size_col"][1] is None
    assert result["size_col"][2] == 3

    # Test with array of structs
    struct_data = {
        "id": [1, 2, 3],
        "struct_array": [
            [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}],
            [{"name": "Charlie", "age": 35}],
            None,
        ],
    }
    struct_df = local_session.create_dataframe(struct_data)
    struct_result = struct_df.with_column(
        "array_size", array_size(col("struct_array"))
    ).to_polars()
    assert struct_result["array_size"].to_list() == [2, 1, None]


def test_array_size_error_cases(local_session):
    with pytest.raises(TypeMismatchError):
        data = {"my_col": [1, 2, 3]}
        df = local_session.create_dataframe(data)
        df.with_column("size_col", array_size(col("my_col"))).to_polars()


def test_array_contains_literal(local_session):
    data = {"my_col": [["a", "b", "c"], ["d", "e"], None]}
    df = local_session.create_dataframe(data)
    result = df.with_column(
        "contains_col", array_contains(col("my_col"), "b")
    ).to_polars()

    assert result["contains_col"].to_list() == [True, False, None]
    result = df.with_column(
        "contains_col", array_contains(col("my_col"), lit("b"))
    ).to_polars()
    assert result["contains_col"].to_list() == [True, False, None]

    data = {"my_col": [["a", "b", "c"], ["d", "e"], None], "value": ["b", "f", "g"]}
    df = local_session.create_dataframe(data)
    result = df.with_column(
        "contains_col", array_contains(col("my_col"), col("value"))
    ).to_polars()
    assert result["contains_col"].to_list() == [True, False, None]


def test_array_contains_struct(local_session):
    df = local_session.create_dataframe(
        {
            "a": [
                [{"b": 1, "c": [2, 3]}, {"b": 3, "c": [4, 5]}],
                [None, {"b": 7, "c": [8, 9]}],
            ],
            "b": [{"b": 1, "c": [2, 3]}, {"b": 3, "c": [4, 5]}],
        }
    )
    result = df.with_column(
        "contains_col", array_contains(col("a"), col("b"))
    ).to_polars()
    assert result["contains_col"].to_list() == [True, False]

    result = df.with_column(
        "contains_col", array_contains(col("a"), lit({"b": 1, "c": [2, 3]}))
    ).to_polars()
    assert result["contains_col"].to_list() == [True, False]

    result = df.with_column(
        "contains_col",
        array_contains(
            col("a"), struct(lit(1).alias("b"), array(lit(2), lit(3)).alias("c"))
        ),
    ).to_polars()
    assert result["contains_col"].to_list() == [True, False]


def test_array_contains_error_cases(local_session):
    with pytest.raises(ValueError):
        data = {"my_col": ["a", "b", "c"]}
        df = local_session.create_dataframe(data)
        df.with_column(
            "contains_col", array_contains(col("my_col"), col("value"))
        ).to_polars()

    with pytest.raises(TypeMismatchError):
        data = {"my_col": [["a", "b", "c"], ["d", "e"], None]}
        df = local_session.create_dataframe(data)
        df.with_column("contains_col", array_contains(col("my_col"), lit(1))).to_polars()
