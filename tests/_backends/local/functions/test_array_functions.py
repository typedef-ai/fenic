import pytest

import fenic as fc
from fenic import array, col, lit, struct
from fenic.core.error import TypeMismatchError


def test_array_size_happy_path(local_session):
    # Test with simple string array
    data = {"text_col": [["hello", "bar"], None, ["hello", "foo", "bar"]]}
    df = local_session.create_dataframe(data)
    result = df.with_column("size_col", fc.arr.size(col("text_col"))).to_polars()
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
        "array_size", fc.arr.size(col("struct_array"))
    ).to_polars()
    assert struct_result["array_size"].to_list() == [2, 1, None]


def test_array_size_error_cases(local_session):
    with pytest.raises(TypeMismatchError):
        data = {"my_col": [1, 2, 3]}
        df = local_session.create_dataframe(data)
        df.with_column("size_col", fc.arr.size(col("my_col"))).to_polars()


def test_array_contains_literal(local_session):
    data = {"my_col": [["a", "b", "c"], ["d", "e"], None]}
    df = local_session.create_dataframe(data)
    result = df.with_column(
        "contains_col", fc.arr.contains(col("my_col"), "b")
    ).to_polars()

    assert result["contains_col"].to_list() == [True, False, None]


def test_array_distinct_primitives_and_structs(local_session):
    data = {
        "nums": [[1, 2, 2, 3], [4, 4, 4], None],
        "objs": [
            [{"a": 1, "b": 2}, {"a": 1, "b": 2}, {"a": 2, "b": 3}],
            [{"a": 1, "b": 2}],
            None,
        ],
    }
    df = local_session.create_dataframe(data)
    out = df.select(
        fc.arr.distinct(col("nums")).alias("u_nums"),
        fc.arr.distinct(col("objs")).alias("u_objs"),
    ).to_polars()
    assert out["u_nums"].to_list()[0] == [1, 2, 3]
    assert out["u_nums"].to_list()[1] == [4]
    assert out["u_nums"].to_list()[2] is None

    # Structs preserve order of first appearances and remove duplicates. Order is not necessarily preserved.
    expected = [{"a": 1, "b": 2}, {"a": 2, "b": 3}]
    actual =out["u_objs"].to_list()[0]
    assert len(actual) == len(expected)
    assert all(item in expected for item in actual)
    assert all(item in actual for item in expected)
    assert out["u_objs"].to_list()[1] == [{"a": 1, "b": 2}]
    assert out["u_objs"].to_list()[2] is None


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
        "contains_col", fc.arr.contains(col("a"), col("b"))
    ).to_polars()
    assert result["contains_col"].to_list() == [True, False]

    result = df.with_column(
        "contains_col", fc.arr.contains(col("a"), lit({"b": 1, "c": [2, 3]}))
    ).to_polars()
    assert result["contains_col"].to_list() == [True, False]

    result = df.with_column(
        "contains_col",
        fc.arr.contains(
            col("a"), struct(lit(1).alias("b"), array(lit(2), lit(3)).alias("c"))
        ),
    ).to_polars()
    assert result["contains_col"].to_list() == [True, False]


def test_array_contains_error_cases(local_session):
    with pytest.raises(ValueError):
        data = {"my_col": ["a", "b", "c"]}
        df = local_session.create_dataframe(data)
        df.with_column(
            "contains_col", fc.arr.contains(col("my_col"), col("value"))
        ).to_polars()

    with pytest.raises(TypeMismatchError):
        data = {"my_col": [["a", "b", "c"], ["d", "e"], None]}
        df = local_session.create_dataframe(data)
        df.with_column("contains_col", fc.arr.contains(col("my_col"), lit(1))).to_polars()


# =============================================================================
# New Array Function Tests
# =============================================================================


def test_array_max_min(local_session):
    """Test array_max and array_min functions."""
    df = local_session.create_dataframe({"nums": [[3, 1, 5, 2], [10, 20], None]})
    result = df.select(
        fc.arr.max("nums").alias("max_val"), fc.arr.min("nums").alias("min_val")
    ).to_polars()

    assert result["max_val"].to_list() == [5, 20, None]
    assert result["min_val"].to_list() == [1, 10, None]


def test_array_sort(local_session):
    """Test array_sort function."""
    df = local_session.create_dataframe({"nums": [[3, 1, 2], [5, 4, 6], None]})
    result = df.select(fc.arr.sort("nums").alias("sorted")).to_polars()

    assert result["sorted"].to_list() == [[1, 2, 3], [4, 5, 6], None]


def test_array_reverse(local_session):
    """Test reverse function."""
    df = local_session.create_dataframe({"arr": [[1, 2, 3], [4, 5], None]})
    result = df.select(fc.arr.reverse("arr").alias("reversed")).to_polars()

    assert result["reversed"].to_list() == [[3, 2, 1], [5, 4], None]


def test_array_remove(local_session):
    """Test array_remove function."""
    df = local_session.create_dataframe({"arr": [[1, 2, 3, 2], [4, 2, 5], None]})
    result = df.select(fc.arr.remove("arr", 2).alias("removed")).to_polars()

    # First array removes all 2s
    assert result["removed"].to_list()[0] == [1, 3]
    assert result["removed"].to_list()[1] == [4, 5]
    assert result["removed"].to_list()[2] is None


def test_array_union(local_session):
    """Test array_union function."""
    df = local_session.create_dataframe({
        "arr1": [[1, 2, 3], [4, 5]],
        "arr2": [[2, 3, 4], [5, 6]]
    })
    result = df.select(fc.arr.union("arr1", "arr2").alias("union")).to_polars()

    # Union should have distinct elements
    assert set(result["union"].to_list()[0]) == {1, 2, 3, 4}
    assert set(result["union"].to_list()[1]) == {4, 5, 6}


def test_array_intersect(local_session):
    """Test array_intersect function."""
    df = local_session.create_dataframe({
        "arr1": [[1, 2, 3], [4, 5, 6]],
        "arr2": [[2, 3, 4], [5, 6, 7]]
    })
    result = df.select(fc.arr.intersect("arr1", "arr2").alias("intersect")).to_polars()

    assert set(result["intersect"].to_list()[0]) == {2, 3}
    assert set(result["intersect"].to_list()[1]) == {5, 6}


def test_array_except(local_session):
    """Test array_except function."""
    df = local_session.create_dataframe({
        "arr1": [[1, 2, 3], [4, 5, 6]],
        "arr2": [[2, 3], [5, 6]]
    })
    result = df.select(fc.arr.except_("arr1", "arr2").alias("except")).to_polars()

    assert result["except"].to_list()[0] == [1]
    assert result["except"].to_list()[1] == [4]


def test_array_compact(local_session):
    """Test array_compact function."""
    df = local_session.create_dataframe({"arr": [[1, None, 2, None, 3], [None, None], None]})
    result = df.select(fc.arr.compact("arr").alias("compact")).to_polars()

    assert result["compact"].to_list() == [[1, 2, 3], [], None]


def test_array_repeat(local_session):
    """Test array_repeat function."""
    df = local_session.create_dataframe({"val": ["x", "y"], "count": [3, 2]})
    result = df.select(fc.arr.repeat(fc.col("val"), fc.col("count")).alias("repeated")).to_polars()

    repeated_col = result["repeated"].to_list()
    assert repeated_col[0] == ["x", "x", "x"]
    assert repeated_col[1] == ["y", "y"]

def test_array_repeat_nested(local_session):
    """Test array_repeat function with nested arrays."""
    df = local_session.create_dataframe({"val": [["x", "y"], ["z"]], "count": [3, 2]})
    result = df.select(fc.arr.repeat(fc.col("val"), fc.col("count")).alias("repeated")).to_polars()

    repeated_col = result["repeated"].to_list()
    assert repeated_col[0] == [["x", "y"], ["x", "y"], ["x", "y"]]
    assert repeated_col[1] == [["z"], ["z"]]

def test_array_repeat_struct(local_session):
    """Test array_repeat function with structs."""
    df = local_session.create_dataframe({"val": [{"a": "x", "b": "y"}, {"a": "z"}], "count": [3, 2]})
    result = df.select(fc.arr.repeat(fc.col("val"), fc.col("count")).alias("repeated")).to_polars()

    repeated_col = result["repeated"].to_list()
    assert repeated_col[0] == [{"a": "x", "b": "y"}, {"a": "x", "b": "y"}, {"a": "x", "b": "y"}]
    # b is None because the struct type is {a: str, b: str}, even if b is not populated in the second row
    assert repeated_col[1] == [{"a": "z", "b": None}, {"a": "z", "b": None}]


def test_flatten(local_session):
    """Test flatten function."""
    df = local_session.create_dataframe({"nested": [[[1, 2], [3, 4]], [[5]], None]})
    result = df.select(fc.flatten("nested").alias("flat")).to_polars()

    assert result["flat"].to_list() == [[1, 2, 3, 4], [5], None]


def test_slice(local_session):
    """Test slice function with PySpark 1-based indexing."""
    df = local_session.create_dataframe({"arr": [[1, 2, 3, 4, 5]]})

    # PySpark uses 1-based indexing: slice(arr, 2, 3) gets elements from position 2, length 3
    result = df.select(fc.arr.slice("arr", 2, 3).alias("sliced")).to_polars()
    assert result["sliced"].to_list() == [[2, 3, 4]]


def test_element_at(local_session):
    """Test element_at function with PySpark 1-based indexing."""
    df = local_session.create_dataframe({"arr": [[10, 20, 30, 40]]})

    # PySpark uses 1-based indexing: element_at(arr, 1) gets first element
    result = df.select(
        fc.arr.element_at("arr", 1).alias("first"),
        fc.arr.element_at("arr", -1).alias("last")
    ).to_polars()

    assert result["first"].to_list() == [10]
    assert result["last"].to_list() == [40]


def test_arrays_overlap(local_session):
    """Test arrays_overlap function."""
    df = local_session.create_dataframe({
        "arr1": [[1, 2, 3], [4, 5], [7, 8]],
        "arr2": [[3, 4], [6, 7], [9, 10]]
    })
    result = df.select(fc.arr.overlap("arr1", "arr2").alias("overlap")).to_polars()

    assert result["overlap"].to_list() == [True, False, False]
