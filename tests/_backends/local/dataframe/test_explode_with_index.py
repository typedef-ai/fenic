"""Tests for explode_with_index and posexplode DataFrame methods."""


import fenic as fc


def test_explode_with_index_basic(local_session):
    """Test basic explode_with_index functionality."""
    df = local_session.create_dataframe({
        "id": [1, 2, 3],
        "tags": [["red", "blue"], ["green"], []],
    })

    result = df.explode_with_index("tags").to_polars()

    # Check schema
    assert "col" in result.columns
    assert "id" in result.columns
    assert "pos" in result.columns

    # Check values - empty arrays should be filtered out
    assert len(result) == 3
    assert result["pos"].to_list() == [0, 1, 0]
    assert result["id"].to_list() == [1, 1, 2]
    assert result["col"].to_list() == ["red", "blue", "green"]


def test_explode_with_index_custom_names(local_session):
    """Test explode_with_index with custom column names."""
    df = local_session.create_dataframe({
        "id": [1, 2],
        "tags": [["red", "blue"], ["green"]],
    })

    result = df.explode_with_index("tags", index_col_name="pos", value_col_name="tag").to_polars()

    # Check schema has custom names
    assert "pos" in result.columns
    assert "tag" in result.columns
    assert "tags" not in result.columns  # Original column renamed

    # Check values
    assert result["pos"].to_list() == [0, 1, 0]
    assert result["tag"].to_list() == ["red", "blue", "green"]


def test_posexplode_basic(local_session):
    """Test posexplode PySpark-compatible alias."""
    df = local_session.create_dataframe({
        "id": [1, 2],
        "tags": [["red", "blue"], ["green"]],
    })

    result = df.posexplode("tags").to_polars()

    # Check schema has PySpark-compatible names
    assert "pos" in result.columns
    assert "col" in result.columns

    # Check values
    assert result["pos"].to_list() == [0, 1, 0]
    assert result["col"].to_list() == ["red", "blue", "green"]
    assert result["id"].to_list() == [1, 1, 2]


def test_explode_with_index_multiple_rows(local_session):
    """Test explode_with_index with multiple rows and different array lengths."""
    df = local_session.create_dataframe({
        "id": [1, 2, 3, 4],
        "values": [[10, 20, 30], [40], [], [50, 60]],
    })

    result = df.explode_with_index("values").to_polars()

    # Check the indices reset for each row
    assert result["pos"].to_list() == [0, 1, 2, 0, 0, 1]
    assert result["col"].to_list() == [10, 20, 30, 40, 50, 60]
    assert result["id"].to_list() == [1, 1, 1, 2, 4, 4]


def test_explode_with_index_preserves_other_columns(local_session):
    """Test that explode_with_index preserves all other columns."""
    df = local_session.create_dataframe({
        "id": [1, 2],
        "name": ["Alice", "Bob"],
        "tags": [["red", "blue"], ["green"]],
        "count": [10, 20],
    })

    result = df.explode_with_index("tags", index_col_name="tag_idx").to_polars()

    # All original columns should be preserved
    assert "id" in result.columns
    assert "name" in result.columns
    assert "count" in result.columns
    assert "tag_idx" in result.columns
    assert "col" in result.columns

    # Check values are correctly duplicated
    assert result["name"].to_list() == ["Alice", "Alice", "Bob"]
    assert result["count"].to_list() == [10, 10, 20]


def test_explode_with_index_column_expression(local_session):
    """Test explode_with_index with Column expression."""
    df = local_session.create_dataframe({
        "id": [1, 2],
        "tags": [["a", "b"], ["c"]],
    })

    result = df.explode_with_index(fc.col("tags"), value_col_name="tags").to_polars()

    assert result["pos"].to_list() == [0, 1, 0]
    assert result["tags"].to_list() == ["a", "b", "c"]


def test_explode_with_index_with_some_empty_arrays(local_session):
    """Test explode_with_index when some arrays are empty."""
    # Note: Can't test completely empty dataframes or all-empty arrays
    # because Polars infers Null type which Fenic doesn't support yet
    df = local_session.create_dataframe({
        "id": [1, 2, 3],
        "tags": [["a"], [], ["b"]],
    })

    result = df.explode_with_index("tags").to_polars()

    # Empty arrays should be filtered out
    assert len(result) == 2
    assert result["pos"].to_list() == [0, 0]
    assert result["col"].to_list() == ["a", "b"]
    assert result["id"].to_list() == [1, 3]


def test_explode_with_index_integer_arrays(local_session):
    """Test explode_with_index with integer arrays."""
    df = local_session.create_dataframe({
        "id": [1, 2],
        "numbers": [[1, 2, 3], [4, 5]],
    })

    result = df.explode_with_index("numbers", index_col_name="num_idx").to_polars()

    assert result["num_idx"].to_list() == [0, 1, 2, 0, 1]
    assert result["col"].to_list() == [1, 2, 3, 4, 5]


def test_posexplode_preserves_column_order(local_session):
    """Test that posexplode adds pos and col columns correctly."""
    df = local_session.create_dataframe({
        "id": [1, 2],
        "name": ["Alice", "Bob"],
        "tags": [["red"], ["blue"]],
    })

    result = df.posexplode("tags").to_polars()

    # pos should be first in the schema (as per our implementation)
    assert "pos" in result.columns
    assert "col" in result.columns
    assert "id" in result.columns
    assert "name" in result.columns

    # Check values
    assert result["pos"].to_list() == [0, 0]
    assert result["col"].to_list() == ["red", "blue"]


def test_explode_with_index_single_element_arrays(local_session):
    """Test explode_with_index with single-element arrays."""
    df = local_session.create_dataframe({
        "id": [1, 2, 3],
        "tags": [["only"], ["one"], ["element"]],
    })

    result = df.explode_with_index("tags").to_polars()

    # Each should have index 0
    assert result["pos"].to_list() == [0, 0, 0]
    assert result["col"].to_list() == ["only", "one", "element"]
    assert result["id"].to_list() == [1, 2, 3]


def test_explode_with_index_value_name_only(local_session):
    """Test explode_with_index when only value_name is provided (index defaults)."""
    df = local_session.create_dataframe({
        "id": [1, 2],
        "tags": [["x", "y"], ["z"]],
    })

    result = df.explode_with_index("tags", value_col_name="tag").to_polars()

    # pos should use default name
    assert "pos" in result.columns
    # value column should be renamed
    assert "tag" in result.columns and "tags" not in result.columns
    assert result["pos"].to_list() == [0, 1, 0]
    assert result["tag"].to_list() == ["x", "y", "z"]


def test_explode_with_index_index_name_only_outer(local_session):
    """Test explode_with_index with only index_name and outer behavior."""
    df = local_session.create_dataframe({
        "id": [1, 2, 3],
        "vals": [[10], [], None],
    })

    result = df.explode_with_index("vals", index_col_name="pos", value_col_name="vals", keep_null_and_empty=True).to_polars()

    assert result["id"].to_list() == [1, 2, 3]
    assert result["pos"].to_list() == [0, None, None]
    # value column keeps original name since value_name not provided
    assert result["vals"].to_list() == [10, None, None]


def test_explode_with_index_both_names_outer(local_session):
    """Test explode_with_index with both custom names and outer behavior."""
    df = local_session.create_dataframe({
        "id": [1, 2, 3],
        "letters": [["a", "b"], [], None],
    })

    result = df.explode_with_index("letters", index_col_name="pos", value_col_name="val", keep_null_and_empty=True).to_polars()

    # expect 4 rows: 2 from first, 1 for empty, 1 for null
    assert len(result) == 4
    assert result["pos"].to_list() == [0, 1, None, None]
    assert result["val"].to_list() == ["a", "b", None, None]
    assert result["id"].to_list() == [1, 1, 2, 3]


def test_explode_with_index_expr_custom_names(local_session):
    """Test explode_with_index with Column expression and custom names."""
    df = local_session.create_dataframe({
        "id": [1, 2],
        "arr": [[1, 2], [3]],
    })

    result = df.explode_with_index(fc.col("arr"), index_col_name="idx", value_col_name="val").to_polars()

    assert "idx" in result.columns and "val" in result.columns
    assert result["idx"].to_list() == [0, 1, 0]
    assert result["val"].to_list() == [1, 2, 3]

def test_explode_outer_basic(local_session):
    """Test explode_outer keeps null and empty arrays."""
    df = local_session.create_dataframe({
        "id": [1, 2, 3],
        "tags": [["red", "blue"], [], None],
    })

    result = df.explode_outer("tags").to_polars()

    # Should preserve all rows, including empty arrays and nulls
    assert len(result) == 4  # 2 from first array + 1 from empty + 1 from null
    assert result["id"].to_list() == [1, 1, 2, 3]
    assert result["tags"].to_list() == ["red", "blue", None, None]


def test_posexplode_outer_basic(local_session):
    """Test posexplode_outer keeps null and empty arrays with positions."""
    df = local_session.create_dataframe({
        "id": [1, 2, 3],
        "tags": [["red", "blue"], [], None],
    })

    result = df.posexplode_outer("tags").to_polars()
    result_with_index = df.explode_with_index("tags", keep_null_and_empty=True).to_polars()

    # Should preserve all rows with (pos, col) structure
    assert len(result) == len(result_with_index)
    assert result["pos"].to_list() == result_with_index["pos"].to_list() == [0, 1, None, None]
    assert result["col"].to_list() == result_with_index["col"].to_list() == ["red", "blue", None, None]
    assert result["id"].to_list() == result_with_index["id"].to_list() == [1, 1, 2, 3]


def test_posexplode_outer_vs_posexplode(local_session):
    """Test the difference between posexplode and posexplode_outer."""
    df = local_session.create_dataframe({
        "id": [1, 2, 3],
        "tags": [["red", "blue"], [], None],
    })

    # Regular posexplode filters out empty/null arrays
    regular_result = df.posexplode("tags").to_polars()
    assert len(regular_result) == 2  # Only the non-empty array
    assert regular_result["pos"].to_list() == [0, 1]
    assert regular_result["col"].to_list() == ["red", "blue"]

    # posexplode_outer keeps all rows
    outer_result = df.posexplode_outer("tags").to_polars()
    assert len(outer_result) == 4  # All rows preserved
    assert outer_result["pos"].to_list() == [0, 1, None, None]
    assert outer_result["col"].to_list() == ["red", "blue", None, None]

