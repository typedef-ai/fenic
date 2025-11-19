import re

import pandas as pd
import polars as pl
import pytest

from fenic import IntegerType, col, lit, udf
from fenic.core.error import ExecutionError, PlanError, ValidationError


def test_select_columns(sample_df):
    result = sample_df.select("name", "city").to_polars()
    assert set(result.columns) == {"name", "city"}
    assert len(result) == 3


def test_select_with_column_expr(sample_df):
    result = sample_df.select(col("name"), col("city")).to_polars()
    assert set(result.columns) == {"name", "city"}
    assert len(result) == 3


def test_select_with_wildcard(sample_df):
    result = sample_df.select("*").to_polars()
    assert set(result.columns) == {"name", "age", "city"}
    assert len(result) == 3

    with pytest.raises(PlanError) as _:
        sample_df.select("name", "*").to_polars()

    result = sample_df.select("*", col("name").alias("name2")).to_polars()
    assert set(result.columns) == {"name", "age", "city", "name2"}
    assert len(result) == 3


def test_filter_simple(sample_df):
    result = sample_df.filter(col("age") > 30).to_polars()
    assert len(result) == 1
    assert result["age"][0] == 35


def test_where_simple(sample_df):
    result = sample_df.where(col("age") > 30).to_polars()
    assert len(result) == 1
    assert result["age"][0] == 35


def test_filter_compound(sample_df):
    result = sample_df.filter((col("age") > 25) & (col("age") < 35)).to_polars()
    assert len(result) == 1
    assert result["age"][0] == 30


def test_with_column(sample_df):
    result = sample_df.with_column("age_plus_1", col("age") + lit(1)).to_polars()
    assert "age_plus_1" in result.columns
    assert result["age_plus_1"][0] == 26

@pytest.fixture
def with_column_sample_df(local_session):
  data = {"name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35]}
  return local_session.create_dataframe(data)

def test_with_column_replace_existing(with_column_sample_df):
    """Test replacing an existing column."""
    result = with_column_sample_df.with_column("age", 50).to_polars()
    assert result["age"].to_list() == [50, 50, 50]
    # Replace with expression
    result = with_column_sample_df.with_column("age", col("age") * 2).to_polars()
    assert result["age"].to_list() == [50, 60, 70]

def test_with_column_series_length_too_long(with_column_sample_df):
  """Test that a Series with a length longer than the DataFrame raises an error."""
  expected_msg = (
    "Column 'next_age' was created from a Series of length 4, but the DataFrame has 3 rows. "
    "Series data must match the DataFrame height."
  )
  with pytest.raises(ExecutionError, match=re.escape(expected_msg)):
    with_column_sample_df.with_column("next_age", pl.Series("next_age", [1, 2, 3, 4])).to_polars()

def test_with_column_series_length_too_short(with_column_sample_df):
  """Test that a Series with a length shorter than the DataFrame raises an error."""
  expected_msg = (
    "Column 'next_age' was created from a Series of length 2, but the DataFrame has 3 rows. "
    "Series data must match the DataFrame height."
  )
  with pytest.raises(ExecutionError, match=re.escape(expected_msg)):
    with_column_sample_df.with_column("next_age", pl.Series("next_age", [1, 2])).to_polars()

def test_with_column_series_length_one_is_treated_as_literal(with_column_sample_df):
  """Test that a Series with a length one is treated as a literal."""
  result = with_column_sample_df.with_column("next_age", pl.Series("next_age", [1])).to_polars()
  assert result["next_age"].to_list() == [1, 1, 1]

def test_with_empty_series_raises_error(with_column_sample_df):
  """Test that a Series with an empty DataFrame raises an error."""
  with pytest.raises(ValidationError, match=re.escape("Series length cannot be 0. To add a column with null values, use `null(DoubleType)` instead.")):
    with_column_sample_df.with_column("age", pl.Series("age", [], dtype=pl.Float64)).to_polars()
  with pytest.raises(ValidationError, match=re.escape("Series length cannot be 0. To add a column with empty values, use `empty(ArrayType(element_type=IntegerType))` instead.")):
    with_column_sample_df.with_column("empty_column", pl.Series("empty_column", [], dtype=pl.List(pl.Int64))).to_polars()

def test_with_series_all_nulls_raises_error_without_dtype(with_column_sample_df):
  """Test that a Series with all nulls raises an error."""
  # with dtype, should work
  result = with_column_sample_df.with_column("age", pl.Series("age", [None, None, None], dtype=pl.Float64)).to_polars()
  assert result["age"].to_list() == [None, None, None]

  with pytest.raises(
    ValidationError, match=re.escape(
      "Series cannot contain all nulls unless a dtype is specified. Use `null(dtype)` (for primitive types) or `empty(ArrayType(...))` or `empty(StructType(...))` (for collections) instead, or specify a dtype for the Series."
    )
  ):
    with_column_sample_df.with_column("age", pl.Series("age", [None, None, None])).to_polars()

def test_with_columns_add_multiple(sample_df):
    """Test adding multiple new columns."""
    result = sample_df.with_columns({
        "age_plus_1": col("age") + lit(1),
        "double_age": col("age") * 2,
        "constant": lit(10)
    }).to_polars()

    assert "age_plus_1" in result.columns
    assert "double_age" in result.columns
    assert "constant" in result.columns
    assert result["age_plus_1"][0] == 26
    assert result["double_age"][0] == 50
    assert result["constant"][0] == 10


def test_with_columns_replace_and_add(local_session):
    """Test replacing existing column and adding new ones."""
    data = {"name": ["Alice", "Bob"], "age": [25, 30]}
    df = local_session.create_dataframe(data)

    result = df.with_columns({
        "age": col("age") + 1,
        "double_age": col("age") * 2
    }).to_polars()

    # age should be replaced with age + 1
    assert result["age"].to_list() == [26, 31]
    # double_age should be computed from the original age (not the replaced age)
    assert result["double_age"].to_list() == [50, 60]


def test_with_columns_literals(local_session):
    """Test adding columns with literal values."""
    data = {"name": ["Alice", "Bob"], "age": [25, 30]}
    df = local_session.create_dataframe(data)

    result = df.with_columns({
        "constant1": 1,
        "constant2": "test"
    }).to_polars()

    assert result["constant1"].to_list() == [1, 1]
    assert result["constant2"].to_list() == ["test", "test"]


def test_with_columns_empty(local_session):
    """Test calling with_columns with empty dict."""
    data = {"name": ["Alice", "Bob"], "age": [25, 30]}
    df = local_session.create_dataframe(data)

    result = df.with_columns({}).to_polars()
    assert result.equals(df.to_polars())


def test_with_columns_preserve_order(local_session):
    """Test that existing columns maintain their order."""
    data = {"name": ["Alice", "Bob"], "age": [25, 30], "city": ["NY", "LA"]}
    df = local_session.create_dataframe(data)

    result = df.with_columns({
        "double_age": col("age") * 2,
        "name_length": lit(5)
    }).to_polars()

    # Original columns should come first in their original order
    expected_cols = ["name", "age", "city", "double_age", "name_length"]
    assert list(result.columns) == expected_cols


def test_with_columns_pyspark_example(local_session):
    """Test PySpark-compatible example from documentation."""
    data = {"age": [2, 5], "name": ["Alice", "Bob"]}
    df = local_session.create_dataframe(data)

    # Example from PySpark docs: df.withColumns({'age2': df.age + 2, 'age3': df.age + 3})
    result = df.with_columns({
        "age2": col("age") + 2,
        "age3": col("age") + 3
    }).to_polars()

    assert "age2" in result.columns
    assert "age3" in result.columns
    assert result["age2"].to_list() == [4, 7]
    assert result["age3"].to_list() == [5, 8]


def test_with_columns_cannot_reference_new_columns(local_session):
    """Test that new columns cannot depend on other new columns in the same with_columns call."""
    import pytest

    data = {"age": [25, 30], "name": ["Alice", "Bob"]}
    df = local_session.create_dataframe(data)

    # This should fail because age_plus_2 tries to reference age_plus_1,
    # but age_plus_1 doesn't exist in the original DataFrame
    with pytest.raises(ValueError, match="Column 'age_plus_1' not found in schema"):
        df.with_columns({
            "age_plus_1": col("age") + 1,
            "age_plus_2": col("age_plus_1") + 1  # age_plus_1 not yet defined
        }).to_polars()

def test_with_columns_new_column_order(local_session):
    """Test that new columns maintain their insertion order."""
    df = local_session.create_dataframe({"a": [1]})
    result = df.with_columns({
        "z_col": lit(1),
        "y_col": lit(2),
        "x_col": lit(3)
    }).to_polars()

    # Original column first, then new columns in insertion order
    assert list(result.columns) == ["a", "z_col", "y_col", "x_col"]



def test_drop_column(sample_df):
    result = sample_df.drop("age").to_polars()
    assert "age" not in result.columns
    assert set(result.columns) == {"name", "city"}


def test_drop_multiple_columns(local_session):
    """Test dropping multiple columns by name."""
    data = {
        "age": [14, 23, 16],
        "name": ["Tom", "Alice", "Bob"],
        "height": [180, 165, 170],
    }
    df = local_session.create_dataframe(data)

    result = df.drop("age", "name").to_polars()

    assert set(result.columns) == {"height"}
    assert result["height"].to_list() == [180, 165, 170]


def test_drop_empty_args(local_session):
    """Test dropping with no arguments (should return same DataFrame)."""
    data = {"age": [14, 23, 16], "name": ["Tom", "Alice", "Bob"]}
    df = local_session.create_dataframe(data)

    result = df.drop().to_polars()

    assert result.equals(df.to_polars())


def test_dropping_non_existent_column_raises_error(local_session):
    """Test dropping a column that doesn't exist (should be no-op)."""
    data = {"age": [14, 23, 16], "name": ["Tom", "Alice", "Bob"]}
    df = local_session.create_dataframe(data)

    with pytest.raises(ValueError, match="Column 'nonexistent' not found in DataFrame"):
        df.drop("nonexistent").to_polars()


def test_dropping_all_columns_raises_error(local_session):
    """Test dropping a column that doesn't exist (should be no-op)."""
    data = {"age": [14, 23, 16], "name": ["Tom", "Alice", "Bob"]}
    df = local_session.create_dataframe(data)

    with pytest.raises(ValueError, match="Cannot drop all columns from DataFrame"):
        df.drop("age", "name").to_polars()


def test_column_access_bracket(sample_df):
    col_expr = sample_df["age"]
    assert col_expr._logical_expr.name == "age"


def test_column_access_dot(sample_df):
    col_expr = sample_df.age
    assert col_expr._logical_expr.name == "age"


def test_limit(sample_df):
    result = sample_df.limit(2).to_polars()
    assert len(result) == 2

    result = sample_df.limit(-2).to_polars()
    assert len(result) == 0

    result = sample_df.limit(0).to_polars()
    assert len(result) == 0

    result = sample_df.limit(1).to_polars()
    assert len(result) == 1


def test_cache_dataframe(local_session):
    computation_count = 0

    @udf(return_type=IntegerType)
    def counting_transform(x):
        nonlocal computation_count
        computation_count += 1
        return x * 2

    source = local_session.create_dataframe({"id": list(range(5)), "value": list(range(5))})

    # Create a DataFrame with our counting transform
    df = source.with_column("transformed", counting_transform(col("value"))).filter(
        col("id") > 1
    )
    # First execution without cache - should compute
    _ = df.to_polars()
    initial_count = computation_count
    assert initial_count > 0  # Verify our transform was actually called

    # Second execution without cache - should compute again
    _ = df.to_polars()
    assert computation_count > initial_count  # Counter should increase

    # Now cache the DataFrame
    cached_df = df.cache()

    # Reset the computation count
    computation_count = 0

    # First execution with cache - should compute and store
    cache_result1 = cached_df.to_polars()
    cache_computation_count = computation_count
    assert cache_computation_count > 0  # Verify computed once

    # Second execution with cache - should use cached result
    cache_result2 = cached_df.to_polars()
    assert computation_count == cache_computation_count  # Counter should not increase

    # Third execution with cache - should still use cached result
    cache_result3 = cached_df.to_polars()
    assert computation_count == cache_computation_count  # Counter should not increase

    # Verify all results are the same
    assert cache_result1.equals(cache_result2)
    assert cache_result2.equals(cache_result3)

    plus_one_df = cached_df.with_column("plus_one", col("transformed") + 1)
    result = plus_one_df.to_polars()

    # Should still use cached results for 'transformed' column
    assert computation_count == initial_count
    assert result["plus_one"].to_list() == [5, 7, 9]


def test_with_column_renamed(sample_df):
    renamed_df = sample_df.with_column_renamed("age", "age_in_years")
    assert renamed_df.columns == ["name", "age_in_years", "city"]

    df = renamed_df.with_column("age_plus_1", col("age_in_years") + 1)
    result = df.to_polars()
    assert "age_plus_1" in result.columns
    assert result["age_plus_1"][0] == 26


def test_with_column_renamed_non_existent(sample_df):
    result = sample_df.with_column_renamed("nonexistent", "new_name").to_polars()
    assert "nonexistent" not in result.columns
    assert "new_name" not in result.columns


# Phase 1: Tests for with_column with Series support


def test_with_column_polars_series(local_session):
    """Test adding a Polars Series to a DataFrame."""
    data = {"name": ["Alice", "Bob"], "age": [25, 30]}
    df = local_session.create_dataframe(data)

    # Create a Polars Series
    bonus_series = pl.Series("bonus", [100, 200])

    result = df.with_column("bonus", bonus_series).to_polars()

    assert "bonus" in result.columns
    assert result["bonus"].to_list() == [100, 200]


def test_with_column_pandas_series(local_session):
    """Test adding a pandas Series to a DataFrame."""
    data = {"name": ["Alice", "Bob"], "age": [25, 30]}
    df = local_session.create_dataframe(data)

    # Create a pandas Series
    bonus_series = pd.Series([100, 200], name="bonus")

    result = df.with_column("bonus", bonus_series).to_polars()

    assert "bonus" in result.columns
    assert result["bonus"].to_list() == [100, 200]


def test_with_column_series_replace_existing(local_session):
    """Test replacing an existing column with a Series."""
    data = {"name": ["Alice", "Bob"], "age": [25, 30]}
    df = local_session.create_dataframe(data)

    # Replace age column with a Series
    new_age_series = pl.Series("new_age", [26, 31])

    result = df.with_column("age", new_age_series).to_polars()

    assert result["age"].to_list() == [26, 31]


def test_with_column_series_dtypes(local_session):
    """Test Series with different data types."""
    data = {"id": [1, 2, 3]}
    df = local_session.create_dataframe(data)

    # Test with different dtype Series
    result_float = df.with_column("score", pl.Series([1.5, 2.5, 3.5])).to_polars()
    assert result_float["score"].dtype == pl.Float64

    result_str = df.with_column("label", pl.Series(["A", "B", "C"])).to_polars()
    assert result_str["label"].dtype == pl.Utf8

    result_bool = df.with_column("flag", pl.Series([True, False, True])).to_polars()
    assert result_bool["flag"].dtype == pl.Boolean


# Phase 2: Tests for with_columns with Series support


def test_with_columns_polars_series(local_session):
    """Test adding multiple Polars Series to a DataFrame."""
    data = {"name": ["Alice", "Bob"], "age": [25, 30]}
    df = local_session.create_dataframe(data)

    # Create multiple Polars Series
    result = df.with_columns({
        "bonus": pl.Series([100, 200]),
        "score": pl.Series([85.5, 92.0])
    }).to_polars()

    assert "bonus" in result.columns
    assert "score" in result.columns
    assert result["bonus"].to_list() == [100, 200]
    assert result["score"].to_list() == [85.5, 92.0]


def test_with_columns_pandas_series(local_session):
    """Test adding multiple pandas Series to a DataFrame."""
    data = {"name": ["Alice", "Bob"], "age": [25, 30]}
    df = local_session.create_dataframe(data)

    # Create multiple pandas Series
    result = df.with_columns({
        "bonus": pd.Series([100, 200]),
        "score": pd.Series([85.5, 92.0])
    }).to_polars()

    assert "bonus" in result.columns
    assert "score" in result.columns
    assert result["bonus"].to_list() == [100, 200]
    assert result["score"].to_list() == [85.5, 92.0]


def test_with_columns_mixed_series_and_columns(local_session):
    """Test mixing Series with Column expressions."""
    data = {"name": ["Alice", "Bob"], "age": [25, 30]}
    df = local_session.create_dataframe(data)

    result = df.with_columns({
        "bonus": pl.Series([100, 200]),
        "double_age": col("age") * 2,
        "constant": 1
    }).to_polars()

    assert result["bonus"].to_list() == [100, 200]
    assert result["double_age"].to_list() == [50, 60]
    assert result["constant"].to_list() == [1, 1]


def test_with_columns_series_replace_existing(local_session):
    """Test replacing existing columns with Series."""
    data = {"name": ["Alice", "Bob"], "age": [25, 30]}
    df = local_session.create_dataframe(data)

    result = df.with_columns({
        "age": pl.Series([26, 31]),
        "bonus": pl.Series([100, 200])
    }).to_polars()

    # age should be replaced
    assert result["age"].to_list() == [26, 31]
    # bonus should be added
    assert result["bonus"].to_list() == [100, 200]

def test_with_columns_with_invalid_series(local_session):
    """Test replacing existing columns with Series."""
    data = {"name": ["Alice", "Bob"], "age": [25, 30]}
    df = local_session.create_dataframe(data)

    expected_msg = (
      "Column 'age' was created from a Series of length 3, but the DataFrame has 2 rows. "
      "Series data must match the DataFrame height."
    )
    with pytest.raises(ExecutionError, match=re.escape(expected_msg)):
      df.with_columns({
        "age": pl.Series([26, 31, 32]),
        "bonus": pl.Series([100, 200])
      }).to_polars()



# Phase 3: Tests for pandas-specific edge cases


def test_with_column_pandas_series_with_index(local_session):
    """Test pandas Series with custom index is handled correctly."""
    data = {"name": ["Alice", "Bob"], "age": [25, 30]}
    df = local_session.create_dataframe(data)

    # Create pandas Series with custom index (should be ignored, data used positionally)
    bonus_series = pd.Series([100, 200], index=["x", "y"])

    result = df.with_column("bonus", bonus_series).to_polars()

    assert result["bonus"].to_list() == [100, 200]


def test_with_column_pandas_series_nullable_dtypes(local_session):
    """Test pandas Series with nullable integer types."""
    data = {"name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35]}
    df = local_session.create_dataframe(data)

    # Create pandas Series with nullable integer type
    score_series = pd.Series([100, None, 200], dtype="Int64")

    result = df.with_column("score", score_series).to_polars()

    assert result["score"][0] == 100
    assert result["score"][1] is None
    assert result["score"][2] == 200


def test_with_column_pandas_categorical(local_session):
    """Test pandas Series with categorical dtype."""
    data = {"name": ["Alice", "Bob"], "age": [25, 30]}
    df = local_session.create_dataframe(data)

    # Create pandas Series with categorical dtype
    category_series = pd.Series(["A", "B"], dtype="category")

    result = df.with_column("category", category_series).to_polars()

    # Polars should convert categorical to string
    assert result["category"].to_list() == ["A", "B"]


def test_with_column_pandas_datetime(local_session):
    """Test pandas Series with datetime dtype."""
    data = {"name": ["Alice", "Bob"], "age": [25, 30]}
    df = local_session.create_dataframe(data)

    # Create pandas Series with datetime
    date_series = pd.Series(pd.to_datetime(["2023-01-01", "2023-01-02"]))

    result = df.with_column("date", date_series).to_polars()

    assert "date" in result.columns
    assert result["date"].dtype == pl.Datetime
