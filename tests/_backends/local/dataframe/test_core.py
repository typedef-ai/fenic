import pytest

from fenic import IntegerType, col, lit, udf
from fenic.core.error import PlanError


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


def test_with_column_replace_existing(local_session):
    """Test replacing an existing column."""
    data = {"name": ["Alice", "Bob"], "age": [25, 30]}
    df = local_session.create_dataframe(data)

    # Replace with literal
    result = df.with_column("age", 50).to_polars()
    assert result["age"].to_list() == [50, 50]

    # Replace with expression
    result = df.with_column("age", col("age") * 2).to_polars()
    assert result["age"].to_list() == [50, 60]


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
