import pytest

from fenic import col
from fenic.core._logical_plan.plans.base import PlanError


def test_basic_union(sample_df):
    df1 = sample_df.filter(col("age") <= 25)
    df2 = sample_df.filter(col("age") > 25)
    result = df1.union(df2).to_polars()
    assert len(result) == 3
    assert set(result["age"]) == {25, 30, 35}


def test_union_different_column_order(local_session):
    """Test union works with different column ordering."""
    data1 = {"id": [1, 2], "value": ["a", "b"]}
    data2 = {"value": ["c", "d"], "id": [3, 4]}
    df1 = local_session.create_dataframe(data1)
    df2 = local_session.create_dataframe(data2)

    result = df1.union(df2).to_polars()

    # Check result has same column order as first DataFrame
    assert result.columns == df1.columns
    assert result["id"].to_list() == [1, 2, 3, 4]
    assert result["value"].to_list() == ["a", "b", "c", "d"]


def test_union_with_duplicates(local_session):
    """Test union preserves duplicate rows (like UNION ALL)."""
    data1 = {"id": [1, 2], "value": ["a", "b"]}
    data2 = {"id": [1, 2], "value": ["a", "b"]}  # Same data as data1
    df1 = local_session.create_dataframe(data1)
    df2 = local_session.create_dataframe(data2)

    result = df1.union(df2).to_polars()

    # Should have 4 rows (duplicates preserved)
    assert result.shape == (4, 2)
    assert result["id"].to_list() == [1, 2, 1, 2]
    assert result["value"].to_list() == ["a", "b", "a", "b"]


def test_union_different_column_names(local_session):
    """Test union fails with incompatible schemas."""
    data1 = {"id": [1, 2], "value": ["a", "b"]}
    data2 = {"id": [3, 4], "different_column": ["c", "d"]}
    df1 = local_session.create_dataframe(data1)
    df2 = local_session.create_dataframe(data2)

    # Should raise PlanError due to different columns
    with pytest.raises(PlanError, match="Cannot union DataFrames: DataFrame #1 has different columns than DataFrame #0."):
        df1.union(df2).to_polars()

def test_union_different_column_types(local_session):
    """Test union fails with incompatible schemas."""
    data1 = {"id": [1, 2], "value": ["a", "b"]}
    data2 = {"id": [3, 4], "value": [1, 2]}
    df1 = local_session.create_dataframe(data1)
    df2 = local_session.create_dataframe(data2)

    # Should raise PlanError due to different columns
    with pytest.raises(PlanError, match="Cannot union DataFrames: DataFrame #1 has incompatible column types."):
        df1.union(df2).to_polars()
