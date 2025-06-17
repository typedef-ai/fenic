"""Tests for DataFrame export methods (to_polars, to_pandas, to_arrow, etc.)."""

import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from fenic.core.metrics import QueryMetrics


@pytest.fixture
def sample_export_df(local_session):
    """Create a sample DataFrame for export testing."""
    data = {
        "id": [1, 2, 3, 4],
        "name": ["Alice", "Bob", "Charlie", "David"],
        "age": [25, 30, 35, 28],
        "score": [95.5, 88.0, 92.3, 87.1],
        "active": [True, False, True, False]
    }
    return local_session.create_dataframe(data)


def test_to_polars_basic(sample_export_df):
    """Test basic to_polars() without query metrics."""
    result = sample_export_df.to_polars()

    # Check return type
    assert isinstance(result, pl.DataFrame)

    # Check schema
    expected_columns = {"id", "name", "age", "score", "active"}
    assert set(result.columns) == expected_columns

    # Check data
    assert len(result) == 4
    assert result["name"].to_list() == ["Alice", "Bob", "Charlie", "David"]
    assert result["age"].to_list() == [25, 30, 35, 28]


def test_collect_polars(sample_export_df):
    """Test collect() method with polars data type."""
    result = sample_export_df.collect("polars")

    # Check return type
    assert hasattr(result, 'data')
    assert hasattr(result, 'metrics')

    # Check data type
    assert isinstance(result.data, pl.DataFrame)
    assert isinstance(result.metrics, QueryMetrics)

    # Check data
    assert len(result.data) == 4
    assert result.data["name"].to_list() == ["Alice", "Bob", "Charlie", "David"]


def test_to_pandas_basic(sample_export_df):
    """Test basic to_pandas() without query metrics."""
    result = sample_export_df.to_pandas()

    # Check return type
    assert isinstance(result, pd.DataFrame)

    # Check schema
    expected_columns = {"id", "name", "age", "score", "active"}
    assert set(result.columns) == expected_columns

    # Check data
    assert len(result) == 4
    assert result["name"].tolist() == ["Alice", "Bob", "Charlie", "David"]
    assert result["age"].tolist() == [25, 30, 35, 28]


def test_collect_pandas(sample_export_df):
    """Test collect() method with pandas data type."""
    result = sample_export_df.collect("pandas")

    # Check return type
    assert hasattr(result, 'data')
    assert hasattr(result, 'metrics')

    # Check data type
    assert isinstance(result.data, pd.DataFrame)
    assert isinstance(result.metrics, QueryMetrics)

    # Check data
    assert len(result.data) == 4
    assert result.data["name"].tolist() == ["Alice", "Bob", "Charlie", "David"]


def test_to_arrow_basic(sample_export_df):
    """Test basic to_arrow() without query metrics."""
    result = sample_export_df.to_arrow()

    # Check return type
    assert isinstance(result, pa.Table)

    # Check schema
    expected_columns = {"id", "name", "age", "score", "active"}
    assert set(result.column_names) == expected_columns

    # Check data
    assert len(result) == 4
    assert result["name"].to_pylist() == ["Alice", "Bob", "Charlie", "David"]
    assert result["age"].to_pylist() == [25, 30, 35, 28]


def test_collect_arrow(sample_export_df):
    """Test collect() method with arrow data type."""
    result = sample_export_df.collect("arrow")

    # Check return type
    assert hasattr(result, 'data')
    assert hasattr(result, 'metrics')

    # Check data type
    assert isinstance(result.data, pa.Table)
    assert isinstance(result.metrics, QueryMetrics)

    # Check data
    assert len(result.data) == 4
    assert result.data["name"].to_pylist() == ["Alice", "Bob", "Charlie", "David"]


def test_to_pydict_basic(sample_export_df):
    """Test basic to_pydict() without query metrics."""
    result = sample_export_df.to_pydict()

    # Check return type
    assert isinstance(result, dict)

    # Check schema
    expected_columns = {"id", "name", "age", "score", "active"}
    assert set(result.keys()) == expected_columns

    # Check data structure - should be dict of lists
    assert isinstance(result["name"], list)
    assert len(result["name"]) == 4
    assert result["name"] == ["Alice", "Bob", "Charlie", "David"]
    assert result["age"] == [25, 30, 35, 28]


def test_collect_pydict(sample_export_df):
    """Test collect() method with pydict data type."""
    result = sample_export_df.collect("pydict")

    # Check return type
    assert hasattr(result, 'data')
    assert hasattr(result, 'metrics')

    # Check data type
    assert isinstance(result.data, dict)
    assert isinstance(result.metrics, QueryMetrics)

    # Check data
    assert result.data["name"] == ["Alice", "Bob", "Charlie", "David"]


def test_to_pylist_basic(sample_export_df):
    """Test basic to_pylist() without query metrics."""
    result = sample_export_df.to_pylist()

    # Check return type
    assert isinstance(result, list)

    # Check data structure - should be list of dicts
    assert len(result) == 4
    assert all(isinstance(row, dict) for row in result)

    # Check first row
    first_row = result[0]
    expected_columns = {"id", "name", "age", "score", "active"}
    assert set(first_row.keys()) == expected_columns
    assert first_row["name"] == "Alice"
    assert first_row["age"] == 25

    # Check all names
    names = [row["name"] for row in result]
    assert names == ["Alice", "Bob", "Charlie", "David"]


def test_collect_pylist(sample_export_df):
    """Test collect() method with pylist data type."""
    result = sample_export_df.collect("pylist")

    # Check return type
    assert hasattr(result, 'data')
    assert hasattr(result, 'metrics')

    # Check data type
    assert isinstance(result.data, list)
    assert isinstance(result.metrics, QueryMetrics)

    # Check data
    assert len(result.data) == 4
    names = [row["name"] for row in result.data]
    assert names == ["Alice", "Bob", "Charlie", "David"]


def test_export_methods_with_transformations(sample_export_df):
    """Test export methods work correctly after DataFrame transformations."""
    # Apply some transformations
    transformed_df = (sample_export_df
                     .filter(sample_export_df["age"] > 27)
                     .select("name", "age", "score")
                     .sort("age"))

    # Test all export methods
    polars_result = transformed_df.to_polars()
    assert len(polars_result) == 3
    assert polars_result["name"].to_list() == ["David", "Bob", "Charlie"]

    # Test with pandas
    pandas_result = transformed_df.to_pandas()
    assert len(pandas_result) == 3
    assert pandas_result["name"].tolist() == ["David", "Bob", "Charlie"]

    # Test arrow
    arrow_result = transformed_df.to_arrow()
    assert len(arrow_result) == 3
    assert arrow_result["name"].to_pylist() == ["David", "Bob", "Charlie"]

    # Test pydict
    pydict_result = transformed_df.to_pydict()
    assert len(pydict_result["name"]) == 3
    assert pydict_result["name"] == ["David", "Bob", "Charlie"]

    # Test pylist
    pylist_result = transformed_df.to_pylist()
    assert len(pylist_result) == 3
    names = [row["name"] for row in pylist_result]
    assert names == ["David", "Bob", "Charlie"]

def test_export_methods_consistency(sample_export_df):
    """Test that all export methods return consistent data."""
    # Get data in all formats
    polars_df = sample_export_df.to_polars()
    arrow_table = sample_export_df.to_arrow()
    pydict = sample_export_df.to_pydict()
    pylist = sample_export_df.to_pylist()

    # Test pandas
    pandas_df = sample_export_df.to_pandas()
    pandas_names = pandas_df["name"].tolist()

    # Check that all formats return the same data
    polars_names = polars_df["name"].to_list()
    arrow_names = arrow_table["name"].to_pylist()
    pydict_names = pydict["name"]
    pylist_names = [row["name"] for row in pylist]

    expected_names = ["Alice", "Bob", "Charlie", "David"]

    assert polars_names == expected_names
    assert arrow_names == expected_names
    assert pydict_names == expected_names
    assert pylist_names == expected_names
    assert pandas_names == expected_names


def test_export_methods_with_null_values(local_session):
    """Test export methods handle null values correctly."""
    # Create DataFrame with null values
    data = {
        "id": [1, 2, 3],
        "name": ["Alice", None, "Charlie"],
        "age": [25, 30, None],
        "score": [95.5, None, 92.3]
    }
    df = local_session.create_dataframe(data)

    # Test all export methods handle nulls
    polars_result = df.to_polars()
    assert polars_result["name"][1] is None
    assert polars_result["age"][2] is None

    # Test with pandas
    pandas_result = df.to_pandas()
    assert pd.isna(pandas_result["name"].iloc[1])
    assert pd.isna(pandas_result["age"].iloc[2])

    # Test arrow
    arrow_result = df.to_arrow()
    assert arrow_result["name"][1].as_py() is None
    assert arrow_result["age"][2].as_py() is None

    # Test pydict
    pydict_result = df.to_pydict()
    assert pydict_result["name"][1] is None
    assert pydict_result["age"][2] is None

    # Test pylist
    pylist_result = df.to_pylist()
    assert pylist_result[1]["name"] is None
    assert pylist_result[2]["age"] is None


def test_collect_default(sample_export_df):
    """Test collect() method with default data type (polars)."""
    result = sample_export_df.collect()

    # Check return type
    assert hasattr(result, 'data')
    assert hasattr(result, 'metrics')

    # Check data type (default should be polars)
    assert isinstance(result.data, pl.DataFrame)
    assert isinstance(result.metrics, QueryMetrics)

    # Check metrics object has expected methods/attributes
    assert hasattr(result.metrics, 'get_summary')

    # Get summary should return a string
    summary = result.metrics.get_summary()
    assert isinstance(summary, str)
    assert len(summary) > 0


def test_export_methods_use_collect(sample_export_df):
    """Test that to_* methods return same data as collect().data."""
    # Test polars
    polars_direct = sample_export_df.to_polars()
    polars_collect = sample_export_df.collect("polars").data
    assert polars_direct.equals(polars_collect)

    # Test pandas
    pandas_direct = sample_export_df.to_pandas()
    pandas_collect = sample_export_df.collect("pandas").data
    assert pandas_direct.equals(pandas_collect)

    # Test arrow
    arrow_direct = sample_export_df.to_arrow()
    arrow_collect = sample_export_df.collect("arrow").data
    assert arrow_direct.equals(arrow_collect)

    # Test pydict
    pydict_direct = sample_export_df.to_pydict()
    pydict_collect = sample_export_df.collect("pydict").data
    assert pydict_direct == pydict_collect

    # Test pylist
    pylist_direct = sample_export_df.to_pylist()
    pylist_collect = sample_export_df.collect("pylist").data
    assert pylist_direct == pylist_collect
