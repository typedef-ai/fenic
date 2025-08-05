import re

import pytest

from fenic.api.functions import (
    asc,
    asc_nulls_first,
    asc_nulls_last,
    col,
    desc,
    desc_nulls_last,
    text,
)
from fenic.core.error import PlanError


def test_sort_basic(sample_df_dups_and_nulls):
    """Test basic sorting functionality with single column."""
    result = sample_df_dups_and_nulls.sort("name").to_polars()
    assert result.item(0, "name") is None
    assert result.item(1, "name") == "Alice"

    result = sample_df_dups_and_nulls.sort("name", ascending=False).to_polars()
    assert result.item(0, "name") is None
    assert result.item(1, "name") == "David"

    result = sample_df_dups_and_nulls.sort([]).to_polars()
    result_orig = sample_df_dups_and_nulls.to_polars()
    assert result.equals(result_orig)

    result = sample_df_dups_and_nulls.sort().to_polars()
    result_orig = sample_df_dups_and_nulls.to_polars()
    assert result.equals(result_orig)

    result = sample_df_dups_and_nulls.sort([], ascending=False).to_polars()
    assert result.equals(result_orig)


def test_sort_multiple_columns(sample_df_dups_and_nulls):
    """Test sorting with multiple columns and different ascending boolean arguments."""
    result = sample_df_dups_and_nulls.sort(["group", col("name")]).to_polars()
    assert result.item(1, "group") == 100
    assert result.item(3, "name") == "Alice"
    assert result.item(5, "name") == "Charlie"

    result = sample_df_dups_and_nulls.sort(
        ["group", col("name")], ascending=False
    ).to_polars()
    assert result.item(1, "group") == 300
    assert result.item(2, "name") == "Alice"
    assert result.item(0, "name") == "Charlie"

    result = sample_df_dups_and_nulls.sort(
        ["group", col("name")], ascending=[False, True]
    ).to_polars()
    assert result.item(1, "group") == 300
    assert result.item(0, "name") == "Alice"
    assert result.item(2, "name") == "Charlie"


def test_sort_with_asc_desc_expressions(sample_df_dups_and_nulls):
    """Test sorting with various asc/desc column expressions."""
    result = sample_df_dups_and_nulls.sort(
        [col("group").desc(), asc("name"), asc_nulls_first(col("city"))]
    ).to_polars()
    assert result.item(0, "group") == 300
    assert result.item(2, "name") == "Charlie"

    result = sample_df_dups_and_nulls.sort(
        [desc_nulls_last(col("age") * col("group")), "city"]
    ).to_polars()
    assert result.item(0, "city") == "Largest Product"
    assert result.item(5, "city") == "Product with Null"

    result = sample_df_dups_and_nulls.sort(
        [col("group").desc_nulls_first(), desc("name"), asc_nulls_last(col("city"))]
    ).to_polars()
    assert result.item(0, "group") == 300
    assert result.item(0, "name") == "Charlie"


def test_sort_complex_expressions(sample_df_dups_and_nulls):
    """Test sorting with complex column expressions."""
    result = sample_df_dups_and_nulls.sort(
        [col("age") * col("group"), col("group")]
    ).to_polars()
    assert result.item(0, "city") == "Product with Null"
    assert result.item(5, "city") == "Largest Product"

    result = sample_df_dups_and_nulls.sort(
        desc_nulls_last(col("age") * col("group"))
    ).to_polars()
    assert result.item(5, "city") == "Product with Null"
    assert result.item(0, "city") == "Largest Product"


def test_sort_invalid_inputs(sample_df_dups_and_nulls):
    """Test error cases."""
    # Test error cases when mixing asc/desc expressions with boolean ascending arguments.
    with pytest.raises(TypeError, match="Cannot specify both"):
        sample_df_dups_and_nulls.sort(asc("age"), ascending=True).to_polars()

    with pytest.raises(TypeError, match="Cannot specify both"):
        sample_df_dups_and_nulls.sort(
            ["age", col("group").asc()], ascending=[True, False]
        ).to_polars()
    # Test error cases when using asc/desc expressions in non-sort operations.
    with pytest.raises(
        PlanError, match="Sort expressions are not allowed in `projection`"
    ):
        sample_df_dups_and_nulls.select(col("age").asc()).to_polars()

    with pytest.raises(
        PlanError, match=re.escape("Invalid use of `.asc()`, `.desc()`, or related sort functions")
    ):
        sample_df_dups_and_nulls.select(
            text.concat(desc_nulls_last(col("age")), col("name"))
        ).to_polars()
