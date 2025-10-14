import math
import re

import polars as pl
import pytest

from fenic import (
    ColumnField,
    DoubleType,
    EmbeddingType,
    IntegerType,
    Session,
    StringType,
    approx_count_distinct,
    avg,
    col,
    count,
    count_distinct,
    first,
    lit,
    max,
    mean,
    min,
    stddev,
    struct,
    sum,
    sum_distinct,
)
from fenic.core.error import PlanError, TypeMismatchError
from fenic.core.types.datatypes import (
    JsonType,
    MarkdownType,
)


def test_sum_aggregation(sample_df):
    result = sample_df.group_by("city").agg(sum(col("age"))).to_polars()
    assert len(result) == 2
    assert "sum(age)" in result.columns

    sf_row = result.filter(pl.col("city") == "San Francisco").row(0)
    seattle_row = result.filter(pl.col("city") == "Seattle").row(0)

    assert sf_row[1] == 55
    assert seattle_row[1] == 35

    result = sample_df.group_by("city").agg(sum("age")).to_polars()
    assert len(result) == 2
    assert "sum(age)" in result.columns

    sf_row = result.filter(pl.col("city") == "San Francisco").row(0)
    seattle_row = result.filter(pl.col("city") == "Seattle").row(0)

    assert sf_row[1] == 55
    assert seattle_row[1] == 35


def test_avg_aggregation(sample_df):
    result = sample_df.group_by("city").agg(avg(col("age"))).to_polars()
    assert len(result) == 2
    assert "avg(age)" in result.columns

    sf_row = result.filter(pl.col("city") == "San Francisco").row(0)
    seattle_row = result.filter(pl.col("city") == "Seattle").row(0)

    assert sf_row[1] == 27.5
    assert seattle_row[1] == 35

    result = sample_df.group_by("city").agg(mean(col("age"))).to_polars()
    assert len(result) == 2
    assert "avg(age)" in result.columns

    sf_row = result.filter(pl.col("city") == "San Francisco").row(0)
    seattle_row = result.filter(pl.col("city") == "Seattle").row(0)

    assert sf_row[1] == 27.5
    assert seattle_row[1] == 35


def test_min_aggregation(sample_df):
    result = sample_df.group_by("city").agg(min(col("age"))).to_polars()
    assert len(result) == 2
    assert "min(age)" in result.columns

    sf_row = result.filter(pl.col("city") == "San Francisco").row(0)
    seattle_row = result.filter(pl.col("city") == "Seattle").row(0)

    assert sf_row[1] == 25
    assert seattle_row[1] == 35


def test_max_aggregation(sample_df):
    result = sample_df.group_by("city").agg(max(col("age"))).to_polars()
    assert len(result) == 2
    assert "max(age)" in result.columns

    sf_row = result.filter(pl.col("city") == "San Francisco").row(0)
    seattle_row = result.filter(pl.col("city") == "Seattle").row(0)

    assert sf_row[1] == 30
    assert seattle_row[1] == 35


def test_count_aggregation(sample_df):
    result = sample_df.group_by("city").agg(count(col("age"))).to_polars()
    assert len(result) == 2
    assert "count(age)" in result.columns

    sf_row = result.filter(pl.col("city") == "San Francisco").row(0)
    seattle_row = result.filter(pl.col("city") == "Seattle").row(0)

    assert sf_row[1] == 2
    assert seattle_row[1] == 1


def test_count_aggregation_wildcard(local_session):
    data = {
        "department": [
            "Sales",
            "Engineering",
            "Marketing",
            "Sales",
            "Engineering",
            None,
        ],
        "salary": [50000, 85000, 60000, 55000, None, 75000],
        "bonus": [5000, None, 3000, 4500, 8000, None],
    }

    df = local_session.create_dataframe(data)

    result = (
        df.group_by("department")
        .agg(
            count("*").alias("total_rows"),
            count(lit("cat")).alias("total_rows_2"),
            count(lit(1)).alias("total_rows_3"),
            count("salary").alias("salary_count"),
            count("bonus").alias("bonus_count"),
        )
        .to_polars()
    )

    assert len(result) == 4  # Sales, Engineering, Marketing, and None

    eng_row = result.filter(pl.col("department") == "Engineering")
    assert eng_row["total_rows"][0] == 2
    assert eng_row["total_rows_2"][0] == 2
    assert eng_row["total_rows_3"][0] == 2
    assert eng_row["salary_count"][0] == 1
    assert eng_row["bonus_count"][0] == 1

    sales_row = result.filter(pl.col("department") == "Sales")
    assert sales_row["total_rows"][0] == 2
    assert sales_row["total_rows_2"][0] == 2
    assert sales_row["total_rows_3"][0] == 2
    assert sales_row["salary_count"][0] == 2
    assert sales_row["bonus_count"][0] == 2

    marketing_row = result.filter(pl.col("department") == "Marketing")
    assert marketing_row["total_rows"][0] == 1
    assert marketing_row["total_rows_2"][0] == 1
    assert marketing_row["total_rows_3"][0] == 1
    assert marketing_row["salary_count"][0] == 1
    assert marketing_row["bonus_count"][0] == 1

    null_row = result.filter(pl.col("department").is_null())
    assert null_row["total_rows"][0] == 1
    assert null_row["total_rows_2"][0] == 1
    assert null_row["total_rows_3"][0] == 1
    assert null_row["salary_count"][0] == 1
    assert null_row["bonus_count"][0] == 0


def test_global_agg_with_dict(local_session):
    """Test global aggregation with dictionary syntax."""
    data = {"age": [25, 30, 35, 28, 32], "salary": [50000, 45000, 60000, 45000, 55000]}
    df = local_session.create_dataframe(data)

    result = df.agg({"age": "min", "salary": "max"}).to_polars()

    assert len(result) == 1
    assert result["min(age)"][0] == 25
    assert result["max(salary)"][0] == 60000


def test_grouped_agg_with_expressions(local_session):
    """Test grouped aggregation with Column expressions."""
    data = {
        "department": ["IT", "HR", "IT", "HR", "IT"],
        "age": [25, 30, 35, 28, 32],
        "salary": [50000, 45000, 60000, 45000, 55000],
    }
    df = local_session.create_dataframe(data)

    result = (
        df.group_by("department")
        .agg(min(col("age")).alias("min_age"), max(col("salary")).alias("max_salary"))
        .to_polars()
    )

    assert len(result) == 2  # Two departments
    it_row = result.filter(pl.col("department") == "IT")
    assert it_row["min_age"][0] == 25
    assert it_row["max_salary"][0] == 60000


def test_grouped_agg_with_dict(local_session):
    """Test grouped aggregation with dictionary syntax."""
    data = {
        "department": ["IT", "HR", "IT", "HR", "IT"],
        "age": [25, 30, 35, 28, 32],
        "salary": [50000, 45000, 60000, 45000, 55000],
    }
    df = local_session.create_dataframe(data)

    result = df.group_by("department").agg({"age": "min", "salary": "max"}).to_polars()

    assert len(result) == 2
    hr_row = result.filter(pl.col("department") == "HR")
    assert hr_row["min(age)"][0] == 28
    assert hr_row["max(salary)"][0] == 45000


def test_agg_validation(local_session):
    """Test validation in agg() method."""
    data = {"age": [25, 30, 35]}
    df = local_session.create_dataframe(data)

    # Test with invalid function name
    with pytest.raises(ValueError, match="Unsupported aggregation function"):
        df.agg({"age": "invalid_func"}).to_polars()

    # Test with non-aggregation expression
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Expression (age + lit(1)) is not an aggregation. Aggregation expressions must use aggregate functions like sum(), avg(), min(), max(), count(). For example: df.agg(sum('col'), avg('col2'))"
        ),
    ):
        df.group_by().agg(col("age") + 1).to_polars()


def test_empty_groupby(local_session):
    """Test groupBy() with no columns is same as global aggregation."""
    data = {"age": [25, 30, 35]}
    df = local_session.create_dataframe(data)

    direct_result = df.agg(min(col("age"))).to_polars()
    grouped_result = df.group_by().agg(min(col("age"))).to_polars()

    assert direct_result.equals(grouped_result)

def test_groupby_derived_columns(local_session):
    """Test groupBy() with a derived column."""
    data = {
        "age": [25, 30, 30],
        "salary": [5, 10, 15],
    }
    df = local_session.create_dataframe(data)
    result = (
        df.group_by((col("age") + 1).alias("age_plus_one"))
        .agg(sum("age"), sum("salary"), sum(col("age") + col("salary")))
        .sort(col("age_plus_one"))
        .to_polars()
    )
    assert result.schema == {
        "age_plus_one": pl.Int64,
        "sum(age)": pl.Int64,
        "sum(salary)": pl.Int64,
        "sum((age + salary))": pl.Int64,
    }
    expected = pl.DataFrame(
        {
            "age_plus_one": [26, 31],
            "sum(age)": [25, 60],
            "sum(salary)": [5, 25],
            "sum((age + salary))": [30, 85],
        }
    )
    assert result.equals(expected), "DataFrame does not match expected"


def test_groupby_nested_aggregation(local_session):
    """Test that groupBy() with a nested aggregation raises an error."""
    data = {
        "age": [25, 30, 30],
        "salary": [5, 10, 15],
    }
    df = local_session.create_dataframe(data)
    with pytest.raises(
        PlanError, match="Invalid use of aggregate expressions"
    ):
        df.group_by("age").agg(sum(sum("salary"))).to_polars()

def test_avg_embedding_aggregation(local_session):
    """Test that avg() works correctly on EmbeddingType columns."""
    data = {
        "group": ["A", "A", "B", "B"],
        "vectors": [
            [1.0, 2.0, 3.0],    # Group A
            [3.0, 4.0, 5.0],    # Group A - avg should be [2.0, 3.0, 4.0]
            [2.0, 0.0, 1.0],    # Group B
            [4.0, 2.0, 3.0],    # Group B - avg should be [3.0, 1.0, 2.0]
        ]
    }
    df = local_session.create_dataframe(data)
    embedding_type = EmbeddingType(dimensions=3, embedding_model="test")

    # Cast vectors to embedding type and compute group-wise averages
    fenic_df = (
        df.select(
            col("group"),
            col("vectors").cast(embedding_type).alias("embeddings")
        )
        .group_by("group")
        .agg(avg("embeddings").alias("avg_embedding"))
        .sort("group")
    )
    assert fenic_df.schema.column_fields == [
        ColumnField("group", StringType),
        ColumnField("avg_embedding", EmbeddingType(dimensions=3, embedding_model="test")),
    ]

    result = fenic_df.to_polars()
    assert result.schema == {
        "group": pl.Utf8,
        "avg_embedding": pl.Array(pl.Float32, 3)
    }

    # Float-friendly comparisons
    group_a_result = result.filter(pl.col("group") == "A")["avg_embedding"][0].to_list()
    group_b_result = result.filter(pl.col("group") == "B")["avg_embedding"][0].to_list()

    assert group_a_result == pytest.approx([2.0, 3.0, 4.0], rel=1e-6)
    assert group_b_result == pytest.approx([3.0, 1.0, 2.0], rel=1e-6)


def test_avg_embedding_with_nulls(local_session):
    """Test that avg() on EmbeddingType handles null values correctly."""
    data = {
        "group": ["A", "A", "A", "B", "B"],
        "vectors": [
            [1.0, 2.0],     # Group A
            None,           # Group A - should be ignored
            [3.0, 4.0],     # Group A - avg should be [2.0, 3.0]
            [2.0, 0.0],     # Group B
            [4.0, 2.0],     # Group B - avg should be [3.0, 1.0]
        ]
    }
    df = local_session.create_dataframe(data)
    embedding_type = EmbeddingType(dimensions=2, embedding_model="test")

    fenic_df = (
        df.select(
            col("group"),
            col("vectors").cast(embedding_type).alias("embeddings")
        )
        .group_by("group")
        .agg(avg("embeddings").alias("avg_embedding"))
        .sort("group")
    )
    assert fenic_df.schema.column_fields == [
        ColumnField("group", StringType),
        ColumnField("avg_embedding", EmbeddingType(dimensions=2, embedding_model="test")),
    ]
    result = fenic_df.to_polars()
    assert result.schema == {
        "group": pl.Utf8,
        "avg_embedding": pl.Array(pl.Float32, 2)
    }
    group_a_result = result.filter(pl.col("group") == "A")["avg_embedding"][0].to_list()
    group_b_result = result.filter(pl.col("group") == "B")["avg_embedding"][0].to_list()
    assert group_a_result == pytest.approx([2.0, 3.0], rel=1e-6)
    assert group_b_result == pytest.approx([3.0, 1.0], rel=1e-6)

def test_first_aggregation(local_session):
    data = {
        "age": [25, 25, 30],
        "salary": [5, 5, 15],
    }
    df = local_session.create_dataframe(data)
    result = df.group_by("age").agg(first("salary")).sort("age").to_polars()
    expected = pl.DataFrame({
        "age": [25, 30],
        "first(salary)": [5, 15],
    })
    assert result.schema == {
        "age": pl.Int64,
        "first(salary)": pl.Int64,
    }
    assert result.equals(expected)

def test_stddev_aggregation(local_session):
    data = {
        "age": [25, 25, 30, 30, 30],
        "salary": [10, 20, 30, 40, 50],
    }
    df = local_session.create_dataframe(data)

    fenic_df = (
        df
        .select(
            col("age"),
            col("salary")
        )
        .group_by("age")
        .agg(stddev("salary"))
        .sort("age")
    )

    assert fenic_df.schema.column_fields == [
        ColumnField("age", IntegerType),
        ColumnField("stddev(salary)", DoubleType),
    ]
    result = fenic_df.to_polars()

    expected = pl.DataFrame({
        "age": [25, 30],
        "stddev(salary)": [math.sqrt(50), 10.0],
    })

    assert result.schema == {
        "age": pl.Int64,
        "stddev(salary)": pl.Float64,
    }

    # Check group keys match
    assert result["age"].to_list() == expected["age"].to_list()

    for res_val, exp_val in zip(result["stddev(salary)"], expected["stddev(salary)"], strict=True):
        assert res_val == pytest.approx(exp_val, rel=1e-9)


def test_count_distinct_aggregation(local_session: Session):
    # Larger dataset with numbers and strings
    groups = ["A"] * 1000 + ["B"] * 1000
    nums = [i % 10 for i in range(1000)] + [None if i % 50 == 0 else (i % 20) for i in range(1000)]
    data = {
        "group": groups,
        "value": nums,
        "text": [f"k{(i % 5)}" for i in range(2000)],
    }
    df = local_session.create_dataframe(data)
    result = (
        df.group_by("group")
        .agg(
            count_distinct("value").alias("cd"),
            count_distinct("text").alias("cd_text"),
        )
        .sort("group")
        .to_polars()
    )

    assert result.schema["cd"] in (pl.Int64, pl.UInt32, pl.Int32)
    assert result.schema["cd_text"] in (pl.Int64, pl.UInt32, pl.Int32)

    a_row = result.filter(pl.col("group") == "A").row(0)
    b_row = result.filter(pl.col("group") == "B").row(0)

    # For group A: value cycles 0..9 (10 distinct, no nulls)
    assert a_row[result.columns.index("cd")] == 10
    # For group B: value cycles 0..19 with some None; nulls are ignored for distinct
    assert b_row[result.columns.index("cd")] == 20
    # text cycles k0..k4 (5 distinct) in both groups (no nulls)
    assert a_row[result.columns.index("cd_text")] == 5
    assert b_row[result.columns.index("cd_text")] == 5


def test_count_distinct_multi_columns(local_session: Session):
    data = {
        "a": [1, 1, 1, 2, 2, None],
        "b": [1, 2, None, 1, 1, 2],
    }
    df = local_session.create_dataframe(data)
    # Distinct pairs by PySpark semantics (ignore rows where any column is null):
    # (1,1), (1,2), (2,1) => 3 distinct
    result = df.agg(count_distinct("a", "b").alias("cd_pairs")).to_polars()
    assert result["cd_pairs"][0] == 3

def test_sum_distinct_with_bools(local_session: Session):
    data = {
        "k": ["x"] * 100 + ["y"] * 100,
        "v": [True] * 100 + [False] * 100,
    }
    df = local_session.create_dataframe(data)
    result = df.agg(sum_distinct("v").alias("sd")).to_polars()
    assert result["sd"][0] == 1

def test_sum_distinct_aggregation(local_session: Session):
    data = {
        "k": ["x"] * 100 + ["y"] * 100,
        "v": [1] * 100 + [i % 5 for i in range(100)],
        "arr": [[1, 2]] * 200,
    }
    df = local_session.create_dataframe(data)
    result = (
        df.group_by("k")
        .agg(
            sum_distinct("v").alias("sd"),
        )
        .sort("k")
        .to_polars()
    )

    x_row = result.filter(pl.col("k") == "x").row(0)
    y_row = result.filter(pl.col("k") == "y").row(0)
    assert x_row[result.columns.index("sd")] == 1
    # y group has v cycling 0..4, distinct sum is 0+1+2+3+4=10
    assert y_row[result.columns.index("sd")] == 10

    # Arrays are not supported for sum_distinct (should error at signature time if attempted)
    with pytest.raises(TypeMismatchError):
        # type checker/validator should reject arrays for sum_distinct
        _ = df.group_by("k").agg(sum_distinct("arr")).to_polars()

    # Structs are not supported for sum_distinct (should error at signature time if attempted)
    with pytest.raises(TypeMismatchError):
        _ = df.group_by("k").agg(sum_distinct(struct("v", "k"))).to_polars()

@pytest.mark.parametrize("test_cardinality", [(1_000, "low cardinality"), (10_000, "medium cardinality"), (100_000, "high cardinality"), (1_000_000, "very high cardinality")])
def test_approx_count_distinct_approximation(local_session: Session, test_cardinality: tuple[int, str]):
    """Test that approx_count_distinct actually uses approximation with HyperLogLog++.

    This test verifies that the approximation is close to the exact count (within expected error bounds)
    """
    cardinality, description = test_cardinality
    # Create dataset with known cardinality
    # Repeat values to ensure we have a reasonable dataset size
    data = {
        "value": list(range(cardinality)),
    }
    df = local_session.create_dataframe(data)

    result = (
        df
        .agg(
            count_distinct("value").alias("exact_count"),
            approx_count_distinct("value").alias("approx_count"),
        )
        .to_polars()
    )

    exact = result["exact_count"][0]
    approx = result["approx_count"][0]

    # Verify exact count is correct
    assert exact == cardinality, f"{description}: exact count should be {cardinality}"

    # Calculate relative error
    relative_error = abs(approx - exact) / exact
    print(f"{description}: relative error {relative_error:.2%}")

    # HyperLogLog++ typical error rate is around 1.15% with default settings
    # We allow up to 5% to be safe (accounting for edge cases)
    max_allowed_error = 0.05

    assert relative_error <= max_allowed_error, (
        f"{description}: approximation error {relative_error:.2%} exceeds max allowed "
        f"{max_allowed_error:.2%} (exact={exact}, approx={approx})"
    )

    print(f"{description}: exact={exact}, approx={approx} relative error={relative_error:.2%}")

def test_approx_count_distinct_with_nulls_and_duplicates(local_session: Session):
    """Test that approx_count_distinct handles nulls correctly in approximate mode."""
    # Create a dataset with known duplicates and nulls
    data = {
        "group": ["A"] * 20_000,
        "value": (
            # 10,000 unique values (0-9999)
            list(range(10_000))
            # 9,000 duplicates of existing values
            + [i % 10_000 for i in range(9_000)]
            # 1,000 nulls
            + [None] * 1_000
        ),
    }
    df = local_session.create_dataframe(data)

    result = (
        df.group_by("group")
        .agg(
            count_distinct("value").alias("exact_count"),
            approx_count_distinct("value").alias("approx_count"),
        )
        .to_polars()
    )

    exact = result["exact_count"][0]
    approx = result["approx_count"][0]

    # Exact count should be 10,000 (nulls are ignored, duplicates are counted once)
    assert exact == 10_000

    # Approximation should be within 5% of exact
    relative_error = abs(approx - exact) / exact
    assert relative_error <= 0.05, (
        f"Approximation error {relative_error:.2%} exceeds 5% "
        f"(exact={exact}, approx={approx})"
    )

def test_datatype_compatibility_with_count_distinct(local_session: Session):
    # tests if count distinct can handle structs/logical types, etc.
    data = {
        "v": [1] * 100 + [i % 5 for i in range(100)],
        "arr": [[1, 2]] * 200,
        "struct": [{"a": x, "b": y} for x, y in zip(range(200), range(200), strict=True)],
        "markdown": ["# Hello"] * 200,
        "json": ["{\"a\": 1, \"b\": 2}"] * 200,
    }
    df = local_session.create_dataframe(data)
    df = df.select(
        col("v"),
        col("struct"),
        col("markdown").cast(MarkdownType).alias("markdown"),
        col("json").cast(JsonType).alias("json"),
    )
    result = df.agg(
        count_distinct("v").alias("cd"),
        count_distinct("struct").alias("cd_struct"),
        count_distinct("markdown").alias("cd_markdown"),
        count_distinct("json").alias("cd_json"),
    ).to_polars()
    print(result)
    assert result["cd"][0] == 5
    assert result["cd_struct"][0] == 200
    assert result["cd_markdown"][0] == 1
    assert result["cd_json"][0] == 1

    result_approx = df.agg(
        approx_count_distinct("v").alias("approx_cd"),
        approx_count_distinct("markdown").alias("approx_cd_markdown"),
        approx_count_distinct("json").alias("approx_cd_json"),
    ).to_polars()
    assert result_approx["approx_cd"][0] == 5
    assert result_approx["approx_cd_markdown"][0] == 1
    assert result_approx["approx_cd_json"][0] == 1

    with pytest.raises(TypeMismatchError):
        _ = df.agg(approx_count_distinct("struct")).to_polars()
