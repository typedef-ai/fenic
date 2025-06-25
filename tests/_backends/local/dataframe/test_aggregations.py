import math
import re

import polars as pl
import pytest

from fenic import (
    ColumnField,
    DoubleType,
    EmbeddingType,
    IntegerType,
    StringType,
    avg,
    col,
    count,
    first,
    lit,
    max,
    mean,
    min,
    semantic,
    stddev,
    sum,
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


def test_semantic_reduce_with_groupby(local_session):
    """Test semantic.reduce() method."""
    data = {
        "date": ["2024-01-01", "2024-01-01", "2024-01-02"],
        "notes": [
            "Q4 Sales Review Discussion: Revenue exceeded targets by 12%. John mentioned concerns about EMEA pipeline. Team agreed John will conduct deep-dive analysis by Friday. Alice suggested meeting with key clients to gather feedback.",
            "Product Planning: Discussed upcoming features for Q1. Team debated prioritization of mobile vs desktop improvements. Bob noted sprint board needs restructuring. Agreed to have product roadmap ready for next board meeting.",
            "Marketing Sync: Campaign performance trending well. Creative assets need final revisions before launch next week. Sarah raised concerns about Q1 budget - needs executive approval for additional spend.",
        ],
        "num_attendees": [10, 15, 20],
    }
    df = local_session.create_dataframe(data)

    result = df.group_by("date").agg(
        semantic.reduce("Summarize the main action items from these {notes}").alias(
            "summary"
        ),
        sum("num_attendees").alias("num_attendees"),
    )
    result = result.to_polars()

    assert result.schema == {
        "date": pl.Utf8,
        "summary": pl.Utf8,
        "num_attendees": pl.Int64,
    }
    assert result.filter(pl.col("date") == "2024-01-01")["num_attendees"][0] == 25
    assert result.filter(pl.col("date") == "2024-01-02")["num_attendees"][0] == 20

    result = df.agg(
        semantic.reduce("Summarize the main action items from these {notes}").alias(
            "summary"
        ),
        sum("num_attendees").alias("num_attendees"),
    )
    result = result.to_polars()

    assert result.schema == {
        "summary": pl.Utf8,
        "num_attendees": pl.Int64,
    }


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
        ValueError, match="Nested aggregation functions are not allowed"
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
