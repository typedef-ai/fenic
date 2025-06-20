import re

import polars as pl
import pytest

from fenic import (
    ColumnField,
    EmbeddingType,
    IntegerType,
    StringType,
    avg,
    col,
    collect_list,
    count,
    lit,
    max,
    mean,
    min,
    semantic,
    sum,
)
from fenic.core.error import TypeMismatchError


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


def test_semantic_grouping(local_session):
    source = local_session.create_dataframe(
        {
            "blurb": [
                "Rust is a memory-safe systems programming language with zero-cost abstractions.",
                "Tokio is an asynchronous runtime for Rust that powers many high-performance applications.",
                "Hiking in the Alps offers breathtaking views and serene landscapes",
                None,
            ],
        }
    )
    df = (
        source.with_column("embeddings", semantic.embed(col("blurb")))
        .semantic.group_by(col("embeddings"), 2)
        .agg(collect_list(col("blurb")).alias("blurbs"))
    )
    result = df.to_polars()
    assert result.schema == {
        "_cluster_id": pl.Int64,
        "blurbs": pl.List(pl.String),
    }
    assert set(result["_cluster_id"].to_list()) == {0, 1, None}

    with pytest.raises(
        TypeMismatchError,
        match="semantic.group_by grouping expression must be an embedding column type",
    ):
        df = source.semantic.group_by(col("blurb"), 2).agg(
            collect_list(col("blurb")).alias("blurbs")
        )
        result = df.to_polars()


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


def test_semantic_cluster_and_reduce(local_session):
    """Test combining semantic clustering with semantic reduction."""
    data = {
        "feedback": [
            "The mobile app crashes frequently when uploading photos. Very frustrating experience.",
            "App keeps freezing during image uploads. Need urgent fix for the crash issues.",
            "Love the new dark mode theme! The UI is much easier on the eyes now.",
            "Great update with the dark mode. The contrast is perfect for night time use.",
            "Customer service was unhelpful and took days to respond to my ticket.",
            "Support team is slow to respond. Had to wait 3 days for a simple question.",
        ],
        "submission_date": ["2024-03-01"] * 6,
        "user_id": [1, 2, 3, 4, 5, 6],
    }
    df = local_session.create_dataframe(data)

    # First cluster the feedback, then summarize each cluster
    result = (
        df.with_column("embeddings", semantic.embed(col("feedback")))
        .semantic.group_by(col("embeddings"), 2)
        .agg(
            count(col("user_id")).alias("feedback_count"),
            semantic.reduce("Summarize my app's product feedback: {feedback}?").alias(
                "theme_summary"
            ),
        )
        .to_polars()
    )

    assert result.schema == {
        "_cluster_id": pl.Int64,
        "feedback_count": pl.UInt32,
        "theme_summary": pl.Utf8,
    }


def test_groupby_saved_embeddings_table(local_session):
    """Test groupBy() with a saved embeddings table."""
    data = {
        "feedback": [
            "The mobile app crashes frequently when uploading photos. Very frustrating experience.",
            "App keeps freezing during image uploads. Need urgent fix for the crash issues.",
            "Love the new dark mode theme! The UI is much easier on the eyes now.",
            "Great update with the dark mode. The contrast is perfect for night time use.",
            "Customer service was unhelpful and took days to respond to my ticket.",
            "Support team is slow to respond. Had to wait 3 days for a simple question.",
        ],
        "submission_date": ["2024-03-01"] * 6,
        "user_id": [1, 2, 3, 4, 5, 6],
    }
    df = local_session.create_dataframe(data)
    df.with_column("embeddings", semantic.embed(col("feedback"))).write.save_as_table(
        "feedback_embeddings", mode="overwrite"
    )
    df_embeddings = local_session.table("feedback_embeddings")
    assert df_embeddings.schema.column_fields == [
        ColumnField("feedback", StringType),
        ColumnField("submission_date", StringType),
        ColumnField("user_id", IntegerType),
        ColumnField("embeddings", EmbeddingType(embedding_model="openai/text-embedding-3-small", dimensions=1536)),
    ]
    result = (
        df_embeddings.semantic.group_by(col("embeddings"), 2)
        .agg(
            count(col("user_id")).alias("feedback_count"),
            semantic.reduce("Summarize my app's product feedback: {feedback}?").alias(
                "grouped_feedback"
            ),
        )
        .to_polars()
    )
    assert len(result) == 2


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
