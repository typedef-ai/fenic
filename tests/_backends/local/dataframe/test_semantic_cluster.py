
import polars as pl
import pytest

from fenic import (
    ColumnField,
    EmbeddingType,
    IntegerType,
    StringType,
    col,
    collect_list,
    count,
    semantic,
)
from fenic.core.error import TypeMismatchError


def test_semantic_cluster_with_centroids(local_session):
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
        .semantic.cluster(col("embeddings"), 2, return_centroids=True)
    )

    assert df.schema.column_fields == [
        ColumnField("blurb", StringType),
        ColumnField("embeddings", EmbeddingType(embedding_model="openai/text-embedding-3-small", dimensions=1536)),
        ColumnField("_cluster_id", IntegerType),
        ColumnField("_cluster_centroid", EmbeddingType(embedding_model="openai/text-embedding-3-small", dimensions=1536)),
    ]
    polars_df = df.to_polars()
    assert polars_df.schema == {
        "blurb": pl.Utf8,
        "embeddings": pl.Array(pl.Float32, 1536),
        "_cluster_id": pl.Int64,
        "_cluster_centroid": pl.Array(pl.Float32, 1536),
    }

def test_semantic_clustering_groups_by_cluster_id_with_aggregation(local_session):
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
        .semantic.cluster(col("embeddings"), 2)
        .group_by(col("_cluster_id"))
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
        match="semantic.cluster by expression must be an embedding column type",
    ):
        df = source.semantic.cluster(col("blurb"), 2).group_by(col("_cluster_id")).agg(
            collect_list(col("blurb")).alias("blurbs")
        ).to_polars()

def test_semantic_clustering_with_semantic_reduction_aggregation(local_session):
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
        .semantic.cluster(col("embeddings"), 2)
        .group_by(col("_cluster_id"))
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


def test_semantic_clustering_on_persisted_embeddings_table(local_session):
    """Test group_by() on a semantic cluster id with a saved embeddings table."""
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
        df_embeddings.semantic.cluster(col("embeddings"), 2)
        .group_by(col("_cluster_id"))
        .agg(
            count(col("user_id")).alias("feedback_count"),
            semantic.reduce("Summarize my app's product feedback: {feedback}?").alias(
                "grouped_feedback"
            ),
        )
        .to_polars()
    )
    assert len(result) == 2
