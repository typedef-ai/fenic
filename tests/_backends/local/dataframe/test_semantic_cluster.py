
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
from fenic.core.error import TypeMismatchError, ValidationError


def test_semantic_cluster_with_centroids(local_session, embedding_model_name_and_dimensions):
    embedding_model_name, embedding_dimensions = embedding_model_name_and_dimensions
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
        .semantic.with_cluster_labels(col("embeddings"), num_clusters=2, num_init=5, max_iter=100, centroid_column="cluster_centroid")
    )

    assert df.schema.column_fields == [
        ColumnField("blurb", StringType),
        ColumnField("embeddings", EmbeddingType(embedding_model=embedding_model_name, dimensions=embedding_dimensions)),
        ColumnField("cluster_label", IntegerType),
        ColumnField("cluster_centroid", EmbeddingType(embedding_model=embedding_model_name, dimensions=embedding_dimensions)),
    ]
    polars_df = df.to_polars()
    assert polars_df.schema == {
        "blurb": pl.Utf8,
        "embeddings": pl.Array(pl.Float32, 1536),
        "cluster_label": pl.Int32,
        "cluster_centroid": pl.Array(pl.Float32, 1536),
    }

def test_semantic_cluster_derived_column(local_session):
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
    df = source.semantic.with_cluster_labels(semantic.embed(col("blurb")), num_clusters=2)

    assert df.schema.column_fields == [
        ColumnField("blurb", StringType),
        ColumnField("cluster_label", IntegerType),
    ]
    polars_df = df.to_polars()
    assert polars_df.schema == {
        "blurb": pl.Utf8,
        "cluster_label": pl.Int32,
    }

def test_semantic_clustering_groups_by_cluster_label_with_aggregation(local_session):
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
        .semantic.with_cluster_labels(col("embeddings"), num_clusters=2)
        .group_by(col("cluster_label"))
        .agg(collect_list(col("blurb")).alias("blurbs"))
    )
    result = df.to_polars()
    assert result.schema == {
        "cluster_label": pl.Int32,
        "blurbs": pl.List(pl.String),
    }
    assert set(result["cluster_label"].to_list()) == {0, 1, None}

    with pytest.raises(
        TypeMismatchError,
        match="semantic.with_cluster_labels by expression must be an embedding column type",
    ):
        df = source.semantic.with_cluster_labels(col("blurb"), 2).group_by(col("cluster_label")).agg(
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
        .semantic.with_cluster_labels(col("embeddings"), 2)
        .group_by(col("cluster_label"))
        .agg(
            count(col("user_id")).alias("feedback_count"),
            semantic.reduce("Summarize my app's product feedback", col("feedback")).alias(
                "theme_summary"
            ),
        )
        .to_polars()
    )

    assert result.schema == {
        "cluster_label": pl.Int32,
        "feedback_count": pl.UInt32,
        "theme_summary": pl.Utf8,
    }


def test_semantic_clustering_on_persisted_embeddings_table(local_session, embedding_model_name_and_dimensions):
    """Test group_by() on a semantic cluster id with a saved embeddings table."""
    embedding_model_name, embedding_dimensions = embedding_model_name_and_dimensions
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
        ColumnField("embeddings", EmbeddingType(embedding_model=embedding_model_name, dimensions=embedding_dimensions)),
    ]
    result = (
        df_embeddings.semantic.with_cluster_labels(col("embeddings"), 2)
        .group_by(col("cluster_label"))
        .agg(
            count(col("user_id")).alias("feedback_count"),
            semantic.reduce("Summarize my app's product feedback", col("feedback")).alias(
                "grouped_feedback"
            ),
        )
        .to_polars()
    )
    assert len(result) == 2

def test_semantic_cluster_with_invalid_parameters(local_session):
    source = local_session.create_dataframe(
        {
            "embeddings": [
                [0.1, 0.5],
                [0.2, 0.8],
            ],
        }
    ).with_column("embeddings", col("embeddings").cast(EmbeddingType(embedding_model="dummy_model", dimensions=2)))
    with pytest.raises(ValidationError, match="`num_clusters` must be a positive integer."):
        source.semantic.with_cluster_labels(col("blurb"), num_clusters=0)
    with pytest.raises(ValidationError, match="`max_iter` must be a positive integer."):
        source.semantic.with_cluster_labels(col("blurb"), num_clusters=2, max_iter=0)
    with pytest.raises(ValidationError, match="`num_init` must be a positive integer."):
        source.semantic.with_cluster_labels(col("blurb"), num_clusters=2, num_init=0)
