from random import choice
from string import ascii_lowercase

import numpy as np
import polars as pl
import pytest

from fenic import (
    ColumnField,
    EmbeddingType,
    FloatType,
    col,
    embedding,
    lit,
    semantic,
    text,
)
from fenic._backends.local.session_state import LocalSessionState
from fenic.api.session import (
    OpenAILanguageModel,
    SemanticConfig,
    Session,
    SessionConfig,
)
from fenic.core._inference.model_catalog import ModelProvider
from fenic.core._logical_plan.resolved_types import ResolvedModelAlias
from fenic.core.error import TypeMismatchError, ValidationError


def test_embeddings(extract_data_df, embedding_model_name_and_dimensions):
    embedding_model_name, embedding_dimensions = embedding_model_name_and_dimensions
    df = extract_data_df.select(semantic.embed(col("review")).alias("embeddings"))
    assert df.schema.column_fields == [
        ColumnField(name="embeddings", data_type=EmbeddingType(dimensions=embedding_dimensions, embedding_model=embedding_model_name))
    ]

    result = df.to_polars()
    assert result.schema["embeddings"] == pl.Array(pl.Float32, embedding_model_name_and_dimensions[1])
    df = extract_data_df.select(
        semantic.embed(
            text.concat(
                lit("review: {"),
                col("review"),
                lit("}"),
            )
        ).alias("embeddings")
    )
    assert df.schema.column_fields == [
        ColumnField(name="embeddings", data_type=EmbeddingType(dimensions=embedding_dimensions, embedding_model=embedding_model_name))
    ]
    result = df.to_polars()
    assert result.schema["embeddings"] == pl.Array(pl.Float32, embedding_model_name_and_dimensions[1])

def test_embedding_very_long_string(local_session, embedding_model_name_and_dimensions):
    embedding_model_name, _ = embedding_model_name_and_dimensions
    if ModelProvider.OPENAI.value in embedding_model_name:
        string_val = "".join((" " if i%5 == 0 else choice(ascii_lowercase)) for i in range(32768))
        data = {
            "review": [string_val],
        }
        df = local_session.create_dataframe(data)
        df = df.select(semantic.embed(col("review")).alias("embeddings"))
        with pytest.raises(Exception, match="Failed to execute query: Error code: 400"):
            df.to_polars()


def test_embedding_without_models():
    """Test that an error is raised if no embedding models are configured."""
    session_config = SessionConfig(
        app_name="embedding_without_models",
    )
    session = Session.get_or_create(session_config)
    with pytest.raises(ValidationError, match="No embedding models configured."):
        session.create_dataframe({"text": ["hello"]}).select(semantic.embed(col("text")).alias("embeddings"))
    session.stop()

    session_config = SessionConfig(
        app_name="embedding_with_models",
        semantic=SemanticConfig(
            language_models={"mini" :OpenAILanguageModel(model_name="gpt-4o-mini", rpm=500, tpm=200_000)},
        ),
    )
    session = Session.get_or_create(session_config)
    with pytest.raises(ValidationError, match="No embedding models configured."):
        session.create_dataframe({"text": ["hello"]}).select(semantic.embed(col("text")).alias("embeddings"))
    session.stop()


def test_normalization(local_session):
    """Test vector normalization functionality."""
    data = {
        "vectors": [
            [3.0, 4.0],       # magnitude 5 -> [0.6, 0.8]
            [1.0, 0.0],       # magnitude 1 -> [1.0, 0.0]
            [0.0, 0.0],       # zero vector -> [NaN, NaN]
        ]
    }
    df = local_session.create_dataframe(data)
    df = df.select(col("vectors").cast(EmbeddingType(dimensions=2, embedding_model="test")).alias("embeddings"))

    result_df = df.select(embedding.normalize(col("embeddings")).alias("normalized"))
    result = result_df.to_polars()
    normalized = result["normalized"].to_list()

    # Check magnitudes are 1 for non-zero vectors
    assert abs(np.linalg.norm(normalized[0]) - 1.0) < 1e-6
    assert abs(np.linalg.norm(normalized[1]) - 1.0) < 1e-6

    # Check specific values
    assert abs(normalized[0][0] - 0.6) < 1e-6
    assert abs(normalized[0][1] - 0.8) < 1e-6

    # Zero vector should become NaN
    assert np.isnan(normalized[2][0])
    assert np.isnan(normalized[2][1])


def test_semantic_search_workflow(local_session):
    """Test complete semantic search workflow with ordering."""
    data = {
        "document": ["doc1", "doc2", "doc3"],
        "vectors": [
            [1.0, 0.0, 0.0],    # Most similar
            [0.1, 0.9, 0.1],    # Least similar
            [0.9, 0.1, 0.0],    # Moderately similar
        ]
    }
    df = local_session.create_dataframe(data)
    df = df.select(
        "document",
        col("vectors").cast(EmbeddingType(dimensions=3, embedding_model="test")).alias("embeddings")
    )

    query_vector = [1.0, 0.0, 0.0]
    result_df = df.select(
        col("document"),
        embedding.compute_similarity(col("embeddings"), query_vector).alias("similarity")
    ).order_by(col("similarity").desc()).limit(3)

    result = result_df.to_polars()
    docs = result["document"].to_list()
    similarities = result["similarity"].to_list()

    # Should be ordered by descending similarity
    assert docs == ["doc1", "doc3", "doc2"]
    assert similarities[0] > similarities[1] > similarities[2]

@pytest.mark.parametrize("metric,expected_values", [
    ("dot", [1.0, 0.0, 0.0, 0.5]),
    ("cosine", [1.0, 0.0, 0.0, 0.7071067811865475]),  # 0.5/sqrt(0.5) for [0.5,0.5,0]
    ("l2", [0.0, 1.4142135623730951, 1.4142135623730951, 0.7071067811865476])  # sqrt(2), sqrt(2), sqrt(0.5)
])
def test_similarity_metrics_vector_query(local_session, metric, expected_values):
    """Test all similarity metrics with vector queries."""
    data = {
        "vectors": [
            [1.0, 0.0, 0.0],    # Same as query
            [0.0, 1.0, 0.0],    # Orthogonal
            [0.0, 0.0, 1.0],    # Orthogonal
            [0.5, 0.5, 0.0],    # 45 degrees
        ]
    }
    df = local_session.create_dataframe(data)
    df = df.select(col("vectors").cast(EmbeddingType(dimensions=3, embedding_model="test")).alias("embeddings"))

    query_vector = [1.0, 0.0, 0.0]
    result_df = df.select(
        embedding.compute_similarity(col("embeddings"), query_vector, metric=metric).alias("similarity")
    )
    result = result_df.to_polars()
    similarities = result["similarity"].to_list()

    for i, expected in enumerate(expected_values):
        assert abs(similarities[i] - expected) < 1e-6


@pytest.mark.parametrize("metric", ["dot", "cosine", "l2"])
def test_similarity_column_to_column(local_session, metric):
    """Test similarity computation between two columns."""
    data = {
        "vectors1": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        "vectors2": [[1.0, 0.0], [1.0, 0.0], [0.5, 0.5]]
    }
    df = local_session.create_dataframe(data)
    df = df.select(
        col("vectors1").cast(EmbeddingType(dimensions=2, embedding_model="test")).alias("embeddings1"),
        col("vectors2").cast(EmbeddingType(dimensions=2, embedding_model="test")).alias("embeddings2")
    )

    result_df = df.select(
        embedding.compute_similarity(col("embeddings1"), col("embeddings2"), metric=metric).alias("similarity")
    )
    result = result_df.to_polars()
    similarities = result["similarity"].to_list()

    # All metrics should compute without error
    assert len(similarities) == 3
    assert all(sim is not None for sim in similarities)

    # Check specific expected values for dot product
    if metric == "dot":
        assert abs(similarities[0] - 1.0) < 1e-6  # [1,0] · [1,0] = 1
        assert abs(similarities[1] - 0.0) < 1e-6  # [0,1] · [1,0] = 0
        assert abs(similarities[2] - 1.0) < 1e-6  # [1,1] · [0.5,0.5] = 1

@pytest.mark.parametrize("metric", ["dot", "cosine", "l2"])
def test_similarity_null_handling(local_session, metric):
    """Test that null values are preserved across all operations."""
    data = {
        "vectors1": [[1.0, 0.0], None, [0.0, 1.0]],
        "vectors2": [[1.0, 0.0], [0.0, 1.0], None]
    }
    df = local_session.create_dataframe(data)
    df = df.select(
        col("vectors1").cast(EmbeddingType(dimensions=2, embedding_model="test")).alias("embeddings1"),
        col("vectors2").cast(EmbeddingType(dimensions=2, embedding_model="test")).alias("embeddings2")
    )

    # Test column-to-column similarity preserves nulls
    result_df = df.select(
        embedding.compute_similarity(col("embeddings1"), col("embeddings2"), metric=metric).alias("similarity")
    )
    result = result_df.to_polars()
    similarities = result["similarity"].to_list()

    assert similarities[0] is not None  # Both non-null
    assert similarities[1] is None      # embeddings1 is null
    assert similarities[2] is None      # embeddings2 is null

    # Test normalization preserves nulls
    norm_df = df.select(embedding.normalize(col("embeddings1")).alias("normalized"))
    result = norm_df.to_polars()
    normalized = result["normalized"].to_list()

    assert normalized[0] is not None
    assert normalized[1] is None
    assert normalized[2] is not None


def test_similarity_validation_errors(local_session):
    """Test validation and type checking errors."""
    data = {"vectors": [[1.0, 0.0]]}
    df = local_session.create_dataframe(data)
    df = df.select(col("vectors").cast(EmbeddingType(dimensions=2, embedding_model="test")).alias("embeddings"))

    # Test NaN in query vector
    with pytest.raises(ValidationError, match="Query vector cannot contain NaN values"):
        embedding.compute_similarity(col("embeddings"), [1.0, float('nan')])

    # Test wrong dimensions
    with pytest.raises(ValidationError, match="Query vector dimensions.*must match embedding dimensions"):
        df.select(embedding.compute_similarity(col("embeddings"), [1.0, 0.0, 0.0]).alias("similarity"))

    # Test mismatched embedding types for column-to-column
    data2 = {"vectors2": [[1.0, 0.0]]}
    df2 = local_session.create_dataframe(data2)
    df2 = df2.select(col("vectors2").cast(EmbeddingType(dimensions=2, embedding_model="different")).alias("embeddings2"))

    combined = df.join(df2, how="cross")
    with pytest.raises(TypeMismatchError, match="embedding.compute_similarity does not match any valid signature*"):
        combined.select(embedding.compute_similarity(col("embeddings"), col("embeddings2")).alias("similarity"))


def test_type_preservation(local_session):
    """Test that operations preserve correct types."""
    data = {"vectors": [[1.0, 0.0]]}
    df = local_session.create_dataframe(data)
    original_type = EmbeddingType(dimensions=2, embedding_model="test-model")
    df = df.select(col("vectors").cast(original_type).alias("embeddings"))

    # Test normalization preserves embedding type
    norm_df = df.select(embedding.normalize(col("embeddings")).alias("normalized"))
    assert norm_df.schema.column_fields == [
        ColumnField(name="normalized", data_type=original_type)
    ]

    # Test similarity returns correct float type
    sim_df = df.select(
        embedding.compute_similarity(col("embeddings"), [1.0, 0.0]).alias("similarity")
    )
    assert sim_df.schema.column_fields == [
        ColumnField(name="similarity", data_type=FloatType)
    ]

    result = sim_df.to_polars()
    assert result.schema["similarity"] == pl.Float32


def test_similarity_numpy_array_input(local_session):
    """Test that numpy arrays work as query vectors."""
    data = {"vectors": [[1.0, 0.0], [0.0, 1.0]]}
    df = local_session.create_dataframe(data)
    df = df.select(col("vectors").cast(EmbeddingType(dimensions=2, embedding_model="test")).alias("embeddings"))

    # Use numpy array as query vector
    query_vector = np.array([1.0, 0.0])
    result_df = df.select(
        embedding.compute_similarity(col("embeddings"), query_vector).alias("similarity")
    )
    result = result_df.to_polars()
    similarities = result["similarity"].to_list()

    assert abs(similarities[0] - 1.0) < 1e-6
    assert abs(similarities[1] - 0.0) < 1e-6


def test_cosine_similarity_special_cases(local_session):
    """Test cosine similarity edge cases."""
    data = {
        "vectors": [
            [0.0, 0.0],      # Zero vector
            [1.0, 0.0],      # Normal vector
            [-1.0, 0.0],     # Opposite direction
        ]
    }
    df = local_session.create_dataframe(data)
    df = df.select(col("vectors").cast(EmbeddingType(dimensions=2, embedding_model="test")).alias("embeddings"))

    # Test with normal query
    result_df = df.select(
        embedding.compute_similarity(col("embeddings"), [1.0, 0.0], metric="cosine").alias("cosine_sim")
    )
    result = result_df.to_polars()
    similarities = result["cosine_sim"].to_list()
    assert np.isnan(similarities[0])              # Zero vector -> NaN
    assert abs(similarities[1] - 1.0) < 1e-6     # Same direction -> 1.0
    assert abs(similarities[2] - (-1.0)) < 1e-6  # Opposite direction -> -1.0

    # Test with zero query vector - all should be NaN
    zero_result_df = df.select(
        embedding.compute_similarity(col("embeddings"), [0.0, 0.0], metric="cosine").alias("cosine_sim")
    )
    result = zero_result_df.to_polars()
    similarities = result["cosine_sim"].to_list()

    assert all(np.isnan(sim) for sim in similarities)

def test_fetch_embedding_model_by_alias(local_session):
    session_state: LocalSessionState = local_session._session_state
    embedding_model = session_state.get_embedding_model(alias=ResolvedModelAlias(name="embedding"))
    assert embedding_model
