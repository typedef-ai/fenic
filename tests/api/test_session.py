from pathlib import Path, PosixPath
from urllib.parse import urlparse

import pandas as pd
import polars as pl
import pytest

from fenic import (
    ColumnField,
    IntegerType,
    OpenAIEmbeddingModel,
    SemanticConfig,
    Session,
    SessionConfig,
    StringType,
    col,
)
from fenic.api.session.config import (
    GoogleDeveloperEmbeddingModel,
    GoogleVertexEmbeddingModel,
    OpenAILanguageModel,
)
from fenic.core._logical_plan.plans import InMemorySource
from fenic.core.error import ConfigurationError
from fenic.core.error import ValidationError as FenicValidationError


def test_session_with_db_path(temp_dir, local_session_config):
    """Test session creation with custom database path."""
    db_path = temp_dir.path
    session = Session.get_or_create(local_session_config)
    if (
        type(db_path) is PosixPath
        or urlparse(db_path).scheme == "file"
        or urlparse(db_path).scheme == ""
    ):
        # s3 path may not exist until duckdb writes to it
        assert Path(db_path).exists()
    session.stop()


def test_create_dataframe_from_polars(local_session):
    """Test creating DataFrame from a Polars DataFrame."""
    pl_df = pl.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
    df = local_session.create_dataframe(pl_df)
    assert df.schema.column_fields == [
        ColumnField("name", StringType),
        ColumnField("age", IntegerType),
    ]
    df = df.to_polars()

    assert df.shape == (2, 2)
    assert df.columns == ["name", "age"]
    assert df["name"].to_list() == ["Alice", "Bob"]
    assert df["age"].to_list() == [25, 30]

def test_create_dataframe_from_pandas(local_session):
    """Test creating DataFrame from a Pandas DataFrame."""
    df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
    df = local_session.create_dataframe(df)
    assert df.schema.column_fields == [
        ColumnField("name", StringType),
        ColumnField("age", IntegerType),
    ]
    df = df.to_polars()

    assert df.shape == (2, 2)
    assert df.columns == ["name", "age"]
    assert df["name"].to_list() == ["Alice", "Bob"]
    assert df["age"].to_list() == [25, 30]


def test_create_dataframe_from_dict(local_session):
    """Test creating DataFrame from a dictionary."""
    data = {"name": ["Alice", "Bob"], "age": [25, 30]}
    df = local_session.create_dataframe(data)
    assert df.schema.column_fields == [
        ColumnField("name", StringType),
        ColumnField("age", IntegerType),
    ]
    df = df.to_polars()

    assert df.shape == (2, 2)
    assert df.columns == ["name", "age"]
    assert df["name"].to_list() == ["Alice", "Bob"]
    assert df["age"].to_list() == [25, 30]


def test_create_dataframe_from_list_of_dicts(local_session):
    """Test creating DataFrame from a list of dictionaries."""
    data = [{"name": "Alice", "age": 25}, {"name": "Bob", "age": 30}]
    df = local_session.create_dataframe(data)
    assert df.schema.column_fields == [
        ColumnField("name", StringType),
        ColumnField("age", IntegerType),
    ]
    df = df.to_polars()

    assert df.shape == (2, 2)
    assert sorted(df.columns) == ["age", "name"]
    assert df["name"].to_list() == ["Alice", "Bob"]
    assert df["age"].to_list() == [25, 30]


def test_create_dataframe_from_arrow(local_session):
    """Test creating DataFrame from a PyArrow Table."""
    import pyarrow as pa

    # Create a PyArrow table
    table = pa.table({
        "name": ["Alice", "Bob"],
        "age": [25, 30]
    })

    df = local_session.create_dataframe(table)
    assert df.schema.column_fields == [
        ColumnField("name", StringType),
        ColumnField("age", IntegerType),
    ]
    df = df.to_polars()

    assert df.shape == (2, 2)
    assert df.columns == ["name", "age"]
    assert df["name"].to_list() == ["Alice", "Bob"]
    assert df["age"].to_list() == [25, 30]


def test_create_dataframe_empty_list(local_session):
    """Test that creating DataFrame from empty list fails."""
    with pytest.raises(
        FenicValidationError, match="Cannot create DataFrame from empty list"
    ):
        local_session.create_dataframe([])


def test_create_dataframe_unsupported_type(local_session):
    """Test that creating DataFrame from unsupported type fails."""
    with pytest.raises(FenicValidationError):
        local_session.create_dataframe(42)  # int is not supported


def test_local_session_with_language_models_only():
    """Verify that a local_session is created successfully when we only supply 'language_models' in semantic_config."""
    session_config = SessionConfig(
        app_name="test_app",
        semantic=SemanticConfig(
            language_models={"mini" :OpenAILanguageModel(model_name="gpt-4o-mini", rpm=500, tpm=200_000)},
            default_language_model="mini"
        ),
    )
    session = Session.get_or_create(session_config)
    session.stop()

def test_local_session_with_no_semantic_config():
    """Verify that a local_session is created successfully if we supply no semantic config."""
    session_config = SessionConfig(
        app_name="test_app",
    )
    session = Session.get_or_create(session_config)
    session.create_dataframe({"text": ["hello"]}).select((col("text")).alias("text"))
    session.stop()

def test_local_session_with_embedding_models_only():
    """Verify that a local_session is created successfully if we supply only embedding models."""
    session_config = SessionConfig(
        app_name="test_app",
        semantic=SemanticConfig(embedding_models={"oai-small": OpenAIEmbeddingModel(model_name="text-embedding-3-small", rpm=3000, tpm=1_000_000)}),
    )
    session = Session.get_or_create(session_config)
    session.stop()

def test_local_session_with_single_lm_no_explicit_default():
    """Verify that a local_session is created successfully if we supply one language model and no default."""
    session_config = SessionConfig(
        app_name="test_app",
        semantic=SemanticConfig(
            language_models={"mini" : OpenAILanguageModel(model_name="gpt-4o-mini", rpm=500, tpm=200_000)},
        ),
    )
    assert session_config.semantic.default_language_model == "mini"
    assert session_config.semantic.language_models["mini"].model_name == "gpt-4o-mini"

def test_local_session_with_ambiguous_default_lm():
    """Verify that a local session creation error is raised if we supply two language models with no default."""
    with pytest.raises(ConfigurationError):
        SessionConfig(
            app_name="test_app",
            semantic=SemanticConfig(
                language_models={"mini" :OpenAILanguageModel(model_name="gpt-4o-mini", rpm=500, tpm=200_000),
                                 "nano" : OpenAILanguageModel(model_name="gpt-4.1-nano", rpm=500, tpm=200_000)},
            ),
        )

def test_inmemory_source(local_session):
    """Test the in-memory source by creating a DataFrame from a Polars DataFrame.
    This verifies that the InMemorySource logical node returns the correct schema
    and sample rows without any file I/O.
    """
    data = {"col1": [1, 2, 3], "col2": ["a", "b", "c"]}
    # Create a Polars DataFrame directly.
    pl_df = pl.DataFrame(data)
    df = local_session.create_dataframe(pl_df)

    # Check that the logical plan is an InMemorySource.
    assert isinstance(
        df._logical_plan, InMemorySource
    ), "Expected an InMemorySource logical node."

    # Verify the schema.
    schema = df.schema
    expected_columns = {"col1", "col2"}
    actual_columns = {field.name for field in schema.column_fields}
    assert (
        actual_columns == expected_columns
    ), f"Expected columns {expected_columns}, got {actual_columns}"


def test_session_config_with_unsupported_embedding_profile_dimensionality():
    """Test that session configuration validation rejects embedding profiles with unsupported dimensionality.
    
    Google's gemini-embedding-001 model supports dimensions: [768, 1536, 3072].
    This tests the validate_models logic, not just Pydantic validation.
    """
    # Test unsupported dimension (1024 is not in [768, 1536, 3072])
    with pytest.raises(ConfigurationError, match="The dimensionality of the Embeddings model profile.*is invalid"):
        SessionConfig(
            app_name="test_app",
            semantic=SemanticConfig(
                embedding_models={
                    "google_embed": GoogleVertexEmbeddingModel(
                        model_name="gemini-embedding-001",
                        rpm=100,
                        tpm=1000,
                        profiles={
                            "invalid": GoogleVertexEmbeddingModel.Profile(output_dimensionality=1024)
                        }
                    )
                }
            )
        )


def test_session_config_with_multiple_invalid_embedding_profiles():
    """Test that session configuration validation catches all invalid profile dimensions."""
    # Test with multiple profiles, some valid and some invalid
    with pytest.raises(ConfigurationError, match="The dimensionality of the Embeddings model profile.*is invalid"):
        SessionConfig(
            app_name="test_app",
            semantic=SemanticConfig(
                embedding_models={
                    "google_embed": GoogleVertexEmbeddingModel(
                        model_name="gemini-embedding-001",
                        rpm=100,
                        tpm=1000,
                        profiles={
                            "valid": GoogleVertexEmbeddingModel.Profile(output_dimensionality=768),
                            "invalid": GoogleVertexEmbeddingModel.Profile(output_dimensionality=1000),  # Not in [768, 1536, 3072]
                            "also_valid": GoogleVertexEmbeddingModel.Profile(output_dimensionality=3072)
                        },
                        default_profile="valid"  # Need to specify default when multiple profiles exist
                    )
                }
            )
        )


def test_google_developer_embedding_unsupported_dimensionality():
    """Test Google Developer embedding model with unsupported dimensionality."""
    with pytest.raises(ConfigurationError, match="The dimensionality of the Embeddings model profile.*is invalid"):
        SessionConfig(
            app_name="test_app",
            semantic=SemanticConfig(
                embedding_models={
                    "google_embed": GoogleDeveloperEmbeddingModel(
                        model_name="gemini-embedding-001",
                        rpm=100,
                        tpm=1000,
                        profiles={
                            "invalid": GoogleDeveloperEmbeddingModel.Profile(output_dimensionality=2048)  # Not in [768, 1536, 3072]
                        }
                    )
                }
            )
        )


def test_session_config_with_valid_embedding_profile_dimensions():
    """Test that session configuration accepts all valid embedding profile dimensions."""
    # This should succeed as all dimensions are valid for gemini-embedding-001
    config = SessionConfig(
        app_name="test_app",
        semantic=SemanticConfig(
            embedding_models={
                "google_embed": GoogleVertexEmbeddingModel(
                    model_name="gemini-embedding-001",
                    rpm=100,
                    tpm=1000,
                    profiles={
                        "small": GoogleVertexEmbeddingModel.Profile(output_dimensionality=768),
                        "medium": GoogleVertexEmbeddingModel.Profile(output_dimensionality=1536),
                        "large": GoogleVertexEmbeddingModel.Profile(output_dimensionality=3072)
                    },
                    default_profile="medium"
                )
            }
        )
    )
    
    # Verify the configuration was created successfully
    assert config.semantic.embedding_models["google_embed"].profiles["small"].output_dimensionality == 768
    assert config.semantic.embedding_models["google_embed"].profiles["medium"].output_dimensionality == 1536
    assert config.semantic.embedding_models["google_embed"].profiles["large"].output_dimensionality == 3072
    assert config.semantic.embedding_models["google_embed"].default_profile == "medium"


def test_embedding_profile_with_none_dimensionality():
    """Test that embedding profiles with None dimensionality (default) are accepted."""
    # This should succeed as None means use the model's default dimensionality
    config = SessionConfig(
        app_name="test_app",
        semantic=SemanticConfig(
            embedding_models={
                "google_embed": GoogleVertexEmbeddingModel(
                    model_name="gemini-embedding-001",
                    rpm=100,
                    tpm=1000,
                    profiles={
                        "default": GoogleVertexEmbeddingModel.Profile()  # output_dimensionality=None
                    }
                )
            }
        )
    )
    
    # Verify None is preserved (will use model's default)
    assert config.semantic.embedding_models["google_embed"].profiles["default"].output_dimensionality is None


