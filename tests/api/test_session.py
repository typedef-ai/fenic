from pathlib import Path, PosixPath
from urllib.parse import urlparse

import pandas as pd
import polars as pl
import pytest
from pydantic import ValidationError as PydanticValidationError

from fenic import (
    AnthropicLanguageModel,
    CohereEmbeddingModel,
    ColumnField,
    GoogleDeveloperEmbeddingModel,
    GoogleDeveloperLanguageModel,
    GoogleVertexEmbeddingModel,
    GoogleVertexLanguageModel,
    IntegerType,
    OpenAIEmbeddingModel,
    OpenAILanguageModel,
    SemanticConfig,
    Session,
    SessionConfig,
    StringType,
    col,
)
from fenic.core._inference.model_catalog import ModelProvider
from fenic.core._logical_plan.plans import InMemorySource
from fenic.core.error import ConfigurationError
from fenic.core.error import ValidationError as FenicValidationError
from tests.conftest import EMBEDDING_MODEL_PROVIDER_ARG


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
    session.stop(skip_usage_summary=True)


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


def test_local_session_with_language_models_only(tmp_path):
    """Verify that a local_session is created successfully when we only supply 'language_models' in semantic_config."""
    session_config = SessionConfig(
        app_name="test_local_session_with_language_models_only",
        semantic=SemanticConfig(
            language_models={"mini" :OpenAILanguageModel(model_name="gpt-4o-mini", rpm=500, tpm=200_000)},
            default_language_model="mini"
        ),
        db_path=tmp_path,
    )
    session = Session.get_or_create(session_config)
    session.stop(skip_usage_summary=True)

def test_local_session_with_no_semantic_config(tmp_path):
    """Verify that a local_session is created successfully if we supply no semantic config."""
    session_config = SessionConfig(
        app_name="test_local_session_with_no_semantic_config",
        db_path=tmp_path,
    )
    session = Session.get_or_create(session_config)
    session.create_dataframe({"text": ["hello"]}).select((col("text")).alias("text"))
    session.stop(skip_usage_summary=True)

def test_local_session_with_embedding_models_only(tmp_path):
    """Verify that a local_session is created successfully if we supply only embedding models."""
    session_config = SessionConfig(
        app_name="test_local_session_with_embedding_models_only",
        db_path=tmp_path,
        semantic=SemanticConfig(embedding_models={"oai-small": OpenAIEmbeddingModel(model_name="text-embedding-3-small", rpm=3000, tpm=1_000_000)}),
    )
    session = Session.get_or_create(session_config)
    session.stop(skip_usage_summary=True)

def test_local_session_with_single_lm_no_explicit_default(tmp_path):
    """Verify that a local_session is created successfully if we supply one language model and no default."""
    session_config = SessionConfig(
        app_name="test_local_session_with_single_lm_no_explicit_default",
        db_path=tmp_path,
        semantic=SemanticConfig(
            language_models={"mini" : OpenAILanguageModel(model_name="gpt-4o-mini", rpm=500, tpm=200_000)},
        ),
    )
    assert session_config.semantic.default_language_model == "mini"
    assert session_config.semantic.language_models["mini"].model_name == "gpt-4o-mini"
    session = Session.get_or_create(session_config)
    session.stop(skip_usage_summary=True)

def test_local_session_with_ambiguous_default_lm(tmp_path):
    """Verify that a local session creation error is raised if we supply two language models with no default."""
    with pytest.raises(ConfigurationError):
        SessionConfig(
            app_name="test_local_session_with_ambiguous_default_lm",
            db_path=tmp_path,
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
            app_name="test_session_config_with_unsupported_embedding_profile_dimensionality",
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
            app_name="test_session_config_with_multiple_invalid_embedding_profiles",
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
            app_name="test_google_developer_embedding_unsupported_dimensionality",
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

def test_cohere_embedding_unsupported_dimensionality():
    """Test Cohere embedding model with unsupported dimensionality."""
    with pytest.raises(PydanticValidationError, match="Input should be less than or equal to 1536"):
        SessionConfig(
            app_name="test_cohere_embedding_unsupported_dimensionality",
            semantic=SemanticConfig(
                embedding_models={
                    "cohere_embed": CohereEmbeddingModel(
                        model_name="embed-v4.0",
                        rpm=100,
                        tpm=1000,
                        profiles={
                            "invalid": CohereEmbeddingModel.Profile(output_dimensionality=2048)  # higher than 1536
                        }
                    )
                }
            )
        )
    with pytest.raises(ConfigurationError, match="The dimensionality of the Embeddings model profile invalid is invalid."):
        SessionConfig(
            app_name="test_cohere_embedding_unsupported_dimensionality2",
            semantic=SemanticConfig(
                embedding_models={
                    "cohere_embed": CohereEmbeddingModel(
                        model_name="embed-v4.0",
                        rpm=100,
                        tpm=1000,
                        profiles={
                            "invalid": CohereEmbeddingModel.Profile(output_dimensionality=768)  # not in [256, 512, 1024, 1536]
                        }
                    )
                }
            )
        )

def test_cohere_embedding_unsupported_input_type():
    """Test Cohere embedding model with unsupported input type."""
    with pytest.raises(PydanticValidationError, match="Input should be 'search_document', 'search_query', 'classification' or 'clustering'"):
        SessionConfig(
            app_name="test_cohere_embedding_unsupported_input_type",
            semantic=SemanticConfig(
                embedding_models={
                    "cohere_embed": CohereEmbeddingModel(
                        model_name="embed-v4.0",
                        rpm=100,
                        tpm=1000,
                        profiles={
                            "invalid": CohereEmbeddingModel.Profile(input_type="hallucinate")  # Not in [search_query, search_document, classification, clustering]
                        }
                    )
                }
            )
        )


def test_cohere_embedding_profile_rejects_arbitrary_args():
    """Test that CohereEmbeddingModel.Profile rejects arbitrary arguments."""
    # This should raise an error now that we've added extra='forbid'
    with pytest.raises(PydanticValidationError, match="Extra inputs are not permitted"):
        CohereEmbeddingModel.Profile(
            output_dimensionality=1024,
            input_type="classification",
            arbitrary_field="should_not_be_accepted",  # This should cause an error
            another_field=123,  # This should also cause an error
            yet_another_field={"nested": "data"}  # This should also cause an error
        )

    # Valid profile should still work
    profile = CohereEmbeddingModel.Profile(
        output_dimensionality=1024,
        input_type="classification"
    )
    assert profile.output_dimensionality == 1024
    assert profile.input_type == "classification"


def test_google_embeddings_profile_rejects_arbitrary_args(tmp_path):
    """Test that GoogleVertexEmbeddingModel.Profile rejects arbitrary arguments."""
    # This should raise an error now that we've added extra='forbid'
    with pytest.raises(PydanticValidationError, match="Extra inputs are not permitted"):
        GoogleVertexEmbeddingModel.Profile(
            output_dimensionality=1536,
            task_type="SEMANTIC_SIMILARITY",
            arbitrary_field="should_not_be_accepted",  # This should cause an error
            another_field=123,  # This should also cause an error
            yet_another_field={"nested": "data"}  # This should also cause an error
        )

    # Valid profile should still work
    profile = GoogleVertexEmbeddingModel.Profile(
        output_dimensionality=1536,
        task_type="SEMANTIC_SIMILARITY"
    )
    assert profile.output_dimensionality == 1536
    assert profile.task_type == "SEMANTIC_SIMILARITY"


def test_session_config_with_valid_embedding_profile_dimensions():
    """Test that session configuration accepts all valid embedding profile dimensions."""
    # This should succeed as all dimensions are valid for gemini-embedding-001
    config = SessionConfig(
        app_name="test_session_config_with_valid_embedding_profile_dimensions",
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
        app_name="test_embedding_profile_with_none_dimensionality",
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


def test_embedding_with_no_profile(request):
    """Test that embedding profiles with no profile (when one is possible) are accepted."""
    # This should succeed as None means use the model's default dimensionality
    embedding_model_provider = ModelProvider(request.config.getoption(EMBEDDING_MODEL_PROVIDER_ARG))
    if embedding_model_provider != ModelProvider.GOOGLE_VERTEX:
        pytest.skip("This test only runs for Google Vertex embedding models")

    config = SessionConfig(
        app_name="test_embedding_with_no_profile",
        semantic=SemanticConfig(
            embedding_models={
                "google_embed": GoogleVertexEmbeddingModel(
                    model_name="gemini-embedding-001",
                    rpm=100,
                    tpm=1000,
                )
            }
        )
    )

    # Verify None is preserved (will use model's default)
    assert config.semantic.embedding_models["google_embed"].profiles is None

def test_model_profile_validation():
    """Test that model profile validation works for providers with multiple profiles and for models that do not use profiles."""
    # Test profile on model that doesn't support profiles
    with pytest.raises(ConfigurationError, match="Model 'gpt-4o-mini' does not support parameter profiles. Please remove the Profile configuration."):
        SessionConfig(
            app_name="test_model_profile_validation",
            semantic=SemanticConfig(
                language_models={"gpt-4o-mini": OpenAILanguageModel(model_name="gpt-4o-mini", rpm=100, tpm=1000, profiles={"fast": OpenAILanguageModel.Profile(reasoning_effort="low")})}
            )
        )

    # Test setting verbosity on model that doesn't support verbosity
    with pytest.raises(ConfigurationError, match="Model 'o3' does not support verbosity. Please remove verbosity from 'fast'."):
        SessionConfig(
            app_name="test_model_profile_validation",
            semantic=SemanticConfig(
                language_models={"o3": OpenAILanguageModel(model_name="o3", rpm=100, tpm=1000, profiles={"fast": OpenAILanguageModel.Profile(reasoning_effort="low", verbosity="low")})}
            )
        )

    # Test setting minimal reasoning on model that doesn't support minimal reasoning
    with pytest.raises(ConfigurationError, match="Model 'o3' does not support 'minimal' reasoning. Please set reasoning_effort on 'fast' to 'low', 'medium', or 'high' instead."):
        SessionConfig(
            app_name="test_model_profile_validation",
            semantic=SemanticConfig(
                language_models={"o3": OpenAILanguageModel(model_name="o3", rpm=100, tpm=1000, profiles={"fast": OpenAILanguageModel.Profile(reasoning_effort="minimal")})}
            )
        )

    # Test unsetting reasoning on model that doesn't support unsetting reasoning
    with pytest.raises(ConfigurationError, match="Model '2.5-pro' does not support disabling reasoning. Please set thinking_token_budget on 'fast' to a non-zero value."):
        SessionConfig(
            app_name="test_model_profile_validation",
            semantic=SemanticConfig(
                language_models={"2.5-pro": GoogleDeveloperLanguageModel(model_name="gemini-2.5-pro", rpm=100, tpm=1000, profiles={"fast": GoogleDeveloperLanguageModel.Profile(thinking_token_budget=0)})}
            )
        )

    # Test profile from wrong model class
    with pytest.raises(PydanticValidationError, match="Input should be a valid dictionary or instance of Profile"):
        SessionConfig(
            app_name="test_model_profile_validation",
            semantic=SemanticConfig(
                language_models={"o3": OpenAILanguageModel(model_name="o3", rpm=100, tpm=1000, profiles={"fast": GoogleDeveloperLanguageModel.Profile(thinking_token_budget=0)})}
            )
        )

    # test that you cannot set thinking_level on a model that doesn't support it
    with pytest.raises(ConfigurationError, match="Model 'gemini-2.5-pro' does not support thinking_level. Please use thinking_token_budget on 'high' instead."):
        SessionConfig(
            app_name="test_model_profile_validation",
            semantic=SemanticConfig(
                language_models={"gemini-2.5-pro": GoogleVertexLanguageModel(model_name="gemini-2.5-pro", rpm=100, tpm=1000, profiles={"high": GoogleVertexLanguageModel.Profile(thinking_level="high")})}
            )
        )

    # test that you cannot set both thinking_token_budget and thinking_level
    with pytest.raises(ConfigurationError, match="Model 'gemini-3-pro-preview' uses thinking_level instead of thinking_token_budget. Please set thinking_level on 'high' instead."):
        SessionConfig(
            app_name="test_model_profile_validation",
            semantic=SemanticConfig(
                language_models={"gemini-3-pro-preview": GoogleDeveloperLanguageModel(model_name="gemini-3-pro-preview", rpm=100, tpm=1000, profiles={"high": GoogleDeveloperLanguageModel.Profile(thinking_token_budget=100)})}
            )
        )


    # Test that gpt-5.1 works with 'none' reasoning (default)
    SessionConfig(
        app_name="test_model_profile_validation",
        semantic=SemanticConfig(
            language_models={"gpt-5.1": OpenAILanguageModel(model_name="gpt-5.1", profiles={"disabled_reasoning": OpenAILanguageModel.Profile(reasoning_effort="none")}, rpm=100, tpm=1000)}
        )
    )
    # Test that previous models that support reasoning cannot disable reasoning
    with pytest.raises(ConfigurationError, match="Model 'gpt-5-nano' does not support 'none' \\(disabled\\) reasoning. Please set reasoning_effort on 'disabled_reasoning' to 'minimal', 'low', 'medium', or 'high' instead."):
        SessionConfig(
            app_name="test_model_profile_validation",
            semantic=SemanticConfig(
                language_models={"gpt-5-nano": OpenAILanguageModel(model_name="gpt-5-nano", rpm=100, tpm=1000, profiles={"disabled_reasoning": OpenAILanguageModel.Profile(reasoning_effort="none")})}
            )
        )

def test_session_config_with_invalid_api_keys(tmp_path, monkeypatch):
    """Test that session configuration validation rejects models with invalid API keys."""
    monkeypatch.setenv("OPENAI_API_KEY", "__invalid__")
    # test openai chat completions client
    with pytest.raises(ConfigurationError, match="Incorrect API key provided: __invalid__."):
        config = SessionConfig(
            app_name="test_session_config_with_invalid_api_keys",
            db_path=tmp_path,
            semantic=SemanticConfig(
                language_models={"o3": OpenAILanguageModel(model_name="o3", rpm=100, tpm=1000)}
            )
        )
        _ = Session.get_or_create(config)

    # test openai embedding client
    with pytest.raises(ConfigurationError, match="Incorrect API key provided: __invalid__."):
        config = SessionConfig(
            app_name="test_session_config_with_invalid_api_keys",
            db_path=tmp_path,
            semantic=SemanticConfig(
                embedding_models={"oai-small": OpenAIEmbeddingModel(model_name="text-embedding-3-small", rpm=100, tpm=1000)}
            )
        )
        _ = Session.get_or_create(config)

def test_session_config_with_invalid_gemini_api_key(tmp_path, monkeypatch):
    """Test that session configuration validation rejects models with invalid Gemini API keys."""
    pytest.importorskip("google.genai")

    monkeypatch.setenv("GEMINI_API_KEY", "__invalid__")
    # test google developer chat completions client
    with pytest.raises(ConfigurationError, match="API key not valid. Please pass a valid API key."):
        config = SessionConfig(
            app_name="test_session_config_with_invalid_gemini_api_key_1",
            db_path=tmp_path,
            semantic=SemanticConfig(
                language_models={"gemini_2.5_pro": GoogleDeveloperLanguageModel(model_name="gemini-2.5-pro", rpm=100, tpm=1000)}
            )
        )
        _ = Session.get_or_create(config)

    # test google developer embedding client
    with pytest.raises(ConfigurationError, match="API key not valid. Please pass a valid API key."):
        config = SessionConfig(
            app_name="test_session_config_with_invalid_gemini_api_key_2",
            db_path=tmp_path,
            semantic=SemanticConfig(
                embedding_models={"google_embed": GoogleDeveloperEmbeddingModel(model_name="gemini-embedding-001", rpm=100, tpm=1000)}
            )
        )
        _ = Session.get_or_create(config)

    # test google developer chat completions client
    # mock default credentials error
    import google.auth
    from google.auth.exceptions import DefaultCredentialsError
    monkeypatch.setattr(
        google.auth,
        "default",
        lambda *a, **kw: (_ for _ in ()).throw(DefaultCredentialsError("No ADC"))
    )
    with pytest.raises(ConfigurationError, match="401 UNAUTHENTICATED"):
        config = SessionConfig(
            app_name="test_session_config_with_invalid_cohere_api_key_2",
            db_path=tmp_path,
            semantic=SemanticConfig(
                language_models={"gemini_2.5_pro": GoogleVertexLanguageModel(model_name="gemini-2.5-pro", rpm=100, tpm=1000)}
            )
        )
        _ = Session.get_or_create(config)

    # test google vertex embedding client
    with pytest.raises(ConfigurationError, match="401 UNAUTHENTICATED"):
        config = SessionConfig(
            app_name="test_session_config_with_invalid_gemini_api_key_3",
            db_path=tmp_path,
            semantic=SemanticConfig(
                embedding_models={"google_embed": GoogleVertexEmbeddingModel(model_name="gemini-embedding-001", rpm=100, tpm=1000)}
            )
        )
        _ = Session.get_or_create(config)

def test_session_config_with_invalid_cohere_api_key(tmp_path, monkeypatch):
    pytest.importorskip("cohere")

    monkeypatch.setenv("COHERE_API_KEY", "__invalid__")
    # test cohere embedding client
    with pytest.raises(ConfigurationError, match="invalid api token"):
        config = SessionConfig(
            app_name="test_session_config_with_invalid_cohere_api_key",
            db_path=tmp_path,
            semantic=SemanticConfig(
                embedding_models={"cohere_embed": CohereEmbeddingModel(model_name="embed-v4.0", rpm=100, tpm=1000)}
            )
        )
        _ = Session.get_or_create(config)

def test_session_config_with_invalid_anthropic_api_key(tmp_path, monkeypatch):
    pytest.importorskip("anthropic")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "__invalid__")
    # test anthropic chat completions client
    with pytest.raises(ConfigurationError, match="'type': 'authentication_error'"):
        config = SessionConfig(
            app_name="test_session_config_with_invalid_anthropic_api_key",
            db_path=tmp_path,
            semantic=SemanticConfig(
                language_models={"claude": AnthropicLanguageModel(model_name="claude-opus-4-1", rpm=100, input_tpm=100, output_tpm=1000)}
            )
        )
        _ = Session.get_or_create(config)
