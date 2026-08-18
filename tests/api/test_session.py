from pathlib import Path, PosixPath
from urllib.parse import urlparse

import pandas as pd
import polars as pl
import pytest
from pydantic import ValidationError as PydanticValidationError

from fenic import (
    AnthropicLanguageModel,
    ArrayType,
    CohereEmbeddingModel,
    ColumnField,
    EmbeddingType,
    FloatType,
    GoogleDeveloperEmbeddingModel,
    GoogleDeveloperLanguageModel,
    GoogleVertexEmbeddingModel,
    GoogleVertexLanguageModel,
    IntegerType,
    JsonType,
    MarkdownType,
    OpenAIEmbeddingModel,
    OpenAILanguageModel,
    OpenRouterLanguageModel,
    Schema,
    SemanticConfig,
    Session,
    SessionConfig,
    StringType,
    StructField,
    StructType,
    col,
    semantic,
)
from fenic.core._inference.model_catalog import ModelProvider, model_catalog
from fenic.core._logical_plan.plans import InMemorySource
from fenic.core.error import ConfigurationError, PlanError
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


def test_case_create_dataframe_from_list_of_dicts(local_session):
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


def test_create_dataframe_with_schema_from_dict_coerces_and_orders_columns(local_session):
    schema = Schema([
        ColumnField("age", IntegerType),
        ColumnField("name", StringType),
        ColumnField("score", FloatType),
    ])
    df = local_session.create_dataframe(
        {"name": ["Alice"], "score": [1], "age": ["42"]},
        schema=schema,
    )

    assert df.schema == schema
    result = df.to_polars()
    assert result.columns == ["age", "name", "score"]
    assert result.schema["age"] == pl.Int64
    assert result.schema["score"] == pl.Float32
    assert result.to_dict(as_series=False) == {
        "age": [42],
        "name": ["Alice"],
        "score": [1.0],
    }


def test_create_dataframe_with_schema_from_polars_uses_provided_schema(local_session):
    schema = Schema([ColumnField("name", StringType)])
    df = local_session.create_dataframe(pl.DataFrame({"name": ["Alice"]}), schema=schema)
    assert df.schema == schema
    assert df.to_polars().schema["name"] == pl.String


def test_create_dataframe_with_schema_from_list_of_dicts_allows_missing_row_keys(local_session):
    schema = Schema([ColumnField("name", StringType), ColumnField("age", IntegerType)])
    df = local_session.create_dataframe([{"name": "Alice"}, {"age": 30}], schema=schema)
    assert df.schema == schema
    assert df.to_polars().to_dict(as_series=False) == {
        "name": ["Alice", None],
        "age": [None, 30],
    }


def test_create_dataframe_with_schema_backfills_column_absent_from_all_rows(local_session):
    schema = Schema([ColumnField("id", IntegerType), ColumnField("note", StringType)])
    df = local_session.create_dataframe([{"id": 1}, {"id": 2}], schema=schema)
    assert df.schema == schema
    assert df.to_polars().to_dict(as_series=False) == {
        "id": [1, 2],
        "note": [None, None],
    }


def test_create_dataframe_with_schema_from_pandas_data(local_session):
    schema = Schema([ColumnField("name", StringType), ColumnField("age", IntegerType)])
    df = local_session.create_dataframe(
        pd.DataFrame({"age": [42, 43], "name": ["Alice", "Bob"]}),
        schema=schema,
    )
    assert df.schema == schema
    result = df.to_polars()
    assert result.columns == ["name", "age"]
    assert result.to_dict(as_series=False) == {
        "name": ["Alice", "Bob"],
        "age": [42, 43],
    }


def test_create_dataframe_with_schema_from_arrow_data(local_session):
    import pyarrow as pa

    schema = Schema([ColumnField("name", StringType), ColumnField("age", IntegerType)])
    df = local_session.create_dataframe(
        pa.table({"age": [25, 30], "name": ["Alice", "Bob"]}),
        schema=schema,
    )
    assert df.schema == schema
    result = df.to_polars()
    assert result.columns == ["name", "age"]
    assert result.to_dict(as_series=False) == {
        "name": ["Alice", "Bob"],
        "age": [25, 30],
    }


def test_create_dataframe_with_schema_allows_empty_list(local_session):
    schema = Schema([ColumnField("name", StringType), ColumnField("age", IntegerType)])
    df = local_session.create_dataframe([], schema=schema)
    assert df.schema == schema
    assert df.to_polars().schema == {"name": pl.String, "age": pl.Int64}
    assert df.to_polars().height == 0


def test_create_dataframe_with_schema_allows_empty_polars_dataframe(local_session):
    schema = Schema([ColumnField("name", StringType), ColumnField("age", IntegerType)])
    df = local_session.create_dataframe(pl.DataFrame(), schema=schema)
    assert df.schema == schema
    assert df.to_polars().schema == {"name": pl.String, "age": pl.Int64}
    assert df.to_polars().height == 0


def test_create_dataframe_with_schema_allows_empty_dict(local_session):
    schema = Schema([ColumnField("name", StringType), ColumnField("age", IntegerType)])
    df = local_session.create_dataframe({}, schema=schema)
    assert df.schema == schema
    assert df.to_polars().schema == {"name": pl.String, "age": pl.Int64}
    assert df.to_polars().height == 0


def test_create_dataframe_with_schema_allows_empty_pandas_dataframe(local_session):
    schema = Schema([ColumnField("name", StringType), ColumnField("age", IntegerType)])
    df = local_session.create_dataframe(pd.DataFrame(), schema=schema)
    assert df.schema == schema
    assert df.to_polars().schema == {"name": pl.String, "age": pl.Int64}
    assert df.to_polars().height == 0


def test_create_dataframe_with_schema_allows_empty_pyarrow_table(local_session):
    import pyarrow as pa

    schema = Schema([ColumnField("name", StringType), ColumnField("age", IntegerType)])
    df = local_session.create_dataframe(pa.table({}), schema=schema)
    assert df.schema == schema
    assert df.to_polars().schema == {"name": pl.String, "age": pl.Int64}
    assert df.to_polars().height == 0


def test_create_dataframe_with_schema_column_oriented_missing_field_raises(local_session):
    schema = Schema([ColumnField("id", IntegerType), ColumnField("name", StringType)])
    with pytest.raises(FenicValidationError, match="missing columns"):
        local_session.create_dataframe({"id": [1]}, schema=schema)


def test_create_dataframe_with_schema_column_oriented_extra_field_raises(local_session):
    schema = Schema([ColumnField("id", IntegerType), ColumnField("name", StringType)])
    with pytest.raises(FenicValidationError, match="extra columns"):
        local_session.create_dataframe(
            {"id": [1], "name": ["Alice"], "extra": [1]},
            schema=schema,
        )


def test_create_dataframe_with_schema_late_row_extra_key_raises(local_session):
    schema = Schema([ColumnField("id", IntegerType)])
    rows = [{"id": i} for i in range(101)] + [{"id": 101, "extra": 3}]
    with pytest.raises(FenicValidationError, match="extra columns"):
        local_session.create_dataframe(rows, schema=schema)


def test_create_dataframe_with_schema_row_oriented_extra_key_raises(local_session):
    schema = Schema([ColumnField("id", IntegerType)])
    with pytest.raises(FenicValidationError, match="extra columns"):
        local_session.create_dataframe([{"id": 1, "extra": 3}], schema=schema)


def test_create_dataframe_no_schema_later_list_value_keeps_plan_error(local_session):
    with pytest.raises(PlanError, match="Failed to create DataFrame"):
        local_session.create_dataframe([{"id": 1}, "bad"])


def test_create_dataframe_with_schema_later_list_value_raises(local_session):
    schema = Schema([ColumnField("id", IntegerType)])
    with pytest.raises(FenicValidationError, match="list of non-dict values"):
        local_session.create_dataframe([{"id": 1}, "bad"], schema=schema)


def test_create_dataframe_with_schema_wrong_type_raises_fenic_error(local_session):
    with pytest.raises(FenicValidationError, match="schema must be a fenic Schema"):
        local_session.create_dataframe(
            {"id": [1]},
            schema=[ColumnField("id", IntegerType)],
        )


def test_create_dataframe_with_schema_unsupported_type_raises(local_session):
    schema = Schema([ColumnField("id", IntegerType)])
    with pytest.raises(FenicValidationError, match="Unsupported data type"):
        local_session.create_dataframe(42, schema=schema)


def test_create_dataframe_with_schema_uncastable_value_raises_plan_error(local_session):
    schema = Schema([ColumnField("id", IntegerType)])
    with pytest.raises(PlanError, match="provided schema"):
        local_session.create_dataframe({"id": ["not-an-int"]}, schema=schema)


def test_create_dataframe_with_schema_duplicate_names_use_plan_validation(local_session):
    schema = Schema([
        ColumnField("id", IntegerType),
        ColumnField("id", StringType),
    ])
    with pytest.raises(PlanError, match="Duplicate column names"):
        local_session.create_dataframe({"id": [1]}, schema=schema)


def test_create_dataframe_with_json_schema_exposes_logical_type(local_session):
    schema = Schema([ColumnField("json_col", JsonType)])
    df = local_session.create_dataframe(
        {"json_col": ['{"user": "Alice"}']},
        schema=schema,
    )
    assert df.schema == schema
    assert df.to_polars().schema["json_col"] == pl.String


def test_create_dataframe_with_markdown_schema_exposes_logical_type(local_session):
    schema = Schema([ColumnField("md_col", MarkdownType)])
    df = local_session.create_dataframe(
        {"md_col": ["# Title"]},
        schema=schema,
    )
    assert df.schema == schema
    assert df.to_polars().schema["md_col"] == pl.String


def test_create_dataframe_with_struct_schema_round_trips_to_polars(local_session):
    schema = Schema([
        ColumnField(
            "profile",
            StructType([
                StructField("name", StringType),
                StructField("age", IntegerType),
            ]),
        )
    ])

    df = local_session.create_dataframe(
        {"profile": [{"name": "Alice", "age": "42"}]},
        schema=schema,
    )

    result = df.to_polars()
    assert df.schema == schema
    assert result.schema["profile"] == pl.Struct([
        pl.Field("name", pl.String),
        pl.Field("age", pl.Int64),
    ])
    assert result["profile"].to_list() == [{"name": "Alice", "age": 42}]


def test_create_dataframe_with_array_schema_round_trips_to_polars(local_session):
    schema = Schema([ColumnField("scores", ArrayType(IntegerType))])

    df = local_session.create_dataframe(
        {"scores": [["1", "2"], ["3"]]},
        schema=schema,
    )

    result = df.to_polars()
    assert df.schema == schema
    assert result.schema["scores"] == pl.List(pl.Int64)
    assert result["scores"].to_list() == [[1, 2], [3]]


def test_create_dataframe_with_embedding_schema_preserves_polars_array(local_session):
    embedding_type = EmbeddingType(dimensions=3, embedding_model="test")
    schema = Schema([ColumnField("embedding", embedding_type)])

    df = local_session.create_dataframe(
        {"embedding": [[1.0, 2.0, 3.0]]},
        schema=schema,
    )

    assert df.schema == schema
    assert df.to_polars().schema["embedding"] == pl.Array(pl.Float32, 3)


def test_create_dataframe_with_embedding_schema_dimension_mismatch_raises_plan_error(local_session):
    embedding_type = EmbeddingType(dimensions=3, embedding_model="test")
    schema = Schema([ColumnField("embedding", embedding_type)])

    with pytest.raises(PlanError, match="provided schema"):
        local_session.create_dataframe(
            {"embedding": [[1.0, 2.0]]},
            schema=schema,
        )


def test_create_dataframe_alias_accepts_schema(local_session):
    schema = Schema([ColumnField("id", IntegerType)])
    df = local_session.createDataFrame({"id": ["1"]}, schema=schema)
    assert df.schema == schema
    assert df.to_polars().to_dict(as_series=False) == {"id": [1]}


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
    with pytest.raises(ConfigurationError, match="Model 'gemini-3.1-pro-preview' uses thinking_level instead of thinking_token_budget. Please set thinking_level on 'high' instead."):
        SessionConfig(
            app_name="test_model_profile_validation",
            semantic=SemanticConfig(
                language_models={"gemini-3.1-pro-preview": GoogleDeveloperLanguageModel(model_name="gemini-3.1-pro-preview", rpm=100, tpm=1000, profiles={"high": GoogleDeveloperLanguageModel.Profile(thinking_token_budget=100)})}
            )
        )
    with pytest.raises(PydanticValidationError, match="Input should be 'low', 'medium' or 'high'"):
        GoogleDeveloperLanguageModel.Profile(media_resolution="ultra_high")
    with pytest.raises(ConfigurationError, match="Model 'gemini-2.5-flash' does not support media_resolution. Please remove media_resolution from 'high'."):
        SessionConfig(
            app_name="test_model_profile_validation",
            semantic=SemanticConfig(
                language_models={"gemini-2.5-flash": GoogleDeveloperLanguageModel(model_name="gemini-2.5-flash", rpm=100, tpm=1000, profiles={"high": GoogleDeveloperLanguageModel.Profile(media_resolution="high")})}
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
    # Test that latest OpenAI models support xhigh reasoning
    SessionConfig(
        app_name="test_model_profile_validation",
        semantic=SemanticConfig(
            language_models={"gpt-5.5": OpenAILanguageModel(model_name="gpt-5.5", profiles={"deep": OpenAILanguageModel.Profile(reasoning_effort="xhigh")}, rpm=100, tpm=1000)}
        )
    )
    SessionConfig(
        app_name="test_model_profile_validation",
        semantic=SemanticConfig(
            language_models={"gpt-5.6-sol": OpenAILanguageModel(model_name="gpt-5.6-sol", profiles={"deep": OpenAILanguageModel.Profile(reasoning_effort="max")}, rpm=100, tpm=1000)}
        )
    )
    # Test that older OpenAI reasoning models reject xhigh reasoning
    with pytest.raises(ConfigurationError, match="Model 'gpt-5.2' does not support 'xhigh' reasoning. Please set reasoning_effort on 'deep' to 'none', 'low', 'medium', or 'high' instead."):
        SessionConfig(
            app_name="test_model_profile_validation",
            semantic=SemanticConfig(
                language_models={"gpt-5.2": OpenAILanguageModel(model_name="gpt-5.2", rpm=100, tpm=1000, profiles={"deep": OpenAILanguageModel.Profile(reasoning_effort="xhigh")})}
            )
        )
    # Test that adaptive thinking Claude models reject legacy manual thinking budget profiles
    with pytest.raises(ConfigurationError, match="Model 'claude-opus-4-8' uses adaptive thinking and does not support manual thinking_token_budget profiles. Please remove thinking_token_budget from 'deep' and set effort instead."):
        SessionConfig(
            app_name="test_model_profile_validation",
            semantic=SemanticConfig(
                language_models={"claude-opus-4-8": AnthropicLanguageModel(model_name="claude-opus-4-8", rpm=100, input_tpm=1000, output_tpm=1000, profiles={"deep": AnthropicLanguageModel.Profile(thinking_token_budget=1024)})}
            )
        )
    with pytest.raises(ConfigurationError, match="Model 'claude-sonnet-4-6' uses adaptive thinking and does not support manual thinking_token_budget profiles. Please remove thinking_token_budget from 'deep' and set effort instead."):
        SessionConfig(
            app_name="test_model_profile_validation",
            semantic=SemanticConfig(
                language_models={"claude-sonnet-4-6": AnthropicLanguageModel(model_name="claude-sonnet-4-6", rpm=100, input_tpm=1000, output_tpm=1000, profiles={"deep": AnthropicLanguageModel.Profile(thinking_token_budget=1024)})}
            )
        )
    # Test that latest Claude models support effort profiles
    SessionConfig(
        app_name="test_model_profile_validation",
        semantic=SemanticConfig(
            language_models={
                "claude-opus-4-8": AnthropicLanguageModel(
                    model_name="claude-opus-4-8",
                    rpm=100,
                    input_tpm=1000,
                    output_tpm=1000,
                    profiles={"deep": AnthropicLanguageModel.Profile(effort="xhigh")},
                )
            }
        ),
    )
    SessionConfig(
        app_name="test_model_profile_validation",
        semantic=SemanticConfig(
            language_models={
                "claude-sonnet-4-6": AnthropicLanguageModel(
                    model_name="claude-sonnet-4-6",
                    rpm=100,
                    input_tpm=1000,
                    output_tpm=1000,
                    profiles={"deep": AnthropicLanguageModel.Profile(effort="max")},
                )
            }
        ),
    )
    SessionConfig(
        app_name="test_model_profile_validation",
        semantic=SemanticConfig(
            language_models={
                "claude-opus-4-5": AnthropicLanguageModel(
                    model_name="claude-opus-4-5",
                    rpm=100,
                    input_tpm=1000,
                    output_tpm=1000,
                    profiles={
                        "deep": AnthropicLanguageModel.Profile(
                            thinking_token_budget=4096,
                            effort="high",
                        )
                    },
                )
            }
        ),
    )
    with pytest.raises(ConfigurationError, match="Model 'claude-sonnet-4-6' does not support effort='xhigh'."):
        SessionConfig(
            app_name="test_model_profile_validation",
            semantic=SemanticConfig(
                language_models={
                    "claude-sonnet-4-6": AnthropicLanguageModel(
                        model_name="claude-sonnet-4-6",
                        rpm=100,
                        input_tpm=1000,
                        output_tpm=1000,
                        profiles={"deep": AnthropicLanguageModel.Profile(effort="xhigh")},
                    )
                }
            ),
        )
    with pytest.raises(ConfigurationError, match="Model 'claude-haiku-4-5' does not support effort profiles."):
        SessionConfig(
            app_name="test_model_profile_validation",
            semantic=SemanticConfig(
                language_models={
                    "claude-haiku-4-5": AnthropicLanguageModel(
                        model_name="claude-haiku-4-5",
                        rpm=100,
                        input_tpm=1000,
                        output_tpm=1000,
                        profiles={"fast": AnthropicLanguageModel.Profile(effort="low")},
                    )
                }
            ),
        )
    # OpenRouter supports the expanded reasoning.effort enum from its current chat API.
    OpenRouterLanguageModel.Profile(reasoning_effort="none")
    OpenRouterLanguageModel.Profile(reasoning_effort="minimal")
    OpenRouterLanguageModel.Profile(reasoning_effort="xhigh")
    OpenRouterLanguageModel.Profile(reasoning_effort="max")

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
    with pytest.raises(ConfigurationError, match="Incorrect API key provided"):
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
                language_models={"claude": AnthropicLanguageModel(model_name="claude-opus-4-8", rpm=100, input_tpm=100, output_tpm=1000)}
            )
        )
        _ = Session.get_or_create(config)


# --- base_url resolution tests ---


def test_openai_language_model_base_url_resolves():
    """Test that base_url on OpenAILanguageModel is threaded through to the resolved config."""
    config = SessionConfig(
        app_name="test_openai_base_url",
        semantic=SemanticConfig(
            language_models={
                "gpt": OpenAILanguageModel(
                    model_name="gpt-4o-mini",
                    rpm=100,
                    tpm=100,
                    base_url="https://my-proxy.example.com/v1",
                )
            }
        ),
    )
    resolved = config._to_resolved_config()
    model_config = resolved.semantic.language_models.model_configs["gpt"]
    assert model_config.base_url == "https://my-proxy.example.com/v1"


def test_openai_language_model_base_url_defaults_to_none():
    """Test that base_url defaults to None when not provided."""
    config = SessionConfig(
        app_name="test_openai_base_url_default",
        semantic=SemanticConfig(
            language_models={
                "gpt": OpenAILanguageModel(
                    model_name="gpt-4o-mini", rpm=100, tpm=100
                )
            }
        ),
    )
    resolved = config._to_resolved_config()
    model_config = resolved.semantic.language_models.model_configs["gpt"]
    assert model_config.base_url is None


def test_openai_embedding_model_base_url_resolves():
    """Test that base_url on OpenAIEmbeddingModel is threaded through to the resolved config."""
    config = SessionConfig(
        app_name="test_openai_embedding_base_url",
        semantic=SemanticConfig(
            embedding_models={
                "embed": OpenAIEmbeddingModel(
                    model_name="text-embedding-3-small",
                    rpm=100,
                    tpm=100,
                    base_url="https://my-proxy.example.com/v1",
                )
            }
        ),
    )
    resolved = config._to_resolved_config()
    model_config = resolved.semantic.embedding_models.model_configs["embed"]
    assert model_config.base_url == "https://my-proxy.example.com/v1"


def test_anthropic_language_model_base_url_resolves():
    """Test that base_url on AnthropicLanguageModel is threaded through to the resolved config."""
    config = SessionConfig(
        app_name="test_anthropic_base_url",
        semantic=SemanticConfig(
            language_models={
                "claude": AnthropicLanguageModel(
                    model_name="claude-sonnet-4-6",
                    rpm=100,
                    input_tpm=100,
                    output_tpm=100,
                    base_url="https://my-proxy.example.com",
                )
            }
        ),
    )
    resolved = config._to_resolved_config()
    model_config = resolved.semantic.language_models.model_configs["claude"]
    assert model_config.base_url == "https://my-proxy.example.com"


def test_anthropic_language_model_base_url_defaults_to_none():
    """Test that base_url defaults to None on Anthropic when not provided."""
    config = SessionConfig(
        app_name="test_anthropic_base_url_default",
        semantic=SemanticConfig(
            language_models={
                "claude": AnthropicLanguageModel(
                    model_name="claude-sonnet-4-6",
                    rpm=100,
                    input_tpm=100,
                    output_tpm=100,
                )
            }
        ),
    )
    resolved = config._to_resolved_config()
    model_config = resolved.semantic.language_models.model_configs["claude"]
    assert model_config.base_url is None


def test_base_url_preserved_in_multi_model_config():
    """Test that base_url is preserved per-model when mixing models with and without custom URLs."""
    config = SessionConfig(
        app_name="test_multi_model_base_url",
        semantic=SemanticConfig(
            language_models={
                "gpt-proxy": OpenAILanguageModel(
                    model_name="gpt-4o-mini",
                    rpm=100,
                    tpm=100,
                    base_url="https://proxy.example.com/v1",
                ),
                "gpt-direct": OpenAILanguageModel(
                    model_name="gpt-4.1-nano",
                    rpm=100,
                    tpm=100,
                ),
            },
            default_language_model="gpt-proxy",
        ),
    )
    resolved = config._to_resolved_config()
    proxy_config = resolved.semantic.language_models.model_configs["gpt-proxy"]
    direct_config = resolved.semantic.language_models.model_configs["gpt-direct"]
    assert proxy_config.base_url == "https://proxy.example.com/v1"
    assert direct_config.base_url is None


# --- OpenAI-compatible endpoint tests ---
#
# The model catalog is process-wide, so each test below uses a distinct model name to stay
# independent of the others.


def test_openai_language_model_accepts_custom_model_with_base_url():
    """Test that a model outside the OpenAI catalog is accepted with base_url and model_parameters."""
    config = SessionConfig(
        app_name="test_openai_compatible_language_model",
        semantic=SemanticConfig(
            language_models={
                "local": OpenAILanguageModel(
                    model_name="test-compatible-completions",
                    rpm=100,
                    tpm=100,
                    base_url="https://my-endpoint.example.com/v1",
                    model_parameters=OpenAILanguageModel.ModelParameters(
                        context_window_length=32768,
                        max_output_tokens=4096,
                    ),
                )
            }
        ),
    )
    resolved = config._to_resolved_config()
    model_config = resolved.semantic.language_models.model_configs["local"]
    assert model_config.model_name == "test-compatible-completions"
    assert model_config.base_url == "https://my-endpoint.example.com/v1"

    catalog_parameters = model_catalog.get_completion_model_parameters(
        ModelProvider.OPENAI, "test-compatible-completions"
    )
    assert catalog_parameters is not None
    assert catalog_parameters.context_window_length == 32768
    assert catalog_parameters.max_output_tokens == 4096
    assert catalog_parameters.input_token_cost == 0.0
    assert catalog_parameters.output_token_cost == 0.0


def test_openai_language_model_custom_model_costs_are_declared():
    """Test that declared token costs are used for a model outside the OpenAI catalog."""
    OpenAILanguageModel(
        model_name="test-compatible-priced",
        rpm=100,
        tpm=100,
        base_url="https://my-endpoint.example.com/v1",
        model_parameters=OpenAILanguageModel.ModelParameters(
            context_window_length=8192,
            max_output_tokens=1024,
            input_token_cost=2e-7,
            output_token_cost=8e-7,
        ),
    )
    cost = model_catalog.calculate_completion_model_cost(
        model_provider=ModelProvider.OPENAI,
        model_name="test-compatible-priced",
        uncached_input_tokens=1000,
        cached_input_tokens_read=0,
        output_tokens=1000,
    )
    assert cost == pytest.approx(1e-3)


def test_openai_language_model_custom_model_requires_model_parameters():
    """Test that a model outside the OpenAI catalog is rejected without model_parameters."""
    with pytest.raises(ConfigurationError, match="is not supported for openai"):
        OpenAILanguageModel(
            model_name="test-compatible-undeclared",
            rpm=100,
            tpm=100,
            base_url="https://my-endpoint.example.com/v1",
        )


def test_openai_language_model_model_parameters_requires_base_url():
    """Test that model_parameters without base_url is rejected."""
    with pytest.raises(ConfigurationError, match="requires 'base_url'"):
        OpenAILanguageModel(
            model_name="test-compatible-no-base-url",
            rpm=100,
            tpm=100,
            model_parameters=OpenAILanguageModel.ModelParameters(
                context_window_length=32768,
                max_output_tokens=4096,
            ),
        )


def test_openai_language_model_unknown_model_name_still_rejected():
    """Test that an unrecognized model name is still rejected when no endpoint is configured."""
    with pytest.raises(ConfigurationError, match="is not supported for openai"):
        OpenAILanguageModel(model_name="gpt-4.1-nanoo", rpm=100, tpm=100)


def test_openai_embedding_model_accepts_custom_model_with_base_url():
    """Test that an embedding model outside the OpenAI catalog is accepted with base_url."""
    config = SessionConfig(
        app_name="test_openai_compatible_embedding_model",
        semantic=SemanticConfig(
            embedding_models={
                "local": OpenAIEmbeddingModel(
                    model_name="test-compatible-embeddings",
                    rpm=100,
                    tpm=100,
                    base_url="https://my-endpoint.example.com/v1",
                    model_parameters=OpenAIEmbeddingModel.ModelParameters(
                        output_dimensions=768,
                        max_input_size=512,
                    ),
                )
            }
        ),
    )
    resolved = config._to_resolved_config()
    model_config = resolved.semantic.embedding_models.model_configs["local"]
    assert model_config.model_name == "test-compatible-embeddings"
    assert model_config.base_url == "https://my-endpoint.example.com/v1"

    catalog_parameters = model_catalog.get_embedding_model_parameters(
        ModelProvider.OPENAI, "test-compatible-embeddings"
    )
    assert catalog_parameters is not None
    assert catalog_parameters.default_dimensions == 768
    assert catalog_parameters.max_input_size == 512
    assert catalog_parameters.input_token_cost == 0.0


def test_openai_embedding_model_custom_model_requires_model_parameters():
    """Test that an embedding model outside the OpenAI catalog needs model_parameters."""
    with pytest.raises(ConfigurationError, match="is not supported for openai"):
        OpenAIEmbeddingModel(
            model_name="test-compatible-embeddings-undeclared",
            rpm=100,
            tpm=100,
            base_url="https://my-endpoint.example.com/v1",
        )


def test_openai_embedding_model_custom_model_rejects_invalid_dimensions():
    """Test that non-positive embedding output dimensions are rejected."""
    with pytest.raises(PydanticValidationError, match="output_dimensions must be positive"):
        OpenAIEmbeddingModel.ModelParameters(output_dimensions=0, max_input_size=512)


def test_openai_compatible_model_enforces_declared_output_token_limit():
    """Test that a declared max_output_tokens is enforced for an out-of-catalog model.

    Also covers the case that matters for a self-hosted endpoint: because base_url is set,
    the session is built without reaching the endpoint, so this needs no reachable server.
    """
    config = SessionConfig(
        app_name="test_openai_compatible_output_limit",
        semantic=SemanticConfig(
            language_models={
                "local": OpenAILanguageModel(
                    model_name="test-compatible-output-limit",
                    rpm=100,
                    tpm=100,
                    base_url="https://my-endpoint.example.com/v1",
                    model_parameters=OpenAILanguageModel.ModelParameters(
                        context_window_length=8192,
                        max_output_tokens=512,
                    ),
                )
            }
        ),
    )
    session = Session.get_or_create(config)
    try:
        source = session.create_dataframe({"city": ["Paris"]})
        with pytest.raises(
            FenicValidationError,
            match="max_output_tokens must be a positive integer less than or equal to 512",
        ):
            source.select(
                col("city"),
                semantic.map(
                    "What is the typical weather in {{city}} in summer?",
                    city=col("city"),
                    max_output_tokens=1024,
                ).alias("weather"),
            ).to_polars()
    finally:
        session.stop()
