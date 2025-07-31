import polars as pl
import pytest
from polars import Schema as PolarsSchema
from pydantic import BaseModel, Field

from fenic import (
    ColumnField,
    MapExample,
    MapExampleCollection,
    OpenAIEmbeddingModel,
    Schema,
    StringType,
    StructField,
    StructType,
    col,
    semantic,
)
from fenic.api.session import (
    SemanticConfig,
    Session,
    SessionConfig,
)
from fenic.core.error import ValidationError


class ProductSummary(BaseModel):
    """Test BaseModel for structured output."""
    name: str = Field(description="Product name")
    description: str = Field(description="One-line description")
    category: str = Field(description="Product category")

class PersonInfo(BaseModel):
    """Another test BaseModel for structured output."""
    first_name: str = Field(description="First name")
    last_name: str = Field(description="Last name")
    age: int = Field(description="Age in years")


def test_semantic_map(local_session):
    source = local_session.create_dataframe({"name": ["Alice"], "city": ["New York"]})
    state_prompt = "What state does {name} live in given that they live in {city}?"
    df_select = source.select(
        semantic.map(state_prompt).alias("state"),
        col("name"),
        semantic.map(instruction="What is the typical weather in {city} in summer?").alias("weather"),
    )
    result = df_select.to_polars()
    assert result.schema == {
        "state": pl.String,
        "name": pl.String,
        "weather": pl.String,
    }

    weather_prompt = "What is the typical weather in {city} in summer?"
    df_with_column = source.with_column(
        "weather",
        semantic.map(instruction=weather_prompt),
    )
    result = df_with_column.to_polars()
    assert result.schema == {
        "name": pl.String,
        "city": pl.String,
        "weather": pl.String,
    }


def test_semantic_map_with_examples(local_session):
    source = local_session.create_dataframe({"name": ["Alice"], "city": ["New York"]})
    weather_prompt = "What is the weather in {city}?"
    weather_collection = MapExampleCollection()
    weather_collection.create_example(
        MapExample(
            input={"city": "Seattle"},
            output="It is rainy and 60 degrees",
        )
    ).create_example(
        MapExample(
            input={"city": "Los Angeles"},
            output="It is sunny and 70 degrees",
        )
    )
    df_with_column = source.with_column(
        "weather",
        semantic.map(
            instruction=weather_prompt,
            examples=weather_collection,
        ),
    )
    result = df_with_column.to_polars()
    assert result.schema == {
        "name": pl.String,
        "city": pl.String,
        "weather": pl.String,
    }


def test_semantic_map_with_nulls(local_session):
    # have a data source with some nulls.
    source = local_session.create_dataframe(
        {"name": ["Alice", "Bob"], "city": ["New York", None]}
    )
    state_prompt = "What state does {name} live in given that they live in {city}?"
    df_select = source.select(
        col("name"),
        semantic.map(state_prompt).alias("state"),
    )
    result = df_select.to_polars()
    assert result.schema == {
        "name": pl.String,
        "state": pl.String,
    }
    result_list = result["state"].to_list()
    assert len(result_list) == 2
    # Make sure that Bob's state is None.
    assert result_list[1] is None


def test_semantic_map_without_models():
    """Test that an error is raised if no language models are configured."""
    session_config = SessionConfig(
        app_name="semantic_map_without_models",
    )
    session = Session.get_or_create(session_config)
    with pytest.raises(ValidationError, match="No language models configured."):
        source = session.create_dataframe(
            {"name": ["Alice", "Bob"]}
        )
        state_prompt = "What state does {name} live in?"
        source.select(semantic.map(state_prompt).alias("map"))
    session.stop()

    session_config = SessionConfig(
        app_name="semantic_map_with_models",
        semantic=SemanticConfig(
            embedding_models={"oai-small" :OpenAIEmbeddingModel(model_name="text-embedding-3-small", rpm=3000, tpm=1_000_000)},
        ),
    )
    session = Session.get_or_create(session_config)
    with pytest.raises(ValidationError, match="No language models configured."):
        source = session.create_dataframe(
            {"name": ["Alice", "Bob"]}
        )
        state_prompt = "What state does {name} live in?"
        source.select(semantic.map(state_prompt).alias("map"))
    session.stop()

def test_semantic_map_with_schema(local_session):
    source = local_session.create_dataframe(
        {"name": ["GlowMate"], "details": ["A rechargeable bedside lamp"]}
    )
    result = source.select(
        semantic.map(
            "Given product name {name} and details {details}, create a product summary.",
            schema=ProductSummary
        ).alias("summary")
    )

    _validate_product_summary_schema(result.schema, result.to_polars().schema)


def test_semantic_map_schema_validation_with_basemodel_examples(local_session):
    """Test that semantic.map validates BaseModel examples match schema."""
    source = local_session.create_dataframe(
        {"name": ["GlowMate"], "details": ["A rechargeable bedside lamp"]}
    )

    # Create valid BaseModel example
    product_example = ProductSummary(
        name="GlowMate",
        description="Modern touch-controlled lamp for better sleep",
        category="lighting"
    )

    # Create examples collection with BaseModel output
    examples = MapExampleCollection()
    examples.create_example(MapExample(
        input={"name": "GlowMate", "details": "A rechargeable bedside lamp"},
        output=product_example
    ))

    # This should work - BaseModel example matches schema
    result = source.select(
        semantic.map(
            "Given product name {name} and details {details}, create a product summary.",
            examples=examples,
            schema=ProductSummary
        ).alias("summary")
    )

    _validate_product_summary_schema(result.schema, result.to_polars().schema)

def test_semantic_map_schema_validation_mismatch_error(local_session):
    """Test that semantic.map raises error when BaseModel examples don't match schema."""
    source = local_session.create_dataframe(
        {"name": ["John"], "details": ["A person"]}
    )

    # Create PersonInfo example (different type)
    person_example = PersonInfo(
        first_name="John",
        last_name="Doe",
        age=30
    )

    # Create examples collection with PersonInfo output
    examples = MapExampleCollection()
    examples.create_example(MapExample(
        input={"name": "John", "details": "A person"},
        output=person_example
    ))

    # This should raise ValidationError - PersonInfo example doesn't match ProductSummary schema
    with pytest.raises(ValidationError, match="all examples are required to have outputs of the same BaseModel type"):
        source.select(
            semantic.map(
                "Given name {name} and details {details}, create a product summary.",
                examples=examples,
                schema=ProductSummary  # Different type than example
            ).alias("summary")
        )


def test_semantic_map_schema_validation_string_examples_with_schema(local_session):
    """Test that semantic.map raises error when string examples are used with schema."""
    source = local_session.create_dataframe(
        {"name": ["BasicLamp"], "details": ["Simple desk lamp"]}
    )

    # Create examples collection with string output
    examples = MapExampleCollection()
    examples.create_example(MapExample(
        input={"name": "BasicLamp", "details": "Simple desk lamp"},
        output="A simple desk lamp for basic lighting needs"  # String output
    ))

    # This should raise ValidationError - string examples don't match BaseModel schema
    with pytest.raises(ValidationError, match="all examples are required to have outputs of the same BaseModel type"):
        source.select(
            semantic.map(
                "Given name {name} and details {details}, create a product summary.",
                examples=examples,
                schema=ProductSummary  # BaseModel schema but string example
            ).alias("summary")
        )

def test_semantic_map_invalid_basemodel_examples_no_schema(local_session):
    """Test that semantic.map raises ValidationError when BaseModel examples are used without schema."""
    source = local_session.create_dataframe(
        {"name": ["Product1"], "details": ["Detail1"]}
    )

    # Create valid BaseModel example
    product_example = ProductSummary(
        name="Product1",
        description="A product description",
        category="electronics"
    )

    # Create examples collection with BaseModel output
    examples = MapExampleCollection()
    examples.create_example(MapExample(
        input={"name": "Product1", "details": "Detail1"},
        output=product_example  # BaseModel output
    ))

    with pytest.raises(ValidationError, match="all examples are required to have outputs of type `str`"):
        source.select(
            semantic.map(
                "Given name {name} and details {details}, create a description.",
                examples=examples
                # No schema parameter
            ).alias("description")
        )

def _validate_product_summary_schema(df_schema: Schema, polars_schema: PolarsSchema):
    """Validate that the schema of a DataFrame is as expected."""
    assert df_schema.column_fields == [
        ColumnField(name="summary", data_type=StructType([
            StructField(name="name", data_type=StringType),
            StructField(name="description", data_type=StringType),
            StructField(name="category", data_type=StringType),
        ]))
    ]
    expected_schema = pl.Schema({
        "summary": pl.Struct({
            "name": pl.String,
            "description": pl.String,
            "category": pl.String,
        })
    })

    assert polars_schema == expected_schema
