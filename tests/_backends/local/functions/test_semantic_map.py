from textwrap import dedent

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
from fenic._constants import MAX_MODEL_CLIENT_TIMEOUT
from fenic.api.session import (
    SemanticConfig,
    Session,
    SessionConfig,
)
from fenic.core.error import InvalidExampleCollectionError, ValidationError


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
    source = local_session.create_dataframe({
        "user": [
            {"preferences": {"formal": True}, "name": "John"},
            {"preferences": {"formal": False}, "name": "Alice"}
        ],
        "tasks": [
            [{"title": "Task 1", "priority": "High"}, {"title": "Task 2", "priority": "Medium"}],
            [{"title": "Write report", "priority": "High"}, {"title": "Review code", "priority": "Low"}, {"title": "Team meeting", "priority": "Medium"}]
        ],
        "index": [1, 2]
    })
    df = source.select(
        semantic.map(
            dedent("""\
                {% if user.preferences.formal %}
                Please respond in a formal, professional tone.
                {% else %}
                Please respond in a casual, friendly tone.
                {% endif %}

                ## Tasks to complete:
                {% for task in tasks %}
                {{ loop.index }}. {{ task.title }} (Priority: {{ task.priority }})
                {% endfor %}

                Construct a plan to execute the tasks above.
            """),
            user=col("user"),
            tasks=col("tasks"),
            index=col("index"),
        ).alias("plan")
    )
    assert df.schema.column_fields == [
        ColumnField(name="plan", data_type=StringType),
    ]
    df = df.to_polars()
    assert df.schema == {
        "plan": pl.String,
    }

def test_semantic_map_with_examples(local_session):
    source = local_session.create_dataframe({
        "user": [
            {"preferences": {"formal": True}, "name": "John"},
            {"preferences": {"formal": False}, "name": "Alice"}
        ],
        "tasks": [
            [{"title": "Task 1", "priority": "High"}, {"title": "Task 2", "priority": "Medium"}],
            [{"title": "Write report", "priority": "High"}, {"title": "Review code", "priority": "Low"}, {"title": "Team meeting", "priority": "Medium"}]
        ]
    })
    examples = MapExampleCollection()
    # foo should be ignored
    examples.create_example(MapExample(
        input={"user": {"preferences": {"formal": True}, "name": "Lisa"}, "tasks": [{"title": "Generate report", "priority": "High"}, {"title": "Review code", "priority": "Low"}, {"title": "Team meeting", "priority": "Medium"}], "another_column": 1},
        output="First, I will write a report on the tasks. Then, I will review the code. Finally, I will attend the team meeting."
    ))
    prompt = dedent("""\
                {% if user.preferences.formal %}
                Please respond in a formal, professional tone.
                {% else %}
                Please respond in a casual, friendly tone.
                {% endif %}

                ## Tasks to complete:
                {% for task in tasks %}
                {{ loop.index }}. {{ task.title }} (Priority: {{ task.priority }})
                {% endfor %}

                Construct a plan to execute the tasks above.
            """)
    df = source.select(
        semantic.map(
            prompt,
            user=col("user"),
            tasks=col("tasks"),
            examples=examples,
        ).alias("plan")
    )
    assert df.schema.column_fields == [
        ColumnField(name="plan", data_type=StringType),
    ]
    df = df.to_polars()
    assert df.schema == {
        "plan": pl.String,
    }

    bad_examples = MapExampleCollection()
    bad_examples.create_example(MapExample(
        input={"user": {"not_preferences": {"hello": "world"}}, "tasks": [{"title": "Generate report", "priority": "High"}, {"title": "Review code", "priority": "Low"}, {"title": "Team meeting", "priority": "Medium"}]},
        output="First, I will write a report on the tasks. Then, I will review the code. Finally, I will attend the team meeting."
    ))
    with pytest.raises(InvalidExampleCollectionError, match="Field 'user' type mismatch: operator expects"):
        source.select(semantic.map(prompt, user=col("user"), tasks=col("tasks"), examples=bad_examples).alias("plan"))


def test_semantic_map_with_nulls(local_session):
    # have a data source with some nulls.
    source = local_session.create_dataframe(
        {"name": ["Alice", "Bob"], "city": ["New York", None]}
    )
    state_prompt = "What state does {{name}} live in given that they live in {{city}}?"
    df_select = source.select(
        col("name"),
        semantic.map(state_prompt, name=col("name"), city=col("city")).alias("state"),
    )
    result = df_select.to_polars()
    assert result.schema == {
        "name": pl.String,
        "state": pl.String,
    }
    result_list = result["state"].to_list()
    assert len(result_list) == 2
    assert result_list[1] is None

    df_select = source.select(
        semantic.map(state_prompt, strict=False, name=col("name"), city=col("city")).alias("state"),
    )
    result = df_select.to_polars()
    result_list = result["state"].to_list()
    assert len(result_list) == 2
    assert result_list[1] is not None


def test_semantic_map_without_models(tmp_path):
    """Test that an error is raised if no language models are configured."""
    session_config = SessionConfig(
        app_name="semantic_map_without_models",
        db_path=tmp_path,
    )
    session = Session.get_or_create(session_config)
    with pytest.raises(ValidationError, match="No language models configured."):
        source = session.create_dataframe(
            {"name": ["Alice", "Bob"]}
        )
        state_prompt = "What state does {{name}} live in?"
        source.select(semantic.map(state_prompt, name=col("name")).alias("map"))
    session.stop()

    session_config = SessionConfig(
        app_name="semantic_map_with_models",
        semantic=SemanticConfig(
            embedding_models={"oai-small" :OpenAIEmbeddingModel(model_name="text-embedding-3-small", rpm=3000, tpm=1_000_000)},
        ),
        db_path=tmp_path,
    )
    session = Session.get_or_create(session_config)
    with pytest.raises(ValidationError, match="No language models configured."):
        source = session.create_dataframe(
            {"name": ["Alice", "Bob"]}
        )
        state_prompt = "What state does {{name}} live in?"
        source.select(semantic.map(state_prompt, name=col("name")).alias("map"))
    session.stop()

def test_semantic_map_with_response_format(local_session):
    source = local_session.create_dataframe(
        {"name": ["GlowMate"], "details": ["A rechargeable bedside lamp"]}
    )
    result = source.select(
        semantic.map(
            "Given product name: '{{name}}' and details: '{{details}}', create a product summary.",
            name=col("name"),
            details=col("details"),
            response_format=ProductSummary
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
            "Given product name {{name}} and details {{details}}, create a product summary.",
            name=col("name"),
            details=col("details"),
            examples=examples,
            response_format=ProductSummary
        ).alias("summary")
    )

    _validate_product_summary_schema(result.schema, result.to_polars().schema)

def test_semantic_map_fails_with_bad_request_timeouts(local_session):
    source = local_session.create_dataframe(
        {"name": ["GlowMate"], "details": ["A rechargeable bedside lamp"]}
    )
    with pytest.raises(ValidationError, match="The `request_timeout` argument must be a positive number."):
        source.select(
            semantic.map(
                "Given product name {{name}} and details {{details}}, create a product summary.",
                name=col("name"),
                details=col("details"),
                request_timeout=0
            ).alias("summary")
        )
    with pytest.raises(ValidationError, match=f"The `request_timeout` argument can't be greater than the system's max timeout of {MAX_MODEL_CLIENT_TIMEOUT} seconds."):
        source.select(
            semantic.map(
                "Given product name {{name}} and details {{details}}, create a product summary.",
                name=col("name"),
                details=col("details"),
                request_timeout=MAX_MODEL_CLIENT_TIMEOUT + 1
            ).alias("summary")
        )

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

    # This should raise InvalidExampleCollectionError - PersonInfo example doesn't match ProductSummary schema
    with pytest.raises(InvalidExampleCollectionError, match="Expected `semantic.map` example output type to be the same as the `response_format` type"):
        source.select(
            semantic.map(
                "Given name {{name}} and details {{details}}, create a product summary.",
                name=col("name"),
                details=col("details"),
                examples=examples,
                response_format=ProductSummary  # Different type than example
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

    # This should raise InvalidExampleCollectionError - string examples don't match BaseModel schema
    with pytest.raises(InvalidExampleCollectionError, match="Expected `semantic.map` example output to be a Pydantic BaseModel"):
        source.select(
            semantic.map(
                "Given name {{name}} and details {{details}}, create a product summary.",
                name=col("name"),
                details=col("details"),
                examples=examples,
                response_format=ProductSummary  # BaseModel schema but string example
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

    with pytest.raises(ValidationError, match="Expected `semantic.map` example output to be a string, but got"):
        source.select(
            semantic.map(
                "Given name {{name}} and details {{details}}, create a description.",
                name=col("name"),
                details=col("details"),
                examples=examples
                # No schema parameter
            ).alias("description")
        )

def test_semantic_map_invalid_jinja_template(local_session):
    source = local_session.create_dataframe({"name": ["GlowMate"], "details": ["A rechargeable bedside lamp"]})
    with pytest.raises(ValidationError, match="The `prompt` argument to `semantic.map` cannot be empty."):
        source.select(
            semantic.map("", name=col("name"), details=col("details")).alias("summary")
        )
    with pytest.raises(ValidationError, match="`semantic.map` prompt requires at least one template variable."):
        source.select(
            semantic.map("hello", name=col("name")).alias("summary")
        )

def test_semantic_map_missing_column_arguments(local_session):
    source = local_session.create_dataframe({"name": ["GlowMate"], "details": ["A rechargeable bedside lamp"]})
    with pytest.raises(ValidationError, match="`semantic.map` requires at least one named column argument"):
        source.select(
            semantic.map("{{name}}").alias("summary")
        )

def test_semantic_map_missing_jinja_variable(local_session):
    source = local_session.create_dataframe({"name": ["GlowMate"], "details": ["A rechargeable bedside lamp"]})
    with pytest.raises(ValidationError, match="Template variable 'details' is not defined."):
        source.select(
            semantic.map("{{name}}{{details}}", name=col("name")).alias("summary")
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
