import json
from dataclasses import dataclass
from textwrap import dedent
from typing import List, Literal, Optional, Union
from unittest.mock import MagicMock, patch

import polars as pl
import pytest
from pydantic import BaseModel, ConfigDict, Field

from fenic import (
    BooleanType,
    IntegerType,
    OpenAIEmbeddingModel,
    StringType,
    col,
    semantic,
)
from fenic.api.session import (
    SemanticConfig,
    Session,
    SessionConfig,
)
from fenic.core.error import ValidationError as FenicValidationError
from fenic.core.types import ArrayType, ColumnField, StructField, StructType


def test_extract_primitive_types(extract_data_df):
    """Test semantic extraction with primitive types."""
    class BasicReviewModel(BaseModel):
        product_name: str = Field(description="The name of the product mentioned in the review")
        phone_version: int = Field(description="specific product number")
        contains_negative_feedback: bool = Field(description="the review contains some negative feedback")

    df = extract_data_df.select(
        semantic.extract(col("review"), BasicReviewModel).alias("review")
    )
    assert df.schema.column_fields == [
        ColumnField(name="review", data_type=StructType([
            StructField(name="product_name", data_type=StringType),
            StructField(name="phone_version", data_type=IntegerType),
            StructField(name="contains_negative_feedback", data_type=BooleanType),
        ]))
    ]
    result = df.to_polars()

    expected_schema = pl.Schema({
        "review": pl.Struct({
            "product_name": pl.String,
            "phone_version": pl.Int64,
            "contains_negative_feedback": pl.Boolean,
        })
    })
    assert result.schema == expected_schema
    assert result["review"][0] is not None

def test_extract_lists(local_session):
    """Test extraction with complex nested Pydantic models containing lists and lists of objects."""
    class Triple(BaseModel):
        subject: str = Field(description="The subject of the triple")
        predicate: str = Field(description="The predicate or relation")
        object: str = Field(description="The object of the triple")

    class KGResult(BaseModel):
        triples: List[Triple] = Field(description="List of extracted knowledge graph triples")
        entities: list[str] = Field(description="Flat list of all detected named entities")

    df = local_session.create_dataframe({
        "blurb": [
            "Anthropic, co-founded by Dario Amodei, is a leading AI research company. "
            "Harrison Chase is the CEO of Langchain."
        ]
    })

    df = df.select(semantic.extract(col("blurb"), KGResult).alias("blurb"))
    assert df.schema.column_fields == [
        ColumnField(name="blurb", data_type=StructType([
            StructField(name="triples", data_type=ArrayType(element_type=StructType([
                StructField(name="subject", data_type=StringType),
                StructField(name="predicate", data_type=StringType),
                StructField(name="object", data_type=StringType),
            ]))),
            StructField(name="entities", data_type=ArrayType(element_type=StringType)),
        ]))
    ]
    result = df.to_polars()

    assert result.schema == pl.Schema({"blurb": pl.Struct({
        "triples": pl.List(pl.Struct({
            "subject": pl.String,
            "predicate": pl.String,
            "object": pl.String,
        })),
        "entities": pl.List(pl.String),
    })})
    assert result["blurb"][0] is not None

def test_extract_nested_objects_with_optional_fields(local_session):
    class WorkExperience(BaseModel):
        company: str = Field(description="Name of the company")
        title: str = Field(description="Job title held")
        start_year: int = Field(description="Year the job started")
        end_year: Optional[int] = Field(None, description="Year the job ended (if applicable)")
        description: Optional[str] = Field(None, description="Short description of responsibilities")

    class Education(BaseModel):
        institution: str = Field(description="Name of the educational institution")
        major: str = Field(description="Field of study")
        graduation_year: Optional[int] = Field(None, description="Year of graduation")

    class Resume(BaseModel):
        name: str = Field(description="Full name of the candidate")
        work_experience: List[WorkExperience] = Field(description="List of work experiences")
        skills: List[str] = Field(description="List of individual skills mentioned")
        education: Optional[Education] = Field(None, description="Education details")

    df = local_session.create_dataframe({
        "text": [dedent("""
            Jane Doe is a software engineer with over 6 years of experience. She worked at OpenAI from 2021 to 2024 as a Machine Learning Engineer, focusing on NLP research.
            Before that, she was at Google as a Software Engineer from 2018 to 2021, building distributed systems.

            She is skilled in Python, PyTorch, and distributed computing. Also familiar with Rust and Kubernetes.

            She graduated from MIT in 2017 with a degree in Computer Science.
        """).strip()]
    })

    df = df.select(semantic.extract(col("text"), Resume).alias("resume"))
    assert df.schema.column_fields == [
        ColumnField(name="resume", data_type=StructType([
            StructField(name="name", data_type=StringType),
            StructField(name="work_experience", data_type=ArrayType(element_type=StructType([
                StructField(name="company", data_type=StringType),
                StructField(name="title", data_type=StringType),
                StructField(name="start_year", data_type=IntegerType),
                StructField(name="end_year", data_type=IntegerType),
                StructField(name="description", data_type=StringType),
            ]))),
            StructField(name="skills", data_type=ArrayType(element_type=StringType)),
            StructField(name="education", data_type=StructType([
                StructField(name="institution", data_type=StringType),
                StructField(name="major", data_type=StringType),
                StructField(name="graduation_year", data_type=IntegerType),
            ]))
        ]))
    ]

    result = df.to_polars()

    assert result.schema == pl.Schema({"resume": pl.Struct({
        "name": pl.String,
        "work_experience": pl.List(pl.Struct({
            "company": pl.String,
            "title": pl.String,
            "start_year": pl.Int64,
            "end_year": pl.Int64,
            "description": pl.String,
        })),
        "skills": pl.List(pl.String),
        "education": pl.Struct({
            "institution": pl.String,
            "major": pl.String,
            "graduation_year": pl.Int64,
        }),
    })})
    assert result["resume"][0] is not None

def test_pydantic_model_with_literal_types(local_session):
    """Test extraction with Pydantic models containing Literal types."""
    class SentimentModel(BaseModel):
        sentiment: Literal["positive", "neutral", "negative"] = Field(
            description="The sentiment classification of the review text"
        )

    df = local_session.create_dataframe({
        "review": [
            "This product is amazing and exceeded my expectations!",
            "It's okay, does the job.",
            "Terrible experience, would not recommend."
        ]
    })

    df = df.select(
        semantic.extract(col("review"), SentimentModel).alias("sentiment_out")
    )
    assert df.schema.column_fields == [
        ColumnField(name="sentiment_out", data_type=StructType([
            StructField(name="sentiment", data_type=StringType),
        ]))
    ]

    results = df.to_polars()

    assert results.schema == pl.Schema({"sentiment_out": pl.Struct({
        "sentiment": pl.String,
    })})


def test_null_input_handling(local_session):
    """Test how semantic extraction handles None/null inputs."""
    df = local_session.create_dataframe({
        "review": [
            "The iPhone 13 has a great camera but average battery life.",
            "I love my Samsung Galaxy S21!",
            None,  # Test null handling
        ],
    })

    class ReviewModel(BaseModel):
        product_name: str = Field(description="The name of the product mentioned")
        contains_negative_feedback: bool = Field(description="the review contains negative feedback")

    df = df.select(semantic.extract(col("review"), ReviewModel).alias("review"))
    result = df.to_polars()

    # Verify that null input produces null output
    assert result["review"].to_list()[2] is None

def test_extract_schema_validation_errors(extract_data_df):
    """Test that Pydantic models without descriptions raise validation errors."""
    # Missing description
    with pytest.raises(FenicValidationError):
        class BadModel(BaseModel):
            invalid_field: str = Field(...)

        extract_data_df.select(
            semantic.extract(col("review"), BadModel).alias("review_out")
        ).to_polars()

    # Union not supported
    with pytest.raises(FenicValidationError):
        class BadModel(BaseModel):
            invalid_field: Union[str, int] = Field(description="a description that doesn't matter")

        extract_data_df.select(
            semantic.extract(col("review"), BadModel).alias("review_out")
        ).to_polars()

    # Dict not supported
    with pytest.raises(FenicValidationError):
        class BadModel(BaseModel):
            invalid_field: dict[str, int] = Field(description="a description that doesn't matter")

        extract_data_df.select(
            semantic.extract(col("review"), BadModel).alias("review_out")
        ).to_polars()

    # Dataclass not supported
    @dataclass
    class Foo:
        foo: str
    with pytest.raises(FenicValidationError):
        class BadModel(BaseModel):
            invalid_field: Foo = Field(description="a description that doesn't matter")

        extract_data_df.select(
            semantic.extract(col("review"), BadModel).alias("review_out")
        ).to_polars()

    # Custom class not supported
    class Foo:
        foo: str
    with pytest.raises(FenicValidationError):
        class BadModel(BaseModel):
            invalid_field: Foo = Field(description="a description that doesn't matter")
            model_config = ConfigDict(arbitrary_types_allowed=True)

        extract_data_df.select(
            semantic.extract(col("review"), BadModel).alias("review_out")
        ).to_polars()


def test_semantic_extract_without_models(tmp_path):
    """Test that an error is raised if no language models are configured."""
    class ExtractSchema(BaseModel):
        first_character: str = Field(description="The first character of the input text")

    session_config = SessionConfig(
        app_name="semantic_extract_without_models",
        db_path=tmp_path,
    )
    session = Session.get_or_create(session_config)
    with pytest.raises(FenicValidationError, match="No language models configured."):
        session.create_dataframe({"text": ["hello"]}).select(semantic.extract(col("text"), ExtractSchema).alias("extracted_text"))
    session.stop(skip_usage_summary=True)

    session_config = SessionConfig(
        app_name="semantic_extract_with_models",
        semantic=SemanticConfig(
            embedding_models={"oai-small": OpenAIEmbeddingModel(model_name="text-embedding-3-small", rpm=3000, tpm=1_000_000)},
        ),
        db_path=tmp_path,
    )
    session = Session.get_or_create(session_config)
    with pytest.raises(FenicValidationError, match="No language models configured."):
        session.create_dataframe({"text": ["hello"]}).select(semantic.extract(col("text"), ExtractSchema).alias("extracted_text"))
    session.stop(skip_usage_summary=True)


def test_semantic_extract_with_mismatched_rows(local_session: Session):
    """Test that semantic.extract can handle rows with different structures."""
    class Item(BaseModel):
        """A simple item with optional metadata."""
        name: str = Field(description="Item name")
        count: Optional[int] = Field(None, description="Item count")
        tags: List[str] = Field(default_factory=list, description="Item tags")

    class Analysis(BaseModel):
        """Analysis result with variable structure."""
        items: List[Item] = Field(default_factory=list, description="List of items")
        summary: Optional[str] = Field(None, description="Optional summary")

    mock_responses = [
        json.dumps({"items": [{"name": "item1"}]}),
        json.dumps({
            "items": [
                {"name": "item2", "count": 5, "tags": ["tag1", "tag2"]}
            ],
            "summary": "Found 1 item"
        }),
    ]

    df = local_session.create_dataframe({
        "text": ["text1", "text2"]
    })

    response_iter = iter(mock_responses)

    def mock_get_completions(messages, operation_name, **kwargs):
        results = []
        for _ in messages:
            try:
                response_text = next(response_iter)
                mock_response = MagicMock()
                mock_response.completion = response_text
                results.append(mock_response)
            except StopIteration:
                results.append(None)
        return results

    with patch.object(
        local_session._session_state.get_language_model(),
        'get_completions',
        side_effect=mock_get_completions
    ):
        result = df.with_column(
            "analysis",
            semantic.extract(col("text"), Analysis)
        ).to_polars()

        assert len(result) == 2
        assert result["analysis"][0] is not None
        assert result["analysis"][1] is not None
