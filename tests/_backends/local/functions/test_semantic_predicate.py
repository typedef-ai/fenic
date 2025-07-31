import polars as pl
import pytest

from fenic import (
    BooleanType,
    ColumnField,
    IntegerType,
    PredicateExample,
    PredicateExampleCollection,
    StringType,
    col,
    semantic,
)
from fenic.api.session import OpenAIModelConfig, SemanticConfig, Session, SessionConfig
from fenic.core.error import ValidationError


def test_single_semantic_filter(local_session):
    claim = "Review: {{review}}. The review has positive sentiment about apache spark."
    source = local_session.create_dataframe(
        {
            "blurb": [
                "Apache Spark is the worst piece of software I've ever used. It's so slow and inefficient and I hate the JVM.",
                "Apache Spark is amazing. It's so fast and effortlessly scales to petabytes of data. Couldn't be happier.",
            ],
            "a_boolean_column": [
                True,
                False,
            ],
            "a_numeric_column": [
                1,
                -1,
            ],
        }
    )
    df = source.filter(
        semantic.predicate(claim, review=col("blurb"))
        & (col("a_boolean_column"))
        & (col("a_numeric_column") > 0)
    )
    assert df.schema.column_fields == [
        ColumnField(name="blurb", data_type=StringType),
        ColumnField(name="a_boolean_column", data_type=BooleanType),
        ColumnField(name="a_numeric_column", data_type=IntegerType),
    ]
    result = df.to_polars()
    assert result.schema == {
        "blurb": pl.String,
        "a_boolean_column": pl.Boolean,
        "a_numeric_column": pl.Int64,
    }

    df = source.select(semantic.predicate(claim, review=col("blurb")).alias("sentiment"))
    result = df.to_polars()
    assert result.schema == {
        "sentiment": pl.Boolean,
    }


def test_semantic_filter_with_examples(local_session):
    claim = (
        "Review: {{part1}}. {{part2}}. The review has positive sentiment about apache spark."
    )
    source = local_session.create_dataframe(
        {
            "blurb1": [
                "Apache Spark is the worst piece of software I've ever used. It's so slow and inefficient and I hate the JVM.",
                "Apache Spark is amazing. It's so fast and effortlessly scales to petabytes of data. Couldn't be happier.",
            ],
            "blurb2": [
                "Apache Spark is the best thing since sliced bread.",
                "Apache Spark is the worst thing since sliced bread.",
            ],
        }
    )
    sentiment_collection = PredicateExampleCollection().create_example(
        PredicateExample(
            input={
                "part1": "Apache Spark has an amazing community.",
                "part2": "Apache Spark has good fault tolerance.",
            },
            output=True,
        )
    )
    df = source.filter(semantic.predicate(claim, part1=col("blurb1"), part2=col("blurb2"), examples=sentiment_collection))
    result = df.to_polars()
    assert result.schema == {
        "blurb1": pl.String,
        "blurb2": pl.String,
    }


def test_many_semantic_filter_or(local_session):
    source = local_session.create_dataframe(
        {
            "review": [
                "Apache Spark runs incredibly fast on our cluster, processing terabytes in minutes.",
                "Apache Spark has never crashed in production, running stable for months.",
                "Apache Spark's documentation is confusing and hard to follow.",
            ]
        }
    )

    df = source.filter(
        semantic.predicate("Review: {{review}}. The review discusses performance or speed", review=col("review"))
        | semantic.predicate("Review: {{review}}. The review discusses reliability or stability", review=col("review"))
    )
    result = df.to_polars()

    # Should match first two reviews (performance and reliability) but not the third (documentation)
    assert result.schema == {
        "review": pl.String,
    }

def test_semantic_predicate_without_models():
    """Test that an error is raised if no language models are configured."""
    session_config = SessionConfig(
        app_name="semantic_predicate_without_models",
    )
    session = Session.get_or_create(session_config)
    with pytest.raises(ValidationError, match="No language models configured."):
        source = session.create_dataframe(
            {"name": ["Alice", "Bob"]}
        )
        predicate_prompt = "The name: {{name}} has 10 letters."
        source.select(semantic.predicate(predicate_prompt, name=col("name")).alias("predicate"))
    session.stop()

    session_config = SessionConfig(
        app_name="semantic_predicate_with_models",
        semantic=SemanticConfig(
            embedding_models={"oai-small": OpenAIModelConfig(model_name="text-embedding-3-small", rpm=3000, tpm=1_000_000)},
        ),
    )
    session = Session.get_or_create(session_config)
    with pytest.raises(ValidationError, match="No language models configured."):
        source = session.create_dataframe(
            {"name": ["Alice", "Bob"]}
        )
        predicate_prompt = "The name: {{name}} has 10 letters."
        source.select(semantic.predicate(predicate_prompt, name=col("name")).alias("predicate"))
    session.stop()
