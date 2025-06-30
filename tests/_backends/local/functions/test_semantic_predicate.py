import polars as pl
from _utils.serde_utils import _test_df_serialization

from fenic import PredicateExample, PredicateExampleCollection, col, semantic


def test_single_semantic_filter(local_session):
    instruction = "This {blurb} has positive sentiment about apache spark."
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
        semantic.predicate(instruction)
        & (col("a_boolean_column"))
        & (col("a_numeric_column") > 0)
    )
    deserialized_df = _test_df_serialization(df, local_session._session_state)
    assert deserialized_df
    result = df.to_polars()
    assert result.schema == {
        "blurb": pl.String,
        "a_boolean_column": pl.Boolean,
        "a_numeric_column": pl.Int64,
    }

    df = source.select(semantic.predicate(instruction).alias("sentiment"))
    result = df.to_polars()
    assert result.schema == {
        "sentiment": pl.Boolean,
    }


def test_semantic_filter_with_examples(local_session):
    instruction = (
        "This {blurb1} and this {blurb2} have positive sentiment about apache spark."
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
                "blurb1": "Apache Spark has an amazing community.",
                "blurb2": "Apache Spark has good fault tolerance.",
            },
            output=True,
        )
    )
    df = source.filter(semantic.predicate(instruction, examples=sentiment_collection))
    deserialized_df = _test_df_serialization(df, local_session._session_state)
    assert deserialized_df
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
        semantic.predicate("This {review} discusses performance or speed")
        | semantic.predicate("This {review} discusses reliability or stability")
    )
    result = df.to_polars()

    # Should match first two reviews (performance and reliability) but not the third (documentation)
    assert result.schema == {
        "review": pl.String,
    }


def test_single_semantic_filter_with_none(local_session):
    instruction = "This {blurb} has positive sentiment about apache spark."
    source = local_session.create_dataframe(
        {
            "blurb": [
                "Apache Spark is the worst piece of software I've ever used. It's so slow and inefficient and I hate the JVM.",
                "Apache Spark is amazing. It's so fast and effortlessly scales to petabytes of data. Couldn't be happier.",
                None,
            ],
            "a_boolean_column": [
                True,
                False,
                False,
            ],
            "a_numeric_column": [
                1,
                -1,
                0,
            ],
        }
    )
    df = source.select(semantic.predicate(instruction).alias("sentiment"))
    result = df.to_polars()
    assert result["sentiment"].to_list()[2] is None
