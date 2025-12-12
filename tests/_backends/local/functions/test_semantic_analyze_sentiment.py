import polars as pl
import pytest

from fenic import OpenAIEmbeddingModel, col, lit, semantic, text
from fenic.api.session import (
    SemanticConfig,
    Session,
    SessionConfig,
)
from fenic.core.error import ValidationError
from fenic.core.types import ColumnField, StringType


def test_semantic_analyze_sentiment(local_session):
    comments_data = {
        "user_comments": [
            "best product ever",
            "can't wait to use it again",
            "I'm so angry about this product",
            "I'm ok with the product but wouldn't recommend it to others",
        ]
    }

    comments_df = local_session.create_dataframe(comments_data)
    categorized_comments_df = comments_df.select(
        col("user_comments"),
        semantic.analyze_sentiment(text.concat(col("user_comments"), lit(" "))).alias(
            "sentiment"
        ),
    )
    assert categorized_comments_df.schema.column_fields == [
        ColumnField(name="user_comments", data_type=StringType),
        ColumnField(name="sentiment", data_type=StringType),
    ]
    result = categorized_comments_df.to_polars()

    assert result.schema == {
        "user_comments": pl.String,
        "sentiment": pl.String,
    }

    for value in result.select(pl.col("sentiment"))["sentiment"].to_list():
        assert value in ["positive", "negative", "neutral"]


def test_semantic_analyze_sentiment_with_none(local_session):
    comments_data = {
        "user_comments": [
            "best product ever",
            "can't wait to use it again",
            "I'm so angry about this product",
            "I'm ok with the product but wouldn't recommend it to others",
            None,
        ]
    }

    comments_df = local_session.create_dataframe(comments_data)
    categorized_comments_df = comments_df.select(
        col("user_comments"),
        semantic.analyze_sentiment(col("user_comments")).alias("sentiment"),
    )
    result = categorized_comments_df.to_polars()

    assert result.schema == {
        "user_comments": pl.String,
        "sentiment": pl.String,
    }
    result_list = result.select(pl.col("sentiment"))["sentiment"].to_list()
    assert len(result_list) == 5
    assert result_list[4] is None
    for result in result_list[:4]:
        assert result in ["positive", "negative", "neutral"]

def test_semantic_analyze_sentiment_without_models(tmp_path):
    """Test that an error is raised if no language models are configured."""
    session_config = SessionConfig(
        app_name="semantic_analyze_sentiment_without_models",
        db_path=tmp_path,
    )
    session = Session.get_or_create(session_config)
    with pytest.raises(ValidationError, match="No language models configured."):
        session.create_dataframe({"text": ["hello"]}).select(semantic.analyze_sentiment(col("text")).alias("sentiment"))
    session.stop(skip_usage_summary=True)

    session_config = SessionConfig(
        app_name="semantic_analyze_sentiment_with_models",
        semantic=SemanticConfig(
            embedding_models={"oai-small": OpenAIEmbeddingModel(model_name="text-embedding-3-small", rpm=3000, tpm=1_000_000)},
        ),
        db_path=tmp_path,
    )
    session = Session.get_or_create(session_config)
    with pytest.raises(ValidationError, match="No language models configured."):
        session.create_dataframe({"text": ["hello"]}).select(semantic.analyze_sentiment(col("text")).alias("sentiment"))
    session.stop(skip_usage_summary=True)
