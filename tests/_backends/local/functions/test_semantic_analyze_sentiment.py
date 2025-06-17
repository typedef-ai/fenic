import polars as pl

from fenic import col, lit, semantic, text


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
    result = categorized_comments_df.to_polars()

    assert result.schema == {
        "user_comments": pl.String,
        "sentiment": pl.String,
    }
    _check_results_in_enum(result, "sentiment", ["positive", "negative", "neutral"])


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
    _check_results_in_enum(
        result,
        "sentiment",
        ["positive", "negative", "neutral", "None"],
        allow_none=True,
    )
    assert result["sentiment"].to_list()[4] is None


def _check_results_in_enum(
    result: pl.DataFrame,
    col_name: str,
    possible_results: list[str],
    allow_none: bool = False,
):
    result_list = result.select(pl.col(col_name))[col_name].to_list()
    for result in result_list:
        if allow_none and result is None:
            continue
        elif result is None:
            raise ValueError("Result is None, but allow_none is False")

        assert result in possible_results
