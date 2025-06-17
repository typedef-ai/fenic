import polars as pl
import pytest

from fenic import col, text


def test_count_tokens_basic(large_text_df):
    df = large_text_df

    df = df.with_column(
        "token_count",
        text.count_tokens(col("text")),
    )

    result = df.to_polars()

    assert result.schema == {
        "text": pl.String,
        "token_count": pl.UInt32,
    }

    token_count = result["token_count"][0]
    assert token_count == 177201


def test_count_tokens_raises_on_numeric(sample_df):
    df = sample_df

    with pytest.raises(TypeError):
        df = df.with_column(
            "token_count",
            text.count_tokens(col("age")),
        )
        df.to_polars()


def test_count_tokens_empty_string(local_session):
    df = local_session.create_dataframe({"text": ["", "     "]})

    df = df.with_column(
        "token_count",
        text.count_tokens(col("text")),
    )

    result = df.to_polars()

    assert result.schema == {
        "text": pl.String,
        "token_count": pl.UInt32,
    }

    token_count_empty_string = result["token_count"][0]
    token_count_string_with_whitespace = result["token_count"][1]

    assert token_count_empty_string == 0
    assert token_count_string_with_whitespace == 1


def test_count_tokens_escape_characters(local_session):
    df = local_session.create_dataframe({"text": ["\n\n\t\b\r"]})

    df = df.with_column(
        "token_count",
        text.count_tokens(col("text")),
    )

    result = df.to_polars()

    assert result.schema == {
        "text": pl.String,
        "token_count": pl.UInt32,
    }

    token_count = result["token_count"][0]
    assert token_count == 4


def test_count_tokens_null_values(local_session):
    data = {"text": [None, "hello"]}
    df = local_session.create_dataframe(data)

    df = df.with_column(
        "token_count",
        text.count_tokens(col("text")),
    )

    result = df.to_polars()

    assert result.schema == {
        "text": pl.String,
        "token_count": pl.UInt32,
    }

    token_count = result["token_count"][0]
    assert token_count is None
