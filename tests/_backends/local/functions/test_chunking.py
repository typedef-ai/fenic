import polars as pl
import requests

from fenic import col, text


def test_token_chunk(local_session):
    data = {
        "text_col": [
            "hello world foo bar",
            "world foo bar",
            "foo bar",
            "bar",
        ]
    }
    df = local_session.create_dataframe(data)

    result = df.with_column(
        "token_chunk_col",
        text.token_chunk(col("text_col"), chunk_size=2, chunk_overlap_percentage=50),
    ).to_polars()

    # Expected outputs updated to include the leading spaces on tokens after the first one.
    expected_outputs = [
        ["hello world", " world foo", " foo bar", " bar"],
        ["world foo", " foo bar", " bar"],
        ["foo bar", " bar"],
        ["bar"],
    ]

    for i, expected in enumerate(expected_outputs):
        actual = result["token_chunk_col"][i]
        if hasattr(actual, "to_list"):
            actual = actual.to_list()
        assert (
            actual == expected
        ), f"Row {i} mismatch: expected {expected}, got {actual}"


def test_recursive_text_chunking(large_text_df):
    df = large_text_df

    df = df.with_column(
        "character_chunks",
        text.recursive_character_chunk(
            col("text"), chunk_size=2500, chunk_overlap_percentage=10
        ),
    )
    df = df.with_column(
        "word_chunks",
        text.recursive_word_chunk(
            col("text"), chunk_size=500, chunk_overlap_percentage=10
        ),
    )
    df = df.with_column(
        "token_chunks",
        text.recursive_token_chunk(
            col("text"), chunk_size=500, chunk_overlap_percentage=10
        ),
    )
    result = df.to_polars()

    assert result.schema == {
        "text": pl.String,
        "character_chunks": pl.List(pl.String),
        "word_chunks": pl.List(pl.String),
        "token_chunks": pl.List(pl.String),
    }

    character_chunks = result["character_chunks"][0]
    word_chunks = result["word_chunks"][0]
    token_chunks = result["token_chunks"][0]
    assert len(character_chunks) == 344
    assert len(word_chunks) == 290
    assert len(token_chunks) == 438


def test_recursive_text_chunking_empty_input(local_session):
    df = local_session.create_dataframe({"text": ["", None]})
    result = df.with_column(
        "character_chunks",
        text.recursive_character_chunk(
            col("text"), chunk_size=2500, chunk_overlap_percentage=10
        ),
    ).to_polars()
    assert result["character_chunks"].to_list() == [[], None]


def test_sliding_window_text_chunking(large_text_df):
    df = large_text_df

    df = df.with_column(
        "character_chunks",
        text.character_chunk(col("text"), chunk_size=500, chunk_overlap_percentage=10),
    )
    df = df.with_column(
        "word_chunks",
        text.word_chunk(col("text"), chunk_size=500, chunk_overlap_percentage=10),
    )
    df = df.with_column(
        "token_chunks",
        text.token_chunk(col("text"), chunk_size=500, chunk_overlap_percentage=10),
    )
    result = df.to_polars()

    assert result.schema == {
        "text": pl.String,
        "character_chunks": pl.List(pl.String),
        "word_chunks": pl.List(pl.String),
        "token_chunks": pl.List(pl.String),
    }

    character_chunks = result["character_chunks"][0]
    word_chunks = result["word_chunks"][0]
    token_chunks = result["token_chunks"][0]
    assert len(character_chunks) == 1696
    assert len(word_chunks) == 290
    assert len(token_chunks) == 394


def test_chunk_large_text(local_session):
    pp_url = "https://typedef-assets.s3.us-west-2.amazonaws.com/example_texts/pride_and_prejudice"
    response = requests.get(pp_url)
    response.raise_for_status()
    pp_content = response.text

    cap_url = "https://typedef-assets.s3.us-west-2.amazonaws.com/example_texts/crime_and_punishment"
    cap_response = requests.get(cap_url)
    cap_response.raise_for_status()
    cap_content = cap_response.text
    df = pl.DataFrame({"text": [pp_content, cap_content]})
    df = local_session.create_dataframe(df)
    result = df.with_column(
        "extracted",
        text.recursive_word_chunk(
            col("text"),
            chunk_size=500,
            chunk_overlap_percentage=10,
        )
    ).to_polars()
    pp_chunks = result["extracted"][0].to_list()
    cap_chunks = result["extracted"][1].to_list()

    assert len(pp_chunks) == 290
    assert len(cap_chunks) == 459
