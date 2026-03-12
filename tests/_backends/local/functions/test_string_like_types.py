"""Tests that text functions work on Markdown and Json types without explicit casting.

Covers TD-1830: text functions (regex, ilike/rlike, etc) should work on Markdown and Json types.
"""
import pytest

from fenic import col, text
from fenic.core.types.datatypes import JsonType, MarkdownType


@pytest.fixture
def markdown_df(local_session):
    data = {"content": ["# Hello World", "## Section Two", "some **bold** text"]}
    df = local_session.create_dataframe(data)
    return df.select(col("content").cast(MarkdownType).alias("content"))


@pytest.fixture
def json_df(local_session):
    data = {"content": ['{"name": "Alice"}', '{"name": "Bob"}', '{"name": "Charlie"}']}
    df = local_session.create_dataframe(data)
    return df.select(col("content").cast(JsonType).alias("content"))


class TestContainsOnStringLikeTypes:
    def test_contains_markdown(self, markdown_df):
        result = markdown_df.filter(col("content").contains("Hello")).to_polars()
        assert len(result) == 1
        assert "Hello" in result["content"][0]

    def test_contains_json(self, json_df):
        result = json_df.filter(col("content").contains("Alice")).to_polars()
        assert len(result) == 1


class TestRLikeOnStringLikeTypes:
    def test_rlike_markdown(self, markdown_df):
        result = markdown_df.filter(col("content").rlike(r"^#\s")).to_polars()
        assert len(result) == 1
        assert "Hello" in result["content"][0]

    def test_rlike_json(self, json_df):
        result = json_df.filter(col("content").rlike(r'"name":\s*"[AB]')).to_polars()
        assert len(result) == 2


class TestLikeOnStringLikeTypes:
    def test_like_markdown(self, markdown_df):
        result = markdown_df.filter(col("content").like("# Hello%")).to_polars()
        assert len(result) == 1

    def test_ilike_markdown(self, markdown_df):
        result = markdown_df.filter(col("content").ilike("# hello%")).to_polars()
        assert len(result) == 1

    def test_ilike_json(self, json_df):
        result = json_df.filter(col("content").ilike('%"name": "alice"%')).to_polars()
        assert len(result) == 1


class TestStartsWithEndsWithOnStringLikeTypes:
    def test_starts_with_markdown(self, markdown_df):
        result = markdown_df.filter(col("content").starts_with("#")).to_polars()
        assert len(result) == 2  # "# Hello World" and "## Section Two"

    def test_ends_with_markdown(self, markdown_df):
        result = markdown_df.filter(col("content").ends_with("World")).to_polars()
        assert len(result) == 1

    def test_starts_with_json(self, json_df):
        result = json_df.filter(col("content").starts_with('{"name"')).to_polars()
        assert len(result) == 3

    def test_ends_with_json(self, json_df):
        result = json_df.filter(col("content").ends_with('"}')).to_polars()
        assert len(result) == 3


class TestStringFunctionsOnStringLikeTypes:
    def test_length_markdown(self, markdown_df):
        result = markdown_df.select(text.length(col("content")).alias("len")).to_polars()
        assert result["len"].to_list() == [13, 14, 18]

    def test_length_json(self, json_df):
        result = json_df.select(text.length(col("content")).alias("len")).to_polars()
        assert all(length > 0 for length in result["len"].to_list())

    def test_upper_markdown(self, markdown_df):
        result = markdown_df.select(text.upper(col("content")).alias("up")).to_polars()
        assert result["up"][0] == "# HELLO WORLD"

    def test_lower_json(self, json_df):
        result = json_df.select(text.lower(col("content")).alias("low")).to_polars()
        assert '"alice"' in result["low"][0]

    def test_replace_markdown(self, markdown_df):
        result = markdown_df.select(
            text.replace(col("content"), "Hello", "Goodbye").alias("replaced")
        ).to_polars()
        assert result["replaced"][0] == "# Goodbye World"

    def test_replace_json(self, json_df):
        result = json_df.select(
            text.replace(col("content"), "Alice", "Zara").alias("replaced")
        ).to_polars()
        assert "Zara" in result["replaced"][0]


class TestRegexpFunctionsOnStringLikeTypes:
    def test_regexp_extract_markdown(self, markdown_df):
        result = markdown_df.select(
            text.regexp_extract(col("content"), r"(#+)", 1).alias("hashes")
        ).to_polars()
        assert result["hashes"][0] == "#"
        assert result["hashes"][1] == "##"

    def test_regexp_count_markdown(self, markdown_df):
        result = markdown_df.select(
            text.regexp_count(col("content"), r"#").alias("count")
        ).to_polars()
        assert result["count"][0] == 1
        assert result["count"][1] == 2

    def test_regexp_substr_json(self, json_df):
        result = json_df.select(
            text.regexp_substr(col("content"), r'"name": "(\w+)"').alias("match")
        ).to_polars()
        assert result["match"][0] == '"name": "Alice"'


class TestChunkingOnStringLikeTypes:
    def test_chunk_markdown(self, local_session):
        long_text = "# Title\n\n" + "word " * 200
        data = {"content": [long_text]}
        df = local_session.create_dataframe(data)
        df = df.select(col("content").cast(MarkdownType).alias("content"))
        result = df.select(
            text.character_chunk(col("content"), chunk_size=100, chunk_overlap_percentage=10).alias("chunks")
        ).to_polars()
        assert len(result["chunks"][0]) > 1

    def test_count_tokens_markdown(self, markdown_df):
        result = markdown_df.select(
            text.count_tokens(col("content")).alias("tokens")
        ).to_polars()
        assert all(count > 0 for count in result["tokens"].to_list())


class TestStripCharsOnStringLikeTypes:
    def test_strip_markdown(self, markdown_df):
        result = markdown_df.select(
            text.btrim(col("content"), "#").alias("stripped")
        ).to_polars()
        assert result["stripped"][0] == " Hello World"
