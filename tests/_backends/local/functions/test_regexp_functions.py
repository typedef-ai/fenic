import pytest

from fenic import col, text
from fenic.core.error import ExecutionError, ValidationError


@pytest.fixture
def replace_test_df(local_session):
    data = {
        "text_col": [
            "hello world",
            "hello hello world",
            "world hello world",
            "no matches here",
        ]
    }
    return local_session.create_dataframe(data)

def test_regexp_replace_with_column_pattern(local_session):
    df = local_session.create_dataframe(
        {
            "text_col": ["hello123world", "test456example", "sample789text"],
            "pattern": [r"\d+", r"[0-9]+", r"[0-9]{3}"],
            "replacement": ["---", "***", "###"],
        }
    )
    result = df.with_column(
        "replaced_text_col",
        text.regexp_replace(col("text_col"), col("pattern"), col("replacement")),
    ).to_polars()
    assert result["replaced_text_col"][0] == "hello---world"
    assert result["replaced_text_col"][1] == "test***example"
    assert result["replaced_text_col"][2] == "sample###text"


def test_regexp_replace_with_column_pattern_and_fixed_replacement(local_session):
    df = local_session.create_dataframe(
        {
            "text_col": ["hello123world", "test456example", "sample789text"],
            "pattern": [r"\d+", r"[0-9]+", r"[0-9]{3}"],
        }
    )
    result = df.with_column(
        "replaced_text_col",
        text.regexp_replace(col("text_col"), col("pattern"), "---"),
    ).to_polars()
    assert result["replaced_text_col"][0] == "hello---world"
    assert result["replaced_text_col"][1] == "test---example"
    assert result["replaced_text_col"][2] == "sample---text"

def test_regexp_replace_basic(replace_test_df):
    result = replace_test_df.with_column(
        "replaced_text_col", text.regexp_replace(col("text_col"), "^hello", "hi")
    ).to_polars()
    # Should only replace 'hello' at the start of the string
    assert result["replaced_text_col"][0] == "hi world"
    assert result["replaced_text_col"][1] == "hi hello world"
    assert result["replaced_text_col"][2] == "world hello world"
    assert result["replaced_text_col"][3] == "no matches here"


def test_regexp_replace_word_boundaries(replace_test_df):
    result = replace_test_df.with_column(
        "replaced_text_col", text.regexp_replace(col("text_col"), r"\bhello\b", "hi")
    ).to_polars()
    # Should replace 'hello' only when it's a complete word
    assert result["replaced_text_col"][0] == "hi world"
    assert result["replaced_text_col"][1] == "hi hi world"
    assert result["replaced_text_col"][2] == "world hi world"
    assert result["replaced_text_col"][3] == "no matches here"

def test_regexp_count(local_session):
    """Test regexp_count function."""
    data = {"text": ["abc123", "456def789", "no digits", None]}
    df = local_session.create_dataframe(data)

    result = df.select(text.regexp_count("text", r"\d").alias("digit_count")).to_polars()

    assert result["digit_count"][0] == 3
    assert result["digit_count"][1] == 6
    assert result["digit_count"][2] == 0
    assert result["digit_count"][3] is None


def test_regexp_count_words(local_session):
    """Test regexp_count with word pattern."""
    data = {"text": ["hello world", "one two three", ""]}
    df = local_session.create_dataframe(data)

    result = df.select(text.regexp_count("text", r"\w+").alias("word_count")).to_polars()

    assert result["word_count"][0] == 2
    assert result["word_count"][1] == 3
    assert result["word_count"][2] == 0


def test_regexp_extract(local_session):
    """Test regexp_extract function."""
    data = {"email": ["user@domain.com", "admin@example.org", "invalid", None]}
    df = local_session.create_dataframe(data)

    # Extract username (group 1)
    result = df.select(
        text.regexp_extract("email", r"([^@]+)@", 1).alias("username")
    ).to_polars()

    assert result["username"][0] == "user"
    assert result["username"][1] == "admin"
    assert result["username"][2] == ''
    assert result["username"][3] == ''


def test_regexp_extract_entire_match(local_session):
    """Test regexp_extract with group 0 (entire match)."""
    data = {"text": ["Price: $123.45", "Cost: $67.89", "No price"]}
    df = local_session.create_dataframe(data)

    result = df.select(
        text.regexp_extract("text", r"(\$)(\d+)\.(\d+)", 0).alias("price")
    ).to_polars()

    assert result["price"][0] == "$123.45"
    assert result["price"][1] == "$67.89"
    assert result["price"][2] == ''

def test_regexp_extract_with_column_pattern(local_session):
    df = local_session.create_dataframe(
        {
            "text_col": ["Price: $123.45", "Cost: $67.89", "No price"],
            "pattern": [r"(\$)(\d+)\.(\d+)", r"(\$)(\d+)\.(\d+)", r"(\$)(\d+)\.(\d+)"],
        }
    )
    result = df.with_column(
        "price", text.regexp_extract(col("text_col"), col("pattern"), 2)
    ).to_polars()
    assert result["price"][0] == "123"
    assert result["price"][1] == "67"
    assert result["price"][2] == ''

def test_regexp_extract_with_group_idx_out_of_range(local_session):
    df = local_session.create_dataframe(
        {
            "text_col": ["Price: $123.45", "Cost: $67.89", "No price"],
            "pattern": [r"(\$)(\d+)\.(\d+)", r"(\$)(\d+)\.(\d+)", r"(\$)(\d+)\.(\d+)"],
        }
    )
    result = df.with_column(
        "price", text.regexp_extract(col("text_col"), col("pattern"), 4)
    ).to_polars()
    assert result["price"][0] == ''
    assert result["price"][1] == ''
    assert result["price"][2] == ''

def test_regexp_extract_with_pattern_and_or_group(local_session):
    """Test regexp_extract where pattern uses | (OR) and some groups may not match."""
    data = {"text": ["foo.bar", "foo", "bar.baz", "baz"]}
    df = local_session.create_dataframe(data)

    # Pattern with two groups: (\w+)\.(\w+) OR (\w+)
    # For "foo.bar", group 1 == "foo", group 2 == "bar"
    # For "foo", group 1 and group 2 don't match (but (\w+) as group 3 matches "foo")
    result = df.select(
        text.regexp_extract("text", r"(\w+)\.(\w+)|(\w+)", 2).alias("second")
    ).to_polars()

    # Only rows with dots will have group 2 match; others will be empty string
    assert result["second"][0] == "bar"  # "foo.bar" -> "bar"
    assert result["second"][1] == ""     # "foo" -> no group 2
    assert result["second"][2] == "baz"  # "bar.baz" -> "baz"
    assert result["second"][3] == ""     # "baz" -> no group 2


def test_regexp_extract_all(local_session):
    """Test regexp_extract_all function."""
    data = {"text": ["abc123def456", "no digits", "789", None]}
    df = local_session.create_dataframe(data)

    result = df.select(
        text.regexp_extract_all("text", r"\d+", 0).alias("digits")
    ).to_polars()

    digits_list = result["digits"].to_list()
    assert digits_list[0] == ["123", "456"]
    assert digits_list[1] == []
    assert digits_list[2] == ["789"]
    assert digits_list[3] is None


def test_regexp_extract_all_with_groups(local_session):
    """Test regexp_extract_all with capture groups."""
    data = {"post": ["Love #coding and #python", "Just #relaxing", "No hashtags"]}
    df = local_session.create_dataframe(data)

    result = df.select(
        text.regexp_extract_all("post", r"#(\w+)", 1).alias("hashtags")
    ).to_polars()

    hashtags_list = result["hashtags"].to_list()
    assert hashtags_list[0] == ["coding", "python"]
    assert hashtags_list[1] == ["relaxing"]
    assert hashtags_list[2] == []


def test_regexp_extract_all_with_column_pattern(local_session):
    """Test regexp_extract_all with column-based pattern (uses Rust plugin)."""
    data = {
        "text": ["abc123def456", "Love #coding and #python", "test@example.com and hello@world.org"],
        "pattern": [r"\d+", r"#(\w+)", r"(\w+)@"],
        "idx": [0, 1, 1]
    }
    df = local_session.create_dataframe(data)

    result = df.select(
        text.regexp_extract_all(col("text"), col("pattern"), col("idx")).alias("matches")
    ).to_polars()

    matches_list = result["matches"].to_list()
    assert matches_list[0] == ["123", "456"]
    assert matches_list[1] == ["coding", "python"]
    assert matches_list[2] == ["test", "hello"]


def test_regexp_extract_all_with_column_pattern_no_matches(local_session):
    """Test regexp_extract_all with column pattern and no matches."""
    data = {
        "text": ["no digits", "no hashtags", "no email"],
        "pattern": [r"\d+", r"#(\w+)", r"(\w+)@"],
        "idx": [0, 1, 1]
    }
    df = local_session.create_dataframe(data)

    result = df.select(
        text.regexp_extract_all(col("text"), col("pattern"), col("idx")).alias("matches")
    ).to_polars()

    matches_list = result["matches"].to_list()
    assert matches_list[0] == []
    assert matches_list[1] == []
    assert matches_list[2] == []

def test_regexp_extract_all_with_nested_capture_groups(local_session):
    """
    Test regexp_extract_all with nested capture groups.
    Pattern: r'((cat)|dog)\w+(jumps)'
    - Group 1: ((cat)|dog)
    - Group 2: (cat)
    - Group 3: (jumps)
    """

    data = {
        "text": [
            "catzzzjumps",
            "dogzzzjumps",
            "foxjumps",      # no match
            "catdogjumps",   # ambiguous, only first valid match should trigger
            "dog111jumps",
        ]
    }
    pattern = r"((cat)|dog)\w+(jumps)"

    df = local_session.create_dataframe(data)

    # Extract all group 1 matches
    result1 = df.select(
        text.regexp_extract_all("text", pattern, 1).alias("group1")
    ).to_polars()
    assert result1["group1"].to_list() == [
        ["cat"],
        ["dog"],
        [],
        ["cat"],   # "catdogjumps" group 1 = 'cat'
        ["dog"],
    ]

    # Extract all group 2 matches (may be None where (cat) did not match)
    result2 = df.select(
        text.regexp_extract_all("text", pattern, 2).alias("group2")
    ).to_polars()
    assert result2["group2"].to_list() == [
        ["cat"],
        [],    # group 2 is (cat), doesn't exist if match is 'dog'
        [],
        ["cat"],
        [],
    ]

    # Extract all group 3 matches ('jumps')
    result3 = df.select(
        text.regexp_extract_all("text", pattern, 3).alias("group3")
    ).to_polars()
    assert result3["group3"].to_list() == [
        ["jumps"],
        ["jumps"],
        [],
        ["jumps"],
        ["jumps"],
    ]

def test_regexp_extract_all_with_column_pattern_and_nulls(local_session):
    """Test regexp_extract_all with column pattern and null values."""
    data = {
        "text": ["abc123", None, "456xyz"],
        "pattern": [r"\d+", r"\d+", r"\d+"],
        "idx": [0, 0, 0]
    }
    df = local_session.create_dataframe(data)

    result = df.select(
        text.regexp_extract_all(col("text"), col("pattern"), col("idx")).alias("matches")
    ).to_polars()

    matches_list = result["matches"].to_list()
    assert matches_list[0] == ["123"]
    assert matches_list[1] is None
    assert matches_list[2] == ["456"]


def test_regexp_extract_all_with_column_idx(local_session):
    """Test regexp_extract_all with column-based group index."""
    data = {
        "text": ["user@example.com", "admin@test.org", "guest@demo.net"],
        "idx": [1, 2, 1]  # Extract different capture groups
    }
    df = local_session.create_dataframe(data)

    # Pattern: capture username and domain separately
    result = df.select(
        text.regexp_extract_all(col("text"), r"(\w+)@(\w+)", col("idx")).alias("matches")
    ).to_polars()

    matches_list = result["matches"].to_list()
    assert matches_list[0] == ["user"]    # Group 1: username
    assert matches_list[1] == ["test"]    # Group 2: domain
    assert matches_list[2] == ["guest"]   # Group 1: username


def test_regexp_extract_all_with_column_pattern_whole_match(local_session):
    """Test regexp_extract_all with column pattern extracting whole match (idx=0)."""
    data = {
        "text": ["The price is $10 and $20", "Cost: $5.99 and $15.50", "Free!"],
        "pattern": [r"\$\d+", r"\$\d+\.\d+", r"\$\d+"],
        "idx": [0, 0, 0]
    }
    df = local_session.create_dataframe(data)

    result = df.select(
        text.regexp_extract_all(col("text"), col("pattern"), col("idx")).alias("prices")
    ).to_polars()

    prices_list = result["prices"].to_list()
    assert prices_list[0] == ["$10", "$20"]
    assert prices_list[1] == ["$5.99", "$15.50"]
    assert prices_list[2] == []


def test_regexp_instr(local_session):
    """Test regexp_instr function."""
    data = {"text": ["abc123", "no digits", "456xyz", None]}
    df = local_session.create_dataframe(data)

    result = df.select(
        text.regexp_instr("text", r"\d", 0).alias("position")
    ).to_polars()

    assert result["position"][0] == 4  # 1-based position
    assert result["position"][1] == 0  # No match
    assert result["position"][2] == 1  # 1-based position
    assert result["position"][3] is None

def test_regexp_instr_with_column_pattern(local_session):
    df = local_session.create_dataframe(
        {
            "text_col": ["abc123", "no digits", "456xyz", None],
            "pattern": [r"\d", r"\d", r"[a-zA-Z]", r"\d"],
        }
    )

    result = df.with_column(
        "position", text.regexp_instr(col("text_col"), col("pattern"), 0)
    ).to_polars()
    assert result["position"][0] == 4
    assert result["position"][1] == 0
    assert result["position"][2] == 4
    assert result["position"][3] is None

def test_regexp_instr_with_column_pattern_and_idx(local_session):
    df = local_session.create_dataframe(
        {
            "text_col": ["abc123", "no digits", "456xyz", None],
            "pattern": [r"\d", r"\d", r"([a-zA-Z])([a-zA-Z])", r"\d"],
            "idx": [0, 0, 2, 0],
        }
    )

    result = df.with_column(
        "position", text.regexp_instr(col("text_col"), col("pattern"), col("idx"))
    ).to_polars()
    assert result["position"][0] == 4
    assert result["position"][1] == 0
    assert result["position"][2] == 5
    assert result["position"][3] is None


def test_regexp_instr_with_email(local_session):
    """Test regexp_instr finding email position."""
    data = {"text": ["Contact: user@domain.com", "No email here", "Email: admin@test.com"]}
    df = local_session.create_dataframe(data)

    result = df.select(
        text.regexp_instr("text", r"[^@\s]+@[^@\s]+", 0).alias("email_pos")
    ).to_polars()

    assert result["email_pos"][0] == 10  # 1-based
    assert result["email_pos"][1] == 0
    assert result["email_pos"][2] == 8  # 1-based


def test_regexp_substr(local_session):
    """Test regexp_substr function."""
    data = {"text": ["Price: $123.45", "No price", "Cost: $67.89", None]}
    df = local_session.create_dataframe(data)

    result = df.select(
        text.regexp_substr("text", r"\d+\.\d+").alias("number")
    ).to_polars()

    assert result["number"][0] == "123.45"
    assert result["number"][1] is None
    assert result["number"][2] == "67.89"
    assert result["number"][3] is None


def test_regexp_substr_url(local_session):
    """Test regexp_substr extracting URL."""
    data = {"text": ["Visit https://example.com for info", "No URL here", "See http://test.org"]}
    df = local_session.create_dataframe(data)

    result = df.select(
        text.regexp_substr("text", r"https?://[^\s]+").alias("url")
    ).to_polars()

    assert result["url"][0] == "https://example.com"
    assert result["url"][1] is None
    assert result["url"][2] == "http://test.org"

def test_regexp_invalid_pattern(local_session):
    """Test that invalid regex patterns are caught."""
    data = {"text": ["test"]}
    df = local_session.create_dataframe(data)

    with pytest.raises(ValidationError, match="Invalid regex pattern"):
        df.select(text.regexp_count("text", r"[invalid(")).to_polars()
    with pytest.raises(ValidationError, match="Invalid regex pattern"):
        df.select(text.regexp_extract("text", r"[invalid(", 0)).to_polars()
    with pytest.raises(ValidationError, match="Invalid regex pattern"):
        df.select(text.regexp_extract_all("text", r"[invalid(", 0)).to_polars()
    with pytest.raises(ValidationError, match="Invalid regex pattern"):
        df.select(text.regexp_instr("text", r"[invalid(", 0)).to_polars()
    with pytest.raises(ValidationError, match="Invalid regex pattern"):
        df.select(text.regexp_substr("text", r"[invalid(")).to_polars()

def test_regexp_invalid_pattern_with_column_pattern(local_session):
    df = local_session.create_dataframe(
        {
            "text_col": ["test"],
            "pattern": [r"[invalid("],
        }
    )
    # raises an execution time error because we cannot validate the pattern when the logical plan is being
    # constructed.
    # Functions that use Polars built-in methods produce "regex error: regex parse error:" messages
    # Functions that use our Rust plugins produce "Invalid regex pattern" messages
    with pytest.raises(ExecutionError, match="Failed to execute query: regex error: regex parse error:"):
        df.select(text.regexp_count("text_col", col("pattern"))).to_polars()
    with pytest.raises(ExecutionError, match="Failed to execute query: regex error: regex parse error:"):
        df.select(text.regexp_extract("text_col", col("pattern"), 0)).to_polars()
    with pytest.raises(ExecutionError, match="Failed to execute query: .* Invalid regex pattern .* regex parse error"):
        df.select(text.regexp_extract_all("text_col", col("pattern"), 0)).to_polars()
    # regexp_instr with literal idx uses Polars str.extract (fast path), so error comes from Polars
    with pytest.raises(ExecutionError, match="Failed to execute query: regex error: regex parse error:"):
        df.select(text.regexp_instr("text_col", col("pattern"), 0)).to_polars()
    with pytest.raises(ExecutionError, match="Failed to execute query: regex error: regex parse error:"):
        df.select(text.regexp_substr("text_col", col("pattern"))).to_polars()
