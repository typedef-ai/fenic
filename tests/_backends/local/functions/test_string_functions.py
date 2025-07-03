import polars as pl
import pytest

from fenic import (
    col,
    lit,
    text,
)


@pytest.fixture
def split_test_df(local_session):
    data = {"text_col": ["a, list, of, words", "special,,list,,of,,words"]}
    return local_session.create_dataframe(data)


def test_textract_basic(local_session):
    # Create sample data
    data = {
        "text": [
            "Name: John Doe, Age: 30",
            "Name: Jane Smith, Age: 25",
            "Name: Bob, Age: 45",
        ]
    }
    df = local_session.create_dataframe(data)

    # Define template and extract
    template = "Name: ${name:csv}, Age: ${age:none}"
    result = df.select(
        text.extract(col("text"), template).alias("_extracted")
    ).to_polars()
    df = result.select(
        pl.col("_extracted").struct.field("name"),
        pl.col("_extracted").struct.field("age"),
    )
    # Verify results
    assert len(result) == 3
    assert df["name"].to_list() == ["John Doe", "Jane Smith", "Bob"]
    assert df["age"].to_list() == ["30", "25", "45"]


def test_textract_filter(local_session):
    data = {
        "text": [
            "Name: John Doe, Age: 30",
            "Name: Jane Smith, Age: 25",
            "Name: Bob, Age: 45",
            "Name: Alice, Age: 30",
        ]
    }
    df = local_session.create_dataframe(data)
    template = "Name: ${name:csv}, Age: ${age:none}"

    df = df.select(
        col("text"), text.extract(col("text"), template).alias("_extracted")
    ).to_polars()

    unnested = df.select(
        pl.col("_extracted").struct.field("name"),
        pl.col("_extracted").struct.field("age"),
    )
    unnested = unnested.filter(pl.col("age") == "30")
    # Expect rows for "John Doe" and "Alice".
    assert unnested["name"].to_list() == ["John Doe", "Alice"]
    assert unnested["age"].to_list() == ["30", "30"]


def test_textract_empty_input(local_session):
    """Ensure that `text.extract` gracefully handles empty strings by returning a
    struct with ``None`` values for all extracted fields instead of raising a
    conversion error (regression test for tuple vs. dict issue).
    """
    data = {
        "text": [
            "",  # empty string – should not crash
            "Name: Alice, Age: 22",  # normal well-formed row for control
        ]
    }
    df = local_session.create_dataframe(data)

    template = "Name: ${name:csv}, Age: ${age:none}"

    extracted_df = df.select(
        text.extract(col("text"), template).alias("_extracted")
    ).to_polars()

    # Row 0: expect None for both fields
    assert extracted_df["_extracted"][0]["name"] is None
    assert extracted_df["_extracted"][0]["age"] is None

    # Row 1: verify correct extraction for a valid row (sanity check)
    assert extracted_df["_extracted"][1]["name"] == "Alice"
    assert extracted_df["_extracted"][1]["age"] == "22"

def test_concat(local_session):
    data = {
        "col1": ["Hello", "World"],
        "col2": ["Spark", "SQL"],
        "col3": [True, False],
        "col4": [1, 2],
    }
    df = local_session.create_dataframe(data)
    result = df.select(
        text.concat(
            col("col1"),
            lit(" "),
            col("col2"),
            lit(" "),
            col("col3"),
            lit(" "),
            col("col4"),
        ).alias("concat_result")
    ).to_polars()
    assert result["concat_result"].to_list() == [
        "Hello Spark true 1",
        "World SQL false 2",
    ]


def test_concat_ws(local_session):
    data = {
        "col1": ["Hello", "World"],
        "col2": ["Spark", "SQL"],
        "col3": [True, False],
        "col4": [1, 2],
    }
    df = local_session.create_dataframe(data)
    result = df.select(
        text.concat_ws(" ", col("col1"), col("col2"), col("col3"), col("col4")).alias(
            "concat_ws_result"
        )
    ).to_polars()
    assert result["concat_ws_result"].to_list() == [
        "Hello Spark true 1",
        "World SQL false 2",
    ]


def test_array_join(local_session):
    data = {"col1": [["Hello", "World"], ["Spark", "SQL"]]}
    df = local_session.create_dataframe(data)
    result = df.select(
        text.array_join(col("col1"), ",").alias("array_join_result")
    ).to_polars()
    assert result["array_join_result"].to_list() == ["Hello,World", "Spark,SQL"]


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


def test_replace_basic(replace_test_df):
    result = replace_test_df.with_column(
        "replaced_text_col", text.replace(col("text_col"), "hello", "hi")
    ).to_polars()
    assert result["replaced_text_col"][0] == "hi world"
    assert result["replaced_text_col"][1] == "hi hi world"
    assert result["replaced_text_col"][2] == "world hi world"
    assert result["replaced_text_col"][3] == "no matches here"


def test_replace_with_empty_string(replace_test_df):
    result = replace_test_df.with_column(
        "replaced_text_col", text.replace(col("text_col"), "hello", "")
    ).to_polars()
    assert result["replaced_text_col"][0] == " world"
    assert result["replaced_text_col"][1] == "  world"
    assert result["replaced_text_col"][2] == "world  world"


def test_replace_no_matches(replace_test_df):
    result = replace_test_df.with_column(
        "replaced_text_col", text.replace(col("text_col"), "xyz", "abc")
    ).to_polars()
    # Should return original strings when no matches
    assert result["replaced_text_col"][0] == "hello world"
    assert result["replaced_text_col"][1] == "hello hello world"


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


def test_replace_with_column_pattern(local_session):
    df = local_session.create_dataframe(
        {
            "text_col": ["hello world", "goodbye world", "hello earth"],
            "pattern": ["hello", "goodbye", "earth"],
            "replacement": ["hi", "bye", "planet"],
        }
    )
    result = df.with_column(
        "replaced_text_col",
        text.replace(col("text_col"), col("pattern"), col("replacement")),
    ).to_polars()
    assert result["replaced_text_col"][0] == "hi world"
    assert result["replaced_text_col"][1] == "bye world"
    assert result["replaced_text_col"][2] == "hello planet"


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


def test_split(split_test_df):
    result = split_test_df.with_column(
        "split_text_col", text.split(col("text_col"), ",")
    ).to_polars()
    assert result["split_text_col"][0].to_list() == ["a", " list", " of", " words"]
    assert result["split_text_col"][1].to_list() == [
        "special",
        "",
        "list",
        "",
        "of",
        "",
        "words",
    ]


def test_split_limit(split_test_df):
    result_with_limit = split_test_df.with_column(
        "split_text_col", text.split(col("text_col"), ",", limit=2)
    ).to_polars()
    assert result_with_limit["split_text_col"][0].to_list() == ["a", " list, of, words"]
    assert result_with_limit["split_text_col"][1].to_list() == [
        "special",
        ",list,,of,,words",
    ]


def test_split_part_basic(split_test_df):
    result = split_test_df.with_column(
        "split_text_col", text.split_part(col("text_col"), ",", 2)
    ).to_polars()
    assert result["split_text_col"][0] == " list"
    assert result["split_text_col"][1] == ""


def test_split_part_out_of_range(split_test_df):
    result = split_test_df.with_column(
        "split_text_col", text.split_part(col("text_col"), ",", 10)
    ).to_polars()
    assert result["split_text_col"][0] == ""
    assert result["split_text_col"][1] == ""


def test_split_part_negative_index(split_test_df):
    result = split_test_df.with_column(
        "split_text_col", text.split_part(col("text_col"), ",", -1)
    ).to_polars()
    assert result["split_text_col"][0] == " words"
    assert result["split_text_col"][1] == "words"


def test_split_part_no_matches(split_test_df):
    result = split_test_df.with_column(
        "split_text_col", text.split_part(col("text_col"), ", ", 1)
    ).to_polars()
    assert result["split_text_col"][0] == "a"
    assert result["split_text_col"][1] == "special,,list,,of,,words"


def test_split_part_zero_index_error(split_test_df):
    with pytest.raises(ValueError):
        split_test_df.with_column(
            "split_text_col", text.split_part(col("text_col"), ", ", 0)
        ).to_polars()


def test_split_part_with_column_delimiter(local_session):
    df = local_session.create_dataframe(
        {"text_col": ["a,b,c", "x:y:z", "1|2|3"], "delimiter": [",", ":", "|"]}
    )
    result = df.with_column(
        "split_text_col", text.split_part(col("text_col"), col("delimiter"), 2)
    ).to_polars()
    assert result["split_text_col"][0] == "b"
    assert result["split_text_col"][1] == "y"
    assert result["split_text_col"][2] == "2"


def test_split_part_with_column_part_number(local_session):
    df = local_session.create_dataframe(
        {"text_col": ["a,b,c", "d,e,f", "g,h,i"], "part_number": [1, 2, 3]}
    )
    result = df.with_column(
        "split_text_col", text.split_part(col("text_col"), ",", col("part_number"))
    ).to_polars()
    assert result["split_text_col"][0] == "a"
    assert result["split_text_col"][1] == "e"
    assert result["split_text_col"][2] == "i"


def test_split_part_with_both_column_delimiter_and_part(local_session):
    df = local_session.create_dataframe(
        {
            "text_col": ["a,b,c", "x:y:z", "1|2|3"],
            "delimiter": [",", ":", "|"],
            "part_number": [1, 2, 3],
        }
    )
    result = df.with_column(
        "split_text_col",
        text.split_part(col("text_col"), col("delimiter"), col("part_number")),
    ).to_polars()
    assert result["split_text_col"][0] == "a"
    assert result["split_text_col"][1] == "y"
    assert result["split_text_col"][2] == "3"


def test_split_part_with_column_negative_part(local_session):
    df = local_session.create_dataframe(
        {"text_col": ["a,b,c", "d,e,f", "g,h,i"], "part_number": [-1, -2, -3]}
    )
    result = df.with_column(
        "split_text_col", text.split_part(col("text_col"), ",", col("part_number"))
    ).to_polars()
    assert result["split_text_col"][0] == "c"
    assert result["split_text_col"][1] == "e"
    assert result["split_text_col"][2] == "g"


def test_split_part_with_column_out_of_range(local_session):
    df = local_session.create_dataframe(
        {"text_col": ["a,b,c", "d,e,f"], "part_number": [4, -4]}
    )
    result = df.with_column(
        "split_text_col", text.split_part(col("text_col"), ",", col("part_number"))
    ).to_polars()
    # Should return empty string for out of range indices
    assert result["split_text_col"][0] == ""
    assert result["split_text_col"][1] == ""


def test_str_length(local_session):
    data = {"text_col": ["hello", "world", "foo", "bar"]}
    df = local_session.create_dataframe(data)
    result = df.with_column("length_col", text.length(col("text_col"))).to_polars()
    assert result["length_col"][0] == 5
    assert result["length_col"][1] == 5
    assert result["length_col"][2] == 3
    assert result["length_col"][3] == 3


def test_byte_length(local_session):
    data = {"text_col": ["hello", "world", "foo", "bar"]}
    df = local_session.create_dataframe(data)
    result = df.with_column(
        "length_col", text.byte_length(col("text_col"))
    ).to_polars()
    assert result["length_col"][0] == 5
    assert result["length_col"][1] == 5
    assert result["length_col"][2] == 3
    assert result["length_col"][3] == 3


def test_upper(local_session):
    data = {"text_col": ["hello", "world", "foo", "bar"]}
    df = local_session.create_dataframe(data)
    result = df.with_column("upper_col", text.upper(col("text_col"))).to_polars()
    assert result["upper_col"][0] == "HELLO"
    assert result["upper_col"][1] == "WORLD"
    assert result["upper_col"][2] == "FOO"
    assert result["upper_col"][3] == "BAR"


def test_lower(local_session):
    data = {"text_col": ["HELLO", "WORLD", "FOO", "BAR"]}
    df = local_session.create_dataframe(data)
    result = df.with_column("lower_col", text.lower(col("text_col"))).to_polars()
    assert result["lower_col"][0] == "hello"
    assert result["lower_col"][1] == "world"
    assert result["lower_col"][2] == "foo"
    assert result["lower_col"][3] == "bar"


def test_title_case(local_session):
    data = {"text_col": ["hello", "world", "foo", "bar"]}
    df = local_session.create_dataframe(data)
    result = df.with_column("title_col", text.title_case(col("text_col"))).to_polars()
    assert result["title_col"][0] == "Hello"
    assert result["title_col"][1] == "World"
    assert result["title_col"][2] == "Foo"
    assert result["title_col"][3] == "Bar"


@pytest.fixture
def basic_trim_df(local_session):
    data = {"text_col": ["hello", "world", "foo", "bar"]}
    return local_session.create_dataframe(data)


@pytest.fixture
def whitespace_trim_df(local_session):
    data = {"text_col": ["   hello   ", "   world   ", "foo\n\t", "\n\tbar"]}
    return local_session.create_dataframe(data)


def test_btrim(basic_trim_df):
    result = basic_trim_df.with_column(
        "stripped_text_col", text.btrim(col("text_col"), "ho")
    ).to_polars()
    assert result["stripped_text_col"][0] == "ell"
    assert result["stripped_text_col"][1] == "world"
    assert result["stripped_text_col"][2] == "f"
    assert result["stripped_text_col"][3] == "bar"


def test_btrim_with_column_chars(local_session):
    data = {
        "text_col": ["hello", "world", "foo", "bar"],
        "chars": ["ho", "o", "o", "r"],
    }
    df = local_session.create_dataframe(data)
    result = df.with_column(
        "stripped_text_col", text.btrim(col("text_col"), col("chars"))
    ).to_polars()
    assert result["stripped_text_col"][0] == "ell"
    assert result["stripped_text_col"][1] == "world"
    assert result["stripped_text_col"][2] == "f"
    assert result["stripped_text_col"][3] == "ba"


def test_ltrim(whitespace_trim_df):
    result = whitespace_trim_df.with_column(
        "stripped_text_col", text.ltrim(col("text_col"))
    ).to_polars()
    assert result["stripped_text_col"][0] == "hello   "
    assert result["stripped_text_col"][1] == "world   "
    assert result["stripped_text_col"][2] == "foo\n\t"
    assert result["stripped_text_col"][3] == "bar"


def test_rtrim(whitespace_trim_df):
    result = whitespace_trim_df.with_column(
        "stripped_text_col", text.rtrim(col("text_col"))
    ).to_polars()
    assert result["stripped_text_col"][0] == "   hello"
    assert result["stripped_text_col"][1] == "   world"
    assert result["stripped_text_col"][2] == "foo"
    assert result["stripped_text_col"][3] == "\n\tbar"


def test_trim_whitespace(whitespace_trim_df):
    result = whitespace_trim_df.with_column(
        "stripped_text_col", text.trim(col("text_col"))
    ).to_polars()
    assert result["stripped_text_col"][0] == "hello"
    assert result["stripped_text_col"][1] == "world"
    assert result["stripped_text_col"][2] == "foo"
    assert result["stripped_text_col"][3] == "bar"
