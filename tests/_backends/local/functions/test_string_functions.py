import json

import pytest

from fenic import col, lit, text
from fenic import json as json_fc
from fenic.core.error import ValidationError
from fenic.core.types.datatypes import JsonType, StringType
from fenic.core.types.schema import ColumnField


@pytest.fixture
def split_test_df(local_session):
    data = {"text_col": ["a, list, of, words", "special,,list,,of,,words"]}
    return local_session.create_dataframe(data)


def test_textract_basic(local_session):
    """Basic text extraction with simple template."""
    data = {
        "text": [
            "Name: John Doe, Age: 30",
            "Name: Jane Smith, Age: 25",
            "Name: Bob, Age: 45",
        ]
    }
    df = local_session.create_dataframe(data)

    template = "Name: ${name:csv}, Age: ${age:none}"
    result = df.select(
        text.extract(col("text"), template).alias("_extracted")
    ).unnest("_extracted").to_polars()

    assert len(result) == 3
    assert result["name"].to_list() == ["John Doe", "Jane Smith", "Bob"]
    assert result["age"].to_list() == ["30", "25", "45"]


def test_textract_complex_log_parsing(local_session):
    """Complex log parsing with multiple escaping rules."""
    data = {
        "log_line": [
            '[2024-01-15 10:30:45] "ERROR" - DatabaseManager: "Connection timeout occurred" {"retry_count": 3, "duration_ms": 5000}',
            '[2024-01-15 10:31:02] "INFO" - AuthService: "User login successful" {"user_id": 12345, "session_duration": 1800}',
            '[2024-01-15 10:31:15] "WARN" - PaymentProcessor: "Rate limit ""exceeded""" {"requests_per_minute": 150, "limit": 100}',
            '[2024-01-15 10:31:30] "DEBUG" - CacheManager: "Cache miss for key" {"cache_hit_ratio": 0.85, "keys_checked": 200}',
        ]
    }
    df = local_session.create_dataframe(data)

    template = '[${timestamp}] ${level:csv} - ${component}: ${message:csv} ${metadata:json}'
    df = df.select(
        text.extract(col("log_line"), template).alias("parsed")
    ).unnest("parsed")
    df = df.with_column("metadata", json_fc.jq(col("metadata"), ".")[0])
    assert df.schema.column_fields == [
        ColumnField(name="timestamp", data_type=StringType),
        ColumnField(name="level", data_type=StringType),
        ColumnField(name="component", data_type=StringType),
        ColumnField(name="message", data_type=StringType),
        ColumnField(name="metadata", data_type=JsonType),
    ]
    result = df.to_polars()

    # Test timestamp extraction
    expected_timestamps = ["2024-01-15 10:30:45", "2024-01-15 10:31:02", "2024-01-15 10:31:15", "2024-01-15 10:31:30"]
    assert result["timestamp"].to_list() == expected_timestamps

    # Test CSV-quoted level extraction
    expected_levels = ["ERROR", "INFO", "WARN", "DEBUG"]
    assert result["level"].to_list() == expected_levels

    # Test component extraction
    expected_components = ["DatabaseManager", "AuthService", "PaymentProcessor", "CacheManager"]
    assert result["component"].to_list() == expected_components

    # Test CSV-quoted message extraction with escaped quotes
    expected_messages = ["Connection timeout occurred", "User login successful", 'Rate limit "exceeded"', "Cache miss for key"]
    assert result["message"].to_list() == expected_messages

    metadata_dicts = [json.loads(item) if item is not None else None for item in result["metadata"].to_list()]

    assert metadata_dicts[0]["retry_count"] == 3
    assert metadata_dicts[0]["duration_ms"] == 5000

    assert metadata_dicts[1]["user_id"] == 12345
    assert metadata_dicts[1]["session_duration"] == 1800

    assert metadata_dicts[2]["requests_per_minute"] == 150
    assert metadata_dicts[2]["limit"] == 100

    assert metadata_dicts[3]["cache_hit_ratio"] == 0.85
    assert metadata_dicts[3]["keys_checked"] == 200

def test_textract_csv_quoting_variations(local_session):
    """CSV quoting with escaped quotes and edge cases."""
    data = {
        "text": [
            'Name: "John Doe", Age: 30',
            'Name: "John ""Big John"" Doe", Age: 30',
            'Name: John Doe, Age: 30',
            'Name: "", Age: 30',
            'Name: "Quote at end", Age: 30',
        ]
    }
    df = local_session.create_dataframe(data)

    template = "Name: ${name:csv}, Age: ${age}"
    result = df.select(
        text.extract(col("text"), template).alias("parsed")
    ).unnest("parsed").to_polars()

    names = result["name"].to_list()
    assert names[0] == "John Doe"
    assert names[1] == 'John "Big John" Doe'
    assert names[2] == "John Doe"
    assert names[3] == ""
    assert names[4] == "Quote at end"


def test_textract_quoted_field_parsing(local_session):
    """QUOTED escaping rule with strict quote requirements."""
    data = {
        "text": [
            'Message: "Simple text" End',
            'Message: "Text with ""escaped"" quotes" End',
            'Message: "" End',
            'Message: "Multi\nline\ntext" End',
            'Message: unquoted text End',  # Should fail
        ]
    }
    df = local_session.create_dataframe(data)

    template = "Message: ${message:quoted} End"
    result = df.select(
        text.extract(col("text"), template).alias("parsed")
    ).unnest("parsed").to_polars()

    messages = result["message"].to_list()
    assert messages[0] == "Simple text"
    assert messages[1] == 'Text with "escaped" quotes'
    assert messages[2] == ""
    assert "Multi" in messages[3] and "text" in messages[3]
    assert messages[4] is None  # Unquoted should fail


def test_textract_json_field_validation(local_session):
    """JSON field parsing with valid and invalid JSON."""
    data = {
        "log": [
            'Data: {"valid": true}',
            'Data: [1, 2, 3]',
            'Data: "string"',
            'Data: 42',
            'Data: true',
            'Data: null',
            'Data: {invalid_json}',
            'Data: {"unclosed": true',
        ]
    }
    df = local_session.create_dataframe(data)

    template = "Data: ${data:json}"
    result = df.select(
        text.extract(col("log"), template).alias("parsed")
    ).unnest("parsed").to_polars()

    json_data = result["data"].to_list()
    # Valid JSON preserved
    assert json_data[0] is not None
    assert json_data[1] is not None
    assert json_data[2] is not None
    assert json_data[3] is not None
    assert json_data[4] is not None
    assert json_data[5] is not None
    # Invalid JSON returns None
    assert json_data[6] is None
    assert json_data[7] is None


def test_textract_dollar_escaping(local_session):
    """Dollar sign escaping with $$ and $$$ patterns."""
    data = {
        "price_text": [
            "Price: $29.99, Tax: $3.50",
            "Price: $150.00, Tax: $15.00",
            "Price: $0.99, Tax: $0.10",
        ]
    }
    df = local_session.create_dataframe(data)

    # Template using $$$ pattern: $$ (escaped $) + ${variable}
    template = "Price: $$${price}, Tax: $$${tax}"
    result = df.select(
        text.extract(col("price_text"), template).alias("parsed")
    ).unnest("parsed").to_polars()

    assert len(result) == 3
    assert result["price"].to_list() == ["29.99", "150.00", "0.99"]
    assert result["tax"].to_list() == ["3.50", "15.00", "0.10"]


def test_textract_multiple_dollar_escaping(local_session):
    """Various dollar escaping combinations."""
    data = {
        "text": [
            "Cost: $5.99 for item xyz",
            "Amount: $$10.00 total",
            "Price: $10.00 plus $2.00",
            "Total: $$12.00 final",
        ]
    }
    df = local_session.create_dataframe(data)

    test_cases = [
        ("Cost: $$5.99 for item ${item}", 0, "xyz"),
        ("Amount: $$$$${amount} total", 1, "10.00"),
        ("Price: $$${price} plus $$${tax}", 2, "10.00"),
        ("Total: $$$$${total} final", 3, "12.00"),
    ]

    for template, row_idx, expected_value in test_cases:
        result = df.select(
            text.extract(col("text"), template).alias("parsed")
        ).unnest("parsed").to_polars()

        # Get the first column name (variable name)
        col_name = result.columns[0]
        actual_value = result[col_name].to_list()[row_idx]
        assert actual_value == expected_value


def test_textract_complex_delimiters(local_session):
    """Multi-character and unusual delimiters."""
    data = {
        "structured_text": [
            "START>>user:john_doe<<>>email:john@example.com<<END",
            "START>>user:jane_smith<<>>email:jane@company.org<<END",
            "START>>user:bob_wilson<<>>email:bob@domain.net<<END",
        ]
    }
    df = local_session.create_dataframe(data)

    template = "START>>${user}<<>>${email}<<END"
    result = df.select(
        text.extract(col("structured_text"), template).alias("parsed")
    ).unnest("parsed").to_polars()

    assert result["user"].to_list() == ["user:john_doe", "user:jane_smith", "user:bob_wilson"]
    assert result["email"].to_list() == ["email:john@example.com", "email:jane@company.org", "email:bob@domain.net"]

def test_textract_newline_handling(local_session):
    """Newlines in both template and data are handled as regular content."""
    data = {
        "multiline_text": [
            "Name: John Doe\nAge: 30\nCity: Boston",
            "Name: Jane Smith\nAge: 25\nCity: New York",
            "Name: Bob Wilson\nAge: 45\nCity: Chicago",
        ]
    }
    df = local_session.create_dataframe(data)

    # Template with newline delimiters
    template = "Name: ${name}\nAge: ${age}\nCity: ${city}"
    result = df.select(
        text.extract(col("multiline_text"), template).alias("parsed")
    ).unnest("parsed").to_polars()

    assert len(result) == 3
    assert result["name"].to_list() == ["John Doe", "Jane Smith", "Bob Wilson"]
    assert result["age"].to_list() == ["30", "25", "45"]
    assert result["city"].to_list() == ["Boston", "New York", "Chicago"]

def test_textract_malformed_input_cases(local_session):
    """Various malformed and edge case inputs."""
    data = {
        "text": [
            "Name: Alice, Age: 28",  # Valid
            "Name: Bob Age: 30",  # Missing comma
            "No match at all",  # No pattern
            "Name: Charlie,",  # Missing age
            "",  # Empty
            "   ",  # Whitespace only
            "Name: Dave, Age: ",  # Empty age
            "Name: Eve, Age: 25, Extra: stuff",  # Extra content
        ]
    }
    df = local_session.create_dataframe(data)

    template = "Name: ${name:csv}, Age: ${age:none}"
    result = df.select(
        text.extract(col("text"), template).alias("parsed")
    ).unnest("parsed").to_polars()

    names = result["name"].to_list()
    ages = result["age"].to_list()

    assert names[0] == "Alice" and ages[0] == "28"  # Valid
    assert names[1] is None and ages[1] is None  # Missing comma
    assert names[2] is None and ages[2] is None  # No match
    assert names[3] is None and ages[3] is None  # Missing age part
    assert names[4] is None and ages[4] is None  # Empty
    assert names[5] is None and ages[5] is None  # Whitespace
    assert names[6] == "Dave" and ages[6] == ""  # Empty age valid
    assert names[7] == "Eve" and ages[7] == "25, Extra: stuff"  # Extra content


def test_textract_whitespace_handling(local_session):
    """Whitespace trimming and empty field handling."""
    data = {
        "text": [
            "Name: , Age: 30",
            "Name:    , Age: 25",
            "Name: John, Age:   ",
            "Name: John , Age: 30 ",
            "Name:  John  , Age:  25  ",
        ]
    }
    df = local_session.create_dataframe(data)

    template = "Name: ${name}, Age: ${age}"
    result = df.select(
        text.extract(col("text"), template).alias("parsed")
    ).unnest("parsed").to_polars()

    names = result["name"].to_list()
    ages = result["age"].to_list()

    assert names[0] == ""
    assert names[1] == ""
    assert names[2] == "John"
    assert names[3] == "John"
    assert names[4] == "John"

    assert ages[0] == "30"
    assert ages[1] == "25"
    assert ages[2] == ""
    assert ages[3] == "30"
    assert ages[4] == "25"


def test_textract_csv_vs_quoted_comparison(local_session):
    """Direct comparison of CSV vs QUOTED escaping rules."""
    data = {
        "text": [
            'Field: "quoted value"',
            'Field: unquoted value',
            'Field: "value with ""quotes"""',
            'Field: ""',
        ]
    }
    df = local_session.create_dataframe(data)

    # CSV (flexible)
    csv_result = df.select(
        text.extract(col("text"), "Field: ${value:csv}").alias("parsed")
    ).unnest("parsed").to_polars()

    # QUOTED (strict)
    quoted_result = df.select(
        text.extract(col("text"), "Field: ${value:quoted}").alias("parsed")
    ).unnest("parsed").to_polars()

    csv_values = csv_result["value"].to_list()
    quoted_values = quoted_result["value"].to_list()

    # CSV handles all cases
    assert csv_values[0] == "quoted value"
    assert csv_values[1] == "unquoted value"
    assert csv_values[2] == 'value with "quotes"'
    assert csv_values[3] == ""

    # QUOTED only handles quoted cases
    assert quoted_values[0] == "quoted value"
    assert quoted_values[1] is None  # Fails - no quotes
    assert quoted_values[2] == 'value with "quotes"'
    assert quoted_values[3] == ""


def test_textract_nested_delimiter_patterns(local_session):
    """Complex nested and similar delimiter patterns."""
    data = {
        "text": [
            "{{user: john_doe}}, other: data",
            "[[item: content]], type: info",
            "((value: text)), status: ok",
            "BEGIN${content}$END, flag: true",
        ]
    }
    df = local_session.create_dataframe(data)

    templates_and_expected = [
        ("{{${user}}}, ${other}: ${value}", ["user: john_doe", "other", "data"]),
        ("[[${item}]], ${type}: ${info}", ["item: content", "type", "info"]),
        ("((${value})), ${status}: ${state}", ["value: text", "status", "ok"]),
        ("BEGIN${content}$$END, ${flag}: ${state}", ["${content}", "flag", "true"]),
    ]

    for i, (template, expected_values) in enumerate(templates_and_expected):
        result = df.select(
            text.extract(col("text"), template).alias("parsed")
        ).unnest("parsed").to_polars()

        actual_values = [result[col].to_list()[i] for col in result.columns]
        assert actual_values == expected_values


def test_textract_template_parsing_errors(local_session):
    """Malformed templates should raise meaningful errors."""
    data = {"text": ["Name: John, Age: 30"]}
    df = local_session.create_dataframe(data)

    malformed_templates = [
        "Name: ${name, Age: ${age}",  # Missing closing brace
        "Name: $name}, Age: ${age}",  # Missing opening brace
        "Name: ${}, Age: ${age}",  # Empty variable name
        "Name: ${name:}, Age: ${age}",  # Empty format specifier
        "Name: ${name:invalid}, Age: ${age}",  # Invalid format
    ]

    for template in malformed_templates:
        with pytest.raises(ValidationError):
            df.select(
                text.extract(col("text"), template).alias("parsed")
            ).unnest("parsed").to_polars()


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
    with pytest.raises(ValidationError):
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
