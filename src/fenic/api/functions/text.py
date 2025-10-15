"""Text manipulation functions for Fenic DataFrames."""

from typing import List, Optional, Union

from pydantic import ConfigDict, validate_call

from fenic.api.column import Column, ColumnOrName
from fenic.api.functions.core import lit
from fenic.core._logical_plan.expressions import (
    ArrayJoinExpr,
    ByteLengthExpr,
    ChunkCharacterSet,
    ChunkLengthFunction,
    ColumnExpr,
    ConcatExpr,
    CountTokensExpr,
    FuzzyRatioExpr,
    FuzzyTokenSetRatioExpr,
    FuzzyTokenSortRatioExpr,
    JinjaExpr,
    LiteralExpr,
    LogicalExpr,
    RecursiveTextChunkExpr,
    RegexpCountExpr,
    RegexpExtractAllExpr,
    RegexpExtractExpr,
    RegexpInstrExpr,
    RegexpSplitExpr,
    RegexpSubstrExpr,
    ReplaceExpr,
    SplitPartExpr,
    StringCasingExpr,
    StripCharsExpr,
    StrLengthExpr,
    TextChunkExpr,
    TextractExpr,
    TsParseExpr,
)
from fenic.core._logical_plan.expressions.text import (
    RecursiveTextChunkExprConfiguration,
    TextChunkExprConfiguration,
)
from fenic.core.error import ValidationError
from fenic.core.types import StringType
from fenic.core.types.enums import (
    FuzzySimilarityMethod,
    TranscriptFormatType,
)


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def extract(column: ColumnOrName, template: str) -> Column:
    """Extracts structured data from text using template-based pattern matching.

    Matches each string in the input column against a template pattern with named
    placeholders. Each placeholder can specify a format rule to handle different
    data types within the text.

    Args:
        column: Input text column to extract from
        template: Template string with placeholders as ``${field_name}`` or ``${field_name:format}``
                 Available formats: none, csv, json, quoted

    Returns:
        Column: Struct column with fields corresponding to template placeholders.
                All fields are strings except JSON fields which preserve their parsed type.

    Template Syntax:
        - ``${field_name}`` - Extract field as plain text
        - ``${field_name:csv}`` - Parse as CSV field (handles quoted values)
        - ``${field_name:json}`` - Parse as JSON and preserve type
        - ``${field_name:quoted}`` - Extract quoted string (removes outer quotes)
        - ``$`` - Literal dollar sign

    Raises:
        ValidationError: If template syntax is invalid

    Example: Basic extraction
        ```python
        text.extract(col("log"), "${date} ${level} ${message}")
        # Input: "2024-01-15 ERROR Connection failed"
        # Output: {date: "2024-01-15", level: "ERROR", message: "Connection failed"}
        ```

    Example: Mixed format extraction
        ```python
        text.extract(col("data"), 'Name: ${name:csv}, Price: ${price}, Tags: ${tags:json}')
        # Input: 'Name: "Smith, John", Price: 99.99, Tags: ["a", "b"]'
        # Output: {name: "Smith, John", price: "99.99", tags: ["a", "b"]}
        ```

    Example: Quoted field handling
        ```python
        text.extract(col("record"), 'Title: ${title:quoted}, Author: ${author}')
        # Input: 'Title: "To Kill a Mockingbird", Author: Harper Lee'
        # Output: {title: "To Kill a Mockingbird", author: "Harper Lee"}
        ```

    Note:
        If a string doesn't match the template pattern, all extracted fields will be null.
    """
    return Column._from_logical_expr(
        TextractExpr(Column._from_col_or_name(column)._logical_expr, template)
    )

@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def recursive_character_chunk(
    column: ColumnOrName,
    chunk_size: int,
    chunk_overlap_percentage: int,
    chunking_character_set_custom_characters: Optional[list[str]] = None,
) -> Column:
    r"""Chunks a string column into chunks of a specified size (in characters) with an optional overlap.

    The chunking is performed recursively, attempting to preserve the underlying structure of the text
    by splitting on natural boundaries (paragraph breaks, sentence breaks, etc.) to maintain context.
    By default, these characters are ['\n\n', '\n', '.', ';', ':', ' ', '-', ''], but this can be customized.

    Args:
        column: The input string column or column name to chunk
        chunk_size: The size of each chunk in characters
        chunk_overlap_percentage: The overlap between each chunk as a percentage of the chunk size
        chunking_character_set_custom_characters (Optional): List of alternative characters to split on. Note that the characters should be ordered from coarsest to finest desired granularity -- earlier characters in the list should result in fewer overall splits than later characters.

    Returns:
        Column: A column containing the chunks as an array of strings

    Example: Default character chunking
        ```python
        # Create chunks of at most 100 characters with 20% overlap
        df.select(
            text.recursive_character_chunk(col("text"), 100, 20).alias("chunks")
        )
        ```

    Example: Custom character chunking
        ```python
        # Create chunks with custom split characters
        df.select(
            text.recursive_character_chunk(
                col("text"),
                100,
                20,
                ['\n\n', '\n', '.', ' ', '']
            ).alias("chunks")
        )
        ```
    """
    if chunking_character_set_custom_characters is None:
        chunking_character_set_name = ChunkCharacterSet.ASCII
    else:
        chunking_character_set_name = ChunkCharacterSet.CUSTOM

    return Column._from_logical_expr(
        RecursiveTextChunkExpr(
            Column._from_col_or_name(column)._logical_expr,
            chunking_configuration=RecursiveTextChunkExprConfiguration(
                desired_chunk_size=chunk_size,
                chunk_overlap_percentage=chunk_overlap_percentage,
                chunk_length_function_name=ChunkLengthFunction.CHARACTER,
                chunking_character_set_name=chunking_character_set_name,
                chunking_character_set_custom_characters=chunking_character_set_custom_characters,
            )
        )
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def recursive_word_chunk(
    column: ColumnOrName,
    chunk_size: int,
    chunk_overlap_percentage: int,
    chunking_character_set_custom_characters: Optional[list[str]] = None,
) -> Column:
    r"""Chunks a string column into chunks of a specified size (in words) with an optional overlap.

    The chunking is performed recursively, attempting to preserve the underlying structure of the text
    by splitting on natural boundaries (paragraph breaks, sentence breaks, etc.) to maintain context.
    By default, these characters are ['\n\n', '\n', '.', ';', ':', ' ', '-', ''], but this can be customized.

    Args:
        column: The input string column or column name to chunk
        chunk_size: The size of each chunk in words
        chunk_overlap_percentage: The overlap between each chunk as a percentage of the chunk size
        chunking_character_set_custom_characters (Optional): List of alternative characters to split on. Note that the characters should be ordered from coarsest to finest desired granularity -- earlier characters in the list should result in fewer overall splits than later characters.

    Returns:
        Column: A column containing the chunks as an array of strings

    Example: Default word chunking
        ```python
        # Create chunks of at most 100 words with 20% overlap
        df.select(
            text.recursive_word_chunk(col("text"), 100, 20).alias("chunks")
        )
        ```

    Example: Custom word chunking
        ```python
        # Create chunks with custom split characters
        df.select(
            text.recursive_word_chunk(
                col("text"),
                100,
                20,
                ['\n\n', '\n', '.', ' ', '']
            ).alias("chunks")
        )
        ```
    """
    if chunking_character_set_custom_characters is None:
        chunking_character_set_name = ChunkCharacterSet.ASCII
    else:
        chunking_character_set_name = ChunkCharacterSet.CUSTOM

    return Column._from_logical_expr(
        RecursiveTextChunkExpr(
            Column._from_col_or_name(column)._logical_expr,
            chunking_configuration=RecursiveTextChunkExprConfiguration(
                desired_chunk_size=chunk_size,
                chunk_overlap_percentage=chunk_overlap_percentage,
                chunk_length_function_name=ChunkLengthFunction.WORD,
                chunking_character_set_name=chunking_character_set_name,
                chunking_character_set_custom_characters=chunking_character_set_custom_characters,
            )
        )
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def recursive_token_chunk(
    column: ColumnOrName,
    chunk_size: int,
    chunk_overlap_percentage: int,
    chunking_character_set_custom_characters: Optional[list[str]] = None,
) -> Column:
    r"""Chunks a string column into chunks of a specified size (in tokens) with an optional overlap.

    The chunking is performed recursively, attempting to preserve the underlying structure of the text
    by splitting on natural boundaries (paragraph breaks, sentence breaks, etc.) to maintain context.
    By default, these characters are ['\n\n', '\n', '.', ';', ':', ' ', '-', ''], but this can be customized.

    Args:
        column: The input string column or column name to chunk
        chunk_size: The size of each chunk in tokens
        chunk_overlap_percentage: The overlap between each chunk as a percentage of the chunk size
        chunking_character_set_custom_characters (Optional): List of alternative characters to split on. Note that the characters should be ordered from coarsest to finest desired granularity -- earlier characters in the list should result in fewer overall splits than later characters.

    Returns:
        Column: A column containing the chunks as an array of strings

    Example: Default token chunking
        ```python
        # Create chunks of at most 100 tokens with 20% overlap
        df.select(
            text.recursive_token_chunk(col("text"), 100, 20).alias("chunks")
        )
        ```

    Example: Custom token chunking
        ```python
        # Create chunks with custom split characters
        df.select(
            text.recursive_token_chunk(
                col("text"),
                100,
                20,
                ['\n\n', '\n', '.', ' ', '']
            ).alias("chunks")
        )
        ```
    """
    if chunking_character_set_custom_characters is None:
        chunking_character_set_name = ChunkCharacterSet.ASCII
    else:
        chunking_character_set_name = ChunkCharacterSet.CUSTOM

    return Column._from_logical_expr(
        RecursiveTextChunkExpr(
            Column._from_col_or_name(column)._logical_expr,
            chunking_configuration=RecursiveTextChunkExprConfiguration(
                desired_chunk_size=chunk_size,
                chunk_overlap_percentage=chunk_overlap_percentage,
                chunk_length_function_name=ChunkLengthFunction.TOKEN,
                chunking_character_set_name=chunking_character_set_name,
                chunking_character_set_custom_characters=chunking_character_set_custom_characters,
            )
        )
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def character_chunk(
    column: ColumnOrName, chunk_size: int, chunk_overlap_percentage: int = 0
) -> Column:
    """Chunks a string column into chunks of a specified size (in characters) with an optional overlap.

    The chunking is done by applying a simple sliding window across the text to create chunks of equal size.
    This approach does not attempt to preserve the underlying structure of the text.

    Args:
        column: The input string column or column name to chunk
        chunk_size: The size of each chunk in characters
        chunk_overlap_percentage: The overlap between chunks as a percentage of the chunk size (Default: 0)

    Returns:
        Column: A column containing the chunks as an array of strings

    Example: Create character chunks
        ```python
        # Create chunks of 100 characters with 20% overlap
        df.select(text.character_chunk(col("text"), 100, 20))
        ```
    """
    return Column._from_logical_expr(
        TextChunkExpr(
            Column._from_col_or_name(column)._logical_expr,
            chunking_configuration=TextChunkExprConfiguration(
                desired_chunk_size=chunk_size,
                chunk_overlap_percentage=chunk_overlap_percentage,
                chunk_length_function_name=ChunkLengthFunction.CHARACTER,
            )
        )
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def word_chunk(
    column: ColumnOrName, chunk_size: int, chunk_overlap_percentage: int = 0
) -> Column:
    """Chunks a string column into chunks of a specified size (in words) with an optional overlap.

    The chunking is done by applying a simple sliding window across the text to create chunks of equal size.
    This approach does not attempt to preserve the underlying structure of the text.

    Args:
        column: The input string column or column name to chunk
        chunk_size: The size of each chunk in words
        chunk_overlap_percentage: The overlap between chunks as a percentage of the chunk size (Default: 0)

    Returns:
        Column: A column containing the chunks as an array of strings

    Example: Create word chunks
        ```python
        # Create chunks of 100 words with 20% overlap
        df.select(text.word_chunk(col("text"), 100, 20))
        ```
    """
    return Column._from_logical_expr(
        TextChunkExpr(
            Column._from_col_or_name(column)._logical_expr,
            chunking_configuration=TextChunkExprConfiguration(
                desired_chunk_size=chunk_size,
                chunk_overlap_percentage=chunk_overlap_percentage,
                chunk_length_function_name=ChunkLengthFunction.WORD,
            )
        )
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def token_chunk(
    column: ColumnOrName, chunk_size: int, chunk_overlap_percentage: int = 0
) -> Column:
    """Chunks a string column into chunks of a specified size (in tokens) with an optional overlap.

    The chunking is done by applying a simple sliding window across the text to create chunks of equal size.
    This approach does not attempt to preserve the underlying structure of the text.

    Args:
        column: The input string column or column name to chunk
        chunk_size: The size of each chunk in tokens
        chunk_overlap_percentage: The overlap between chunks as a percentage of the chunk size (Default: 0)

    Returns:
        Column: A column containing the chunks as an array of strings

    Example: Create token chunks
        ```python
        # Create chunks of 100 tokens with 20% overlap
        df.select(text.token_chunk(col("text"), 100, 20))
        ```
    """
    return Column._from_logical_expr(
        TextChunkExpr(
            Column._from_col_or_name(column)._logical_expr,
            chunking_configuration=TextChunkExprConfiguration(
                desired_chunk_size=chunk_size,
                chunk_overlap_percentage=chunk_overlap_percentage,
                chunk_length_function_name=ChunkLengthFunction.TOKEN,
            )
        )
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def count_tokens(
    column: ColumnOrName,
) -> Column:
    r"""Returns the number of tokens in a string using OpenAI's cl100k_base encoding (tiktoken).

    Args:
        column: The input string column.

    Returns:
        Column: A column with the token counts for each input string.

    Example: Count tokens in text
        ```python
        # Count tokens in a text column
        df.select(text.count_tokens(col("text")))
        ```
    """
    return Column._from_logical_expr(
        CountTokensExpr(Column._from_col_or_name(column)._logical_expr)
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def concat(*cols: ColumnOrName) -> Column:
    """Concatenates multiple columns or strings into a single string.

    Args:
        *cols: Columns or strings to concatenate

    Returns:
        Column: A column containing the concatenated strings

    Example: Concatenate columns
        ```python
        # Concatenate two columns with a space in between
        df.select(text.concat(col("col1"), lit(" "), col("col2")))
        ```
    """
    if not cols:
        raise ValidationError("No columns were provided. Please specify at least one column to use with the concat method.")

    flattened_args = []
    for arg in cols:
        if isinstance(arg, (list, tuple)):
            flattened_args.extend(arg)
        else:
            flattened_args.append(arg)

    flattened_exprs = [
        Column._from_col_or_name(c)._logical_expr for c in flattened_args
    ]
    return Column._from_logical_expr(ConcatExpr(flattened_exprs))



@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def parse_transcript(column: ColumnOrName, format: TranscriptFormatType) -> Column:
    """Parses a transcript from text to a structured format with unified schema.

    Converts transcript text in various formats (srt, webvtt, generic) to a standardized structure
    with fields: index, speaker, start_time, end_time, duration, content, format.
    All timestamps are returned as floating-point seconds from the start.

    Args:
        column: The input string column or column name containing transcript text
        format: The format of the transcript ("srt", "webvtt", or "generic")

    Returns:
        Column: A column containing an array of structured transcript entries with unified schema:

            - index: Optional[int] - Entry index (1-based)
            - speaker: Optional[str] - Speaker name (for generic format)
            - start_time: float - Start time in seconds
            - end_time: Optional[float] - End time in seconds
            - duration: Optional[float] - Duration in seconds
            - content: str - Transcript content/text
            - format: str - Original format ("srt", "webvtt", or "generic")

    Examples:
        >>> # Parse SRT format transcript
        >>> df.select(text.parse_transcript(col("transcript"), "srt"))
        >>> # Parse generic conversation transcript
        >>> df.select(text.parse_transcript(col("transcript"), "generic"))
        >>> # Parse WebVTT format transcript
        >>> df.select(text.parse_transcript(col("transcript"), "webvtt"))
    """
    return Column._from_logical_expr(
        TsParseExpr(Column._from_col_or_name(column)._logical_expr, format)
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def concat_ws(separator: str, *cols: ColumnOrName) -> Column:
    """Concatenates multiple columns or strings into a single string with a separator.

    Args:
        separator: The separator to use
        *cols: Columns or strings to concatenate

    Returns:
        Column: A column containing the concatenated strings

    Example: Concatenate with comma separator
        ```python
        # Concatenate columns with comma separator
        df.select(text.concat_ws(",", col("col1"), col("col2")))
        ```
    """
    if not cols:
        raise ValidationError("No columns were provided. Please specify at least one column to use with the concat_ws method.")

    flattened_args = []
    for arg in cols:
        if isinstance(arg, (list, tuple)):
            flattened_args.extend(arg)
        else:
            flattened_args.append(arg)

    expr_args = []
    for arg in flattened_args:
        expr_args.append(Column._from_col_or_name(arg)._logical_expr)
        expr_args.append(lit(separator)._logical_expr)
    expr_args.pop()
    return Column._from_logical_expr(ConcatExpr(expr_args))


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def array_join(column: ColumnOrName, delimiter: str) -> Column:
    """Joins an array of strings into a single string with a delimiter.

    Args:
        column: The column to join
        delimiter: The delimiter to use
    Returns:
            Column: A column containing the joined strings

    Example: Join array with comma
        ```python
        # Join array elements with comma
        df.select(text.array_join(col("array_column"), ","))
        ```
    """
    return Column._from_logical_expr(
        ArrayJoinExpr(Column._from_col_or_name(column)._logical_expr, delimiter)
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def replace(
    src: ColumnOrName, search: Union[Column, str], replace: Union[Column, str]
) -> Column:
    """Replace all occurrences of a pattern with a new string, treating pattern as a literal string.

    This method creates a new string column with all occurrences of the specified pattern
    replaced with a new string. The pattern is treated as a literal string, not a regular expression.
    If either search or replace is a column expression, the operation is performed dynamically
    using the values from those columns.

    Args:
        src: The input string column or column name to perform replacements on
        search: The pattern to search for (can be a string or column expression)
        replace: The string to replace with (can be a string or column expression)

    Returns:
        Column: A column containing the strings with replacements applied

    Example: Replace with literal string
        ```python
        # Replace all occurrences of "foo" in the "name" column with "bar"
        df.select(text.replace(col("name"), "foo", "bar"))
        ```

    Example: Replace using column values
        ```python
        # Replace all occurrences of the value in the "search" column with the value in the "replace" column, for each row in the "text" column
        df.select(text.replace(col("text"), col("search"), col("replace")))
        ```
    """
    if isinstance(search, Column):
        search_expr = search._logical_expr
    else:
        search_expr = lit(search)._logical_expr
    if isinstance(replace, Column):
        replace_expr = replace._logical_expr
    else:
        replace_expr = lit(replace)._logical_expr
    return Column._from_logical_expr(
        ReplaceExpr(
            Column._from_col_or_name(src)._logical_expr, search_expr, replace_expr, True
        )
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def regexp_replace(
    src: ColumnOrName,
    pattern: Union[Column, str],
    replacement: Union[Column, str],
) -> Column:
    r"""Replace all occurrences of a pattern with a new string, treating pattern as a regular expression.

    This method creates a new string column with all occurrences of the specified pattern
    replaced with a new string. The pattern is treated as a regular expression.
    If either pattern or replacement is a column expression, the operation is performed dynamically
    using the values from those columns.

    Args:
        src: The input string column or column name to perform replacements on
        pattern: The regular expression pattern to search for (can be a string or column expression)
        replacement: The string to replace with (can be a string or column expression)

    Returns:
        Column: A column containing the strings with replacements applied

    Example: Replace digits with dashes
        ```python
        # Replace all digits with dashes
        df.select(text.regexp_replace(col("text"), r"\d+", "--"))
        ```

    Example: Dynamic replacement using column values
        ```python
        # Replace using patterns from columns
        df.select(text.regexp_replace(col("text"), col("pattern"), col("replacement")))
        ```

    Example: Complex pattern replacement
        ```python
        # Replace email addresses with [REDACTED]
        df.select(text.regexp_replace(col("text"), r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "[REDACTED]"))
        ```
    """
    if isinstance(pattern, Column):
        pattern_expr = pattern._logical_expr
    else:
        pattern_expr = lit(pattern)._logical_expr
    if isinstance(replacement, Column):
        replacement_expr = replacement._logical_expr
    else:
        replacement_expr = lit(replacement)._logical_expr
    return Column._from_logical_expr(
        ReplaceExpr(
            Column._from_col_or_name(src)._logical_expr,
            pattern_expr,
            replacement_expr,
            False,
        )
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def regexp_count(src: ColumnOrName, pattern: Union[Column, str]) -> Column:
    r"""Count the number of times a regex pattern is matched in a string.

    Returns the count of matches for each string in the column.

    Args:
        src: The input string column or column name
        pattern: The regex pattern to count (can be a string literal or column expression)

    Returns:
        Column: An integer column containing the count of pattern matches

    Example: Count digits
        ```python
        import fenic as fc

        df = fc.Session.local().create_dataframe({
            "text": ["abc123", "456def789", "no digits"]
        })

        result = df.select(fc.text.regexp_count("text", r"\d"))
        # Output: [3, 6, 0]
        ```

    Example: Count words
        ```python
        df = fc.Session.local().create_dataframe({
            "text": ["hello world", "one two three"]
        })

        result = df.select(fc.text.regexp_count("text", r"\w+"))
        # Output: [2, 3]
        ```
    """
    if isinstance(pattern, Column):
        pattern_expr = pattern._logical_expr
    else:
        pattern_expr = lit(pattern)._logical_expr
    return Column._from_logical_expr(
        RegexpCountExpr(Column._from_col_or_name(src)._logical_expr, pattern_expr)
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def regexp_extract(
    src: ColumnOrName, pattern: Union[Column, str], idx: int
) -> Column:
    r"""Extract a specific regex group from a string.

    Extracts a capture group matched by the regex pattern. Group 0 is the entire match, group 1+ are capture groups.
    If the pattern/group has multiple matches within the same string, the result will be the first match. Use `regexp_extract_all`
    to extract all matches.

    Args:
        src: The input string column or column name
        pattern: The regex pattern with capture groups (can be a string literal or column expression)
        idx: The group index to extract (0 = entire match, 1+ = capture groups)

    Returns:
        Column: A string column containing the extracted group. If no match is found, or
            the specified group did not match, the result will be an empty string.

    Example: Extract email username
        ```python
        import fenic as fc

        df = fc.Session.local().create_dataframe({
            "email": ["user@domain.com", "admin@example.org"]
        })

        result = df.select(fc.text.regexp_extract("email", r"([^@]+)@", 1))
        # Output: ["user", "admin"]
        ```

    Example: Extract phone area code
        ```python
        df = fc.Session.local().create_dataframe({
            "phone": ["(555) 123-4567", "(123) 456-7890"]
        })

        result = df.select(fc.text.regexp_extract("phone", r"\((\d{3})\)", 1))
        # Output: ["555", "123"]
        ```
    """
    if isinstance(pattern, Column):
        pattern_expr = pattern._logical_expr
    else:
        pattern_expr = lit(pattern)._logical_expr
    return Column._from_logical_expr(
        RegexpExtractExpr(
            Column._from_col_or_name(src)._logical_expr, pattern_expr, idx
        )
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def regexp_extract_all(
    src: ColumnOrName, pattern: Union[Column, str], idx: Union[Column, int] = 0
) -> Column:
    r"""Extract all strings matching a regex pattern, optionally from a specific group.

    Returns an array of all matches. Group 0 is the entire match, group 1+ are capture groups.
    If the pattern/group has multiple matches within the same string, the result will be an array of all matches.

    Args:
        src: The input string column or column name
        pattern: The regex pattern with optional capture groups
        idx: The group index to extract (default: 0 for entire match, 1+ for capture groups)

    Returns:
        Column: An array column containing all matches

    Example: Extract all digits
        ```python
        import fenic as fc

        df = fc.Session.local().create_dataframe({
            "text": ["abc123def456", "no digits", "789"]
        })

        result = df.select(fc.text.regexp_extract_all("text", r"\d+"))
        # Output: [["123", "456"], [], ["789"]]
        ```

    Example: Extract all hashtags
        ```python
        df = fc.Session.local().create_dataframe({
            "post": ["Love #coding and #python", "Just #relaxing"]
        })

        result = df.select(fc.text.regexp_extract_all("post", r"#(\w+)", 1))
        # Output: [["coding", "python"], ["relaxing"]]
        ```
    """
    if isinstance(pattern, Column):
        pattern_expr = pattern._logical_expr
    else:
        pattern_expr = lit(pattern)._logical_expr
    if isinstance(idx, Column):
        idx_expr = idx._logical_expr
    else:
        idx_expr = lit(idx)._logical_expr
    return Column._from_logical_expr(
        RegexpExtractAllExpr(
            Column._from_col_or_name(src)._logical_expr, pattern_expr, idx_expr
        )
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def regexp_instr(
    src: ColumnOrName, pattern: Union[Column, str], idx: Union[Column, int] = 0
) -> Column:
    r"""Find the 1-based position of the first regex match in a string.

    Returns the position (1-based) of the first substring matching the pattern.
    Returns 0 if no match is found.

    Args:
        src: The input string column or column name
        pattern: The regex pattern to search for
        idx: The group index to locate (default: 0 for entire match, 1+ for capture groups)

    Returns:
        Column: An integer column with 1-based position (0 if no match)

    Example: Find position of first digit
        ```python
        import fenic as fc

        df = fc.Session.local().create_dataframe({
            "text": ["abc123", "no digits", "456xyz"]
        })

        result = df.select(fc.text.regexp_instr("text", r"\d"))
        # Output: [4, 0, 1]  # 1-based positions
        ```

    Example: Find position of email
        ```python
        df = fc.Session.local().create_dataframe({
            "text": ["Contact: user@domain.com", "No email here"]
        })

        result = df.select(fc.text.regexp_instr("text", r"[^@]+@[^@]+"))
        # Output: [10, 0]
        ```
    """
    if isinstance(pattern, Column):
        pattern_expr = pattern._logical_expr
    else:
        pattern_expr = lit(pattern)._logical_expr
    if isinstance(idx, Column):
        idx_expr = idx._logical_expr
    else:
        idx_expr = lit(idx)._logical_expr
    return Column._from_logical_expr(
        RegexpInstrExpr(
            Column._from_col_or_name(src)._logical_expr, pattern_expr, idx_expr
        )
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def regexp_substr(src: ColumnOrName, pattern: Union[Column, str]) -> Column:
    r"""Extract the first substring matching a regex pattern.

    Returns the first substring that matches the pattern. Returns null if no match is found.

    Args:
        src: The input string column or column name
        pattern: The regex pattern to search for

    Returns:
        Column: A string column containing the first match (null if no match)

    Example: Extract first number
        ```python
        import fenic as fc

        df = fc.Session.local().create_dataframe({
            "text": ["Price: $123.45", "No price", "Cost: $67.89"]
        })

        result = df.select(fc.text.regexp_substr("text", r"\d+\.\d+"))
        # Output: ["123.45", null, "67.89"]
        ```

    Example: Extract first URL
        ```python
        df = fc.Session.local().create_dataframe({
            "text": ["Visit https://example.com for info", "No URL here"]
        })

        result = df.select(fc.text.regexp_substr("text", r"https?://[^\s]+"))
        # Output: ["https://example.com", null]
        ```
    """
    if isinstance(pattern, Column):
        pattern_expr = pattern._logical_expr
    else:
        pattern_expr = lit(pattern)._logical_expr
    return Column._from_logical_expr(
        RegexpSubstrExpr(Column._from_col_or_name(src)._logical_expr, pattern_expr)
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def split(src: ColumnOrName, pattern: str, limit: int = -1) -> Column:
    r"""Split a string column into an array using a regular expression pattern.

    This method creates an array column by splitting each value in the input string column
    at matches of the specified regular expression pattern.

    Args:
        src: The input string column or column name to split
        pattern: The regular expression pattern to split on
        limit: Maximum number of splits to perform (Default: -1 for unlimited).
              If > 0, returns at most limit+1 elements, with remainder in last element.

    Returns:
        Column: A column containing arrays of substrings

    Example: Split on whitespace
        ```python
        # Split on whitespace
        df.select(text.split(col("text"), r"\s+"))
        ```

    Example: Split with limit
        ```python
        # Split on whitespace, max 2 splits
        df.select(text.split(col("text"), r"\s+", limit=2))
        ```
    """
    return Column._from_logical_expr(
        RegexpSplitExpr(Column._from_col_or_name(src)._logical_expr, pattern, limit)
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def split_part(
    src: ColumnOrName, delimiter: Union[Column, str], part_number: Union[Column, int]
) -> Column:
    """Split a string and return a specific part using 1-based indexing.

    Splits each string by a delimiter and returns the specified part.
    If the delimiter is a column expression, the split operation is performed dynamically
    using the delimiter values from that column.

    Behavior:
    - If any input is null, returns null
    - If part_number is out of range of split parts, returns empty string
    - If part_number is 0, throws an error
    - If part_number is negative, counts from the end of the split parts
    - If the delimiter is an empty string, the string is not split

    Args:
        src: The input string column or column name to split
        delimiter: The delimiter to split on (can be a string or column expression)
        part_number: Which part to return (1-based integer index or column expression)

    Returns:
        Column: A column containing the specified part from each split string

    Example: Get second part of comma-separated values
        ```python
        # Get second part of comma-separated values
        df.select(text.split_part(col("text"), ",", 2))
        ```

    Example: Get last part using negative index
        ```python
        # Get last part using negative index
        df.select(text.split_part(col("text"), ",", -1))
        ```

    Example: Use dynamic delimiter from column
        ```python
        # Use dynamic delimiter from column
        df.select(text.split_part(col("text"), col("delimiter"), 1))
        ```
    """
    if isinstance(part_number, int) and part_number == 0:
        raise ValidationError(
            f"`split_part` expects a non-zero integer for the part_number, but got {part_number}."
        )
    if isinstance(part_number, Column):
        part_number_expr = part_number._logical_expr
    else:
        part_number_expr = lit(part_number)._logical_expr

    if isinstance(delimiter, Column):
        delimiter_expr = delimiter._logical_expr
    else:
        delimiter_expr = lit(delimiter)._logical_expr

    return Column._from_logical_expr(
        SplitPartExpr(
            Column._from_col_or_name(src)._logical_expr, delimiter_expr, part_number_expr
        )
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def upper(column: ColumnOrName) -> Column:
    """Convert all characters in a string column to uppercase.

    Args:
        column: The input string column to convert to uppercase

    Returns:
        Column: A column containing the uppercase strings

    Example: Convert text to uppercase
        ```python
        # Convert all text in the name column to uppercase
        df.select(text.upper(col("name")))
        ```
    """
    return Column._from_logical_expr(
        StringCasingExpr(Column._from_col_or_name(column)._logical_expr, "upper")
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def lower(column: ColumnOrName) -> Column:
    """Convert all characters in a string column to lowercase.

    Args:
        column: The input string column to convert to lowercase

    Returns:
        Column: A column containing the lowercase strings

    Example: Convert text to lowercase
        ```python
        # Convert all text in the name column to lowercase
        df.select(text.lower(col("name")))
        ```
    """
    return Column._from_logical_expr(
        StringCasingExpr(Column._from_col_or_name(column)._logical_expr, "lower")
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def title_case(column: ColumnOrName) -> Column:
    """Convert the first character of each word in a string column to uppercase.

    Args:
        column: The input string column to convert to title case

    Returns:
        Column: A column containing the title case strings

    Example: Convert text to title case
        ```python
        # Convert text in the name column to title case
        df.select(text.title_case(col("name")))
        ```
    """
    return Column._from_logical_expr(
        StringCasingExpr(Column._from_col_or_name(column)._logical_expr, "title")
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def trim(column: ColumnOrName) -> Column:
    """Remove whitespace from both sides of strings in a column.

    This function removes all whitespace characters (spaces, tabs, newlines) from
    both the beginning and end of each string in the column.

    Args:
        column: The input string column or column name to trim

    Returns:
        Column: A column containing the trimmed strings

    Example: Remove whitespace from both sides
        ```python
        # Remove whitespace from both sides of text
        df.select(text.trim(col("text")))
        ```
    """
    return Column._from_logical_expr(
        StripCharsExpr(Column._from_col_or_name(column)._logical_expr, None, "both")
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def btrim(col: ColumnOrName, trim: Optional[Union[Column, str]]) -> Column:
    """Remove specified characters from both sides of strings in a column.

    This function removes all occurrences of the specified characters from
    both the beginning and end of each string in the column.
    If trim is a column expression, the characters to remove are determined dynamically
    from the values in that column.

    Args:
        col: The input string column or column name to trim
        trim: The characters to remove from both sides (Default: whitespace)
              Can be a string or column expression.

    Returns:
        Column: A column containing the trimmed strings

    Example: Remove brackets from both sides
        ```python
        # Remove brackets from both sides of text
        df.select(text.btrim(col("text"), "[]"))
        ```

    Example: Remove characters specified in a column
        ```python
        # Remove characters specified in a column
        df.select(text.btrim(col("text"), col("chars")))
        ```
    """
    if trim is None:
        trim_expr = None
    elif isinstance(trim, Column):
        trim_expr = trim._logical_expr
    else:
        trim_expr = lit(trim)._logical_expr
    return Column._from_logical_expr(
        StripCharsExpr(Column._from_col_or_name(col)._logical_expr, trim_expr, "both")
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def ltrim(col: ColumnOrName) -> Column:
    """Remove whitespace from the start of strings in a column.

    This function removes all whitespace characters (spaces, tabs, newlines) from
    the beginning of each string in the column.

    Args:
        col: The input string column or column name to trim

    Returns:
        Column: A column containing the left-trimmed strings

    Example: Remove leading whitespace
        ```python
        # Remove whitespace from the start of text
        df.select(text.ltrim(col("text")))
        ```
    """
    return Column._from_logical_expr(
        StripCharsExpr(Column._from_col_or_name(col)._logical_expr, None, "left")
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def rtrim(col: ColumnOrName) -> Column:
    """Remove whitespace from the end of strings in a column.

    This function removes all whitespace characters (spaces, tabs, newlines) from
    the end of each string in the column.

    Args:
        col: The input string column or column name to trim

    Returns:
        Column: A column containing the right-trimmed strings

    Example: Remove trailing whitespace
        ```python
        # Remove whitespace from the end of text
        df.select(text.rtrim(col("text")))
        ```
    """
    return Column._from_logical_expr(
        StripCharsExpr(Column._from_col_or_name(col)._logical_expr, None, "right")
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def length(column: ColumnOrName) -> Column:
    """Calculate the character length of each string in the column.

    Args:
        column: The input string column to calculate lengths for

    Returns:
        Column: A column containing the length of each string in characters

    Example: Get string lengths
        ```python
        # Get the length of each string in the name column
        df.select(text.length(col("name")))
        ```
    """
    return Column._from_logical_expr(
        StrLengthExpr(Column._from_col_or_name(column)._logical_expr)
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def byte_length(column: ColumnOrName) -> Column:
    """Calculate the byte length of each string in the column.

    Args:
        column: The input string column to calculate byte lengths for

    Returns:
        Column: A column containing the byte length of each string

    Example: Get byte lengths
        ```python
        # Get the byte length of each string in the name column
        df.select(text.byte_length(col("name")))
        ```
    """
    return Column._from_logical_expr(
        ByteLengthExpr(Column._from_col_or_name(column)._logical_expr)
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def jinja(
    jinja_template: str,
    /,
    strict: bool = True,
    **columns: Column
) -> Column:
    """Render a Jinja template using values from the specified columns.

    This function evaluates a Jinja2 template string for each row, using the provided
    columns as template variables. Only a subset of Jinja2 features is supported.

    Args:
        jinja_template: A Jinja2 template string to render for each row.
                        Variables are referenced using double braces: {{ variable_name }}
        strict: If True, when any of the provided columns has a None value for a row,
                the entire row's output will be None (template is not rendered).
                If False, None values are handled using Jinja2's null rendering behavior.
                Default is True.
        **columns: Keyword arguments mapping variable names to columns.
                  Each keyword becomes a variable in the template context.

    Returns:
        Column: A string column containing the rendered template for each row

    Supported Features:
        - Variable substitution: {{ variable }}
        - Struct/object field access: {{ user.name }}
        - Array indexing with literals: {{ items[0] }}, {{ data["key"] }}
        - For loops: {% for item in items %}...{% endfor %}
        - If/elif/else conditionals: {% if condition %}...{% endif %}
        - Loop variables: {{ loop.index }}, {{ loop.first }}, etc.
        - Constants: {{ "literal string" }}, {{ 42 }}

    Not Supported (use column expressions instead):
        - **Filters**: {{ name|upper }} → Use upper_name=fc.upper(col("name"))
        - **Function calls**: {{ len(items) }} → Use item_count=fc.array_size(col("items"))
        - **Operators**: {% if price > 100 %} → Use is_expensive=(col("price") > 100)
        - **Arithmetic**: {{ price * quantity }} → Use total=col("price") * col("quantity")
        - **Dynamic indexing**: {{ items[i] }} → Use item=(fc.col("items").get_item(col("index")))
        - **Variable assignment**: {% set x = 5 %} → Pre-compute as column expression
        - **Macros, includes, extends**: Not supported

    Example: LLM prompt formatting with conditional context and examples
        ```python
        # Format prompts with user query, conditional context, and examples
        prompt_template = '''
        Answer the user's question.

        {% if context %}
        Context: {{ context }}
        {% endif %}

        {% if examples %}
        Few-shot examples:
        {% for ex in examples %}
        Q: {{ ex.question }}
        A: {{ ex.answer }}
        {% endfor %}
        {% endif %}

        Question: {{ query }}

        Please provide a {{ style }} response.'''

        # Generate prompts with varying context based on query type
        result = df.select(
            text.jinja(
                prompt_template,
                # Direct columns
                query=col("user_question"),
                context=col("retrieved_context"),  # Can be null for some rows

                # Column expression for conditional logic
                style=fc.when(col("query_type") == "technical", "detailed and technical")
                      .when(col("query_type") == "casual", "conversational")
                      .otherwise("clear and concise"),

                # Array of examples (struct array)
                examples=col("few_shot_examples")  # Array of {question, answer} structs
            ).alias("llm_prompt")
        )
        ```

    Notes:
        - Template syntax is validated at query planning time
        - Complex operations can use column expressions
        - Arrays can only be iterated with {% for %} or accessed with literal indices
        - Structs can only use literal field names
        - Null values are rendered according to Jinja2's null rendering behavior
    """
    # Convert keyword arguments to column expressions with proper names
    column_exprs: List[LogicalExpr] = []
    for var_name, column in columns.items():
        if isinstance(column._logical_expr, ColumnExpr) and column._logical_expr.name == var_name:
            column_exprs.append(column._logical_expr)
        else:
            column_exprs.append(column.alias(var_name)._logical_expr)

    return Column._from_logical_expr(
        JinjaExpr(column_exprs, jinja_template, strict)
    )

@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def compute_fuzzy_ratio(column: ColumnOrName, other: Union[Column, str], method: FuzzySimilarityMethod = "indel") -> Column:
    """Compute the similarity between two strings using a fuzzy string matching algorithm.

    This function computes a fuzzy similarity score between two string columns (or a string column
    and a literal string) for each row. It supports multiple well-known string similarity metrics,
    including Levenshtein, Damerau-Levenshtein, Jaro, Jaro-Winkler, and Hamming.

    The returned score is a similarity percentage between 0 and 100, where:
        - 100 indicates the strings are identical
        - 0 indicates maximum dissimilarity (as defined by the method)

    Based on https://rapidfuzz.github.io/RapidFuzz/Usage/fuzz.html#rapidfuzz.fuzz.ratio

    Args:
        column: A string column or column name. This is the left-hand side of the comparison.
        other: A second string column or literal string. This is the right-hand side of the comparison.
        method: A string indicating which similarity method to use. Must be one of:
            - `"indel"`: Indel distance — counts only insertions and deletions (no substitutions); based on the Longest Common Subsequence.
            - `"levenshtein"`: Levenshtein distance (edit distance)
            - `"damerau_levenshtein"`: Damerau-Levenshtein distance (includes transpositions)
            - `"jaro"`: Jaro similarity, accounts for transpositions and proximity
            - `"jaro_winkler"`: Jaro-Winkler similarity, gives higher scores for common prefixes
            - `"hamming"`: Hamming distance. Counts differing positions between two equal-length strings, padding shorter string if needed.

    Returns:
        Column: A double column with similarity scores in the range [0, 100].

    Example: Compare two columns
        ```python
        result = df.select(
            compute_fuzzy_ratio(col("a"), col("b"), method="levenshtein").alias("sim")
        )
        ```

    Example: Compare a column to a literal string
        ```python
        result = df.select(
            compute_fuzzy_ratio(col("a"), "world", method="jaro").alias("sim_to_world")
        )
        ```
    """
    if isinstance(other, str):
        other_expr = LiteralExpr(other, StringType)
    else:
        other_expr = other._logical_expr

    return Column._from_logical_expr(FuzzyRatioExpr(Column._from_col_or_name(column)._logical_expr, other_expr, method))

@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def compute_fuzzy_token_sort_ratio(column: ColumnOrName, other: Union[Column, str], method: FuzzySimilarityMethod = "indel") -> Column:
    """Compute fuzzy similarity after sorting tokens in each string.

    Tokenizes strings by whitespace, sorts tokens alphabetically, concatenates
    them back into a string, then applies the specified similarity metric.
    Useful for comparing strings where word order doesn't matter.

    Based on https://rapidfuzz.github.io/RapidFuzz/Usage/fuzz.html#rapidfuzz.fuzz.token_sort_ratio

    Args:
        column: First string column to compare
        other: Second string column or literal string to compare against
        method: Similarity algorithm to use after token sorting

    Returns:
        Double column with similarity scores between 0 and 100

    Example:
        ```python
        # df.select(compute_fuzzy_token_sort_ratio(col("city"), "city  new  york", "levenshtein"))
        # "new york city" → ["new", "york", "city"] → sorted → ["city", "new", "york"] → "city new york"
        # "city new york" → ["city", "new", "york"] → sorted → ["city", "new", "york"] → "city new york"
        # levenshtein similarity("city new york", "city new york") = 100
        ```
    """
    if isinstance(other, str):
        other_expr = LiteralExpr(other, StringType)
    else:
        other_expr = other._logical_expr

    return Column._from_logical_expr(FuzzyTokenSortRatioExpr(Column._from_col_or_name(column)._logical_expr, other_expr, method))

@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def compute_fuzzy_token_set_ratio(column: ColumnOrName, other: Union[Column, str], method: FuzzySimilarityMethod = "indel") -> Column:
    """Compute fuzzy similarity using token set comparison.

    Tokenizes strings by whitespace, creates sets of unique tokens, then
    compares three combinations: diff1 vs diff2, intersection vs left set,
    and intersection vs right set. Returns the maximum similarity score.
    Useful for comparing strings where both word order and duplicates
    don't matter.

    Based on https://rapidfuzz.github.io/RapidFuzz/Usage/fuzz.html#rapidfuzz.fuzz.token_set_ratio

    Args:
        column: First string column to compare
        other: Second string column or literal string to compare against
        method: Similarity algorithm to use for comparison

    Returns:
        Double column with similarity scores between 0 and 100

    Example:
        ```python
        # df.select(compute_fuzzy_token_set_ratio(col("city"), "city of new york", "indel"))
        # "new york city new" → unique tokens: {"city", "new", "york"}
        # "city of new york" → unique tokens: {"city", "new", "of", "york"}
        # intersection: {"city", "new", "york"}
        # diff1: {} (empty)
        # diff2: {"of"}
        # Compares: diff1 vs diff2, intersection vs set1, intersection vs set2
        # Returns max similarity score = 100
        ```
    """
    if isinstance(other, str):
        other_expr = LiteralExpr(other, StringType)
    else:
        other_expr = other._logical_expr

    return Column._from_logical_expr(FuzzyTokenSetRatioExpr(Column._from_col_or_name(column)._logical_expr, other_expr, method))
