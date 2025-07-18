"""Text manipulation functions for Fenic DataFrames."""

from typing import Optional, Union

from pydantic import ConfigDict, validate_call

from fenic.api.column import Column, ColumnOrName
from fenic.api.functions.core import lit
from fenic.core._logical_plan.expressions import (
    ArrayJoinExpr,
    ByteLengthExpr,
    ConcatExpr,
    CountTokensExpr,
    RecursiveTextChunkExpr,
    RegexpSplitExpr,
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
    ChunkCharacterSet,
    ChunkLengthFunction,
)
from fenic.core.error import ValidationError
from fenic.core.types.enums import TranscriptFormatType


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
            desired_chunk_size=chunk_size,
            chunk_overlap_percentage=chunk_overlap_percentage,
            chunk_length_function_name=ChunkLengthFunction.CHARACTER,
            chunking_character_set_name=chunking_character_set_name,
            chunking_character_set_custom_characters=chunking_character_set_custom_characters,
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
            desired_chunk_size=chunk_size,
            chunk_overlap_percentage=chunk_overlap_percentage,
            chunk_length_function_name=ChunkLengthFunction.WORD,
            chunking_character_set_name=chunking_character_set_name,
            chunking_character_set_custom_characters=chunking_character_set_custom_characters,
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
            desired_chunk_size=chunk_size,
            chunk_overlap_percentage=chunk_overlap_percentage,
            chunk_length_function_name=ChunkLengthFunction.TOKEN,
            chunking_character_set_name=chunking_character_set_name,
            chunking_character_set_custom_characters=chunking_character_set_custom_characters,
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
            desired_chunk_size=chunk_size,
            chunk_overlap_percentage=chunk_overlap_percentage,
            chunk_length_function_name=ChunkLengthFunction.CHARACTER,
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
            desired_chunk_size=chunk_size,
            chunk_overlap_percentage=chunk_overlap_percentage,
            chunk_length_function_name=ChunkLengthFunction.WORD,
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
            desired_chunk_size=chunk_size,
            chunk_overlap_percentage=chunk_overlap_percentage,
            chunk_length_function_name=ChunkLengthFunction.TOKEN,
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
        raise ValidationError("At least one column must be provided to concat method")

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

    Converts transcript text in various formats (srt, generic) to a standardized structure
    with fields: index, speaker, start_time, end_time, duration, content, format.
    All timestamps are returned as floating-point seconds from the start.

    Args:
        column: The input string column or column name containing transcript text
        format: The format of the transcript ("srt" or "generic")

    Returns:
        Column: A column containing an array of structured transcript entries with unified schema:

            - index: Optional[int] - Entry index (1-based)
            - speaker: Optional[str] - Speaker name (for generic format)
            - start_time: float - Start time in seconds
            - end_time: Optional[float] - End time in seconds
            - duration: Optional[float] - Duration in seconds
            - content: str - Transcript content/text
            - format: str - Original format ("srt" or "generic")

    Examples:
        >>> # Parse SRT format transcript
        >>> df.select(text.parse_transcript(col("transcript"), "srt"))
        >>> # Parse generic conversation transcript
        >>> df.select(text.parse_transcript(col("transcript"), "generic"))
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
        raise ValidationError("At least one column must be provided to concat_ws method")

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
