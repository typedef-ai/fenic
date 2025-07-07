from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, List, Literal, Optional, Tuple, Union

if TYPE_CHECKING:
    from fenic.core._logical_plan.plans.base import LogicalPlan

from pydantic import BaseModel, Field

from fenic.core._logical_plan.expressions.base import LogicalExpr
from fenic.core._logical_plan.signatures.scalar_function import ScalarFunction
from fenic.core.error import ValidationError
from fenic.core.types import (
    DataType,
    JsonType,
    StringType,
    StructField,
    StructType,
)


class TokenType(Enum):
    DELIMITER = auto()    # Literal text content
    COLUMN = auto()  # Column placeholder with optional format


class EscapingRule(Enum):
    NONE = auto()
    CSV = auto()
    JSON = auto()
    REGEX = auto()
    QUOTED = auto()
    QUALIFIED = auto()

    @classmethod
    def from_string(cls, s: str) -> 'EscapingRule':
        s_upper = s.upper()
        if s_upper in cls.__members__:
            return cls[s_upper]
        valid = ", ".join(m.lower() for m in cls.__members__)
        raise ValueError(f"Invalid escaping rule '{s}'. Valid are {valid}")

@dataclass
class ParsedTemplateFormat:
    delimiters: List[str] = field(default_factory=list)
    escaping_rules: List[EscapingRule] = field(default_factory=list)
    columns: List[str] = field(default_factory=list)
    _original_format: str = field(default="", init=False)

    @classmethod
    def parse(cls, format_string: str) -> 'ParsedTemplateFormat':
        """Parse a template format string like 'prefix${col1}middle${col2:csv}suffix'."""
        instance = cls()
        instance._original_format = format_string

        try:
            tokens = instance._tokenize(format_string)
            instance._process_tokens(tokens)
            return instance
        except ValueError as e:
            # Enhance error with dump() context
            raise ValidationError(f"{e}\n\nDebug:\n{instance.dump()}") from e

    def _tokenize(self, format_string: str) -> List[Tuple[TokenType, str]]:
        """Break format string into TEXT and COLUMN tokens."""
        tokens = []
        pos = 0

        while pos < len(format_string):
            dollar_pos = format_string.find('$', pos)

            if dollar_pos == -1:
                if pos < len(format_string):
                    tokens.append((TokenType.DELIMITER, format_string[pos:]))
                break

            if dollar_pos > pos:
                tokens.append((TokenType.DELIMITER, format_string[pos:dollar_pos]))

            if dollar_pos + 1 >= len(format_string):
                raise ValueError(f"Unexpected end after '$' at position {dollar_pos}")

            next_char = format_string[dollar_pos + 1]
            if next_char == '$':
                tokens.append((TokenType.DELIMITER, '$'))
                pos = dollar_pos + 2
            elif next_char == '{':
                brace_pos = format_string.find('}', dollar_pos + 2)
                if brace_pos == -1:
                    raise ValueError(f"Unmatched opening brace starting at position {dollar_pos + 1}")

                column_content = format_string[dollar_pos + 2:brace_pos]
                tokens.append((TokenType.COLUMN, column_content))
                pos = brace_pos + 1
            else:
                raise ValueError(f"Expected '{{' or '$' after '$' at position {dollar_pos + 1}, got '{next_char}'")

        return tokens

    def _process_tokens(self, tokens: List[Tuple[TokenType, str]]) -> None:
        """Process tokens into delimiters, columns, and escaping rules."""
        self.delimiters.append("")  # Start with empty delimiter

        for token_type, content in tokens:
            if token_type == TokenType.DELIMITER:
                self.delimiters[-1] += content
            elif token_type == TokenType.COLUMN:
                self._process_column_token(content)
                self.delimiters.append("")  # Start new delimiter after column

    def _process_column_token(self, content: str) -> None:
        """Process a column token like 'col_name' or 'col_name:csv'."""
        if ':' in content:
            col_name, format_spec = content.split(':', 1)
            col_name = col_name.strip()
            format_spec = format_spec.strip()
            escaping_rule = EscapingRule.from_string(format_spec)
        else:
            col_name = content.strip()
            escaping_rule = EscapingRule.NONE

        if not col_name:
            raise ValueError("Column name cannot be empty")

        self.columns.append(col_name)
        self.escaping_rules.append(escaping_rule)

    def dump(self) -> str:
        """Return a debug representation of the parsed format."""
        lines = []
        for i, delim in enumerate(self.delimiters):
            lines.append(f"Delimiter {i}: {repr(delim)}")
            if i < len(self.columns):
                lines.append(f"  Column {i}: {self.columns[i]}")
                lines.append(f"  Escaping: {self.escaping_rules[i].name}")
        return "\n".join(lines)

    def to_struct_schema(self) -> StructType:
        return StructType(
            struct_fields=[
                StructField(
                    name=col,
                    data_type=JsonType if self.escaping_rules[i] == EscapingRule.JSON else StringType
                )
                for i, col in enumerate(self.columns)
            ]
        )

class TextractExpr(ScalarFunction):
    function_name = "text.extract"

    def __init__(self, input_expr: LogicalExpr, template: str):
        self.input_expr = input_expr
        self.template = template
        self.parsed_template = ParsedTemplateFormat.parse(template)

        super().__init__(input_expr)

    def __str__(self):
        return f"text.extract('{self.template}', {self.input_expr})"

    def _infer_dynamic_return_type(self, arg_types: List[DataType], plan: LogicalPlan) -> DataType:
        """Return StructType with fields based on parsed template."""
        return self.parsed_template.to_struct_schema()


class ChunkLengthFunction(Enum):
    CHARACTER = "CHARACTER"
    WORD = "WORD"
    # trunk-ignore(bandit/B105): not a token
    TOKEN = "TOKEN"


class ChunkCharacterSet(Enum):
    CUSTOM = "CUSTOM"
    ASCII = "ASCII"
    UNICODE = "UNICODE"


class TextChunkExprConfiguration(BaseModel):
    desired_chunk_size: int = Field(gt=0)
    chunk_overlap_percentage: int = Field(default=0, ge=0, lt=100)
    chunk_length_function_name: ChunkLengthFunction = ChunkLengthFunction.TOKEN


class TextChunkExpr(ScalarFunction):
    function_name = "text.chunk"

    def __init__(
        self,
        input_expr: LogicalExpr,
        desired_chunk_size: int,
        chunk_overlap_percentage: int = 0,
        chunk_length_function_name: ChunkLengthFunction = ChunkLengthFunction.TOKEN
    ):
        self.input_expr = input_expr
        self.chunk_configuration = TextChunkExprConfiguration(
            desired_chunk_size=desired_chunk_size,
            chunk_overlap_percentage=chunk_overlap_percentage,
            chunk_length_function_name=chunk_length_function_name,
        )

        # Only validate the string expression (chunk_configuration is not LogicalExpr)
        super().__init__(input_expr)

    def __str__(self) -> str:
        return f"text_chunk({self.input_expr}, {self.chunk_configuration})"

class RecursiveTextChunkExprConfiguration(TextChunkExprConfiguration):
    chunking_character_set_name: ChunkCharacterSet = ChunkCharacterSet.ASCII
    chunking_character_set_custom_characters: Optional[list[str]] = None


class RecursiveTextChunkExpr(ScalarFunction):
    function_name = "text.recursive_chunk"

    def __init__(
        self,
        input_expr: LogicalExpr,
        desired_chunk_size: int,
        chunk_overlap_percentage: int = 0,
        chunk_length_function_name: ChunkLengthFunction = ChunkLengthFunction.TOKEN,
        chunking_character_set_name: ChunkCharacterSet = ChunkCharacterSet.ASCII,
        chunking_character_set_custom_characters: Optional[list[str]] = None
    ):
        self.input_expr = input_expr
        self.chunking_configuration = RecursiveTextChunkExprConfiguration(
            desired_chunk_size=desired_chunk_size,
            chunk_overlap_percentage=chunk_overlap_percentage,
            chunk_length_function_name=chunk_length_function_name,
            chunking_character_set_name=chunking_character_set_name,
            chunking_character_set_custom_characters=chunking_character_set_custom_characters,
        )

        # Only validate the string expression (chunking_configuration is not LogicalExpr)
        super().__init__(input_expr)

    def __str__(self) -> str:
        return f"text_chunk({self.input_expr}, {self.chunking_configuration})"


class CountTokensExpr(ScalarFunction):
    function_name = "text.count_tokens"

    def __init__(self, input_expr: LogicalExpr):
        self.input_expr = input_expr
        super().__init__(input_expr)

class ConcatExpr(ScalarFunction):
    function_name = "text.concat"

    def __init__(self, exprs: List[LogicalExpr]):
        self.exprs = exprs
        super().__init__(*exprs)


class ArrayJoinExpr(ScalarFunction):
    function_name = "text.array_join"

    def __init__(self, expr: LogicalExpr, delimiter: str):
        self.expr = expr
        self.delimiter = delimiter

        # Only validate the array expression (delimiter is not LogicalExpr)
        super().__init__(expr)

    def __str__(self) -> str:
        return f"array_join({self.expr}, {self.delimiter})"


class ContainsExpr(ScalarFunction):
    """Expression for checking if a string column contains a substring.

    This expression creates a boolean result indicating whether each value in the input
    string column contains the specified substring.

    Args:
        expr: The input string column expression
        substr: The substring to search for within each value

    Raises:
        TypeError: If the input expression is not a string column
    """

    function_name = "text.contains"

    def __init__(self, expr: LogicalExpr, substr: Union[str, LogicalExpr]):
        self.expr = expr
        self.substr = substr

        # Pass appropriate arguments to signature validation
        if isinstance(substr, LogicalExpr):
            super().__init__(expr, substr)  # Both string inputs
        else:
            super().__init__(expr)  # Only main string input


class ContainsAnyExpr(ScalarFunction):
    """Expression for checking if a string column contains any of multiple substrings.

    This expression creates a boolean result indicating whether each value in the input
    string column contains any of the specified substrings.

    Args:
        expr: The input string column expression
        substrs: List of substrings to search for within each value
        case_insensitive: Whether to perform case-insensitive matching (default: True)

    Raises:
        TypeError: If the input expression is not a string column
    """

    function_name = "text.contains_any"

    def __init__(
        self, expr: LogicalExpr, substrs: List[str], case_insensitive: bool = True
    ):
        self.expr = expr
        self.substrs = substrs
        self.case_insensitive = case_insensitive

        # Only validate the main string expression (substrs and case_insensitive are not LogicalExprs)
        super().__init__(expr)

    def __str__(self) -> str:
        return f"contains_any({self.expr}, {', '.join(self.substrs)}, case_insensitive={self.case_insensitive})"


class RLikeExpr(ScalarFunction):
    """Expression for matching a string column against a regular expression pattern.

    This expression creates a boolean result indicating whether each value in the input
    string column matches the specified regular expression pattern.

    Args:
        expr: The input string column expression
        pattern: The regular expression pattern to match against

    Raises:
        TypeError: If the input expression is not a string column
        ValidationError: If the regular expression pattern is invalid
    """

    function_name = "text.rlike"

    def __init__(self, expr: LogicalExpr, pattern: str):
        self.expr = expr
        self.pattern = pattern

        # Validate regex pattern at construction time
        try:
            re.compile(pattern)
        except Exception as e:
            raise ValidationError(f"Invalid regex pattern: {pattern}") from e

        # Only validate the string expression
        super().__init__(expr)

    def __str__(self) -> str:
        return f"rlike({self.expr}, {self.pattern})"


class LikeExpr(ScalarFunction):
    """Expression for matching a string column against a SQL LIKE pattern.

    This expression creates a boolean result indicating whether each value in the input
    string column matches the specified SQL LIKE pattern. The pattern can contain % and _ wildcards.

    Args:
        expr: The input string column expression
        pattern: The SQL LIKE pattern to match against (% for any sequence, _ for single character)

    Raises:
        TypeError: If the input expression is not a string column
        ValidationError: If the LIKE pattern is invalid
    """

    function_name = "text.like"

    def __init__(self, expr: LogicalExpr, pattern: str):
        self.expr = expr
        self.raw_pattern = pattern
        self.pattern = self._convert_to_regex(pattern)

        # Validate the converted pattern
        try:
            re.compile(self.pattern)
        except Exception as e:
            raise ValidationError(f"Invalid LIKE pattern: {self.raw_pattern}") from e

        # Only validate the string expression
        super().__init__(expr)

    def _convert_to_regex(self, pattern: str) -> str:
        # Convert SQL LIKE pattern to regex pattern
        # Escape special regex characters except % and _
        special_chars = r"[](){}^$.|+\\"
        pattern = "".join("\\" + c if c in special_chars else c for c in pattern)
        # Convert SQL wildcards to regex wildcards
        pattern = pattern.replace("%", ".*").replace("_", ".")
        return pattern

    def __str__(self) -> str:
        return f"like({self.expr}, {self.raw_pattern}, {self.pattern})"


class ILikeExpr(ScalarFunction):
    """Expression for case-insensitive matching of a string column against a SQL LIKE pattern.

    This expression creates a boolean result indicating whether each value in the input
    string column matches the specified SQL LIKE pattern, ignoring case. The pattern can
    contain % and _ wildcards.

    Args:
        expr: The input string column expression
        pattern: The SQL LIKE pattern to match against (% for any sequence, _ for single character)

    Raises:
        TypeError: If the input expression is not a string column
        ValidationError: If the LIKE pattern is invalid
    """

    function_name = "text.ilike"

    def __init__(self, expr: LogicalExpr, pattern: str):
        self.expr = expr
        self.raw_pattern = pattern
        self.pattern = self._convert_to_regex(pattern)

        # Validate the converted pattern
        try:
            re.compile(self.pattern)
        except Exception as e:
            raise ValidationError(f"Invalid ILIKE pattern: {self.raw_pattern}") from e

        # Only validate the string expression
        super().__init__(expr)

    def __str__(self) -> str:
        return f"ilike({self.expr}, {self.raw_pattern}, {self.pattern})"

    def _convert_to_regex(self, pattern: str) -> str:
        # Convert SQL LIKE pattern to regex pattern with case insensitivity
        # Escape special regex characters except % and _
        special_chars = r"[](){}^$.|+\\"
        pattern = "".join("\\" + c if c in special_chars else c for c in pattern)
        # Convert SQL wildcards to regex wildcards
        pattern = pattern.replace("%", ".*").replace("_", ".")
        return f"(?i){pattern}"


class TsParseExpr(ScalarFunction):
    function_name = "text.parse_transcript"

    def __init__(self, expr: LogicalExpr, format: str):
        self.expr = expr
        self.format = format

        # Only validate the string expression (format is not a LogicalExpr)
        super().__init__(expr)

    def __str__(self) -> str:
        return f"parse_transcript({self.expr}, {self.format})"


class StartsWithExpr(ScalarFunction):
    """Expression for checking if a string column starts with a substring.

    This expression creates a boolean result indicating whether each value in the input
    string column starts with the specified substring.

    Args:
        expr: The input string column expression
        substr: The substring to check for at the start of each value

    Raises:
        TypeError: If the input expression is not a string column
        ValidationError: If the substring starts with a regular expression anchor (^)
    """

    function_name = "text.starts_with"

    def __init__(self, expr: LogicalExpr, substr: Union[str, LogicalExpr]):
        self.expr = expr
        self.substr = substr

        # Validate substring if it is `str`
        if isinstance(substr, str) and substr.startswith("^"):
            raise ValidationError("substr should not start with a regular expression anchor")

        # Pass appropriate arguments to signature validation
        if isinstance(substr, LogicalExpr):
            super().__init__(expr, substr)  # Both string inputs
        else:
            super().__init__(expr)  # Only main string input

    def __str__(self) -> str:
        return f"starts_with({self.expr}, {self.substr})"


class EndsWithExpr(ScalarFunction):
    """Expression for checking if a string column ends with a substring.

    This expression creates a boolean result indicating whether each value in the input
    string column ends with the specified substring.

    Args:
        expr: The input string column expression
        substr: The substring to check for at the end of each value

    Raises:
        TypeError: If the input expression is not a string column
        ValidationError: If the substring ends with a regular expression anchor ($)
    """

    function_name = "text.ends_with"

    def __init__(self, expr: LogicalExpr, substr: Union[str, LogicalExpr]):
        self.expr = expr
        self.substr = substr

        if isinstance(substr, str) and substr.endswith("$"):
            raise ValidationError("substr should not end with a regular expression anchor")

        # Pass appropriate arguments to signature validation
        if isinstance(substr, LogicalExpr):
            super().__init__(expr, substr)  # Both string inputs
        else:
            super().__init__(expr)  # Only main string input

    def __str__(self) -> str:
        return f"ends_with({self.expr}, {self.substr})"


class RegexpSplitExpr(ScalarFunction):
    """Expression for splitting a string column using a regular expression pattern.

    This expression creates an array of substrings by splitting the input string column
    at matches of the specified regular expression pattern. The implementation uses
    str.replace to convert regex matches to a unique delimiter, then splits on that delimiter.

    Args:
        expr: The input string column expression
        pattern: The regular expression pattern to split on
        limit: Maximum number of splits to perform. -1 for unlimited splits (default).
              If > 0, returns at most limit+1 elements, with remainder in last element.

    Raises:
        TypeError: If the input expression is not a string column
    """

    function_name = "text.regexp_split"

    def __init__(self, expr: LogicalExpr, pattern: str, limit: int = -1):
        self.expr = expr
        self.pattern = pattern
        self.limit = limit

        # Only validate the string expression (pattern and limit are not LogicalExprs)
        super().__init__(expr)

    def __str__(self) -> str:
        return f"regexp_split({self.expr}, {self.pattern}, limit={self.limit})"



class SplitPartExpr(ScalarFunction):
    """Expression for splitting a string column and returning a specific part.

    This expression splits each string by a delimiter and returns the specified part (1-based indexing).
    The delimiter and part number can be either literal values or column expressions.
    When either is a column expression, the operation is performed dynamically using map_batches.

    Behavior:
    - If any input is null, returns null
    - If part_number is out of range of split parts, returns empty string
    - If part_number is 0, throws an error
    - If part_number is negative, counts from the end of the split parts
    - If the delimiter is an empty string, the string is not split

    Args:
        expr: The input string column expression
        delimiter: The delimiter to split on (can be a string or column expression)
        part_number: Which part to return (1-based, can be an integer or column expression)

    Raises:
        TypeError: If the input expression is not a string column
        ValidationError: If part_number is 0
    """

    function_name = "text.split_part"

    def __init__(
        self, expr: LogicalExpr, delimiter: Union[LogicalExpr, str], part_number: int
    ):
        self.expr = expr
        self.delimiter = delimiter
        self.part_number = part_number

        if part_number == 0:
            raise ValidationError("part_number cannot be 0")

        # Only validate LogicalExpr arguments
        if isinstance(delimiter, LogicalExpr):
            super().__init__(expr, delimiter)
        else:
            super().__init__(expr)

    def __str__(self) -> str:
        return (
            f"text_split({self.expr}, {self.delimiter}, part_number={self.part_number})"
        )


class StringCasingExpr(ScalarFunction):
    """Expression for converting the case of a string column.

    This expression creates a new string column with all values converted to the specified case.

    Args:
        expr: The input string column expression
        case: The case to convert the string to ("upper", "lower", "title")

    Raises:
        TypeError: If the input expression is not a string column
    """

    function_name = "text.string_casing"

    def __init__(self, expr: LogicalExpr, case: Literal["upper", "lower", "title"]):
        self.expr = expr
        self.case = case

        # Only validate the string expression (case is not LogicalExpr)
        super().__init__(expr)

    def __str__(self) -> str:
        return f"string_casing({self.expr}, {self.case})"


class StripCharsExpr(ScalarFunction):
    """Expression for removing specified characters from string ends.

    This expression creates a new string column with specified characters removed from
    the beginning and/or end of each string. The characters to remove can be specified
    as a literal string, a column expression, or None for whitespace.

    Args:
        expr: The input string column expression
        chars: The characters to remove (None for whitespace, can be string or column expression)
        side: Which side to strip from ("left", "right", "both")

    Raises:
        TypeError: If the input expression is not a string column
    """

    function_name = "text.strip_chars"

    def __init__(
        self,
        expr: LogicalExpr,
        chars: Union[LogicalExpr, str, None],
        side: Literal["left", "right", "both"] = "both",
    ):
        self.expr = expr
        self.chars = chars
        self.side = side

        # Only validate LogicalExpr arguments (chars might be string literal or None)
        if isinstance(chars, LogicalExpr):
            super().__init__(expr, chars)
        else:
            super().__init__(expr)

    def __str__(self) -> str:
        return f"strip_chars({self.expr}, {self.chars}, side={self.side})"

class ReplaceExpr(ScalarFunction):
    """Expression for replacing substrings in a string column.

    This expression creates a new string column with occurrences of a search pattern
    replaced with a replacement string. Both the search pattern and replacement can be
    either literal values or column expressions. When either is a column expression,
    the operation is performed dynamically using map_batches.

    Args:
        expr: The input string column expression
        search: The pattern to search for (can be a string or column expression)
        replacement: The string to replace with (can be a string or column expression)
        literal: Whether to treat the pattern as a literal string (True) or regex (False)
        replacement_count: Max number of replacements to make. -1 for all occurrences.

    Raises:
        TypeError: If the input expression is not a string column
        ValidationError: If replacement_count is not >= 1 or -1
    """

    function_name = "text.replace"

    def __init__(
        self,
        expr: LogicalExpr,
        search: Union[LogicalExpr, str],
        replacement: Union[LogicalExpr, str],
        literal: bool,
        replacement_count: int,
    ):
        self.expr = expr
        self.search = search
        self.literal = literal
        self.replacement = replacement
        self.replacement_count = replacement_count

        # Validate replacement_count at construction time
        if replacement_count != -1 and replacement_count < 1:
            raise ValidationError("replacement_count must be >= 1 or -1 for all")

        # Only validate LogicalExpr arguments
        logical_args = [expr]
        if isinstance(search, LogicalExpr):
            logical_args.append(search)
        if isinstance(replacement, LogicalExpr):
            logical_args.append(replacement)

        super().__init__(*logical_args)

    def __str__(self) -> str:
        return f"replace({self.expr}, {self.search}, {self.replacement}, {self.replacement_count})"


class StrLengthExpr(ScalarFunction):
    """Expression for calculating the length of a string column.

    This expression creates a new integer column with the number of characters in each value
    of the input string column.

    Args:
        expr: The input string column expression

    Raises:
        TypeError: If the input expression is not a string column
    """

    function_name = "text.str_length"

    def __init__(self, expr: LogicalExpr):
        self.expr = expr
        super().__init__(expr)


class ByteLengthExpr(ScalarFunction):
    """Expression for calculating the length of a string column in bytes.

    This expression creates a new integer column with the number of bytes in each value
    of the input string column.

    Args:
        expr: The input string column expression

    Raises:
        TypeError: If the input expression is not a string column
    """

    function_name = "text.byte_length"

    def __init__(self, expr: LogicalExpr):
        self.expr = expr
        super().__init__(expr)
