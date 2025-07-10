from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, List, Literal, Optional, Tuple, Union

if TYPE_CHECKING:
    from fenic.core._logical_plan import LogicalPlan

from pydantic import BaseModel, Field

from fenic.core._logical_plan.expressions.base import LogicalExpr
from fenic.core.error import TypeMismatchError, ValidationError
from fenic.core.types import (
    ArrayType,
    BooleanType,
    ColumnField,
    DoubleType,
    IntegerType,
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

class TextractExpr(LogicalExpr):
    def __init__(self, input_expr: LogicalExpr, template: str):
        self.input_expr = input_expr
        self.template = template
        self.parsed_template = ParsedTemplateFormat.parse(template)

    def __str__(self):
        return f"text.extract('{self.template}', {self.input_expr})"

    def _validate_types(self, plan: LogicalPlan):
        expr_field = self.input_expr.to_column_field(plan)
        if expr_field.data_type != StringType:
            raise TypeError(
                f"text.extract requires string type for input expression, got {expr_field.data_type}. "
                "The input must be a text column to extract fields from."
            )

    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
        self._validate_types(plan)
        result_field = ColumnField(str(self), self.parsed_template.to_struct_schema())
        return result_field

    def children(self) -> List[LogicalExpr]:
        return [self.input_expr]


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


class TextChunkExpr(LogicalExpr):
    def __init__(
        self,
        input_expr: LogicalExpr,
        chunk_configuration: TextChunkExprConfiguration,
    ):
        self.input_expr = input_expr
        self.chunk_configuration = chunk_configuration

    def __str__(self) -> str:
        return f"text_chunk({self.input_expr}, {self.chunk_configuration})"

    def _validate_types(self, plan: LogicalPlan):
        input_field = self.input_expr.to_column_field(plan)
        if input_field.data_type != StringType:
            raise TypeError(
                f"text_chunk requires a string input, got {input_field.data_type}"
            )

    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
        self._validate_types(plan)
        return ColumnField(name=str(self), data_type=ArrayType(element_type=StringType))

    def children(self) -> List[LogicalExpr]:
        return [self.input_expr]


class RecursiveTextChunkExprConfiguration(TextChunkExprConfiguration):
    chunking_character_set_name: ChunkCharacterSet = ChunkCharacterSet.ASCII
    chunking_character_set_custom_characters: Optional[list[str]] = None


class RecursiveTextChunkExpr(LogicalExpr):
    def __init__(
        self,
        input_expr: LogicalExpr,
        chunking_configuration: RecursiveTextChunkExprConfiguration,
    ):
        self.input_expr = input_expr
        self.chunking_configuration = chunking_configuration

    def __str__(self) -> str:
        return f"text_chunk({self.input_expr}, {self.chunking_configuration})"

    def _validate_types(self, plan: LogicalPlan):
        input_field = self.input_expr.to_column_field(plan)
        if input_field.data_type != StringType:
            raise TypeError(
                f"text_chunk requires a string input, got {input_field.data_type}"
            )

    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
        self._validate_types(plan)
        return ColumnField(name=str(self), data_type=ArrayType(element_type=StringType))

    def children(self) -> List[LogicalExpr]:
        return [self.input_expr]


class CountTokensExpr(LogicalExpr):
    def __init__(
        self,
        input_expr: LogicalExpr,
    ):
        self.input_expr = input_expr

    def __str__(self) -> str:
        return f"count_tokens({self.input_expr})"

    def _validate_types(self, plan: "LogicalPlan"):
        input_field = self.input_expr.to_column_field(plan)
        if input_field.data_type != StringType:
            raise TypeError(
                f"count_tokens requires a string input, got {input_field.data_type}"
            )

    def to_column_field(self, plan: "LogicalPlan") -> ColumnField:
        self._validate_types(plan)
        return ColumnField(name=str(self), data_type=IntegerType)

    def children(self) -> List[LogicalExpr]:
        return [self.input_expr]

class ConcatExpr(LogicalExpr):
    def __init__(self, exprs: List[LogicalExpr]):
        self.exprs = exprs

    def __str__(self) -> str:
        return f"concat({', '.join(str(expr) for expr in self.exprs)})"

    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
        for arg in self.exprs:
            arg_field = arg.to_column_field(plan)
            if isinstance(arg_field.data_type, ArrayType) or isinstance(
                arg_field.data_type, StructType
            ):
                raise TypeError(
                    f"concat requires primitive types castable to strings, got {arg_field.data_type}"
                )
        return ColumnField(name=str(self), data_type=StringType)

    def children(self) -> List[LogicalExpr]:
        return self.exprs


class ArrayJoinExpr(LogicalExpr):
    def __init__(self, expr: LogicalExpr, delimiter: str):
        self.expr = expr
        self.delimiter = delimiter

    def __str__(self) -> str:
        return f"array_join({self.expr}, {self.delimiter})"

    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
        input_field = self.expr.to_column_field(plan)
        if input_field.data_type != ArrayType(element_type=StringType):
            raise TypeError(
                f"array_join requires an array of strings as input, got {input_field.data_type}"
            )
        return ColumnField(name=str(self), data_type=StringType)

    def children(self) -> List[LogicalExpr]:
        return [self.expr]


class ContainsExpr(LogicalExpr):
    """Expression for checking if a string column contains a substring.

    This expression creates a boolean result indicating whether each value in the input
    string column contains the specified substring.

    Args:
        expr: The input string column expression
        substr: The substring to search for within each value

    Raises:
        TypeError: If the input expression is not a string column
    """

    def __init__(self, expr: LogicalExpr, substr: Union[str, LogicalExpr]):
        self.expr = expr
        self.substr = substr

    def __str__(self) -> str:
        return f"contains({self.expr}, {self.substr})"

    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
        input_field = self.expr.to_column_field(plan)
        if input_field.data_type != StringType:
            raise TypeError(
                f"contains requires column of type string as input, got {input_field.data_type}"
            )
        return ColumnField(name=str(self), data_type=BooleanType)

    def children(self) -> List[LogicalExpr]:
        if isinstance(self.substr, str):
            return [self.expr]
        else:
            return [self.expr, self.substr]


class ContainsAnyExpr(LogicalExpr):
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

    def __init__(
        self, expr: LogicalExpr, substrs: List[str], case_insensitive: bool = True
    ):
        self.expr = expr
        self.substrs = substrs
        self.case_insensitive = case_insensitive

    def __str__(self) -> str:
        return f"contains_any({self.expr}, {', '.join(self.substrs)}, case_insensitive={self.case_insensitive})"

    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
        input_field = self.expr.to_column_field(plan)
        if input_field.data_type != StringType:
            raise TypeError(
                f"contains_any requires column of type string as input, got {input_field.data_type}"
            )
        return ColumnField(name=str(self), data_type=BooleanType)

    def children(self) -> List[LogicalExpr]:
        return [self.expr]


class RLikeExpr(LogicalExpr):
    """Expression for matching a string column against a regular expression pattern.

    This expression creates a boolean result indicating whether each value in the input
    string column matches the specified regular expression pattern.

    Args:
        expr: The input string column expression
        pattern: The regular expression pattern to match against

    Raises:
        TypeError: If the input expression is not a string column
        ValueError: If the regular expression pattern is invalid
    """

    def __init__(self, expr: LogicalExpr, pattern: str):
        self.expr = expr
        self.pattern = pattern

    def __str__(self) -> str:
        return f"rlike({self.expr}, {self.pattern})"

    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
        input_field = self.expr.to_column_field(plan)
        if input_field.data_type != StringType:
            raise TypeError(
                f"rlike requires column of type string as input, got {input_field.data_type}"
            )
        try:
            re.compile(self.pattern)
        except Exception as e:
            raise ValueError(f"Invalid regex pattern: {self.pattern}") from e
        return ColumnField(name=str(self), data_type=BooleanType)

    def children(self) -> List[LogicalExpr]:
        return [self.expr]


class LikeExpr(LogicalExpr):
    """Expression for matching a string column against a SQL LIKE pattern.

    This expression creates a boolean result indicating whether each value in the input
    string column matches the specified SQL LIKE pattern. The pattern can contain % and _ wildcards.

    Args:
        expr: The input string column expression
        pattern: The SQL LIKE pattern to match against (% for any sequence, _ for single character)

    Raises:
        TypeError: If the input expression is not a string column
        ValueError: If the LIKE pattern is invalid
    """

    def __init__(self, expr: LogicalExpr, pattern: str):
        self.expr = expr
        self.raw_pattern = pattern
        self.pattern = self._convert_to_regex(pattern)

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

    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
        input_field = self.expr.to_column_field(plan)
        if input_field.data_type != StringType:
            raise TypeError(
                f"like/ilike requires column of type string as input, got {input_field.data_type}"
            )
        try:
            re.compile(self.pattern)
        except Exception as e:
            raise ValueError(f"Invalid LIKE/ILIKE pattern: {self.raw_pattern}") from e
        return ColumnField(name=str(self), data_type=BooleanType)

    def children(self) -> List[LogicalExpr]:
        return [self.expr]


class ILikeExpr(LikeExpr):
    """Expression for case-insensitive matching of a string column against a SQL LIKE pattern.

    This expression creates a boolean result indicating whether each value in the input
    string column matches the specified SQL LIKE pattern, ignoring case. The pattern can
    contain % and _ wildcards.

    Args:
        expr: The input string column expression
        pattern: The SQL LIKE pattern to match against (% for any sequence, _ for single character)

    Raises:
        TypeError: If the input expression is not a string column
        ValueError: If the LIKE pattern is invalid
    """

    def __init__(self, expr: LogicalExpr, pattern: str):
        self.expr = expr
        self.raw_pattern = pattern
        self.pattern = self._convert_to_regex(pattern)

    def __str__(self) -> str:
        return f"ilike({self.expr}, {self.raw_pattern}, {self.pattern})"

    def _convert_to_regex(self, pattern: str) -> str:
        converted_pattern = super()._convert_to_regex(pattern)
        return f"(?i){converted_pattern}"

    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
        return super().to_column_field(plan)

    def count_semantic_predicate_expressions(self) -> int:
        return self.expr.count_semantic_predicate_expressions()

    def children(self) -> List[LogicalExpr]:
        return [self.expr]


class TsParseExpr(LogicalExpr):
    # Unified schema for all transcript formats
    OUTPUT_TYPE = ArrayType(
        element_type=StructType(
            [
                StructField("index", IntegerType),        # Optional[int] - Entry index (1-based)
                StructField("speaker", StringType),       # Optional[str] - Speaker name
                StructField("start_time", DoubleType),    # float - Start time in seconds
                StructField("end_time", DoubleType),      # Optional[float] - End time in seconds
                StructField("duration", DoubleType),      # Optional[float] - Duration in seconds
                StructField("content", StringType),       # str - Transcript content/text
                StructField("format", StringType),        # str - Original format ("srt" or "generic")
            ]
        )
    )

    def __init__(self, expr: LogicalExpr, format: str):
        self.expr = expr
        self.format = format

    def __str__(self) -> str:
        return f"parse_transcript({self.expr}, {self.format})"

    def _validate_types(self, plan: LogicalPlan):
        input_field = self.expr.to_column_field(plan)
        if input_field.data_type != StringType:
            raise TypeMismatchError(
                "text.parse_transcript()",
                self.expr,
                StringType,
            )

    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
        self._validate_types(plan)
        return ColumnField(name=str(self), data_type=self.OUTPUT_TYPE)

    def children(self) -> List[LogicalExpr]:
        return [self.expr]


class StartsWithExpr(LogicalExpr):
    """Expression for checking if a string column starts with a substring.

    This expression creates a boolean result indicating whether each value in the input
    string column starts with the specified substring.

    Args:
        expr: The input string column expression
        substr: The substring to check for at the start of each value

    Raises:
        TypeError: If the input expression is not a string column
        ValueError: If the substring starts with a regular expression anchor (^)
    """

    def __init__(self, expr: LogicalExpr, substr: Union[str, LogicalExpr]):
        self.expr = expr
        self.substr = substr

    def __str__(self) -> str:
        return f"starts_with({self.expr}, {self.substr})"

    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
        input_field = self.expr.to_column_field(plan)
        if input_field.data_type != StringType:
            raise TypeError(
                f"starts_with requires column of type string as input, got {input_field.data_type}"
            )
        if isinstance(self.substr, str) and self.substr.startswith("^"):
            raise ValueError("substr should not start with a regular expression anchor")
        return ColumnField(name=str(self), data_type=BooleanType)

    def children(self) -> List[LogicalExpr]:
        if isinstance(self.substr, str):
            return [self.expr]
        else:
            return [self.expr, self.substr]


class EndsWithExpr(LogicalExpr):
    """Expression for checking if a string column ends with a substring.

    This expression creates a boolean result indicating whether each value in the input
    string column ends with the specified substring.

    Args:
        expr: The input string column expression
        substr: The substring to check for at the end of each value

    Raises:
        TypeError: If the input expression is not a string column
        ValueError: If the substring ends with a regular expression anchor ($)
    """

    def __init__(self, expr: LogicalExpr, substr: Union[str, LogicalExpr]):
        self.expr = expr
        self.substr = substr

    def __str__(self) -> str:
        return f"ends_with({self.expr}, {self.substr})"

    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
        input_field = self.expr.to_column_field(plan)
        if input_field.data_type != StringType:
            raise TypeError(
                f"ends_with requires column of type string as input, got {input_field.data_type}"
            )
        if isinstance(self.substr, str) and self.substr.endswith("$"):
            raise ValueError("substr should not end with a regular expression anchor")
        return ColumnField(name=str(self), data_type=BooleanType)

    def children(self) -> List[LogicalExpr]:
        if isinstance(self.substr, str):
            return [self.expr]
        else:
            return [self.expr, self.substr]


class RegexpSplitExpr(LogicalExpr):
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

    def __init__(self, expr: LogicalExpr, pattern: str, limit: int = -1):
        self.expr = expr
        self.pattern = pattern
        self.limit = limit

    def __str__(self) -> str:
        return f"regexp_split({self.expr}, {self.pattern}, limit={self.limit})"

    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
        input_field = self.expr.to_column_field(plan)
        if input_field.data_type != StringType:
            raise TypeError(
                f"regexp_split requires column of type string as input, got {input_field.data_type}"
            )
        return ColumnField(name=str(self), data_type=ArrayType(element_type=StringType))

    def children(self) -> List[LogicalExpr]:
        return [self.expr]


class SplitPartExpr(LogicalExpr):
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
        ValueError: If part_number is 0
    """

    def __init__(
        self, expr: LogicalExpr, delimiter: Union[LogicalExpr, str], part_number: int
    ):
        self.expr = expr
        self.delimiter = delimiter
        self.part_number = part_number

    def __str__(self) -> str:
        return (
            f"text_split({self.expr}, {self.delimiter}, part_number={self.part_number})"
        )

    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
        input_field = self.expr.to_column_field(plan)
        if input_field.data_type != StringType:
            raise TypeError(
                f"text_split requires column of type string as input, got {input_field.data_type}"
            )
        return ColumnField(name=str(self), data_type=StringType)

    def children(self) -> List[LogicalExpr]:
        if isinstance(self.delimiter, str):
            return [self.expr]
        else:
            return [self.expr, self.delimiter]


class StringCasingExpr(LogicalExpr):
    """Expression for converting the case of a string column.

    This expression creates a new string column with all values converted to the specified case.

    Args:
        expr: The input string column expression
        case: The case to convert the string to ("upper", "lower", "title")

    Raises:
        TypeError: If the input expression is not a string column
    """

    def __init__(self, expr: LogicalExpr, case: Literal["upper", "lower", "title"]):
        self.expr = expr
        self.case = case

    def __str__(self) -> str:
        return f"string_casing({self.expr}, {self.case})"

    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
        input_field = self.expr.to_column_field(plan)
        if input_field.data_type != StringType:
            raise TypeError(
                f"string_casing requires column of type string as input, got {input_field.data_type}"
            )
        return ColumnField(name=str(self), data_type=StringType)

    def children(self) -> List[LogicalExpr]:
        return [self.expr]


class StripCharsExpr(LogicalExpr):
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

    def __init__(
        self,
        expr: LogicalExpr,
        chars: Union[LogicalExpr, str, None],
        side: Literal["left", "right", "both"] = "both",
    ):
        self.expr = expr
        self.chars = chars
        self.side = side

    def __str__(self) -> str:
        return f"strip_chars({self.expr}, {self.chars}, side={self.side})"

    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
        input_field = self.expr.to_column_field(plan)
        if input_field.data_type != StringType:
            raise TypeError(
                f"strip_chars requires column of type string as input, got {input_field.data_type}"
            )
        return ColumnField(name=str(self), data_type=StringType)

    def children(self) -> List[LogicalExpr]:
        return [self.expr]


class ReplaceExpr(LogicalExpr):
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
        ValueError: If replacement_count is not >= 1 or -1
    """

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

    def __str__(self) -> str:
        return f"replace({self.expr}, {self.search}, {self.replacement}, {self.replacement_count})"

    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
        input_field = self.expr.to_column_field(plan)
        if input_field.data_type != StringType:
            raise TypeError(
                f"replace requires column of type string as input, got {input_field.data_type}"
            )
        if self.replacement_count != -1 and self.replacement_count < 1:
            raise ValueError("replacement_count must be >= 1 or -1 for all")
        return ColumnField(name=str(self), data_type=StringType)

    def children(self) -> List[LogicalExpr]:
        return [self.expr]


class StrLengthExpr(LogicalExpr):
    """Expression for calculating the length of a string column.

    This expression creates a new integer column with the number of characters in each value
    of the input string column.

    Args:
        expr: The input string column expression

    Raises:
        TypeError: If the input expression is not a string column
    """

    def __init__(self, expr: LogicalExpr):
        self.expr = expr

    def __str__(self) -> str:
        return f"str_length({self.expr})"

    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
        input_field = self.expr.to_column_field(plan)
        if input_field.data_type != StringType:
            raise TypeError(
                f"str_length requires column of type string as input, got {input_field.data_type}"
            )
        return ColumnField(name=str(self), data_type=IntegerType)

    def children(self) -> List[LogicalExpr]:
        return [self.expr]


class ByteLengthExpr(LogicalExpr):
    """Expression for calculating the length of a string column in bytes.

    This expression creates a new integer column with the number of bytes in each value
    of the input string column.

    Args:
        expr: The input string column expression

    Raises:
        TypeError: If the input expression is not a string column
    """

    def __init__(self, expr: LogicalExpr):
        self.expr = expr

    def __str__(self) -> str:
        return f"byte_length({self.expr})"

    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
        input_field = self.expr.to_column_field(plan)
        if input_field.data_type != StringType:
            raise TypeError(
                f"byte_length requires column of type string as input, got {input_field.data_type}"
            )
        return ColumnField(name=str(self), data_type=IntegerType)

    def children(self) -> List[LogicalExpr]:
        return [self.expr]
