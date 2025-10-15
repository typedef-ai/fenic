from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

from fenic.core._logical_plan.jinja_validation import (
    VariableTree,
)
from fenic.core.types.enums import (
    StringCasingType,
    StripCharsSide,
    TranscriptFormatType,
)

if TYPE_CHECKING:
    from fenic.core._logical_plan.plans.base import LogicalPlan

import logging

from pydantic import BaseModel, Field

from fenic._polars_plugins import py_validate_regex  # noqa: F401
from fenic.core._interfaces.session_state import BaseSessionState
from fenic.core._logical_plan.expressions.base import (
    LogicalExpr,
    UnparameterizedExpr,
    ValidatedDynamicSignature,
    ValidatedSignature,
)
from fenic.core._logical_plan.expressions.basic import (
    AliasExpr,
    ColumnExpr,
    LiteralExpr,
)
from fenic.core._logical_plan.signatures.signature_validator import SignatureValidator
from fenic.core.error import ValidationError
from fenic.core.types import (
    ColumnField,
    DataType,
    FuzzySimilarityMethod,
    JsonType,
    StringType,
    StructField,
    StructType,
)

logger = logging.getLogger(__name__)

class ChunkLengthFunction(Enum):
    CHARACTER = "CHARACTER"
    WORD = "WORD"
    # trunk-ignore(bandit/B105): not a token
    TOKEN = "TOKEN"


class ChunkCharacterSet(Enum):
    CUSTOM = "CUSTOM"
    ASCII = "ASCII"
    UNICODE = "UNICODE"


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

class TextractExpr(ValidatedDynamicSignature, LogicalExpr):
    function_name = "text.extract"

    def __init__(self, input_expr: LogicalExpr, template: str):
        self.input_expr = input_expr
        self.template = template
        self.parsed_template = ParsedTemplateFormat.parse(template)
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self):
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return [self.input_expr]

    def __str__(self):
        return f"{self.function_name}('{self.template}', {self.input_expr})"

    def _infer_dynamic_return_type(self, arg_types: List[DataType], plan: LogicalPlan, session_state: BaseSessionState) -> DataType:
        """Return StructType with fields based on parsed template."""
        return self.parsed_template.to_struct_schema()

    def _eq_specific(self, other: TextractExpr) -> bool:
        return self.template == other.template


class TextChunkExprConfiguration(BaseModel):
    desired_chunk_size: int = Field(gt=0)
    chunk_overlap_percentage: int = Field(default=0, ge=0, lt=100)
    chunk_length_function_name: ChunkLengthFunction = ChunkLengthFunction.TOKEN


class TextChunkExpr(ValidatedSignature, LogicalExpr):
    function_name = "text.chunk"

    def __init__(
        self,
        input_expr: LogicalExpr,
        chunking_configuration: TextChunkExprConfiguration,
    ):
        self.input_expr = input_expr
        # Create the configuration object for internal use
        self.chunking_configuration = chunking_configuration
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self):
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return [self.input_expr]

    def __str__(self) -> str:
        return f"{self.function_name}({self.input_expr}, {self.chunking_configuration})"

    def _eq_specific(self, other: TextChunkExpr) -> bool:
        return self.chunking_configuration == other.chunking_configuration

class RecursiveTextChunkExprConfiguration(TextChunkExprConfiguration):
    chunking_character_set_name: ChunkCharacterSet = ChunkCharacterSet.ASCII
    chunking_character_set_custom_characters: Optional[list[str]] = None


class RecursiveTextChunkExpr(ValidatedSignature, LogicalExpr):
    function_name = "text.recursive_chunk"

    def __init__(
        self,
        input_expr: LogicalExpr,
        chunking_configuration: RecursiveTextChunkExprConfiguration,
    ):
        self.input_expr = input_expr
        # Create the configuration object for internal use
        self.chunking_configuration = chunking_configuration
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self):
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return [self.input_expr]

    def __str__(self) -> str:
        return f"{self.function_name}({self.input_expr}, {self.chunking_configuration})"

    def _eq_specific(self, other: RecursiveTextChunkExpr) -> bool:
        return self.chunking_configuration == other.chunking_configuration


class CountTokensExpr(ValidatedSignature, UnparameterizedExpr, LogicalExpr):
    function_name = "text.count_tokens"

    def __init__(self, input_expr: LogicalExpr):
        self.input_expr = input_expr
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self):
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return [self.input_expr]


class ConcatExpr(ValidatedSignature, UnparameterizedExpr, LogicalExpr):
    function_name = "text.concat"

    def __init__(self, exprs: List[LogicalExpr]):
        self.exprs = exprs
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self):
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return self.exprs


class ArrayJoinExpr(ValidatedSignature, LogicalExpr):
    function_name = "text.array_join"

    def __init__(self, expr: LogicalExpr, delimiter: str):
        self.expr = expr
        self.delimiter = delimiter
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self):
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return [self.expr]

    def __str__(self) -> str:
        return f"{self.function_name}({self.expr}, {self.delimiter})"

    def _eq_specific(self, other: ArrayJoinExpr) -> bool:
        return self.delimiter == other.delimiter


class ContainsExpr(ValidatedSignature, UnparameterizedExpr, LogicalExpr):
    """Expression for checking if a string column contains a substring.

    This expression creates a boolean result indicating whether each value in the input
    string column contains the specified substring.

    Args:
        expr: The input string column expression
        substr: The substring to search for within each value (column expression or LiteralExpr string)

    Raises:
        TypeError: If the input expression is not a string column
    """

    function_name = "text.contains"

    def __init__(self, expr: LogicalExpr, substr: LogicalExpr):
        self.expr = expr
        self.substr = substr
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self):
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return [self.expr, self.substr]

    def __str__(self) -> str:
        return f"{self.function_name}({self.expr}, {self.substr})"


class ContainsAnyExpr(ValidatedSignature, LogicalExpr):
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
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self):
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return [self.expr]

    def __str__(self) -> str:
        return f"{self.function_name}({self.expr}, {', '.join(self.substrs)}, case_insensitive={self.case_insensitive})"

    def _eq_specific(self, other: ContainsAnyExpr) -> bool:
        return self.substrs == other.substrs and self.case_insensitive == other.case_insensitive


class RLikeExpr(ValidatedSignature, LogicalExpr):
    """Expression for matching a string column against a regular expression pattern.

    This expression creates a boolean result indicating whether each value in the input
    string column matches the specified regular expression pattern.

    Args:
        expr: The input string column expression
        pattern: The regular expression pattern to match against

    Raises:
        ValidationError: If the regular expression pattern is invalid
    """

    function_name = "text.rlike"

    def __init__(self, expr: LogicalExpr, pattern: LogicalExpr):
        if isinstance(pattern, LiteralExpr) and pattern.data_type == StringType:
            try:
                py_validate_regex(pattern.literal)
            except Exception as e:
                raise ValidationError(f"Invalid regex pattern: {pattern.literal}") from e

        self.expr = expr
        self.pattern = pattern
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self):
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return [self.expr, self.pattern]

    def __str__(self) -> str:
        return f"{self.function_name}({self.expr}, {self.pattern})"

    def _eq_specific(self, other: RLikeExpr) -> bool:
        return self.pattern == other.pattern


class LikeExpr(ValidatedSignature, LogicalExpr):
    """Expression for matching a string column against a SQL LIKE pattern.

    This expression creates a boolean result indicating whether each value in the input
    string column matches the specified SQL LIKE pattern. The pattern can contain % and _ wildcards.

    Args:
        expr: The input string column expression
        pattern: The SQL LIKE pattern to match against (% for any sequence, _ for single character)

    Raises:
        TypeError: If the input expression is not a literal expression that resolves to a string.
        ValidationError: If the LIKE pattern is invalid
    """

    function_name = "text.like"

    def __init__(self, expr: LogicalExpr, pattern: LogicalExpr):
        self.expr = expr
        self.pattern = pattern

        if isinstance(pattern, LiteralExpr) and pattern.data_type == StringType:
            try:
                py_validate_regex(pattern.literal)
            except Exception as e:
                raise ValidationError(f"Invalid LIKE pattern: {self.pattern}") from e

        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self):
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return [self.expr, self.pattern]

    def __str__(self) -> str:
        return f"{self.function_name}({self.expr}, {self.pattern})"

    def _eq_specific(self, other: LikeExpr) -> bool:
        return self.pattern == other.pattern


class ILikeExpr(ValidatedSignature, LogicalExpr):
    """Expression for case-insensitive matching of a string column against a SQL LIKE pattern.

    This expression creates a boolean result indicating whether each value in the input
    string column matches the specified SQL LIKE pattern, ignoring case. The pattern can
    contain % and _ wildcards.

    Args:
        expr: The input string column expression
        pattern: The SQL LIKE pattern to match against (% for any sequence, _ for single character)

    Raises:
        TypeError: If the input expression is not a literal expression that resolves to a string.
        ValidationError: If the LIKE pattern is invalid.
    """

    function_name = "text.ilike"

    def __init__(self, expr: LogicalExpr, pattern: LogicalExpr):
        self.expr = expr
        self.pattern = pattern

        if isinstance(pattern, LiteralExpr) and pattern.data_type == StringType:
            try:
                py_validate_regex(pattern.literal)
            except Exception as e:
                raise ValidationError(f"Invalid ILIKE pattern: {self.pattern}") from e

        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self):
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return [self.expr, self.pattern]

    def __str__(self) -> str:
        return f"{self.function_name}({self.expr}, {self.pattern})"

    def _eq_specific(self, other: ILikeExpr) -> bool:
        return self.pattern == other.pattern


class TsParseExpr(ValidatedSignature, LogicalExpr):
    function_name = "text.parse_transcript"

    def __init__(self, expr: LogicalExpr, format: TranscriptFormatType):
        self.expr = expr
        self.format = format
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self):
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return [self.expr]

    def __str__(self) -> str:
        return f"{self.function_name}({self.expr}, {self.format})"

    def _eq_specific(self, other: TsParseExpr) -> bool:
        return self.format == other.format


class StartsWithExpr(ValidatedSignature, UnparameterizedExpr, LogicalExpr):
    """Expression for checking if a string column starts with a substring.

    This expression creates a boolean result indicating whether each value in the input
    string column starts with the specified substring.

    Args:
        expr: The input string column expression
        substr: The substring to check for at the start of each value (column expression or LiteralExpr string)

    Raises:
        TypeError: If the input expression is not a string column
        ValidationError: If the substring starts with a regular expression anchor (^)
    """

    function_name = "text.starts_with"

    def __init__(self, expr: LogicalExpr, substr: LogicalExpr):
        self.expr = expr
        self.substr = substr
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self):
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return [self.expr, self.substr]


class EndsWithExpr(ValidatedSignature, UnparameterizedExpr, LogicalExpr):
    """Expression for checking if a string column ends with a substring.

    This expression creates a boolean result indicating whether each value in the input
    string column ends with the specified substring.

    Args:
        expr: The input string column expression
        substr: The substring to check for at the end of each value (column expression or LiteralExpr string)

    Raises:
        TypeError: If the input expression is not a string column
        ValidationError: If the substring ends with a regular expression anchor ($)
    """

    function_name = "text.ends_with"

    def __init__(self, expr: LogicalExpr, substr: LogicalExpr):
        self.expr = expr
        self.substr = substr
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self):
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return [self.expr, self.substr]


class RegexpSplitExpr(ValidatedSignature, LogicalExpr):
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
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self):
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return [self.expr]

    def __str__(self) -> str:
        return f"{self.function_name}({self.expr}, {self.pattern}, limit={self.limit})"

    def _eq_specific(self, other: RegexpSplitExpr) -> bool:
        return self.pattern == other.pattern and self.limit == other.limit

class SplitPartExpr(ValidatedSignature, UnparameterizedExpr, LogicalExpr):
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
        delimiter: The delimiter to split on (column expression or LiteralExpr string)
        part_number: Which part to return (1-based, column expression or LiteralExpr integer)

    Raises:
        TypeMismatchError: If the input expression is not a string column
        ValidationError: If part_number is 0
    """

    function_name = "text.split_part"

    def __init__(
        self, expr: LogicalExpr, delimiter: LogicalExpr, part_number: LogicalExpr
    ):
        self.expr = expr
        self.delimiter = delimiter
        self.part_number = part_number
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self):
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return [self.expr, self.delimiter, self.part_number]


class StringCasingExpr(ValidatedSignature, LogicalExpr):
    """Expression for converting the case of a string column.

    This expression creates a new string column with all values converted to the specified case.

    Args:
        expr: The input string column expression
        case: The case to convert the string to ("upper", "lower", "title")

    Raises:
        TypeError: If the input expression is not a string column
    """

    function_name = "text.string_casing"

    def __init__(self, expr: LogicalExpr, case: StringCasingType):
        self.expr = expr
        self.case = case
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self):
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return [self.expr]

    def _eq_specific(self, other: StringCasingExpr) -> bool:
        return self.case == other.case


class StripCharsExpr(ValidatedSignature, LogicalExpr):
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
        chars: Optional[LogicalExpr],
        side: StripCharsSide = "both",
    ):
        self.expr = expr
        self.chars = chars
        self.side = side
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self):
        return self._validator

    def children(self) -> List[LogicalExpr]:
        if self.chars is not None:
            return [self.expr, self.chars]
        else:
            return [self.expr]

    def __str__(self) -> str:
        return f"{self.function_name}({self.expr}, {self.chars}, side={self.side})"

    def _eq_specific(self, other: StripCharsExpr) -> bool:
        return self.side == other.side

class ReplaceExpr(ValidatedSignature, LogicalExpr):
    """Expression for replacing substrings in a string column.

    This expression creates a new string column with occurrences of a search pattern
    replaced with a replacement string. Both the search pattern and replacement can be
    either literal values or column expressions. When either is a column expression,
    the operation is performed dynamically using map_batches.

    Args:
        expr: The input string column expression
        search: The pattern to search for (column expression or LiteralExpr string)
        replacement: The string to replace with (column expression or LiteralExpr string)
        literal: Whether to treat the pattern as a literal string (True) or regex (False)

    Raises:
        TypeError: If the input expression is not a string column
        ValidationError: If replacement_count is not >= 1 or -1
    """

    function_name = "text.replace"

    def __init__(
        self,
        expr: LogicalExpr,
        search: LogicalExpr,
        replacement: LogicalExpr,
        literal: bool,
    ):
        self.expr = expr
        self.search = search
        self.literal = literal
        self.replacement = replacement

        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self):
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return [self.expr, self.search, self.replacement]

    def __str__(self) -> str:
        return f"{self.function_name}({self.expr}, {self.search}, {self.replacement})"

    def _eq_specific(self, other: ReplaceExpr) -> bool:
        return self.literal == other.literal

class StrLengthExpr(ValidatedSignature, UnparameterizedExpr, LogicalExpr):
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
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self):
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return [self.expr]


class ByteLengthExpr(ValidatedSignature, UnparameterizedExpr, LogicalExpr):
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
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self):
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return [self.expr]


class JinjaExpr(LogicalExpr):
    """Expression for evaluating a Jinja template.

    This expression creates a new string column with the result of evaluating the Jinja template.

    Args:
        exprs: The input string column expressions
        template: The Jinja template to evaluate
    """

    def __init__(self, exprs: List[Union[ColumnExpr, AliasExpr]], template: str, strict: bool):
        self.template = template
        self.strict = strict
        self.variable_tree: VariableTree = VariableTree.from_jinja_template(template)
        self.exprs = self.variable_tree.filter_used_expressions(exprs)

    def children(self) -> List[LogicalExpr]:
        return self.exprs

    def to_column_field(self, plan: LogicalPlan, session_state: BaseSessionState) -> ColumnField:
        for expr in self.exprs:
            data_type = expr.to_column_field(plan, session_state).data_type
            self.variable_tree.validate_jinja_variable(expr.name, data_type)

        return ColumnField(
            name=str(self),
            data_type=StringType,
        )

    def __str__(self) -> str:
        return f"text.jinja({self.template}, {', '.join(str(expr) for expr in self.exprs)})"

    def _eq_specific(self, other: JinjaExpr) -> bool:
        return self.template == other.template and self.strict == other.strict

class FuzzyRatioExpr(ValidatedSignature, LogicalExpr):
    """Expression for computing the similarity between two strings using a fuzzy matching algorithm.

    This expression creates a new float column with the similarity score between the two input strings.
    The similarity score is computed using a fuzzy matching algorithm.

    Args:
        expr: The input string column expression
        other: The other string column expression
    """

    function_name = "text.fuzzy_ratio"

    def __init__(self, expr: LogicalExpr, other: LogicalExpr, method: FuzzySimilarityMethod):
        self.expr = expr
        self.other = other
        self.method = method
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self):
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return [self.expr, self.other]

    def _eq_specific(self, other: FuzzyRatioExpr) -> bool:
        return self.method == other.method

class FuzzyTokenSortRatioExpr(ValidatedSignature, LogicalExpr):
    """Expression for computing the fuzzy token sort ratio between two strings.

    This expression creates a new float column with the fuzzy token sort ratio between the two input strings.
    The fuzzy token sort ratio is computed using a fuzzy matching algorithm.
    """

    function_name = "text.fuzzy_token_sort_ratio"

    def __init__(self, expr: LogicalExpr, other: LogicalExpr, method: FuzzySimilarityMethod):
        self.expr = expr
        self.other = other
        self.method = method
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self):
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return [self.expr, self.other]

    def _eq_specific(self, other: FuzzyTokenSortRatioExpr) -> bool:
        return self.method == other.method

class FuzzyTokenSetRatioExpr(ValidatedSignature, LogicalExpr):
    """Expression for computing the fuzzy token set ratio between two strings.

    This expression creates a new float column with the fuzzy token set ratio between the two input strings.
    The fuzzy token set ratio is computed using a fuzzy matching algorithm.
    """

    function_name = "text.fuzzy_token_set_ratio"

    def __init__(self, expr: LogicalExpr, other: LogicalExpr, method: FuzzySimilarityMethod):
        self.expr = expr
        self.other = other
        self.method = method
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self):
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return [self.expr, self.other]

    def _eq_specific(self, other: FuzzyTokenSetRatioExpr) -> bool:
        return self.method == other.method


class RegexpCountExpr(ValidatedSignature, LogicalExpr):
    """Expression for counting regex pattern matches in a string.

    Counts the number of times the regex pattern is matched in the string.

    Args:
        expr: The input string column expression
        pattern: The regex pattern to count

    Raises:
        ValidationError: If the regex pattern is invalid
    """

    function_name = "text.regexp_count"

    def __init__(self, expr: LogicalExpr, pattern: LogicalExpr):
        if isinstance(pattern, LiteralExpr) and pattern.data_type == StringType:
            try:
                py_validate_regex(pattern.literal)
            except Exception as e:
                raise ValidationError(f"Invalid regex pattern: {pattern.literal}") from e

        self.expr = expr
        self.pattern = pattern
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self):
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return [self.expr, self.pattern]

    def _eq_specific(self, other: RegexpCountExpr) -> bool:
        return self.pattern == other.pattern


class RegexpExtractExpr(ValidatedSignature, LogicalExpr):
    """Expression for extracting a regex group from a string.

    Extracts a specific group matched by the regex pattern from the string.

    Args:
        expr: The input string column expression
        pattern: The regex pattern with capture groups
        idx: The group index to extract (0 = entire match, 1+ = capture groups)

    Raises:
        ValidationError: If the regex pattern is invalid
    """

    function_name = "text.regexp_extract"

    def __init__(self, expr: LogicalExpr, pattern: LogicalExpr, idx: int):
        if isinstance(pattern, LiteralExpr) and pattern.data_type == StringType:
            try:
                py_validate_regex(pattern.literal)
            except Exception as e:
                raise ValidationError(f"Invalid regex pattern: {pattern.literal}") from e

        self.expr = expr
        self.pattern = pattern
        self.idx = idx
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self):
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return [self.expr, self.pattern]

    def _eq_specific(self, other: RegexpExtractExpr) -> bool:
        return self.pattern == other.pattern and self.idx == other.idx


class RegexpExtractAllExpr(ValidatedSignature, LogicalExpr):
    """Expression for extracting all regex matches from a string.

    Extracts all strings that match the regex pattern, optionally from a specific group.
    Returns an array of matches.

    Args:
        expr: The input string column expression
        pattern: The regex pattern with optional capture groups
        idx: The group index to extract (0 = entire match, 1+ = capture groups)

    Raises:
        ValidationError: If the regex pattern is invalid
    """

    function_name = "text.regexp_extract_all"

    def __init__(self, expr: LogicalExpr, pattern: LogicalExpr, idx: LogicalExpr):
        if isinstance(pattern, LiteralExpr) and pattern.data_type == StringType:
            try:
                py_validate_regex(pattern.literal)
            except Exception as e:
                raise ValidationError(f"Invalid regex pattern: {pattern.literal}") from e

        self.expr = expr
        self.pattern = pattern
        self.idx = idx
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self):
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return [self.expr, self.pattern, self.idx]

    def _eq_specific(self, other: RegexpExtractAllExpr) -> bool:
        return self.pattern == other.pattern and self.idx == other.idx


class RegexpInstrExpr(ValidatedSignature, LogicalExpr):
    """Expression for finding the position of a regex match in a string.

    Returns the 1-based position of the first substring that matches the regex pattern.
    Returns 0 if no match is found.

    Args:
        expr: The input string column expression
        pattern: The regex pattern to search for
        idx: The group index to locate (0 = entire match, 1+ = capture groups)

    Raises:
        ValidationError: If the regex pattern is invalid
    """

    function_name = "text.regexp_instr"

    def __init__(self, expr: LogicalExpr, pattern: LogicalExpr, idx: LogicalExpr):
        if isinstance(pattern, LiteralExpr) and pattern.data_type == StringType:
            try:
                py_validate_regex(pattern.literal)
            except Exception as e:
                raise ValidationError(f"Invalid regex pattern: {pattern.literal}") from e

        self.expr = expr
        self.pattern = pattern
        self.idx = idx
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self):
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return [self.expr, self.pattern, self.idx]

    def _eq_specific(self, other: RegexpInstrExpr) -> bool:
        return self.pattern == other.pattern and self.idx == other.idx


class RegexpSubstrExpr(ValidatedSignature, LogicalExpr):
    """Expression for extracting the first regex match from a string.

    Returns the first substring that matches the regex pattern.

    Args:
        expr: The input string column expression
        pattern: The regex pattern to search for

    Raises:
        ValidationError: If the regex pattern is invalid
    """

    function_name = "text.regexp_substr"

    def __init__(self, expr: LogicalExpr, pattern: LogicalExpr):
        if isinstance(pattern, LiteralExpr) and pattern.data_type == StringType:
            try:
                py_validate_regex(pattern.literal)
            except Exception as e:
                raise ValidationError(f"Invalid regex pattern: {pattern.literal}") from e

        self.expr = expr
        self.pattern = pattern
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self):
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return [self.expr, self.pattern]

    def _eq_specific(self, other: RegexpSubstrExpr) -> bool:
        return self.pattern == other.pattern
