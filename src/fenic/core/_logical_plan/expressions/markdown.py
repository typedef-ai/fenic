from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from fenic.core._logical_plan import LogicalPlan

from fenic.core._logical_plan.expressions.base import LogicalExpr
from fenic.core.error import TypeMismatchError
from fenic.core.types import (
    ArrayType,
    ColumnField,
    IntegerType,
    JsonType,
    MarkdownType,
    StringType,
    StructField,
    StructType,
)


class MdToJsonExpr(LogicalExpr):
    def __init__(self, expr: LogicalExpr):
        self.expr = expr

    def __str__(self) -> str:
        return f"md_to_json({self.expr})"

    def _validate_types(self, plan: LogicalPlan):
        input_field = self.expr.to_column_field(plan)
        if input_field.data_type != MarkdownType:
            raise TypeMismatchError(MarkdownType, input_field.data_type, "markdown.to_json()")

    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
        self._validate_types(plan)
        return ColumnField(name=str(self), data_type=JsonType)

    def children(self) -> List[LogicalExpr]:
        return [self.expr]


class MdGetCodeBlocksExpr(LogicalExpr):
    def __init__(self, expr: LogicalExpr, language_filter: Optional[str] = None):
        self.expr = expr
        self.language_filter = language_filter
        self.jq_query = self._build_jq_query(language_filter)

    def _build_jq_query(self, language_filter: Optional[str] = None) -> str:
        if language_filter:
            escaped_filter = language_filter.replace('"', '\\"')
            selection = (
                f'.type? == "code_block" and (.language == "{escaped_filter}" or .language == null)'
            )
        else:
            selection = '.type? == "code_block"'

        query = f'''
    [.. | select({selection}) |
    {{
    code: .text,
    language: .language,
    }}]'''
        return query

    def __str__(self) -> str:
        return f"markdown.get_code_blocks({self.expr})"

    def _validate_types(self, plan: LogicalPlan):
        input_field = self.expr.to_column_field(plan)
        if input_field.data_type != MarkdownType:
            raise TypeMismatchError(MarkdownType, input_field.data_type, "markdown.get_code_blocks()")

    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
        self._validate_types(plan)
        return ColumnField(name=str(self), data_type=ArrayType(StructType([StructField("language", StringType), StructField("code", StringType)])))

    def children(self) -> List[LogicalExpr]:
        return [self.expr]


class MdGenerateTocExpr(LogicalExpr):
    def __init__(self, expr: LogicalExpr, max_level: Optional[int] = None):
        self.expr = expr
        self.max_level = max_level or 6
        self.jq_query = self._build_jq_query(self.max_level)

    def _build_jq_query(self, max_level: int) -> str:
        # Wrap in a simple object to avoid JSON string encoding issues
        query = f'''
    {{toc: ([.. | select(.type? == "heading" and .level <= {max_level}) |
    ("#" * .level) + " " + (.content | map(select(.type? == "text") | .text) | join(""))
    ] | join("\\n"))}}'''
        return query

    def __str__(self) -> str:
        return f"markdown.generate_toc({self.expr})"

    def _validate_types(self, plan: LogicalPlan):
        input_field = self.expr.to_column_field(plan)
        if input_field.data_type != MarkdownType:
            raise TypeMismatchError(MarkdownType, input_field.data_type, "markdown.generate_toc()")

    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
        self._validate_types(plan)
        return ColumnField(name=str(self), data_type=MarkdownType)

    def children(self) -> List[LogicalExpr]:
        return [self.expr]


class MdExtractHeaderChunks(LogicalExpr):
    def __init__(self, expr: LogicalExpr, header_level: int):
        self.expr = expr
        self.header_level = header_level
        self.jq_query = self._build_jq_query(header_level)

    def _build_jq_query(self, header_level: int) -> str:
        query = f'''
# Extract all text content and normalize whitespace
def extract_text: [.. | .text? // empty] | join(" ") | gsub("\\\\s+"; " ") | gsub("^\\\\s+|\\\\s+$"; "");

# Walk AST collecting chunks at target level, tracking breadcrumb path
def walk_headings($node; $path):
  if $node.type == "heading" and $node.level == {header_level} then
    # Target level heading - create chunk with content and breadcrumbs
    {{
      heading: ($node.content | extract_text),
      level: $node.level,
      content: (($node.children // []) | extract_text),
      parent_heading: (if $path | length > 0 then $path[-1] else null end),
      full_path: ($path + [$node.content | extract_text] | join(" > "))
    }}
  elif $node.type == "heading" and $node.level < {header_level} then
    # Higher level heading - add to path and recurse
    ($node.children // []) | map(walk_headings(.; $path + [$node.content | extract_text])) | flatten
  else
    # Content or deeper headings - just recurse
    ($node.children // []) | map(walk_headings(.; $path)) | flatten
  end;

walk_headings(.; [])'''
        return query

    def __str__(self) -> str:
        return f"markdown.extract_header_chunks({self.expr}, header_level={self.header_level})"

    def _validate_types(self, plan: LogicalPlan):
        input_field = self.expr.to_column_field(plan)
        if input_field.data_type != MarkdownType:
            raise TypeMismatchError(MarkdownType, input_field.data_type, "markdown.extract_header_chunks()")

    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
        self._validate_types(plan)
        return ColumnField(
            name=str(self),
            data_type=ArrayType(
                StructType([
                    StructField("heading", StringType),
                    StructField("level", IntegerType),
                    StructField("content", StringType),
                    StructField("parent_heading", StringType),
                    StructField("full_path", StringType),
                ])
            )
        )

    def children(self) -> List[LogicalExpr]:
        return [self.expr]
