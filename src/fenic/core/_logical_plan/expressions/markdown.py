from __future__ import annotations

from typing import Optional

from fenic.core._logical_plan.expressions.base import LogicalExpr
from fenic.core._logical_plan.signatures.scalar_function import ScalarFunction


class MdToJsonExpr(ScalarFunction):
    function_name = "markdown.to_json"

    def __init__(self, expr: LogicalExpr):
        self.expr = expr
        super().__init__(expr)

    def __str__(self) -> str:
        return f"md_to_json({self.expr})"


class MdGetCodeBlocksExpr(ScalarFunction):
    function_name = "markdown.get_code_blocks"

    def __init__(self, expr: LogicalExpr, language_filter: Optional[str] = None):
        self.expr = expr
        self.language_filter = language_filter
        self.jq_query = self._build_jq_query(language_filter)

        # Only validate the markdown expression (language_filter is not LogicalExpr)
        super().__init__(expr)

    def _build_jq_query(self, language_filter: Optional[str] = None) -> str:
        if language_filter:
            escaped_filter = language_filter.replace('"', '\\"')
            selection = (
                f'.type? == "code_block" and (.language == "{escaped_filter}")'
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


class MdGenerateTocExpr(ScalarFunction):
    function_name = "markdown.generate_toc"

    def __init__(self, expr: LogicalExpr, max_level: Optional[int] = None):
        self.expr = expr
        self.max_level = max_level or 6
        self.jq_query = self._build_jq_query(self.max_level)

        # Only validate the markdown expression (max_level is not LogicalExpr)
        super().__init__(expr)

    def _build_jq_query(self, max_level: int) -> str:
        # Wrap in a simple object to avoid JSON string encoding issues
        query = f'''
    {{toc: ([.. | select(.type? == "heading" and .level <= {max_level}) |
    ("#" * .level) + " " + (.content | map(select(.type? == "text") | .text) | join(""))
    ] | join("\\n"))}}'''
        return query

    def __str__(self) -> str:
        return f"markdown.generate_toc({self.expr})"


class MdExtractHeaderChunks(ScalarFunction):
    function_name = "markdown.extract_header_chunks"

    def __init__(self, expr: LogicalExpr, header_level: int):
        self.expr = expr
        self.header_level = header_level
        self.jq_query = self._build_jq_query(header_level)

        # Only validate the markdown expression (header_level is not LogicalExpr)
        super().__init__(expr)

    def _build_jq_query(self, header_level: int) -> str:
        query = f'''
# Extract all text content and normalize whitespace
def extract_text: [.. | .text? // ""] | join(" ") | gsub("\\\\s+"; " ") | gsub("^\\\\s+|\\\\s+$"; "");

# Walk AST collecting chunks at target level, tracking breadcrumb path
def walk_headings($node; $path):
  if $node.type == "heading" and $node.level == {header_level} then
    # Target level heading - create chunk with content and breadcrumbs
    {{
      heading: (($node.content // []) | extract_text),
      level: $node.level,
      content: (($node.children // []) | extract_text),
      parent_heading: (if $path | length > 0 then $path[-1] else null end),
      full_path: ($path + [($node.content // []) | extract_text] | join(" > "))
    }}
  elif $node.type == "heading" and $node.level < {header_level} then
    # Higher level heading - add to path and recurse
    ($node.children // []) | map(walk_headings(.; $path + [($node.content // []) | extract_text])) | flatten
  else
    # Content or deeper headings - just recurse
    ($node.children // []) | map(walk_headings(.; $path)) | flatten
  end;

walk_headings(.; [])'''
        return query

    def __str__(self) -> str:
        return f"markdown.extract_header_chunks({self.expr}, header_level={self.header_level})"
