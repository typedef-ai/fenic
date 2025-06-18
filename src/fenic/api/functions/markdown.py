"""Markdown functions."""
from typing import Optional

from pydantic import ConfigDict, validate_call

from fenic.api.column import Column, ColumnOrName
from fenic.core._logical_plan.expressions import (
    MdExtractHeaderChunks,
    MdGenerateTocExpr,
    MdGetCodeBlocksExpr,
    MdToJsonExpr,
)
from fenic.core.error import ValidationError


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def to_json(column: ColumnOrName) -> Column:
    """Converts a column of Markdown-formatted strings into a hierarchical JSON representation.

    Args:
        column (ColumnOrName): Input column containing Markdown strings.

    Returns:
        Column: A column of JSON-formatted strings representing the structured document tree.

    Notes:
        - This function parses Markdown into a structured JSON format optimized for document chunking,
          semantic analysis, and `jq` queries.
        - The output conforms to a custom schema that organizes content into nested sections based
          on heading levels. This makes it more expressive than flat ASTs like `mdast`.
        - The full JSON schema is available at: docs.fenic.ai/topics/markdown-json

    Supported Markdown Features:
        - Headings with nested hierarchy (e.g., h2 → h3 → h4)
        - Paragraphs with inline formatting (bold, italics, links, code, etc.)
        - Lists (ordered, unordered, task lists)
        - Tables with header alignment and inline content
        - Code blocks with language info
        - Blockquotes, horizontal rules, and inline/flow HTML

    Example: Convert markdown to JSON
        ```python
        df.select(markdown.to_json(col("markdown_text")))
        ```

    Example: Extract all level-2 headings with jq
        ```python
        # Combine with jq to extract all level-2 headings
        df.select(json.jq(markdown.to_json(col("md")), ".. | select(.type == 'heading' and .level == 2)"))
        ```
    """
    return Column._from_logical_expr(
        MdToJsonExpr(Column._from_col_or_name(column)._logical_expr)
    )

@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def get_code_blocks(column: ColumnOrName, language_filter: Optional[str] = None) -> Column:
    """Extracts all code blocks from a column of Markdown-formatted strings.

    Args:
        column (ColumnOrName): Input column containing Markdown strings.
        language_filter (Optional[str]): Optional language filter to extract only code blocks with a specific language. By default, all code blocks are extracted.

    Returns:
        Column: A column of code blocks. The output column type is:
            ArrayType(StructType([
                StructField("language", StringType),
                StructField("code", StringType),
            ]))

    Notes:
        - Code blocks are parsed from fenced Markdown blocks (e.g., triple backticks ```).
        - Language identifiers are optional and may be null if not provided in the original Markdown.
        - Indented code blocks without fences are not currently supported.
        - This function is useful for extracting embedded logic, configuration, or examples
          from documentation or notebooks.

    Example: Extract all code blocks
        ```python
        df.select(markdown.get_code_blocks(col("markdown_text")))
        ```

    Example: Explode code blocks into individual rows
        ```python
        # Explode the list of code blocks into individual rows
        df = df.explode(df.with_column("blocks", markdown.get_code_blocks(col("md"))))
        df = df.select(col("blocks")["language"], col("blocks")["code"])
        ```
    """
    return Column._from_logical_expr(
        MdGetCodeBlocksExpr(Column._from_col_or_name(column)._logical_expr, language_filter)
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def generate_toc(column: ColumnOrName, max_level: Optional[int] = None) -> Column:
    """Generates a table of contents from markdown headings.

    Args:
        column (ColumnOrName): Input column containing Markdown strings.
        max_level (Optional[int]): Maximum heading level to include in the TOC (1-6).
                                 Defaults to 6 (all levels).

    Returns:
        Column: A column of Markdown-formatted table of contents strings.

    Notes:
        - The TOC is generated using markdown heading syntax (# ## ### etc.)
        - Each heading in the source document becomes a line in the TOC
        - The heading level is preserved in the output
        - This creates a valid markdown document that can be rendered or processed further

    Example: Generate a complete TOC
        ```python
        df.select(markdown.generate_toc(col("documentation")))
        ```

    Example: Generate a simplified TOC with only top 2 levels
        ```python
        df.select(markdown.generate_toc(col("documentation"), max_level=2))
        ```

    Example: Add TOC as a new column
        ```python
        df = df.with_column("toc", markdown.generate_toc(col("content"), max_level=3))
        ```
    """
    if max_level and (max_level < 1 or max_level > 6):
        raise ValidationError(f"max_level must be between 1 and 6 (inclusive), but got {max_level}. Use 1 for # headings, 2 for ## headings, etc.")
    return Column._from_logical_expr(
        MdGenerateTocExpr(Column._from_col_or_name(column)._logical_expr, max_level)
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def extract_header_chunks(column: ColumnOrName, header_level: int) -> Column:
    """Splits markdown documents into logical chunks based on heading hierarchy.

    Args:
        column (ColumnOrName): Input column containing Markdown strings.
        header_level (int): Heading level to split on (1-6). Creates a new chunk at every
                            heading of this level, including all nested content and subsections.

    Returns:
        Column: A column of arrays containing chunk objects with the following structure:
            ```python
            ArrayType(StructType([
                StructField("heading", StringType),        # Heading text (clean, no markdown)
                StructField("level", IntegerType),         # Heading level (1-6)
                StructField("content", StringType),        # All content under this heading (clean text)
                StructField("parent_heading", StringType), # Parent heading text (or null)
                StructField("full_path", StringType),      # Full breadcrumb path
            ]))
            ```
    Notes:
        - **Context-preserving**: Each chunk contains all content and subsections under the heading
        - **Hierarchical awareness**: Includes parent heading context for better LLM understanding
        - **Clean text output**: Strips markdown formatting for direct LLM consumption

    Chunking Behavior:
        With `header_level=2`, this markdown:
        ```markdown
        # Introduction
        Overview text

        ## Getting Started
        Setup instructions

        ### Prerequisites
        Python 3.8+ required

        ## API Reference
        Function documentation
        ```
        Produces 2 chunks:

        1. `Getting Started` chunk (includes `Prerequisites` subsection)
        2. `API Reference` chunk

    Example: Split articles into top-level sections
        ```python
        df.select(markdown.extract_header_chunks(col("articles"), header_level=1))
        ```

    Example: Split documentation into feature sections
        ```python
        df.select(markdown.extract_header_chunks(col("docs"), header_level=2))
        ```

    Example: Create fine-grained chunks for detailed analysis
        ```python
        df.select(markdown.extract_header_chunks(col("content"), header_level=3))
        ```

    Example: Explode chunks into individual rows for processing
        ```python
        chunks_df = df.select(
            markdown.extract_header_chunks(col("markdown"), header_level=2).alias("chunks")
        ).explode("chunks")
        chunks_df.select(
            col("chunks").heading,
            col("chunks").content,
            col("chunks").full_path
        )
        ```
    """
    if header_level < 1 or header_level > 6:
        raise ValidationError(f"split_level must be between 1 and 6 (inclusive), but got {header_level}. Use 1 for # headings, 2 for ## headings, etc.")

    return Column._from_logical_expr(
        MdExtractHeaderChunks(Column._from_col_or_name(column)._logical_expr, header_level)
    )
