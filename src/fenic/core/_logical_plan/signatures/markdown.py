"""Markdown function signatures for the fenic signature system.

This module registers function signatures for markdown processing functions,
providing centralized type validation and return type inference.
"""
from fenic.core._logical_plan.signatures.function_signature import FunctionSignature
from fenic.core._logical_plan.signatures.registry import FunctionRegistry
from fenic.core._logical_plan.signatures.type_signature import Exact
from fenic.core.types.datatypes import (
    ArrayType,
    IntegerType,
    JsonType,
    MarkdownType,
    StringType,
    StructField,
    StructType,
)


def register_markdown_signatures():
    """Register all markdown function signatures."""
    # Markdown to JSON conversion
    FunctionRegistry.register(
        "markdown.to_json",
        FunctionSignature(
            function_name="markdown.to_json",
            type_signature=Exact([MarkdownType]),  # markdown input
            return_type=JsonType
        )
    )

    # Get code blocks from markdown
    FunctionRegistry.register(
        "markdown.get_code_blocks",
        FunctionSignature(
            function_name="markdown.get_code_blocks",
            type_signature=Exact([MarkdownType]),  # markdown input (language_filter is literal)
            return_type=ArrayType(StructType([
                StructField("language", StringType),
                StructField("code", StringType)
            ]))
        )
    )

    # Generate table of contents from markdown
    FunctionRegistry.register(
        "markdown.generate_toc",
        FunctionSignature(
            function_name="markdown.generate_toc",
            type_signature=Exact([MarkdownType]),  # markdown input (max_level is literal)
            return_type=MarkdownType
        )
    )

    # Extract header chunks from markdown
    FunctionRegistry.register(
        "markdown.extract_header_chunks",
        FunctionSignature(
            function_name="markdown.extract_header_chunks",
            type_signature=Exact([MarkdownType]),  # markdown input (header_level is literal)
            return_type=ArrayType(StructType([
                StructField("heading", StringType),
                StructField("level", IntegerType),
                StructField("content", StringType),
                StructField("parent_heading", StringType),
                StructField("full_path", StringType)
            ]))
        )
    )

register_markdown_signatures()
