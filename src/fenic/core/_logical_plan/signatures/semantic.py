"""Semantic function signatures for the fenic signature system.

This module registers function signatures for semantic AI functions,
providing centralized type validation and return type inference.
"""
from fenic.core._logical_plan.signatures.function_signature import (
    FunctionSignature,
    ReturnTypeStrategy,
)
from fenic.core._logical_plan.signatures.registry import FunctionRegistry
from fenic.core._logical_plan.signatures.type_signature import Exact
from fenic.core.types.datatypes import MarkdownType, StringType


def register_semantic_signatures():
    """Register all semantic function signatures."""
    # Semantic extract - schema-based information extraction
    FunctionRegistry.register("semantic.extract", FunctionSignature(
        function_name="semantic.extract",
        type_signature=Exact([StringType]),  # String input (schema/template are parameters)
        return_type=ReturnTypeStrategy.DYNAMIC  # Returns StructType based on schema
    ))

    # Semantic classify - classification into labels/enum
    FunctionRegistry.register("semantic.classify", FunctionSignature(
        function_name="semantic.classify",
        type_signature=Exact([StringType]),  # String input (labels are parameters)
        return_type=StringType
    ))

    # Sentiment analysis - analyze sentiment of text
    FunctionRegistry.register("semantic.analyze_sentiment", FunctionSignature(
        function_name="semantic.analyze_sentiment",
        type_signature=Exact([StringType]),  # String input
        return_type=StringType
    ))

    # Embeddings - generate embeddings for text
    FunctionRegistry.register("semantic.embed", FunctionSignature(
        function_name="semantic.embed",
        type_signature=Exact([StringType]),  # String input (model_alias is parameter)
        return_type=ReturnTypeStrategy.DYNAMIC  # Returns EmbeddingType with specific dimensions
    ))

    # Summarize - summarize text
    FunctionRegistry.register("semantic.summarize", FunctionSignature(
        function_name="semantic.summarize",
        type_signature=Exact([StringType]),  # String input (model_alias is parameter)
        return_type=StringType
    ))

    # Parse PDF - parse PDF files with OCR/VLMs
    FunctionRegistry.register("semantic.parse_pdf", FunctionSignature(
        function_name="semantic.parse_pdf",
        type_signature=Exact([StringType]),
        return_type=MarkdownType
    ))

# Register all signatures when module is imported
register_semantic_signatures()
