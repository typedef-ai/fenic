"""Text function signatures for the fenic signature system.

This module registers function signatures for text processing functions,
providing centralized type validation and return type inference.
"""
from fenic.core._logical_plan.signatures.function_signature import (
    FunctionSignature,
    ReturnTypeStrategy,
)
from fenic.core._logical_plan.signatures.registry import FunctionRegistry
from fenic.core._logical_plan.signatures.type_signature import Exact, OneOf, VariadicAny
from fenic.core.types.datatypes import (
    ArrayType,
    BooleanType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

# Unified schema for all transcript formats
TRANSCRIPT_OUTPUT_TYPE = ArrayType(
    element_type=StructType(
        [
            StructField("index", IntegerType),  # Optional[int] - Entry index (1-based)
            StructField("speaker", StringType),  # Optional[str] - Speaker name
            StructField("start_time", DoubleType),  # float - Start time in seconds
            StructField("end_time", DoubleType),  # Optional[float] - End time in seconds
            StructField("duration", DoubleType),  # Optional[float] - Duration in seconds
            StructField("content", StringType),  # str - Transcript content/text
            StructField("format", StringType),  # str - Original format ("srt", "webvtt", or "generic")
        ]
    )
)

def register_text_signatures():
    """Register all text function signatures."""
    # Text extraction - string input with template
    FunctionRegistry.register(
        "text.extract",
        FunctionSignature(
            function_name="text.extract",
            type_signature=Exact([StringType]),  # Takes string input
            return_type=ReturnTypeStrategy.DYNAMIC  # Returns StructType with extracted fields
        )
    )

    # Text chunking - string input returns array of strings
    FunctionRegistry.register(
        "text.chunk",
        FunctionSignature(
            function_name="text.chunk",
            type_signature=Exact([StringType]),  # Takes string input
            return_type=ArrayType(StringType)
        )
    )

    # Recursive text chunking - same as text_chunk
    FunctionRegistry.register(
        "text.recursive_chunk",
        FunctionSignature(
            function_name="text.recursive_chunk",
            type_signature=Exact([StringType]),  # Takes string input
            return_type=ArrayType(StringType)
        )
    )

    # Count tokens - string input returns integer
    FunctionRegistry.register(
        "text.count_tokens",
        FunctionSignature(
            function_name="text.count_tokens",
            type_signature=Exact([StringType]),  # Takes string input
            return_type=IntegerType
        )
    )

    # Concat - variable number of arguments, all must be castable to string
    FunctionRegistry.register(
        "text.concat",
        FunctionSignature(
            function_name="text.concat",
            type_signature=VariadicAny(expected_min_args=1),  # Any types castable to string
            return_type=StringType
        )
    )

    # Array join - array of strings (delimiter is literal)
    FunctionRegistry.register(
        "text.array_join",
        FunctionSignature(
            function_name="text.array_join",
            type_signature=Exact([ArrayType(StringType)]),  # array<string> input only
            return_type=StringType
        )
    )

    # Contains - string + substring (string literal or LogicalExpr)
    FunctionRegistry.register(
        "text.contains",
        FunctionSignature(
            function_name="text.contains",
            type_signature=Exact([StringType, StringType]),
            return_type=BooleanType
        )
    )

    # Contains any - string input (substring list and case_insensitive handled as literals)
    FunctionRegistry.register(
        "text.contains_any",
        FunctionSignature(
            function_name="text.contains_any",
            type_signature=Exact([StringType]),  # string input only
            return_type=BooleanType
        )
    )

    FunctionRegistry.register(
        "text.rlike",
        FunctionSignature(
            function_name="text.rlike",
            type_signature=Exact([StringType, StringType]),  # expr, pattern
            return_type=BooleanType
        )
    )

    FunctionRegistry.register(
        "text.like",
        FunctionSignature(
            function_name="text.like",
            type_signature=Exact([StringType, StringType]),  # expr, pattern
            return_type=BooleanType
        )
    )

    FunctionRegistry.register(
        "text.ilike",
        FunctionSignature(
            function_name="text.ilike",
            type_signature=Exact([StringType, StringType]),  # expr, pattern
            return_type=BooleanType
        )
    )

    # Transcript parsing - string input (format is literal)
    FunctionRegistry.register(
        "text.parse_transcript",
        FunctionSignature(
            function_name="text.parse_transcript",
            type_signature=Exact([StringType]),  # string input only
            return_type=TRANSCRIPT_OUTPUT_TYPE  # Returns specific transcript schema
        )
    )

    # String prefix/suffix checking - string + prefix/suffix (string literal or LogicalExpr)
    FunctionRegistry.register(
        "text.starts_with",
        FunctionSignature(
            function_name="text.starts_with",
            type_signature=Exact([StringType, StringType]),
            return_type=BooleanType
        )
    )

    FunctionRegistry.register(
        "text.ends_with",
        FunctionSignature(
            function_name="text.ends_with",
            type_signature=OneOf([
                Exact([StringType, StringType])  # string input + suffix expr
            ]),
            return_type=BooleanType
        )
    )

    # String splitting - string input (patterns/delimiters handled as literals)
    FunctionRegistry.register(
        "text.regexp_split",
        FunctionSignature(
            function_name="text.regexp_split",
            type_signature=Exact([StringType]),  # string input only
            return_type=ArrayType(StringType)
        )
    )

    # Regex functions
    FunctionRegistry.register(
        "text.regexp_count",
        FunctionSignature(
            function_name="text.regexp_count",
            type_signature=Exact([StringType, StringType]),  # string input + pattern
            return_type=IntegerType
        )
    )

    FunctionRegistry.register(
        "text.regexp_extract",
        FunctionSignature(
            function_name="text.regexp_extract",
            type_signature=Exact([StringType, StringType]),  # string input + pattern
            return_type=StringType
        )
    )

    FunctionRegistry.register(
        "text.regexp_extract_all",
        FunctionSignature(
            function_name="text.regexp_extract_all",
            type_signature=Exact([StringType, StringType, IntegerType]),  # string input + pattern + group index
            return_type=ArrayType(StringType)
        )
    )

    FunctionRegistry.register(
        "text.regexp_instr",
        FunctionSignature(
            function_name="text.regexp_instr",
            type_signature=Exact([StringType, StringType, IntegerType]),  # string input + pattern + group index
            return_type=IntegerType
        )
    )

    FunctionRegistry.register(
        "text.regexp_substr",
        FunctionSignature(
            function_name="text.regexp_substr",
            type_signature=Exact([StringType, StringType]),  # string input + pattern
            return_type=StringType
        )
    )

    FunctionRegistry.register(
        "text.split_part",
        FunctionSignature(
            function_name="text.split_part",
            type_signature=Exact([StringType, StringType, IntegerType]),
            return_type=StringType
        )
    )

    # String casing - string input (case type handled as literal)
    FunctionRegistry.register(
        "text.string_casing",
        FunctionSignature(
            function_name="text.string_casing",
            type_signature=Exact([StringType]),  # string input only
            return_type=StringType
        )
    )

    # String trimming
    FunctionRegistry.register(
        "text.strip_chars",
        FunctionSignature(
            function_name="text.strip_chars",
            type_signature=OneOf([
                Exact([StringType]),  # string input only (chars is None)
                Exact([StringType, StringType])  # string input + chars expr
            ]),
            return_type=StringType
        )
    )

    # String replacement - string input + optional search/replacement expressions
    FunctionRegistry.register(
        "text.replace",
        FunctionSignature(
            function_name="text.replace",
            type_signature= Exact([StringType, StringType, StringType]),  # string input + search expr + replacement expr
            return_type=StringType
        )
    )

    # String length functions
    FunctionRegistry.register(
        "text.str_length",
        FunctionSignature(
            function_name="text.str_length",
            type_signature=Exact([StringType]),  # string
            return_type=IntegerType
        )
    )

    FunctionRegistry.register(
        "text.byte_length",
        FunctionSignature(
            function_name="text.byte_length",
            type_signature=Exact([StringType]),  # string
            return_type=IntegerType
        )
    )

    FunctionRegistry.register(
        "text.fuzzy_ratio",
        FunctionSignature(
            function_name="text.fuzzy_ratio",
            type_signature=Exact([StringType, StringType]),
            return_type=DoubleType
        )
    )

    FunctionRegistry.register(
        "text.fuzzy_token_sort_ratio",
        FunctionSignature(
            function_name="text.fuzzy_token_sort_ratio",
            type_signature=Exact([StringType, StringType]),
            return_type=DoubleType
        )
    )

    FunctionRegistry.register(
        "text.fuzzy_token_set_ratio",
        FunctionSignature(
            function_name="text.fuzzy_token_set_ratio",
            type_signature=Exact([StringType, StringType]),
            return_type=DoubleType
        )
    )


register_text_signatures()
