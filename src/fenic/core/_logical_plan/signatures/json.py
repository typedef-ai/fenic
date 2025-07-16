"""JSON function signatures for the fenic signature system.

This module registers function signatures for JSON processing functions,
providing centralized type validation and return type inference.
"""
from fenic.core._logical_plan.signatures.function_signature import FunctionSignature
from fenic.core._logical_plan.signatures.registry import FunctionRegistry
from fenic.core._logical_plan.signatures.type_signature import Exact
from fenic.core.types.datatypes import ArrayType, BooleanType, JsonType, StringType


def register_json_signatures():
    """Register all JSON function signatures."""
    # JQ query on JSON data
    FunctionRegistry.register(
        "json.jq",
        FunctionSignature(
            function_name="json.jq",
            type_signature=Exact([JsonType]),  # JSON input (query is literal string)
            return_type=ArrayType(JsonType)
        )
    )

    # Get JSON type as string
    FunctionRegistry.register(
        "json.type",
        FunctionSignature(
            function_name="json.type",
            type_signature=Exact([JsonType]),  # JSON input
            return_type=StringType
        )
    )

    # Check if JSON contains a value
    FunctionRegistry.register(
        "json.contains",
        FunctionSignature(
            function_name="json.contains",
            type_signature=Exact([JsonType]),  # JSON input (value is literal string)
            return_type=BooleanType
        )
    )

register_json_signatures()
