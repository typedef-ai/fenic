"""JSON function signatures for the fenic signature system.

This module registers function signatures for JSON processing functions,
providing centralized type validation and return type inference.
"""
from fenic.core._logical_plan.expressions.json import (
    JqExpr,
    JsonContainsExpr,
    JsonTypeExpr,
)
from fenic.core._logical_plan.signatures.registry import FunctionRegistry
from fenic.core._logical_plan.signatures.signature import FunctionSignature
from fenic.core._logical_plan.signatures.types import Exact
from fenic.core.types.datatypes import ArrayType, BooleanType, JsonType, StringType


def register_json_signatures():
    """Register all JSON function signatures for ScalarFunctions."""
    # JQ query on JSON data
    FunctionRegistry.register(
        "json.jq",
        JqExpr,
        FunctionSignature(
            function_name="json.jq",
            type_signature=Exact([JsonType]),  # JSON input (query is literal string)
            return_type=ArrayType(JsonType)
        )
    )
    
    # Get JSON type as string
    FunctionRegistry.register(
        "json.type",
        JsonTypeExpr,
        FunctionSignature(
            function_name="json.type",
            type_signature=Exact([JsonType]),  # JSON input
            return_type=StringType
        )
    )
    
    # Check if JSON contains a value
    FunctionRegistry.register(
        "json.contains",
        JsonContainsExpr,
        FunctionSignature(
            function_name="json.contains",
            type_signature=Exact([JsonType]),  # JSON input (value is literal string)
            return_type=BooleanType
        )
    )