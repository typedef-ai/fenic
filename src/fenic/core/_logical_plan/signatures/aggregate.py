"""Aggregate function signatures for the fenic signature system.

This module registers function signatures for aggregate functions,
providing centralized type validation and return type inference.
"""
from fenic.core._logical_plan.signatures.registry import FunctionRegistry
from fenic.core._logical_plan.signatures.signature import (
    FunctionSignature,
    ReturnTypeStrategy,
)
from fenic.core._logical_plan.signatures.types import Exact, InstanceOf, OneOf
from fenic.core.types.datatypes import (
    BooleanType,
    DoubleType,
    EmbeddingType,
    FloatType,
    IntegerType,
    StringType,
)

# Constants for type validation
SUMMABLE_TYPES = (IntegerType, FloatType, DoubleType, BooleanType)


def register_aggregate_signatures():
    """Register all aggregate function signatures for AggregateFunctions."""
    # Sum - numeric types only, returns same type as input
    FunctionRegistry.register("sum", FunctionSignature(
        function_name="sum",
        type_signature=OneOf([
            Exact([IntegerType]),
            Exact([FloatType]),
            Exact([DoubleType]),
            Exact([BooleanType])
        ]),
        return_type=ReturnTypeStrategy.SAME_AS_INPUT
    ))
    
    # Average - numeric types and embeddings, returns DoubleType for numeric, same type for embeddings
    FunctionRegistry.register("avg", FunctionSignature(
        function_name="avg",
        type_signature=OneOf([
            Exact([IntegerType]),
            Exact([FloatType]),
            Exact([DoubleType]),
            Exact([BooleanType]),
            InstanceOf([EmbeddingType])
        ]),
        return_type=ReturnTypeStrategy.DYNAMIC  # Special logic needed for embeddings vs numeric
    ))
    
    # Min/Max - numeric types only, returns same type as input
    FunctionRegistry.register("min", FunctionSignature(
        function_name="min",
        type_signature=OneOf([
            Exact([IntegerType]),
            Exact([FloatType]),
            Exact([DoubleType]),
            Exact([BooleanType])
        ]),
        return_type=ReturnTypeStrategy.SAME_AS_INPUT
    ))
    
    FunctionRegistry.register("max", FunctionSignature(
        function_name="max",
        type_signature=OneOf([
            Exact([IntegerType]),
            Exact([FloatType]),
            Exact([DoubleType]),
            Exact([BooleanType])
        ]),
        return_type=ReturnTypeStrategy.SAME_AS_INPUT
    ))
    
    # Count - accepts any type, always returns IntegerType
    FunctionRegistry.register("count", FunctionSignature(
        function_name="count",
        type_signature=Exact([object]),  # Accepts any DataType
        return_type=IntegerType
    ))
    
    # List aggregation - accepts any type except literals, returns ArrayType of input element type
    FunctionRegistry.register("list", FunctionSignature(
        function_name="list",
        type_signature=Exact([object]),  # Accepts any DataType (literal check done separately)
        return_type=ReturnTypeStrategy.DYNAMIC  # Returns ArrayType(input_type)
    ))
    
    # First - accepts any type, returns same type as input
    FunctionRegistry.register("first", FunctionSignature(
        function_name="first",
        type_signature=Exact([object]),  # Accepts any DataType
        return_type=ReturnTypeStrategy.SAME_AS_INPUT
    ))
    
    # Standard deviation - numeric types only, returns DoubleType
    FunctionRegistry.register("stddev", FunctionSignature(
        function_name="stddev",
        type_signature=OneOf([
            Exact([IntegerType]),
            Exact([FloatType]),
            Exact([DoubleType]),
            Exact([BooleanType])
        ]),
        return_type=DoubleType
    ))
    
    # Markdown group schema - string input only, returns StringType
    FunctionRegistry.register("md_group_schema", FunctionSignature(
        function_name="md_group_schema",
        type_signature=Exact([StringType]),
        return_type=StringType
    ))


# Register all signatures when module is imported
register_aggregate_signatures()