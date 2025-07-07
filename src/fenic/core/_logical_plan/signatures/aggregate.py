"""Aggregate function signatures for the fenic signature system.

This module registers function signatures for aggregate functions,
providing centralized type validation and return type inference.
"""
from fenic.core._logical_plan.expressions import (
    AvgExpr,
    CountExpr,
    FirstExpr,
    ListExpr,
    MaxExpr,
    MinExpr,
    StdDevExpr,
    SumExpr,
)
from fenic.core._logical_plan.signatures.registry import FunctionRegistry
from fenic.core._logical_plan.signatures.signature import (
    FunctionSignature,
    ReturnTypeStrategy,
)
from fenic.core._logical_plan.signatures.types import (
    Any,
    Exact,
    InstanceOf,
    Numeric,
    OneOf,
)
from fenic.core.types.datatypes import (
    BooleanType,
    DoubleType,
    EmbeddingType,
    FloatType,
    IntegerType,
)

# Constants for type validation
SUMMABLE_TYPES = (IntegerType, FloatType, DoubleType, BooleanType)


def register_aggregate_signatures():
    """Register all aggregate function signatures for AggregateFunctions."""
    # Sum - numeric types only, returns same type as input
    FunctionRegistry.register("sum", SumExpr, FunctionSignature(
        function_name="sum",
        type_signature=OneOf([
            Numeric(1),
            Exact([BooleanType])
        ]),
        return_type=ReturnTypeStrategy.SAME_AS_INPUT
    ))
    
    # Average - numeric types and embeddings, returns DoubleType for numeric, same type for embeddings
    FunctionRegistry.register("avg", AvgExpr, FunctionSignature(
        function_name="avg",
        type_signature=OneOf([
            Numeric(1),
            Exact([BooleanType]),
            InstanceOf([EmbeddingType])
        ]),
        return_type=ReturnTypeStrategy.DYNAMIC  # Special logic needed for embeddings vs numeric
    ))
    
    # Min/Max - numeric types only, returns same type as input
    FunctionRegistry.register("min", MinExpr, FunctionSignature(
        function_name="min",
        type_signature=OneOf([
            Numeric(1),
            Exact([BooleanType])
        ]),
        return_type=ReturnTypeStrategy.SAME_AS_INPUT
    ))
    
    FunctionRegistry.register("max", MaxExpr, FunctionSignature(
        function_name="max",
        type_signature=OneOf([
            Numeric(1),
            Exact([BooleanType])
        ]),
        return_type=ReturnTypeStrategy.SAME_AS_INPUT
    ))
    
    # Count - accepts any type, always returns IntegerType
    FunctionRegistry.register("count", CountExpr, FunctionSignature(
        function_name="count",
        type_signature=Any(1),  # Accepts one argument of any DataType
        return_type=IntegerType
    ))
    
    # List aggregation - accepts any type except literals, returns ArrayType of input element type
    FunctionRegistry.register("list", ListExpr, FunctionSignature(
        function_name="list",
        type_signature=Any(1),  # Accepts any DataType (literal check done separately)
        return_type=ReturnTypeStrategy.DYNAMIC  # Returns ArrayType(input_type)
    ))
    
    # First - accepts any type, returns same type as input
    FunctionRegistry.register("first", FirstExpr, FunctionSignature(
        function_name="first",
        type_signature=Any(1),  # Accepts one argument of any DataType
        return_type=ReturnTypeStrategy.SAME_AS_INPUT
    ))
    
    # Standard deviation - numeric types only, returns DoubleType
    FunctionRegistry.register("stddev", StdDevExpr, FunctionSignature(
        function_name="stddev",
        type_signature=OneOf([
            Numeric(1),
            Exact([BooleanType])
        ]),
        return_type=DoubleType
    ))



# Register all signatures when module is imported
register_aggregate_signatures()