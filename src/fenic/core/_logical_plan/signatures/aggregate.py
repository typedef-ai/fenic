"""Aggregate function signatures for the fenic signature system.

This module registers function signatures for aggregate functions,
providing centralized type validation and return type inference.
"""
from fenic.core._logical_plan.signatures.function_signature import (
    FunctionSignature,
    ReturnTypeStrategy,
)
from fenic.core._logical_plan.signatures.registry import FunctionRegistry
from fenic.core._logical_plan.signatures.type_signature import (
    Any,
    AnyExcludingStructs,
    Exact,
    InstanceOf,
    Numeric,
    OneOf,
    VariadicAny,
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
    """Register all aggregate function signatures."""
    # Sum - numeric types only, returns same type as input
    FunctionRegistry.register("sum", FunctionSignature(
        function_name="sum",
        type_signature=OneOf([
            Numeric(1),
            Exact([BooleanType])
        ]),
        return_type=ReturnTypeStrategy.SAME_AS_INPUT
    ))
    
    # Average - numeric types and embeddings, returns DoubleType for numeric, same type for embeddings
    FunctionRegistry.register("avg", FunctionSignature(
        function_name="avg",
        type_signature=OneOf([
            Numeric(1),
            Exact([BooleanType]),
            InstanceOf([EmbeddingType])
        ]),
        return_type=ReturnTypeStrategy.DYNAMIC  # Special logic needed for embeddings vs numeric
    ))
    
    # Min/Max - numeric types only, returns same type as input
    FunctionRegistry.register("min", FunctionSignature(
        function_name="min",
        type_signature=OneOf([
            Numeric(1),
            Exact([BooleanType])
        ]),
        return_type=ReturnTypeStrategy.SAME_AS_INPUT
    ))
    
    FunctionRegistry.register("max", FunctionSignature(
        function_name="max",
        type_signature=OneOf([
            Numeric(1),
            Exact([BooleanType])
        ]),
        return_type=ReturnTypeStrategy.SAME_AS_INPUT
    ))
    
    # Count - accepts any type, always returns IntegerType
    FunctionRegistry.register("count", FunctionSignature(
        function_name="count",
        type_signature=Any(1),  # Accepts one argument of any DataType
        return_type=IntegerType
    ))
    
    # List aggregation - accepts any type except literals, returns ArrayType of input element type
    FunctionRegistry.register("collect_list", FunctionSignature(
        function_name="collect_list",
        type_signature=Any(1),  # Accepts any DataType (literal check done separately)
        return_type=ReturnTypeStrategy.DYNAMIC  # Returns ArrayType(input_type)
    ))
    
    # First - accepts any type, returns same type as input
    FunctionRegistry.register("first", FunctionSignature(
        function_name="first",
        type_signature=Any(1),  # Accepts one argument of any DataType
        return_type=ReturnTypeStrategy.SAME_AS_INPUT
    ))
    
    # Standard deviation - numeric types only, returns DoubleType
    FunctionRegistry.register("stddev", FunctionSignature(
        function_name="stddev",
        type_signature=OneOf([
            Numeric(1),
            Exact([BooleanType])
        ]),
        return_type=DoubleType
    ))

    # Distinct aggregate functions
    # approx_count_distinct - supports any type except StructType and ArrayType<StructType>
    FunctionRegistry.register("approx_count_distinct", FunctionSignature(
        function_name="approx_count_distinct",
        type_signature=AnyExcludingStructs(expected_num_args=1),
        return_type=IntegerType
    ))

    # count_distinct - supports variadic columns of any type.
    FunctionRegistry.register("count_distinct", FunctionSignature(
        function_name="count_distinct",
        type_signature=VariadicAny(expected_min_args=1),
        return_type=IntegerType
    ))

    # sum_distinct - numeric types (and Boolean for parity with sum), returns SAME_AS_INPUT
    FunctionRegistry.register("sum_distinct", FunctionSignature(
        function_name="sum_distinct",
        type_signature=OneOf([
            Numeric(1),
            Exact([BooleanType])
        ]),
        return_type=ReturnTypeStrategy.SAME_AS_INPUT
    ))



# Register all signatures when module is imported
register_aggregate_signatures()