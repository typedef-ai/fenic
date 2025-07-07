"""Basic expression signatures for the fenic signature system.

This module registers function signatures for basic expressions, providing
centralized type validation and return type inference.
"""
from fenic.core._logical_plan.expressions.basic import (
    AliasExpr,
    ArrayContainsExpr,
    ArrayExpr,
    ArrayLengthExpr,
    CastExpr,
    CoalesceExpr,
    ColumnExpr,
    IndexExpr,
    InExpr,
    IsNullExpr,
    LiteralExpr,
    NotExpr,
    SortExpr,
    StructExpr,
    UDFExpr,
)
from fenic.core._logical_plan.signatures.registry import FunctionRegistry
from fenic.core._logical_plan.signatures.signature import (
    FunctionSignature,
    ReturnTypeStrategy,
)
from fenic.core._logical_plan.signatures.types import (
    ArrayOfAny,
    ArrayWithMatchingElement,
    VariadicAny,
    VariadicUniform,
)
from fenic.core.types.datatypes import BooleanType, IntegerType


def register_basic_signatures():
    """Register all basic expression signatures for ScalarFunctions."""
    # Array functions
    FunctionRegistry.register(
        "array_size",
        ArrayLengthExpr,
        FunctionSignature(function_name="array_size", type_signature=ArrayOfAny(), return_type=IntegerType)
    )
    
    # Array construction - variadic uniform (all elements same type)
    FunctionRegistry.register(
        "array",
        ArrayExpr,
        FunctionSignature(function_name="array", type_signature=VariadicUniform(expected_min_args=1),
                          return_type=ReturnTypeStrategy.DYNAMIC)
    )
    
    # Struct construction - variadic any (different types allowed)
    FunctionRegistry.register(
        "struct",
        StructExpr,
        FunctionSignature(function_name="struct", type_signature=VariadicAny(expected_min_args=1),
                          return_type=ReturnTypeStrategy.DYNAMIC)
    )

    # Coalesce - all arguments must be same type as first
    FunctionRegistry.register(
        "coalesce",
        CoalesceExpr,
        FunctionSignature(function_name="coalesce", type_signature=VariadicUniform(expected_min_args=1),
                          return_type=ReturnTypeStrategy.SAME_AS_INPUT)
    )

    # Array contains - array + matching element type
    FunctionRegistry.register(
        "array_contains",
        ArrayContainsExpr,
        FunctionSignature(function_name="array_contains", type_signature=ArrayWithMatchingElement(),
                          return_type=BooleanType)
    )
    
    # UDF - LogicalExpr subclass
    FunctionRegistry.register("udf", UDFExpr)
    
    # Literal - LogicalExpr subclass 
    FunctionRegistry.register("lit", LiteralExpr)
    
    # Core expression types - LogicalExpr subclasses with own validation
    FunctionRegistry.register("col", ColumnExpr)  # Column reference
    FunctionRegistry.register("alias", AliasExpr)  # Column alias
    FunctionRegistry.register("sort", SortExpr)   # Sort order
    FunctionRegistry.register("index", IndexExpr)  # Array/struct indexing
    FunctionRegistry.register("is_null", IsNullExpr)  # Null check
    FunctionRegistry.register("cast", CastExpr)   # Type casting
    FunctionRegistry.register("not", NotExpr)     # Boolean negation
    FunctionRegistry.register("in", InExpr)       # IN operator