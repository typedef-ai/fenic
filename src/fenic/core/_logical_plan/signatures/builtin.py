"""Built-in function signatures for the fenic signature system.

This module registers function signatures for built-in functions, providing
centralized type validation and return type inference.
"""
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


def register_builtin_signatures():
    """Register all built-in function signatures for ScalarFunctions."""
    # Array functions
    FunctionRegistry.register("array_size", FunctionSignature(
        function_name="array_size",
        type_signature=ArrayOfAny(),  # Any array type
        return_type=IntegerType
    ))
    
    # Array construction - variadic uniform (all elements same type)
    FunctionRegistry.register("array", FunctionSignature(
        function_name="array",
        type_signature=VariadicUniform(expected_min_args=1),  # No required type - inferred from inputs
        return_type=ReturnTypeStrategy.DYNAMIC  # Returns ArrayType(element_type)
    ))
    
    # Struct construction - variadic any (different types allowed)
    FunctionRegistry.register("struct", FunctionSignature(
        function_name="struct", 
        type_signature=VariadicAny(expected_min_args=1),
        return_type=ReturnTypeStrategy.DYNAMIC  # Returns StructType with fields
    ))

    # Coalesce - all arguments must be same type as first
    FunctionRegistry.register("coalesce", FunctionSignature(
        function_name="coalesce",
        type_signature=VariadicUniform(expected_min_args=1),  # No required_type = same as first
        return_type=ReturnTypeStrategy.SAME_AS_INPUT
    ))

    # Array contains - array + matching element type
    FunctionRegistry.register("array_contains", FunctionSignature(
        function_name="array_contains",
        type_signature=ArrayWithMatchingElement(),
        return_type=BooleanType
    ))
    



# Register all signatures when module is imported
register_builtin_signatures()