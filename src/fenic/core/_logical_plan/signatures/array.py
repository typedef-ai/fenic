"""Array expression signatures for the fenic signature system.

This module registers function signatures for array expressions, providing
centralized type validation and return type inference.
"""

from fenic.core._logical_plan.signatures.function_signature import (
    FunctionSignature,
    ReturnTypeStrategy,
)
from fenic.core._logical_plan.signatures.registry import FunctionRegistry
from fenic.core._logical_plan.signatures.type_signature import (
    ArrayOfAny,
    ArrayOfPrimitives,
    ArrayWithMatchingElement,
    Uniform,
    VariadicAny,
)
from fenic.core.types.datatypes import BooleanType, IntegerType


def register_array_signatures():
    """Register all array expression signatures."""
    # Array size - returns the number of elements in an array
    FunctionRegistry.register(
        "array_size",
        FunctionSignature(
            function_name="array_size",
            type_signature=ArrayOfAny(),
            return_type=IntegerType
        )
    )

    # Array distinct - returns array with duplicate elements removed (preserves element type)
    FunctionRegistry.register(
        "array_distinct",
        FunctionSignature(
            function_name="array_distinct",
            type_signature=ArrayOfAny(),
            return_type=ReturnTypeStrategy.SAME_AS_INPUT,
        ),
    )

    # Array contains - array + matching element type
    FunctionRegistry.register(
        "array_contains",
        FunctionSignature(
            function_name="array_contains",
            type_signature=ArrayWithMatchingElement(),
            return_type=BooleanType
        )
    )

    # Array max - returns max element from array (element type)
    # NOTE: Only works on arrays of primitive/comparable types (not structs, etc.)
    FunctionRegistry.register(
        "array_max",
        FunctionSignature(
            function_name="array_max",
            type_signature=ArrayOfPrimitives(),
            return_type=ReturnTypeStrategy.DYNAMIC
        )
    )

    # Array min - returns min element from array (element type)
    # NOTE: Only works on arrays of primitive/comparable types (not structs, etc.)
    FunctionRegistry.register(
        "array_min",
        FunctionSignature(
            function_name="array_min",
            type_signature=ArrayOfPrimitives(),
            return_type=ReturnTypeStrategy.DYNAMIC
        )
    )

    # Array sort - returns sorted array (preserves array type)
    # NOTE: Only works on arrays of primitive/comparable types (not structs, etc.)
    # LIMITATION: Does not support comparator function like PySpark's array_sort(col, comparator)
    FunctionRegistry.register(
        "array_sort",
        FunctionSignature(
            function_name="array_sort",
            type_signature=ArrayOfPrimitives(),
            return_type=ReturnTypeStrategy.SAME_AS_INPUT
        )
    )

    # Array reverse - returns reversed array (preserves array type)
    FunctionRegistry.register(
        "array_reverse",
        FunctionSignature(
            function_name="array_reverse",
            type_signature=ArrayOfAny(),
            return_type=ReturnTypeStrategy.SAME_AS_INPUT
        )
    )

    # Array remove - removes all occurrences of element
    FunctionRegistry.register(
        "array_remove",
        FunctionSignature(
            function_name="array_remove",
            type_signature=ArrayWithMatchingElement(),
            return_type=ReturnTypeStrategy.SAME_AS_INPUT
        )
    )

    # Array union - returns union of two arrays without duplicates
    FunctionRegistry.register(
        "array_union",
        FunctionSignature(
            function_name="array_union",
            type_signature=Uniform(expected_num_args=2),
            return_type=ReturnTypeStrategy.SAME_AS_INPUT
        )
    )

    # Array intersect - returns intersection of two arrays
    FunctionRegistry.register(
        "array_intersect",
        FunctionSignature(
            function_name="array_intersect",
            type_signature=Uniform(expected_num_args=2),
            return_type=ReturnTypeStrategy.SAME_AS_INPUT
        )
    )

    # Array except - returns elements in first array but not in second
    FunctionRegistry.register(
        "array_except",
        FunctionSignature(
            function_name="array_except",
            type_signature=Uniform(expected_num_args=2),
            return_type=ReturnTypeStrategy.SAME_AS_INPUT
        )
    )

    # Array compact - removes null values from array
    FunctionRegistry.register(
        "array_compact",
        FunctionSignature(
            function_name="array_compact",
            type_signature=ArrayOfAny(),
            return_type=ReturnTypeStrategy.SAME_AS_INPUT
        )
    )

    # Array repeat - creates array with element repeated n times
    FunctionRegistry.register(
        "array_repeat",
        FunctionSignature(
            function_name="array_repeat",
            type_signature=VariadicAny(expected_min_args=2),
            return_type=ReturnTypeStrategy.DYNAMIC
        )
    )

    # Slice - extracts subarray from array
    FunctionRegistry.register(
        "slice",
        FunctionSignature(
            function_name="slice",
            type_signature=VariadicAny(expected_min_args=3),
            return_type=ReturnTypeStrategy.DYNAMIC
        )
    )

    # Element at - returns element at given index
    FunctionRegistry.register(
        "element_at",
        FunctionSignature(
            function_name="element_at",
            type_signature=VariadicAny(expected_min_args=2),
            return_type=ReturnTypeStrategy.DYNAMIC
        )
    )

    # Arrays overlap - checks if two arrays have common elements
    FunctionRegistry.register(
        "arrays_overlap",
        FunctionSignature(
            function_name="arrays_overlap",
            type_signature=Uniform(expected_num_args=2),
            return_type=BooleanType
        )
    )


# Auto-register when module is imported
register_array_signatures()
