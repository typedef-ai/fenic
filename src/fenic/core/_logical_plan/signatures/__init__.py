"""Function signature validation system.

This package provides a centralized system for validating function signatures
and inferring return types.
"""

# Import signature modules to register them
from fenic.core._logical_plan.signatures import builtin  # noqa: F401
from fenic.core._logical_plan.signatures.registry import FunctionRegistry
from fenic.core._logical_plan.signatures.scalar_function import ScalarFunction
from fenic.core._logical_plan.signatures.signature import (
    FunctionSignature,
    ReturnTypeStrategy,
)
from fenic.core._logical_plan.signatures.types import (
    # Specialized type signatures
    ArrayOfAny,
    ArrayWithMatchingElement,
    Exact,
    Numeric,
    OneOf,
    # Core signatures
    PositionalSignature,
    StructWithStringKey,
    TypeSignature,
    Uniform,
    VariadicAny,
    VariadicUniform,
)

__all__ = [
    "TypeSignature",
    "PositionalSignature",
    "Exact",
    "Uniform",
    "VariadicUniform",
    "VariadicAny",
    "Numeric",
    "OneOf",
    # Specialized type signatures
    "ArrayOfAny",
    "ArrayWithMatchingElement", 
    "StructWithStringKey",
    "FunctionSignature",
    "ReturnTypeStrategy",
    "FunctionRegistry",
    "ScalarFunction",
]