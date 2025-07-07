"""Function signature validation system.

This package provides a centralized system for validating function signatures
and inferring return types.
"""

# Note: signature modules are imported at the bottom to avoid circular imports
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

# Note: signature modules are registered via registration import in expressions/__init__.py

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

# Note: builtin and core modules are registered via registration import in expressions/__init__.py