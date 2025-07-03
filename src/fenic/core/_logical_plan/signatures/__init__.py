"""Function signature validation system.

This package provides a centralized system for validating function signatures
and inferring return types.
"""

# Import signature modules to register them

from fenic.core._logical_plan.signatures import (
    builtin,  # noqa: F401
    embedding,  # noqa: F401
    json,  # noqa: F401
    markdown,  # noqa: F401
    semantic,  # noqa: F401
    text,  # noqa: F401
)
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
    EqualTypes,
    Exact,
    InstanceOf,
    Numeric,
    OneOf,
    # Core signatures
    PositionalSignature,
    StructWithStringKey,
    TypeSignature,
    Uniform,
    VariadicAny,
    VariadicUniform,
    # Utility functions
    require_equal_types,
)

__all__ = [
    "TypeSignature",
    "PositionalSignature",
    "Exact",
    "InstanceOf",
    "Uniform",
    "VariadicUniform",
    "VariadicAny",
    "Numeric",
    "OneOf",
    # Specialized type signatures
    "ArrayOfAny",
    "ArrayWithMatchingElement", 
    "EqualTypes",
    "StructWithStringKey",
    # Utility functions
    "require_equal_types",
    "FunctionSignature",
    "ReturnTypeStrategy",
    "FunctionRegistry",
    "ScalarFunction",
]