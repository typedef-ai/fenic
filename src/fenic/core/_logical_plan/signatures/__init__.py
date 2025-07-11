"""Function signature validation system.

This package provides a centralized system for validating function signatures
and inferring return types.
"""

from fenic.core._logical_plan.signatures import (  #noqa: F401
    basic,
    embedding,
    json,
    markdown,
    semantic,
    text,
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
    TypeSignature,
    Uniform,
    VariadicAny,
    VariadicUniform,
)

__all__ = [
    "TypeSignature",
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
    "FunctionSignature",
    "ReturnTypeStrategy",
    "FunctionRegistry",
    "ScalarFunction",
]