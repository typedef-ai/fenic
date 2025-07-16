"""Function signature validation system.

This package provides a centralized system for validating function signatures
and inferring return types.
"""
from fenic.core._logical_plan.signatures import (  #noqa: F401
    aggregate,
    basic,
    embedding,
    json,
    markdown,
    semantic,
    text,
)
from fenic.core._logical_plan.signatures.function_signature import (
    FunctionSignature,
    ReturnTypeStrategy,
)
from fenic.core._logical_plan.signatures.registry import FunctionRegistry
from fenic.core._logical_plan.signatures.signature_validator import SignatureValidator
from fenic.core._logical_plan.signatures.type_signature import (
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
    "SignatureValidator",
]
