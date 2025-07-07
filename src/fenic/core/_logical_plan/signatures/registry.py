"""Central registry for function signatures.

This module provides a global registry where function signatures are stored
and retrieved by function name.
"""
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    pass

from fenic.core._logical_plan.signatures.signature import FunctionSignature
from fenic.core.error import InternalError


@dataclass
class FunctionRegistryEntry:
    """Entry in the function registry."""
    expression_class: type
    expression_migrated: bool
    signature: Optional[FunctionSignature] = None

class FunctionRegistry:
    """Central registry for function signatures and expression classes."""
    
    _functions: Dict[str, FunctionRegistryEntry] = {}

    @classmethod
    def register(cls, func_name: str, expression_class: type, signature: Optional[FunctionSignature] = None) -> None:
        """Register a function signature and its expression class."""
        cls._functions[func_name] = FunctionRegistryEntry(expression_class, (signature is not None), signature)

    @classmethod
    def get_signature(cls, func_name: str) -> FunctionSignature:
        """Get a function signature by name."""
        if func_name not in cls._functions:
            raise InternalError(f"Unknown function: {func_name}")
        if cls._functions[func_name].signature is None:
            raise InternalError(f"No signature registered for function: {func_name}")
        return cls._functions[func_name].signature

    @classmethod
    def is_registered(cls, func_name: str) -> bool:
        """Check if a function is registered."""
        return func_name in cls._functions

    @classmethod
    def list_functions(cls) -> List[str]:
        """List all registered function names."""
        return list(cls._functions.keys())

    @classmethod
    def get_expression_class(cls, func_name: str) -> type:
        """Get the expression class for a function name."""
        if func_name not in cls._functions:
            raise InternalError(f"Unknown function: {func_name}")
        return cls._functions[func_name].expression_class

    @classmethod
    def has_expression_class(cls, func_name: str) -> bool:
        """Check if an expression class is registered for a function name."""
        return func_name in cls._functions