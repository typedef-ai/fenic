"""Central registry for function signatures.

This module provides a global registry where function signatures are stored
and retrieved by function name.
"""
from dataclasses import dataclass
from typing import Dict, Optional

from fenic.core._logical_plan.signatures.signature import FunctionSignature
from fenic.core.error import InternalError


@dataclass
class FunctionRegistryEntry:
    """Entry in the function registry."""
    expression_class: type
    signature: Optional[FunctionSignature] = None

class FunctionRegistry:
    """Central registry for function signatures and expression classes."""
    
    _functions: Dict[str, FunctionRegistryEntry] = {}

    @classmethod
    def register(cls, func_name: str, expression_class: type, signature: Optional[FunctionSignature] = None) -> None:
        """Register a function signature and its expression class."""
        cls._functions[func_name] = FunctionRegistryEntry(expression_class, signature)

    @classmethod
    def get_signature(cls, func_name: str) -> FunctionSignature:
        """Get a function signature by name."""
        if func_name not in cls._functions:
            raise InternalError(f"Unknown function: {func_name}")
        if cls._functions[func_name].signature is None:
            raise InternalError(f"No signature registered for function: {func_name}")
        return cls._functions[func_name].signature
