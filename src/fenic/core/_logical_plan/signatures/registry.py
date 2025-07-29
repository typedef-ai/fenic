"""Central registry for function signatures.

This module provides a global registry where function signatures are stored
and retrieved by function name.
"""
from typing import Dict

from fenic.core._logical_plan.signatures.function_signature import FunctionSignature
from fenic.core.error import InternalError


class FunctionRegistry:
    """Central registry for function signatures and expression classes."""

    _signatures: Dict[str, FunctionSignature] = {}

    @classmethod
    def register(cls, func_name: str, signature: FunctionSignature) -> None:
        """Register a function signature and its expression class."""
        cls._signatures[func_name] = signature

    @classmethod
    def get_signature(cls, func_name: str) -> FunctionSignature:
        """Get a function signature by name."""
        if func_name not in cls._signatures:
            raise InternalError(f"Unknown function: {func_name}")
        return cls._signatures[func_name]
