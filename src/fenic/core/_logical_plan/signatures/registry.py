"""Central registry for function signatures.

This module provides a global registry where function signatures are stored
and retrieved by function name.
"""

from typing import Dict, List

from fenic.core._logical_plan.signatures.signature import FunctionSignature
from fenic.core.error import InternalError


class FunctionRegistry:
    """Central registry for function signatures."""
    
    _signatures: Dict[str, FunctionSignature] = {}

    @classmethod
    def register(cls, func_name: str, signature: FunctionSignature) -> None:
        """Register a function signature."""
        cls._signatures[func_name] = signature

    @classmethod
    def get_signature(cls, func_name: str) -> FunctionSignature:
        """Get a function signature by name."""
        if func_name not in cls._signatures:
            raise InternalError(f"Unknown function: {func_name}")
        return cls._signatures[func_name]

    @classmethod
    def is_registered(cls, func_name: str) -> bool:
        """Check if a function is registered."""
        return func_name in cls._signatures

    @classmethod
    def list_functions(cls) -> List[str]:
        """List all registered function names."""
        return list(cls._signatures.keys())