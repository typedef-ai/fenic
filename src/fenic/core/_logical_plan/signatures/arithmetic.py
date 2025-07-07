"""Arithmetic expression signatures for the fenic signature system.

This module registers function signatures for arithmetic expressions 
that have been migrated to ScalarFunction.
"""

from fenic.core._logical_plan.expressions.arithmetic import ArithmeticExpr
from fenic.core._logical_plan.signatures.registry import FunctionRegistry
from fenic.core._logical_plan.signatures.signature import (
    FunctionSignature,
    ReturnTypeStrategy,
)
from fenic.core._logical_plan.signatures.types import Numeric


def register_arithmetic_signatures():
    """Register signatures for arithmetic operations."""
    # Single registration for all arithmetic operations: numeric inputs, promoted return type
    FunctionRegistry.register(
        "arithmetic",
        ArithmeticExpr,
        FunctionSignature(function_name="arithmetic", type_signature=Numeric(expected_num_args=2),
                          return_type=ReturnTypeStrategy.PROMOTED)
    )