"""Arithmetic expression signatures for the fenic signature system.

This module registers function signatures for arithmetic expressions
"""
from fenic.core._logical_plan.expressions import ArithmeticExpr
from fenic.core._logical_plan.signatures import FunctionRegistry


def register_arithmetic_signatures():
    """Register signatures for arithmetic operations."""
    # Single registration for all arithmetic operations: numeric inputs, promoted return type
    FunctionRegistry.register("arithmetic", ArithmeticExpr)
