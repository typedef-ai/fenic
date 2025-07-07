"""Comparison expression signatures for the fenic signature system.

This module registers function signatures for comparison expressions 
that have been migrated to ScalarFunction.
"""

from fenic.core._logical_plan.expressions.comparison import (
    BooleanExpr,
    EqualityComparisonExpr,
    NumericComparisonExpr,
)
from fenic.core._logical_plan.signatures.registry import FunctionRegistry
from fenic.core._logical_plan.signatures.signature import FunctionSignature
from fenic.core._logical_plan.signatures.types import Numeric, Uniform
from fenic.core.types.datatypes import BooleanType


def register_comparison_signatures():
    """Register signatures for comparison operations."""
    # Numeric comparisons: numeric inputs, boolean return
    FunctionRegistry.register(
        "numeric_comparison",
        NumericComparisonExpr,
        FunctionSignature(function_name="numeric_comparison", type_signature=Numeric(expected_num_args=2),
                          return_type=BooleanType)
    )
    
    # Boolean operations: boolean inputs, boolean return
    FunctionRegistry.register(
        "boolean",
        BooleanExpr,
        FunctionSignature(function_name="boolean",
                          type_signature=Uniform(expected_num_args=2, required_type=BooleanType),
                          return_type=BooleanType)
    )
    
    # Equality comparisons: any inputs (as long as they're equal), boolean return
    FunctionRegistry.register(
        "equality_comparison",
        EqualityComparisonExpr,
        FunctionSignature(function_name="equality_comparison", type_signature=Uniform(expected_num_args=2),
                          return_type=BooleanType)
    )