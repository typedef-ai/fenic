"""Comparison expression signatures for the fenic signature system."""
from fenic.core._logical_plan.expressions import (
    BooleanExpr,
    EqualityComparisonExpr,
    NumericComparisonExpr,
)
from fenic.core._logical_plan.signatures import FunctionRegistry


def register_comparison_signatures():
    """Register expressions for comparison operations."""
    # these LogicalExpr subclasses handle their own type validation
    FunctionRegistry.register("numeric_comparison", NumericComparisonExpr)
    FunctionRegistry.register("boolean", BooleanExpr)
    FunctionRegistry.register("equality_comparison", EqualityComparisonExpr)
