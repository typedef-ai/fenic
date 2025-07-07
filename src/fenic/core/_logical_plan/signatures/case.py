"""Case expression signatures for the fenic signature system.

This module registers function signatures for case expressions 
(LogicalExpr subclasses with their own validation).
"""

from fenic.core._logical_plan.expressions.case import OtherwiseExpr, WhenExpr
from fenic.core._logical_plan.signatures.registry import FunctionRegistry


def register_case_signatures():
    """Register signatures for case expressions (LogicalExpr subclasses with own validation)."""
    # when and otherwise are LogicalExpr subclasses that handle their own type validation
    # No signature needed since they're not migrated to ScalarFunction yet
    FunctionRegistry.register("when", WhenExpr)
    FunctionRegistry.register("otherwise", OtherwiseExpr)