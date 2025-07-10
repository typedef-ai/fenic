"""Case expression signatures for the fenic signature system.

This module registers function signatures for case expressions 
(LogicalExpr subclasses with their own validation).
"""

from fenic.core._logical_plan.expressions.case import OtherwiseExpr, WhenExpr
from fenic.core._logical_plan.signatures.registry import FunctionRegistry


def register_case_signatures():
    """Register expressions for case expressions (LogicalExpr subclasses with own validation)."""
    # these LogicalExpr subclasses handle their own type validation
    FunctionRegistry.register("when", WhenExpr)
    FunctionRegistry.register("otherwise", OtherwiseExpr)
