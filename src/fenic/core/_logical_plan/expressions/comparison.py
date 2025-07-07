from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from fenic.core._logical_plan.expressions.base import LogicalExpr, Operator
from fenic.core._logical_plan.signatures.scalar_function import ScalarFunction


class NumericComparisonExpr(ScalarFunction):
    """Numeric comparison expressions (GT, GTE, LT, LTE) - only accepts numeric types."""
    function_name = "numeric_comparison"

    def __init__(self, left: LogicalExpr, right: LogicalExpr, op: Operator):
        self.left = left
        self.right = right
        self.op = op

        super().__init__(left, right)

    def __str__(self):
        return f"({self.left} {self.op.value} {self.right})"


class BooleanExpr(ScalarFunction):
    """Boolean expressions (AND, OR) - only accepts boolean types."""
    function_name = "boolean"

    def __init__(self, left: LogicalExpr, right: LogicalExpr, op: Operator):
        self.left = left
        self.right = right
        self.op = op

        super().__init__(left, right)

    def __str__(self):
        return f"({self.left} {self.op.value} {self.right})"


class EqualityComparisonExpr(ScalarFunction):
    """Equality comparison expressions (EQ, NOT_EQ) - supports any types as long as they're equal."""
    function_name = "equality_comparison"

    def __init__(self, left: LogicalExpr, right: LogicalExpr, op: Operator):
        self.left = left
        self.right = right
        self.op = op

        super().__init__(left, right)

    def __str__(self):
        return f"({self.left} {self.op.value} {self.right})"
