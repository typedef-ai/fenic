from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from fenic.core._logical_plan import LogicalPlan

from fenic.core._logical_plan.expressions.base import LogicalExpr
from fenic.core.types import BooleanType, ColumnField


class WhenExpr(LogicalExpr):
    def __init__(self, expr: Optional[LogicalExpr], condition: LogicalExpr, value: LogicalExpr):
        self.expr = expr
        self.condition = condition
        self.value = value

    def __str__(self):
        return f"{'CASE' if self.expr is None else self.expr} WHEN ({self.condition}) THEN ({self.value})"

    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
        if self.condition.to_column_field(plan).data_type != BooleanType:
            raise TypeError(
                "when() condition must be a boolean expression.  Got type: "
                f"{self.condition.to_column_field(plan).data_type}"
            )

        return self.value.to_column_field(plan)

    def children(self) -> List[LogicalExpr]:
        children = []
        if self.expr is not None:
            children.append(self.expr)
        children.extend([self.condition, self.value])
        return children


class OtherwiseExpr(LogicalExpr):
    def __init__(self, expr: LogicalExpr, value: LogicalExpr):
        self.expr = expr
        self.value = value
        if not isinstance(self.expr, WhenExpr):
            raise TypeError("otherwise() can only be called on when() expressions")

    def __str__(self):
        return f"{self.expr} ELSE ({self.value})"

    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
        return self.value.to_column_field(plan)

    def children(self) -> List[LogicalExpr]:
        return [self.expr, self.value]
