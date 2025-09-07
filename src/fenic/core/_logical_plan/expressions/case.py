from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from fenic.core._logical_plan import LogicalPlan

from fenic.core._interfaces.session_state import BaseSessionState
from fenic.core._logical_plan.expressions.base import LogicalExpr, UnparameterizedExpr
from fenic.core.error import TypeMismatchError, ValidationError
from fenic.core.types import BooleanType, ColumnField


class WhenExpr(UnparameterizedExpr, LogicalExpr):
    def __init__(self, expr: Optional[LogicalExpr], condition: LogicalExpr, value: LogicalExpr):
        self.expr = expr
        self.condition = condition
        self.value = value
        if expr is not None and not isinstance(expr, WhenExpr):
            raise ValidationError("Column.when() can only be called on when() expressions")

    def __str__(self):
        return f"{'CASE' if self.expr is None else self.expr} WHEN ({self.condition}) THEN ({self.value})"

    def to_column_field(self, plan: LogicalPlan, session_state: BaseSessionState) -> ColumnField:
        if self.expr is not None:
            expr_type = self.expr.to_column_field(plan, session_state).data_type
            value_type = self.value.to_column_field(plan, session_state).data_type
            if expr_type != value_type:
                raise TypeMismatchError.from_message(
                    f"Type mismatch in when(): all case branches must return the same type. "
                    f"Previous branch has type {expr_type}, but this branch has type {value_type}."
                )
        condition_type = self.condition.to_column_field(plan, session_state).data_type
        if condition_type != BooleanType:
            raise TypeMismatchError.from_message(
                "when() condition must be a boolean expression.  Got type: "
                f"{condition_type}"
            )

        return self.value.to_column_field(plan, session_state)

    def children(self) -> List[LogicalExpr]:
        children = []
        if self.expr is not None:
            children.append(self.expr)
        children.extend([self.condition, self.value])
        return children


class OtherwiseExpr(UnparameterizedExpr, LogicalExpr):
    def __init__(self, expr: LogicalExpr, value: LogicalExpr):
        self.expr = expr
        self.value = value
        if not isinstance(self.expr, WhenExpr):
            raise ValidationError("Column.otherwise() can only be called on when() expressions")

    def __str__(self):
        return f"{self.expr} ELSE ({self.value})"

    def to_column_field(self, plan: LogicalPlan, session_state: BaseSessionState) -> ColumnField:
        when_expr_type = self.expr.to_column_field(plan, session_state).data_type
        value_field = self.value.to_column_field(plan, session_state)
        if when_expr_type != value_field.data_type:
            raise TypeMismatchError.from_message(
                f"Type mismatch in otherwise(): when/then expression has type "
                f"{when_expr_type}, but otherwise() value has type {value_field.data_type}. "
                f"Both branches must return the same type."
            )
        return value_field

    def children(self) -> List[LogicalExpr]:
        return [self.expr, self.value]
