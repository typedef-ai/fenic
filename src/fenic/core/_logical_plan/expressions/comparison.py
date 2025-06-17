from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fenic.core._logical_plan import LogicalPlan

from fenic.core._logical_plan.expressions.base import BinaryExpr
from fenic.core.types import (
    BooleanType,
    ColumnField,
)
from fenic.core.types.datatypes import is_dtype_numeric


class EqualityComparisonExpr(BinaryExpr):
    def _validate_types(self, plan: LogicalPlan):
        left_type = self.left.to_column_field(plan).data_type
        right_type = self.right.to_column_field(plan).data_type

        if left_type != right_type:
            raise TypeError(
                f"Type mismatch: Cannot apply {self.op} operator to non-matching types. "
                f"Left type: {left_type}, Right type: {right_type}. "
                f"Both operands must be of the same type."
            )
        return

    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
        self._validate_types(plan)
        return ColumnField(str(self), BooleanType)


class NumericComparisonExpr(BinaryExpr):
    def _validate_types(self, plan: LogicalPlan):
        left_type = self.left.to_column_field(plan).data_type
        right_type = self.right.to_column_field(plan).data_type

        if not is_dtype_numeric(left_type) or not is_dtype_numeric(right_type):
            raise TypeError(
                f"Type mismatch: Cannot apply {self.op} operator to non-numeric types. "
                f"Left type: {left_type}, Right type: {right_type}. "
                f"Both operands must be numeric: IntegerType, FloatType, or DoubleType"
            )
        return

    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
        self._validate_types(plan)
        return ColumnField(str(self), BooleanType)


class BooleanExpr(BinaryExpr):
    def _validate_types(self, plan: LogicalPlan):
        left_type = self.left.to_column_field(plan).data_type
        right_type = self.right.to_column_field(plan).data_type

        if left_type != BooleanType or right_type != BooleanType:
            raise TypeError(
                f"Type mismatch: Cannot apply {self.op} operator to non-boolean types. "
                f"Left type: {left_type}, Right type: {right_type}. "
                f"Both operands must be BooleanType)"
            )
        return

    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
        self._validate_types(plan)
        return ColumnField(str(self), BooleanType)
