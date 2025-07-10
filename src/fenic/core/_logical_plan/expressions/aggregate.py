from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from fenic.core._logical_plan import LogicalPlan

from fenic.core._logical_plan.expressions.base import LogicalExpr
from fenic.core._logical_plan.expressions.basic import LiteralExpr
from fenic.core.types import (
    ArrayType,
    BooleanType,
    ColumnField,
    DoubleType,
    EmbeddingType,
    FloatType,
    IntegerType,
)

SUMMABLE_TYPES = (IntegerType, FloatType, DoubleType, BooleanType)


class AggregateExpr(LogicalExpr):
    def __init__(self, agg_name: str, expr: LogicalExpr):
        self.agg_name = agg_name
        self.expr = expr

    def __str__(self):
        return f"{self.agg_name}({str(self.expr)})"

    def children(self) -> List[LogicalExpr]:
        return [self.expr]


class SumExpr(AggregateExpr):
    def __init__(self, expr: LogicalExpr):
        super().__init__("sum", expr)

    def _validate_types(self, plan: LogicalPlan):
        expr_type = self.expr.to_column_field(plan).data_type
        if expr_type not in SUMMABLE_TYPES:
            raise TypeError(
                f"Type mismatch: Cannot apply sum function to non-numeric types. "
                f"Type: {expr_type}. "
                f"Only numeric types ({', '.join(t for t in SUMMABLE_TYPES)}) are supported."
            )

        return

    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
        self._validate_types(plan)
        return ColumnField(str(self), self.expr.to_column_field(plan).data_type)


class AvgExpr(AggregateExpr):
    def __init__(self, expr: LogicalExpr):
        super().__init__("avg", expr)
        self.input_type = None  # Will be set during validation

    def _validate_types(self, plan: LogicalPlan):
        expr_type = self.expr.to_column_field(plan).data_type
        self.input_type = expr_type

        if expr_type not in SUMMABLE_TYPES and not isinstance(expr_type, EmbeddingType):
            raise TypeError(
                f"Type mismatch: Cannot apply avg function to non-numeric types. "
                f"Type: {expr_type}. "
                f"Only numeric types ({', '.join(str(t) for t in SUMMABLE_TYPES)}) and EmbeddingType are supported."
            )
        return

    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
        self._validate_types(plan)

        # If averaging embeddings, return the same embedding type
        if isinstance(self.input_type, EmbeddingType):
            return ColumnField(str(self), self.input_type)
        else:
            return ColumnField(str(self), DoubleType)


class MinExpr(AggregateExpr):
    def __init__(self, expr: LogicalExpr):
        super().__init__("min", expr)

    def _validate_types(self, plan: LogicalPlan):
        expr_type = self.expr.to_column_field(plan).data_type

        if expr_type not in SUMMABLE_TYPES:
            raise TypeError(
                f"Type mismatch: Cannot apply min function to non-numeric types. "
                f"Type: {expr_type}. "
                f"Only numeric types ({', '.join(t for t in SUMMABLE_TYPES)}) are supported."
            )
        return

    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
        self._validate_types(plan)
        return ColumnField(str(self), self.expr.to_column_field(plan).data_type)


class MaxExpr(AggregateExpr):
    def __init__(self, expr: LogicalExpr):
        super().__init__("max", expr)

    def _validate_types(self, plan: LogicalPlan):
        expr_type = self.expr.to_column_field(plan).data_type

        if expr_type not in SUMMABLE_TYPES:
            raise TypeError(
                f"Type mismatch: Cannot apply max function to non-numeric types. "
                f"Type: {expr_type}. "
                f"Only numeric types ({', '.join(t for t in SUMMABLE_TYPES)}) are supported."
            )
        return

    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
        self._validate_types(plan)
        return ColumnField(str(self), self.expr.to_column_field(plan).data_type)


class CountExpr(AggregateExpr):
    def __init__(self, expr: LogicalExpr):
        super().__init__("count", expr)

    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
        self.expr.to_column_field(plan)
        return ColumnField(str(self), IntegerType)

    def children(self) -> List[LogicalExpr]:
        return [self.expr]


class ListExpr(AggregateExpr):
    def __init__(self, expr: LogicalExpr):
        super().__init__("collect_list", expr)

    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
        if isinstance(self.expr, LiteralExpr):
            raise TypeError(
                f"Type mismatch: Cannot apply collect_list function to literal value."
                f"Type: {self.expr.to_column_field(plan).data_type}. "
                f"Only non-literal values are supported."
            )
        else:
            return ColumnField(
                str(self), ArrayType(self.expr.to_column_field(plan).data_type)
            )

class FirstExpr(AggregateExpr):
    def __init__(self, expr: LogicalExpr):
        super().__init__("first", expr)

    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
        return ColumnField(str(self), self.expr.to_column_field(plan).data_type)

class StdDevExpr(AggregateExpr):
    def __init__(self, expr: LogicalExpr):
        super().__init__("stddev", expr)

    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
        return ColumnField(str(self), DoubleType)

    def _validate_types(self, plan: LogicalPlan):
        expr_type = self.expr.to_column_field(plan).data_type
        if expr_type not in SUMMABLE_TYPES:
            raise TypeError(
                f"Type mismatch: Cannot apply stddev function to non-numeric types. "
                f"Type: {expr_type}. "
                f"Only numeric types ({', '.join(str(t) for t in SUMMABLE_TYPES)}) are supported."
            )
