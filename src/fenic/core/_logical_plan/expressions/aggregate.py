from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from fenic.core._logical_plan import LogicalPlan

from fenic.core._logical_plan.expressions.base import LogicalExpr
from fenic.core._logical_plan.expressions.basic import LiteralExpr
from fenic.core._logical_plan.signatures.function_base import AggregateFunction
from fenic.core.types import (
    ArrayType,
    ColumnField,
    DataType,
    DoubleType,
    EmbeddingType,
)

class SumExpr(AggregateFunction):
    function_name = "sum"

    def __init__(self, expr: LogicalExpr):
        super().__init__(expr)


class AvgExpr(AggregateFunction):
    function_name = "avg"

    def __init__(self, expr: LogicalExpr):
        super().__init__(expr)
        self.input_type = None  # Will be set during validation

    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
        """Use signature to validate and get return type, storing input type for transpiler."""
        # Get the input type first
        self.input_type = self.expr.to_column_field(plan).data_type

        # Now use the parent implementation to validate and get return type
        return super().to_column_field(plan)

    def _infer_dynamic_return_type(self, arg_types: List[DataType], plan: LogicalPlan) -> DataType:
        """Return EmbeddingType for embeddings, DoubleType for numeric types."""
        input_type = arg_types[0]
        if isinstance(input_type, EmbeddingType):
            return input_type
        else:
            return DoubleType


class MinExpr(AggregateFunction):
    function_name = "min"

    def __init__(self, expr: LogicalExpr):
        super().__init__(expr)


class MaxExpr(AggregateFunction):
    function_name = "max"

    def __init__(self, expr: LogicalExpr):
        super().__init__(expr)


class CountExpr(AggregateFunction):
    function_name = "count"

    def __init__(self, expr: LogicalExpr):
        super().__init__(expr)


class ListExpr(AggregateFunction):
    function_name = "list"

    def __init__(self, expr: LogicalExpr):
        # Check for literal expressions upfront
        if isinstance(expr, LiteralExpr):
            raise TypeError(
                "Type mismatch: Cannot apply collect_list function to literal value. "
                "Only non-literal values are supported."
            )
        super().__init__(expr)

    def _infer_dynamic_return_type(self, arg_types: List[DataType], plan: LogicalPlan) -> DataType:
        """Return ArrayType with element type matching the input type."""
        return ArrayType(arg_types[0])

    def __str__(self) -> str:
        return f"collect_list({self.expr})"

class FirstExpr(AggregateFunction):
    function_name = "first"

    def __init__(self, expr: LogicalExpr):
        super().__init__(expr)

class StdDevExpr(AggregateFunction):
    function_name = "stddev"

    def __init__(self, expr: LogicalExpr):
        super().__init__(expr)
