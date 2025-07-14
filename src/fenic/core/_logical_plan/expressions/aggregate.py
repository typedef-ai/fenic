from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from fenic.core._logical_plan import LogicalPlan

from fenic.core._logical_plan.expressions.base import (
    AggregateExpr,
    LogicalExpr,
    ValidatedDynamicSignature,
    ValidatedSignature,
)
from fenic.core._logical_plan.expressions.basic import LiteralExpr
from fenic.core._logical_plan.signatures.signature_validator import SignatureValidator
from fenic.core.types import (
    ArrayType,
    ColumnField,
    DataType,
    DoubleType,
    EmbeddingType,
)


class SumExpr(ValidatedSignature, AggregateExpr):
    function_name = "sum"

    def __init__(self, expr: LogicalExpr):
        self.expr = expr
        self._children = [expr]
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self) -> SignatureValidator:
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return self._children

    def __str__(self) -> str:
        args_str = ", ".join(str(arg) for arg in self._children)
        return f"{self.function_name}({args_str})"


class AvgExpr(ValidatedDynamicSignature, AggregateExpr):
    function_name = "avg"

    def __init__(self, expr: LogicalExpr):
        self.expr = expr
        self._children = [expr]
        self._validator = SignatureValidator(self.function_name)
        self.input_type = None  # Will be set during validation

    @property
    def validator(self) -> SignatureValidator:
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return self._children

    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
        """Use signature to validate and get return type, storing input type for transpiler."""
        # Get the input type first
        self.input_type = self.expr.to_column_field(plan).data_type

        # Now use the mixin implementation to validate and get return type
        return super().to_column_field(plan)

    def _infer_dynamic_return_type(self, arg_types: List[DataType], plan: LogicalPlan) -> DataType:
        """Return EmbeddingType for embeddings, DoubleType for numeric types."""
        input_type = arg_types[0]
        if isinstance(input_type, EmbeddingType):
            return input_type
        else:
            return DoubleType

    def __str__(self) -> str:
        args_str = ", ".join(str(arg) for arg in self._children)
        return f"{self.function_name}({args_str})"


class MinExpr(ValidatedSignature, AggregateExpr):
    function_name = "min"

    def __init__(self, expr: LogicalExpr):
        self.expr = expr
        self._children = [expr]
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self) -> SignatureValidator:
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return self._children

    def __str__(self) -> str:
        args_str = ", ".join(str(arg) for arg in self._children)
        return f"{self.function_name}({args_str})"


class MaxExpr(ValidatedSignature, AggregateExpr):
    function_name = "max"

    def __init__(self, expr: LogicalExpr):
        self.expr = expr
        self._children = [expr]
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self) -> SignatureValidator:
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return self._children

    def __str__(self) -> str:
        args_str = ", ".join(str(arg) for arg in self._children)
        return f"{self.function_name}({args_str})"


class CountExpr(ValidatedSignature, AggregateExpr):
    function_name = "count"

    def __init__(self, expr: LogicalExpr):
        self.expr = expr
        self._children = [expr]
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self) -> SignatureValidator:
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return self._children

    def __str__(self) -> str:
        args_str = ", ".join(str(arg) for arg in self._children)
        return f"{self.function_name}({args_str})"


class ListExpr(ValidatedDynamicSignature, AggregateExpr):
    function_name = "list"

    def __init__(self, expr: LogicalExpr):
        # Check for literal expressions upfront
        if isinstance(expr, LiteralExpr):
            raise TypeError(
                "Type mismatch: Cannot apply collect_list function to literal value. "
                "Only non-literal values are supported."
            )
        self.expr = expr
        self._children = [expr]
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self) -> SignatureValidator:
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return self._children

    def _infer_dynamic_return_type(self, arg_types: List[DataType], plan: LogicalPlan) -> DataType:
        """Return ArrayType with element type matching the input type."""
        return ArrayType(arg_types[0])

    def __str__(self) -> str:
        return f"collect_list({self.expr})"

class FirstExpr(ValidatedSignature, AggregateExpr):
    function_name = "first"

    def __init__(self, expr: LogicalExpr):
        self.expr = expr
        self._children = [expr]
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self) -> SignatureValidator:
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return self._children

    def __str__(self) -> str:
        args_str = ", ".join(str(arg) for arg in self._children)
        return f"{self.function_name}({args_str})"

class StdDevExpr(ValidatedSignature, AggregateExpr):
    function_name = "stddev"

    def __init__(self, expr: LogicalExpr):
        self.expr = expr
        self._children = [expr]
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self) -> SignatureValidator:
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return self._children

    def __str__(self) -> str:
        args_str = ", ".join(str(arg) for arg in self._children)
        return f"{self.function_name}({args_str})"
