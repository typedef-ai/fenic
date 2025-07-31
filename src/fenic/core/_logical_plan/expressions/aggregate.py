from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from fenic.core._logical_plan import LogicalPlan

from fenic.core._interfaces.session_state import BaseSessionState
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
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self) -> SignatureValidator:
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return [self.expr]

    def _eq_specific(self, other: SumExpr) -> bool:
        return True


class AvgExpr(ValidatedDynamicSignature, AggregateExpr):
    function_name = "avg"

    def __init__(self, expr: LogicalExpr):
        self.expr = expr
        self._validator = SignatureValidator(self.function_name)
        self.input_type = None  # Will be set during validation

    @property
    def validator(self) -> SignatureValidator:
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return [self.expr]

    def to_column_field(self, plan: LogicalPlan, session_state: BaseSessionState) -> ColumnField:
        """Use signature to validate and get return type, storing input type for transpiler."""
        # Get the input type first
        self.input_type = self.expr.to_column_field(plan, session_state).data_type

        # Now use the mixin implementation to validate and get return type
        return super().to_column_field(plan, session_state)

    def _infer_dynamic_return_type(self, arg_types: List[DataType], plan: LogicalPlan, session_state: BaseSessionState) -> DataType:
        """Return EmbeddingType for embeddings, DoubleType for numeric types."""
        input_type = arg_types[0]
        if isinstance(input_type, EmbeddingType):
            return input_type
        else:
            return DoubleType

    def _eq_specific(self, other: AvgExpr) -> bool:
        return True

class MinExpr(ValidatedSignature, AggregateExpr):
    function_name = "min"

    def __init__(self, expr: LogicalExpr):
        self.expr = expr
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self) -> SignatureValidator:
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return [self.expr]

    def _eq_specific(self, other: MinExpr) -> bool:
        return True

class MaxExpr(ValidatedSignature, AggregateExpr):
    function_name = "max"

    def __init__(self, expr: LogicalExpr):
        self.expr = expr
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self) -> SignatureValidator:
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return [self.expr]

    def _eq_specific(self, other: MaxExpr) -> bool:
        return True

class CountExpr(ValidatedSignature, AggregateExpr):
    function_name = "count"

    def __init__(self, expr: LogicalExpr):
        self.expr = expr
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self) -> SignatureValidator:
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return [self.expr]

    def _eq_specific(self, other: CountExpr) -> bool:
        return True

class ListExpr(ValidatedDynamicSignature, AggregateExpr):
    function_name = "collect_list"

    def __init__(self, expr: LogicalExpr):
        # Check for literal expressions upfront
        if isinstance(expr, LiteralExpr):
            raise TypeError(
                "Type mismatch: Cannot apply collect_list function to literal value. "
                "Only non-literal values are supported."
            )
        self.expr = expr
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self) -> SignatureValidator:
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return [self.expr]

    def _infer_dynamic_return_type(self, arg_types: List[DataType], plan: LogicalPlan, session_state: BaseSessionState) -> DataType:
        """Return ArrayType with element type matching the input type."""
        return ArrayType(arg_types[0])

    def _eq_specific(self, other: ListExpr) -> bool:
        return True

class FirstExpr(ValidatedSignature, AggregateExpr):
    function_name = "first"

    def __init__(self, expr: LogicalExpr):
        self.expr = expr
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self) -> SignatureValidator:
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return [self.expr]

    def _eq_specific(self, other: FirstExpr) -> bool:
        return True

class StdDevExpr(ValidatedSignature, AggregateExpr):
    function_name = "stddev"

    def __init__(self, expr: LogicalExpr):
        self.expr = expr
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self) -> SignatureValidator:
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return [self.expr]

    def _eq_specific(self, other: StdDevExpr) -> bool:
        return True
