from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from fenic.core._logical_plan import LogicalPlan

from fenic.core._interfaces.session_state import BaseSessionState
from fenic.core._logical_plan.expressions.base import (
    AggregateExpr,
    LogicalExpr,
    UnparameterizedExpr,
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


class SumExpr(ValidatedSignature, UnparameterizedExpr, AggregateExpr):
    function_name = "sum"

    def __init__(self, expr: LogicalExpr):
        self.expr = expr
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self) -> SignatureValidator:
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return [self.expr]


class AvgExpr(ValidatedDynamicSignature, UnparameterizedExpr, AggregateExpr):
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

class MinExpr(ValidatedSignature, UnparameterizedExpr, AggregateExpr):
    function_name = "min"

    def __init__(self, expr: LogicalExpr):
        self.expr = expr
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self) -> SignatureValidator:
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return [self.expr]

class MaxExpr(ValidatedSignature, UnparameterizedExpr, AggregateExpr):
    function_name = "max"

    def __init__(self, expr: LogicalExpr):
        self.expr = expr
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self) -> SignatureValidator:
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return [self.expr]

class CountExpr(ValidatedSignature, UnparameterizedExpr, AggregateExpr):
    function_name = "count"

    def __init__(self, expr: LogicalExpr):
        self.expr = expr
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self) -> SignatureValidator:
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return [self.expr]

class ListExpr(ValidatedDynamicSignature, UnparameterizedExpr, AggregateExpr):
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

class FirstExpr(ValidatedSignature, UnparameterizedExpr, AggregateExpr):
    function_name = "first"

    def __init__(self, expr: LogicalExpr):
        self.expr = expr
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self) -> SignatureValidator:
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return [self.expr]

class StdDevExpr(ValidatedSignature, UnparameterizedExpr, AggregateExpr):
    function_name = "stddev"

    def __init__(self, expr: LogicalExpr):
        self.expr = expr
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self) -> SignatureValidator:
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return [self.expr]


class ApproxCountDistinctExpr(ValidatedSignature, UnparameterizedExpr, AggregateExpr):
    function_name = "approx_count_distinct"

    def __init__(self, expr: LogicalExpr):
        self.expr = expr
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self) -> SignatureValidator:
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return [self.expr]


class CountDistinctExpr(ValidatedSignature, UnparameterizedExpr, AggregateExpr):
    function_name = "count_distinct"

    def __init__(self, exprs: List[LogicalExpr], nulls_not_distinct: bool = False):
        self.exprs = exprs
        self._validator = SignatureValidator(self.function_name)
        self.nulls_not_distinct = nulls_not_distinct

    @property
    def validator(self) -> SignatureValidator:
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return list(self.exprs)


class SumDistinctExpr(ValidatedSignature, UnparameterizedExpr, AggregateExpr):
    function_name = "sum_distinct"

    def __init__(self, expr: LogicalExpr):
        self.expr = expr
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self) -> SignatureValidator:
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return [self.expr]
