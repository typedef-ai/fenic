"""Array-specific expression classes for the logical plan.

This module contains expression classes for array manipulation operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from fenic.core._logical_plan import LogicalPlan

from fenic.core._interfaces.session_state import BaseSessionState
from fenic.core._logical_plan.expressions.base import (
    LogicalExpr,
    UnparameterizedExpr,
    ValidatedDynamicSignature,
    ValidatedSignature,
)
from fenic.core._logical_plan.signatures.signature_validator import SignatureValidator
from fenic.core.types import ArrayType, DataType


class ArrayLengthExpr(ValidatedSignature, UnparameterizedExpr, LogicalExpr):
    """Expression representing array length calculation."""

    function_name = "array_size"

    def __init__(self, expr: LogicalExpr):
        self.expr = expr
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self) -> SignatureValidator:
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return [self.expr]


class ArrayDistinctExpr(ValidatedSignature, UnparameterizedExpr, LogicalExpr):
    function_name = "array_distinct"

    def __init__(self, expr: LogicalExpr):
        self.expr = expr
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self) -> SignatureValidator:
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return [self.expr]


class ArrayContainsExpr(ValidatedSignature, UnparameterizedExpr, LogicalExpr):
    """Expression representing array contains check."""

    function_name = "array_contains"

    def __init__(self, expr: LogicalExpr, other: LogicalExpr):
        self.expr = expr
        self.other = other
        self._children = [expr, other]
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self) -> SignatureValidator:
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return self._children


class ArrayMaxExpr(ValidatedDynamicSignature, UnparameterizedExpr, LogicalExpr):
    function_name = "array_max"

    def __init__(self, expr: LogicalExpr):
        self.expr = expr
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self) -> SignatureValidator:
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return [self.expr]

    def _infer_dynamic_return_type(self, arg_types: List[DataType], plan: LogicalPlan, session_state: BaseSessionState) -> DataType:
        """Return the element type of the array."""
        array_type = arg_types[0]
        if isinstance(array_type, ArrayType):
            return array_type.element_type
        raise TypeError(f"Expected ArrayType, got {array_type}")


class ArrayMinExpr(ValidatedDynamicSignature, UnparameterizedExpr, LogicalExpr):
    function_name = "array_min"

    def __init__(self, expr: LogicalExpr):
        self.expr = expr
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self) -> SignatureValidator:
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return [self.expr]

    def _infer_dynamic_return_type(self, arg_types: List[DataType], plan: LogicalPlan, session_state: BaseSessionState) -> DataType:
        """Return the element type of the array."""
        array_type = arg_types[0]
        if isinstance(array_type, ArrayType):
            return array_type.element_type
        raise TypeError(f"Expected ArrayType, got {array_type}")


class ArraySortExpr(ValidatedSignature, UnparameterizedExpr, LogicalExpr):
    function_name = "array_sort"

    def __init__(self, expr: LogicalExpr):
        self.expr = expr
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self) -> SignatureValidator:
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return [self.expr]


class ArrayReverseExpr(ValidatedSignature, UnparameterizedExpr, LogicalExpr):
    function_name = "array_reverse"

    def __init__(self, expr: LogicalExpr):
        self.expr = expr
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self) -> SignatureValidator:
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return [self.expr]


class ArrayRemoveExpr(ValidatedSignature, UnparameterizedExpr, LogicalExpr):
    function_name = "array_remove"

    def __init__(self, expr: LogicalExpr, element: LogicalExpr):
        self.expr = expr
        self.element = element
        self._children = [expr, element]
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self) -> SignatureValidator:
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return self._children


class ArrayUnionExpr(ValidatedSignature, UnparameterizedExpr, LogicalExpr):
    function_name = "array_union"

    def __init__(self, left: LogicalExpr, right: LogicalExpr):
        self.left = left
        self.right = right
        self._children = [left, right]
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self) -> SignatureValidator:
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return self._children


class ArrayIntersectExpr(ValidatedSignature, UnparameterizedExpr, LogicalExpr):
    function_name = "array_intersect"

    def __init__(self, left: LogicalExpr, right: LogicalExpr):
        self.left = left
        self.right = right
        self._children = [left, right]
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self) -> SignatureValidator:
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return self._children


class ArrayExceptExpr(ValidatedSignature, UnparameterizedExpr, LogicalExpr):
    function_name = "array_except"

    def __init__(self, left: LogicalExpr, right: LogicalExpr):
        self.left = left
        self.right = right
        self._children = [left, right]
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self) -> SignatureValidator:
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return self._children


class ArrayCompactExpr(ValidatedSignature, UnparameterizedExpr, LogicalExpr):
    function_name = "array_compact"

    def __init__(self, expr: LogicalExpr):
        self.expr = expr
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self) -> SignatureValidator:
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return [self.expr]


class ArrayRepeatExpr(ValidatedDynamicSignature, UnparameterizedExpr, LogicalExpr):
    function_name = "array_repeat"

    def __init__(self, element: LogicalExpr, count: LogicalExpr):
        self.element = element
        self.count = count
        self._children = [element, count]
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self) -> SignatureValidator:
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return self._children

    def _infer_dynamic_return_type(self, arg_types: List[DataType], plan: LogicalPlan, session_state: BaseSessionState) -> DataType:
        """Return an array of the element type."""
        return ArrayType(arg_types[0])


class ArraySliceExpr(ValidatedDynamicSignature, UnparameterizedExpr, LogicalExpr):
    function_name = "slice"

    def __init__(self, expr: LogicalExpr, start: LogicalExpr, length: LogicalExpr):
        self.expr = expr
        self.start = start
        self.length = length
        self._children = [expr, start, length]
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self) -> SignatureValidator:
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return self._children

    def _infer_dynamic_return_type(self, arg_types: List[DataType], plan: LogicalPlan, session_state: BaseSessionState) -> DataType:
        """Return the same array type as input."""
        return arg_types[0]


class ElementAtExpr(ValidatedDynamicSignature, UnparameterizedExpr, LogicalExpr):
    function_name = "element_at"

    def __init__(self, expr: LogicalExpr, index: LogicalExpr):
        self.expr = expr
        self.index = index
        self._children = [expr, index]
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self) -> SignatureValidator:
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return self._children

    def _infer_dynamic_return_type(self, arg_types: List[DataType], plan: LogicalPlan, session_state: BaseSessionState) -> DataType:
        """Return the element type of the array."""
        array_type = arg_types[0]
        if isinstance(array_type, ArrayType):
            return array_type.element_type
        raise TypeError(f"Expected ArrayType, got {array_type}")


class ArraysOverlapExpr(ValidatedSignature, UnparameterizedExpr, LogicalExpr):
    function_name = "arrays_overlap"

    def __init__(self, left: LogicalExpr, right: LogicalExpr):
        self.left = left
        self.right = right
        self._children = [left, right]
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self) -> SignatureValidator:
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return self._children
