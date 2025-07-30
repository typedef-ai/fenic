from __future__ import annotations

from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from fenic.core._logical_plan import LogicalPlan

from fenic.core._interfaces.session_state import BaseSessionState
from fenic.core._logical_plan.expressions.base import BinaryExpr
from fenic.core.types.datatypes import (
    DataType,
    DoubleType,
    FloatType,
    IntegerType,
    _DoubleType,
    _FloatType,
    _IntegerType,
    _is_dtype_numeric,
)
from fenic.core.types.schema import ColumnField


class ArithmeticExpr(BinaryExpr):
    def _validate_types(self, left_type: DataType, right_type: DataType):
        if not _is_dtype_numeric(left_type) or not _is_dtype_numeric(right_type):
            raise TypeError(
                f"Type mismatch: Cannot apply {self.op} operator to non-numeric types. "
                f"Left type: {left_type}, Right type: {right_type}. "
                f"Both operands must be numeric: IntegerType, FloatType, or DoubleType"
            )
        return

    def _promote_type(
        self,
        left_type: Union[_IntegerType, _FloatType, _DoubleType],
        right_type: Union[_IntegerType, _FloatType, _DoubleType],
    ) -> Union[_IntegerType, _FloatType, _DoubleType]:
        if isinstance(left_type, _DoubleType) or isinstance(right_type, _DoubleType):
            return DoubleType
        elif isinstance(left_type, _FloatType) or isinstance(right_type, _FloatType):
            return FloatType
        else:
            return IntegerType

    def to_column_field(self, plan: LogicalPlan, session_state: BaseSessionState) -> ColumnField:
        left_field = self.left.to_column_field(plan, session_state)
        right_field = self.right.to_column_field(plan, session_state)
        self._validate_types(left_field.data_type, right_field.data_type)
        result_type = self._promote_type(left_field.data_type, right_field.data_type)
        return ColumnField(str(self), result_type)
