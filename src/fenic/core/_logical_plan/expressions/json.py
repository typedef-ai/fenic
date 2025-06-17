from __future__ import annotations

import json as json_lib
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from fenic.core._logical_plan import LogicalPlan

from fenic._polars_plugins import py_validate_jq_query  # noqa: F401
from fenic.core._logical_plan.expressions.base import LogicalExpr
from fenic.core.error import TypeMismatchError, ValidationError
from fenic.core.types import (
    ArrayType,
    BooleanType,
    ColumnField,
    JsonType,
    StringType,
)


class JqExpr(LogicalExpr):
    def __init__(self, expr: LogicalExpr, query: str):
        self.expr = expr
        try:
            py_validate_jq_query(query)
        except ValueError as e:
            raise ValidationError(str(e)) from None
        self.query = query

    def __str__(self) -> str:
        return f"jq({self.expr}, {self.query})"

    def _validate_types(self, plan: LogicalPlan):
        input_field = self.expr.to_column_field(plan)
        if input_field.data_type != JsonType:
            raise TypeMismatchError(JsonType, input_field.data_type, "json.jq()")

    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
        self._validate_types(plan)
        return ColumnField(name=str(self), data_type=ArrayType(element_type=JsonType))

    def children(self) -> List[LogicalExpr]:
        return [self.expr]


class JsonTypeExpr(LogicalExpr):
    def __init__(self, expr: LogicalExpr):
        self.expr = expr
        self.jq_query = "{result: type}"

    def __str__(self) -> str:
        return f"json.type({self.expr})"

    def _validate_types(self, plan: LogicalPlan):
        input_field = self.expr.to_column_field(plan)
        if input_field.data_type != JsonType:
            raise TypeMismatchError(JsonType, input_field.data_type, "json.type()")

    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
        self._validate_types(plan)
        return ColumnField(name=str(self), data_type=StringType)

    def children(self) -> List[LogicalExpr]:
        return [self.expr]


class JsonContainsExpr(LogicalExpr):
    def __init__(self, expr: LogicalExpr, value: str):
        self.expr = expr
        self.value = value
        # Validate that value is valid JSON
        try:
            parsed_value = json_lib.loads(value)
            json_str = json_lib.dumps(parsed_value)
        except json_lib.JSONDecodeError as e:
            raise ValidationError(
                f"json.contains() requires a valid JSON string as the search value. "
                f"Received: {repr(value)}."
            ) from e

        # Use recursive descent with type-aware matching
        self.jq_query = f'{{result: any(..; (type == "object" and contains({json_str})) or (type != "object" and . == {json_str}))}}'

    def __str__(self) -> str:
        return f"json.contains({self.expr}, {self.value})"

    def _validate_types(self, plan: LogicalPlan):
        input_field = self.expr.to_column_field(plan)
        if input_field.data_type != JsonType:
            raise TypeMismatchError(JsonType, input_field.data_type, "json.contains()")

    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
        self._validate_types(plan)
        return ColumnField(name=str(self), data_type=BooleanType)

    def children(self) -> List[LogicalExpr]:
        return [self.expr]
