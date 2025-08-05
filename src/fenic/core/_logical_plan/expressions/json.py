from __future__ import annotations

import json as json_lib
from typing import List

from fenic._polars_plugins import py_validate_jq_query  # noqa: F401
from fenic.core._logical_plan.expressions.base import (
    LogicalExpr,
    UnparameterizedExpr,
    ValidatedSignature,
)
from fenic.core._logical_plan.signatures.signature_validator import SignatureValidator
from fenic.core.error import ValidationError


class JqExpr(ValidatedSignature, LogicalExpr):
    function_name = "json.jq"

    def __init__(self, expr: LogicalExpr, query: str):
        self.expr = expr
        self.query = query

        # Validate jq query at construction time
        try:
            py_validate_jq_query(query)
        except ValueError as e:
            raise ValidationError(str(e)) from None

        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self) -> SignatureValidator:
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return [self.expr]

    def _eq_specific(self, other: JqExpr) -> bool:
        return self.query == other.query

class JsonTypeExpr(ValidatedSignature, UnparameterizedExpr, LogicalExpr):
    function_name = "json.type"

    def __init__(self, expr: LogicalExpr):
        self.expr = expr
        self.jq_query = "{result: type}"
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self) -> SignatureValidator:
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return [self.expr]


class JsonContainsExpr(ValidatedSignature, LogicalExpr):
    function_name = "json.contains"

    def __init__(self, expr: LogicalExpr, value: str):
        self.expr = expr
        self.value = value

        # Validate that value is valid JSON at construction time
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

        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self) -> SignatureValidator:
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return [self.expr]

    def _eq_specific(self, other: JsonContainsExpr) -> bool:
        return self.value == other.value
