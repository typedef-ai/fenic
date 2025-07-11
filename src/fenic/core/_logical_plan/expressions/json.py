from __future__ import annotations

import json as json_lib

from fenic._polars_plugins import py_validate_jq_query  # noqa: F401
from fenic.core._logical_plan.expressions.base import LogicalExpr
from fenic.core._logical_plan.signatures.scalar_function import ScalarFunction
from fenic.core.error import ValidationError


class JqExpr(ScalarFunction):
    function_name = "json.jq"

    def __init__(self, expr: LogicalExpr, query: str):
        self.expr = expr
        self.query = query

        # Validate jq query at construction time
        try:
            py_validate_jq_query(query)
        except ValueError as e:
            raise ValidationError(str(e)) from None

        # Only validate the JSON expression (query is not LogicalExpr)
        super().__init__(expr)

    def __str__(self) -> str:
        return f"jq({self.expr}, {self.query})"


class JsonTypeExpr(ScalarFunction):
    function_name = "json.type"

    def __init__(self, expr: LogicalExpr):
        self.expr = expr
        self.jq_query = "{result: type}"

        super().__init__(expr)

    def __str__(self) -> str:
        return f"json.type({self.expr})"

class JsonContainsExpr(ScalarFunction):
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

        # Only validate the JSON expression (value is not LogicalExpr)
        super().__init__(expr)

    def __str__(self) -> str:
        return f"json.contains({self.expr}, {self.value})"
