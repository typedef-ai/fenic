from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

from pydantic import BaseModel, Field, create_model
from typing_extensions import Literal

from fenic.core._logical_plan import walker
from fenic.core._logical_plan.expressions.basic import UnresolvedLiteralExpr
from fenic.core._logical_plan.plans.base import LogicalPlan
from fenic.core._utils.type_inference import infer_pytype_from_dtype
from fenic.core.error import PlanError
from fenic.core.mcp.types import (
    BoundToolParam,
    TableFormat,
    ToolParam,
    UserDefinedTool,
)
from fenic.core.types.datatypes import ArrayType

LIMIT_DESCRIPTION = "The number of rows to return in the result set. Omit to return the maximum number of rows allowed by the tool."
TABLE_FORMAT_DESCRIPTION = "The format of the table to return in the response. If `structured`, the rows will be returned as a list of JSON objects. If `markdown`, the rows will be returned as a markdown-formatted table."

logger = logging.getLogger(__name__)

def bind_tool(
    name: str,
    description: str,
    params: list[ToolParam],
    result_limit: int,
    query: LogicalPlan
) -> UserDefinedTool:
    """Create a tool from a query and a set of parameters.

    Raises PlanError if the logical plan contains unresolved parameters that are not in the tool parameters.
    """
    unresolved_exprs: list[UnresolvedLiteralExpr] = [
        expr for expr in walker.find_expressions(query, lambda expr: isinstance(expr, UnresolvedLiteralExpr))
    ]

    unresolved_exprs_grouped = defaultdict(list)
    for expr in unresolved_exprs:
        unresolved_exprs_grouped[expr.parameter_name].append(expr)
    unresolved_exprs_by_name = {expr.parameter_name: expr for expr in unresolved_exprs}
    for _, unresolved_exprs in unresolved_exprs_grouped.items():
        if not all(unresolved_expr == unresolved_exprs[0] for unresolved_expr in unresolved_exprs):
            raise PlanError(
                "All unresolved expressions with the same parameter name must have the same configuration values"
            )

    params = {param.name: param for param in params}
    missing_params = unresolved_exprs_by_name.keys() - params.keys()
    if missing_params:
        raise PlanError(f"Tool does not have ToolParam(s) registered for the following placeholders: {missing_params}")
    extra_params = params.keys() - unresolved_exprs_by_name.keys()
    if extra_params:
        logger.warning(f"Extra parameters: {extra_params}")

    resolved_params: list[BoundToolParam] = []
    for unresolved_expr_name, unresolved_expr in unresolved_exprs_by_name.items():
        tool_param_model = params[unresolved_expr_name]
        # Validate allowed values if default present and non-None
        if (
            tool_param_model.allowed_values is not None
            and tool_param_model.has_default
            and tool_param_model.default_value is not None
        ):
            if tool_param_model.default_value not in tool_param_model.allowed_values:
                raise PlanError(
                    f"Default value {tool_param_model.default_value} is not in the allowed values {tool_param_model.allowed_values}"
                )
            # Ensure allowed values are homogeneous with the default's Python type
            if not all(isinstance(value, type(tool_param_model.default_value)) for value in tool_param_model.allowed_values):
                raise PlanError(
                    f"Allowed values {tool_param_model.allowed_values} must all be the same type as the default value {type(tool_param_model.default_value).__name__}"
                )

        resolved_params.append(
            BoundToolParam(
                name=tool_param_model.name,
                description=tool_param_model.description,
                data_type=unresolved_expr.data_type,
                required=tool_param_model.required,
                has_default=tool_param_model.has_default,
                default_value=tool_param_model.default_value,
                allowed_values=tool_param_model.allowed_values,
            )
        )

    return UserDefinedTool(
        name=name,
        description=description,
        params=resolved_params,
        _parameterized_view=query,
        max_result_limit=result_limit,
    )


def create_pydantic_model_for_tool(tool: UserDefinedTool) -> type[BaseModel]:
    """Create a Pydantic model for a tool."""
    model_name = f"{tool.name}_Params"
    model_fields = {}
    for param in tool.params:
        if param.allowed_values:
            literal_values = tuple(param.allowed_values)
            literal_type = Literal[literal_values]  # type: ignore[valid-type]
            if isinstance(param.data_type, ArrayType):
                literal_type = list[literal_type]  # type: ignore[valid-type]
            if param.has_default:
                model_fields[param.name] = (
                    Optional[literal_type],
                    Field(default=param.default_value, description=param.description),
                )
            else:
                model_fields[param.name] = (literal_type, Field(..., description=param.description))
        else:
            py_type = infer_pytype_from_dtype(param.data_type)
            if param.has_default:
                model_fields[param.name] = (
                    Optional[py_type],
                    Field(default=param.default_value, description=param.description),
                )
            else:
                model_fields[param.name] = (py_type, Field(..., description=param.description))

    model_fields["table_format"] = (
        TableFormat,
        Field(default="markdown", description=TABLE_FORMAT_DESCRIPTION),
    )
    model_fields["limit"] = (
        int,
        Field(default=tool.max_result_limit, description=LIMIT_DESCRIPTION),
    )

    return create_model(model_name, **model_fields)
