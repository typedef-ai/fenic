from __future__ import annotations

import logging
from collections import defaultdict
from typing import Callable, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator
from pydantic.dataclasses import dataclass
from typing_extensions import Literal, Self

from fenic.core._logical_plan import walker
from fenic.core._logical_plan.expressions.basic import UnresolvedLiteralExpr
from fenic.core._logical_plan.plans.base import LogicalPlan
from fenic.core._utils.type_inference import infer_pytype_from_dtype
from fenic.core.error import PlanError, ValidationError
from fenic.core.types.datatypes import DataType

logger = logging.getLogger(__name__)

ToolParameterType = Union[str, int, float, bool, list, dict]
TableFormat = Literal["structured", "markdown"]

class ToolParam(BaseModel):
    name: str
    description: str
    allowed_values: Optional[List[ToolParameterType]] = None
    has_default: bool = False
    default_value: Optional[ToolParameterType] = None

    @model_validator(mode='after')
    def check_default_value(self) -> Self:
        # If a default value is provided, mark has_default True
        if self.default_value is not None and not self.has_default:
            self.has_default = True
        # If allowed_values provided and default is explicitly None, disallow
        if self.allowed_values and self.has_default and self.default_value is None:
            raise ValidationError(
                "NoneType cannot be used with default value if allowed_values is not empty."
            )
        return self

    @property
    def required(self) -> bool:
        return not self.has_default


class UnresolvedTool(BaseModel):
    name: str
    description: str
    params: List[ToolParam]
    result_limit: int


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ResolvedToolParam:
    name: str
    description: str
    data_type: DataType
    required: bool
    has_default: bool
    default_value: Optional[ToolParameterType]
    allowed_values: Optional[List[ToolParameterType]]


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ResolvedTool:
    name: str
    description: str
    params: list[ResolvedToolParam]
    query: LogicalPlan
    result_limit: int


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class DynamicTool:
    """A tool implemented as a regular Python callable with explicit parameters.

    The callable must return a LogicalPlan. Collection/formatting is handled by
    the MCP generator wrapper.
    """
    name: str
    description: str
    func: Callable[..., LogicalPlan]
    result_limit: Optional[int]


def create_unresolved_tool(name: str, description: str, params: list[ToolParam], result_limit: int) -> UnresolvedTool:
    """Create an unresolved tool."""
    return UnresolvedTool(name=name, description=description, params=params, result_limit=result_limit)


def resolve_tool(unresolved_tool: UnresolvedTool, query: LogicalPlan) -> ResolvedTool:
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

    tool_params = {param.name: param for param in unresolved_tool.params}
    missing_params = unresolved_exprs_by_name.keys() - tool_params.keys()
    if missing_params:
        raise PlanError(f"Missing parameters: {missing_params}")
    extra_params = tool_params.keys() - unresolved_exprs_by_name.keys()
    if extra_params:
        logger.warning(f"Extra parameters: {extra_params}")

    resolved_params: list[ResolvedToolParam] = []
    for unresolved_expr_name, unresolved_expr in unresolved_exprs_by_name.items():
        tool_param_model = tool_params[unresolved_expr_name]
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
            ResolvedToolParam(
                name=tool_param_model.name,
                description=tool_param_model.description,
                data_type=unresolved_expr.data_type,
                required=tool_param_model.required,
                has_default=tool_param_model.has_default,
                default_value=tool_param_model.default_value,
                allowed_values=tool_param_model.allowed_values,
            )
        )

    return ResolvedTool(
        name=unresolved_tool.name,
        description=unresolved_tool.description,
        params=resolved_params,
        query=query,
        result_limit=unresolved_tool.result_limit,
    )


def create_pydantic_model_for_tool(tool: ResolvedTool) -> type[BaseModel]:
    """Create a Pydantic model for a tool."""
    model_name = f"{tool.name}_Params"
    model_fields = {}
    for param in tool.params:
        if param.allowed_values is not None:
            literal_values = tuple(param.allowed_values)
            literal_type = Literal[literal_values]  # type: ignore[valid-type]
            if param.has_default:
                model_fields[param.name] = (
                    literal_type,
                    Field(default=param.default_value, description=param.description),
                )
            else:
                model_fields[param.name] = (literal_type, Field(..., description=param.description))
        else:
            py_type = infer_pytype_from_dtype(param.data_type)
            if param.has_default:
                model_fields[param.name] = (
                    py_type,
                    Field(default=param.default_value, description=param.description),
                )
            else:
                model_fields[param.name] = (py_type, Field(..., description=param.description))

    model_fields["table_format"] = (
        TableFormat,
        Field(default="structured", description="The format of the table to return in the response."),
    )
    model_fields["limit"] = (
        int,
        Field(default=tool.result_limit, description="The number of rows to return in the result set."),
    )

    return create_model(model_name, **model_fields)
