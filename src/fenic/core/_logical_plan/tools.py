from __future__ import annotations

import logging
from collections import defaultdict
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator
from pydantic.dataclasses import dataclass

from fenic.core._logical_plan import walker
from fenic.core._logical_plan.expressions.basic import UnresolvedLiteralExpr
from fenic.core._logical_plan.plans.base import LogicalPlan
from fenic.core._utils.type_inference import infer_pytype_from_dtype
from fenic.core.error import PlanError
from fenic.core.types.datatypes import DataType

logger = logging.getLogger(__name__)

ToolParameterType = Union[str, int, float, bool, list, dict, None]


class ToolParam(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    has_default: bool = Field(default=False)
    default_value: Optional[ToolParameterType] = Field(default=None)
    allowed_values: Optional[List[ToolParameterType]] = Field(default=None)

    @model_validator(mode="after")
    def _validate(self):
        if self.has_default:
            # default_value may be None, but must be allowed if you constrain values
            if self.allowed_values is not None and self.default_value not in self.allowed_values:
                raise ValueError("default_value must be one of allowed_values")
        else:
            # if user omitted passing it, we can set it properly here.
            if self.default_value is not None:
                self.has_default = True
        return self

    @property
    def required(self):
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


def create_unresolved_tool(name: str, description: str, params: list[ToolParam], result_limit: int) -> UnresolvedTool:
    """Create an unresolved tool.

    An unresolved tool is a tool that has not been bound to a Logical Plan.
    It is used to create a tool that can be used to execute a query.

    Args:
        name: The name of the tool.
        description: The description of the tool.
        params: The parameters of the tool.
        result_limit: The number of rows to return in the result set.

    Returns:
        An UnresolvedTool object.
    """
    return UnresolvedTool(name=name, description=description, params=params, result_limit=result_limit)


def resolve_tool(unresolved_tool: UnresolvedTool, query: LogicalPlan) -> ResolvedTool:
    """Create a tool from a query and a set of parameters.

    Args:
        unresolved_tool: The unresolved tool.
        query: The query to execute.

    Returns:
        A ResolvedTool object.

    Raises:
        PlanError: If the the logical plan contains unresolved parameters that are not in the tool parameters.
    """
    unresolved_exprs: list[UnresolvedLiteralExpr] = \
        [expr for expr in walker.find_expressions(
            query,
            lambda expr: isinstance(expr, UnresolvedLiteralExpr))
         ]

    unresolved_exprs_grouped = defaultdict(list)
    for expr in unresolved_exprs:
        unresolved_exprs_grouped[expr.parameter_name].append(expr)
    unresolved_exprs_by_name = {expr.parameter_name: expr for expr in unresolved_exprs}
    for _, unresolved_exprs in unresolved_exprs_grouped.items():
        if not all(unresolved_expr == unresolved_exprs[0] for unresolved_expr in unresolved_exprs):
            raise PlanError(
                "All unresolved expressions with the same parameter name must have the same configuration values")

    tool_params = {param.name: param for param in unresolved_tool.params}
    missing_params = unresolved_exprs_by_name.keys() - tool_params.keys()
    if missing_params:
        raise PlanError(f"Missing parameters: {missing_params}")
    extra_params = tool_params.keys() - unresolved_exprs_by_name.keys()
    if extra_params:
        logger.warning(f"Extra parameters: {extra_params}")

    resolved_params = []
    for unresolved_expr_name, unresolved_expr in unresolved_exprs_by_name.items():
        tool_param = tool_params[unresolved_expr_name]
        resolved_params.append(
            ResolvedToolParam(
                name=tool_param.name,
                description=tool_param.description,
                data_type=unresolved_expr.data_type,
                required=tool_param.required,
                has_default=tool_param.has_default,
                default_value=tool_param.default_value,
                allowed_values=tool_param.allowed_values,
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
    """Create a Pydantic model for a tool.

    Args:
        tool: The tool to create a Pydantic model for.

    Returns:
        A Pydantic model.
    """
    model_name = f"{tool.name}_Params"
    model_fields = {}
    for param in tool.params:
        if param.allowed_values is not None:
            literal_type = Literal[tuple(param.allowed_values)]
            if param.has_default:
                model_fields[param.name] = (literal_type,
                                            Field(default=param.default_value, description=param.description))
            else:
                model_fields[param.name] = (literal_type, Field(..., description=param.description))
        else:
            py_type = infer_pytype_from_dtype(param.data_type)
            if param.has_default:
                model_fields[param.name] = (py_type, Field(default=param.default_value, description=param.description))
            else:
                model_fields[param.name] = (py_type, Field(..., description=param.description))

    return create_model(model_name, **model_fields)
