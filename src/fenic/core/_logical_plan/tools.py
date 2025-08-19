from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Generic, List, Optional, TypeVar, Union

from pydantic import BaseModel, Field, create_model, ConfigDict
from pydantic.dataclasses import dataclass

from fenic.core._utils.type_inference import infer_dtype_from_pyobj
from fenic.core._logical_plan import walker

from fenic.core._logical_plan.expressions.basic import UnresolvedLiteralExpr
from fenic.core._logical_plan.plans.base import LogicalPlan
from fenic.core.error import PlanError, TypeMismatchError

logger = logging.getLogger(__name__)

ToolParamType = TypeVar("ToolParamType", bound=Union[str, int, float, bool, list, dict])

class ToolParam(BaseModel, Generic[ToolParamType]):
    name: str
    description: str
    type: type[ToolParamType]
    required: bool = True
    default: Optional[ToolParamType] = None

class ToolParams(BaseModel):
    params: List[ToolParam]

@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ValidatedTool:
    name: str
    description: str
    params: ToolParams
    query: LogicalPlan
    result_limit: int

def create_validated_tool(name: str, description: str, params: ToolParams, query: LogicalPlan, result_limit: int = 50) -> ValidatedTool:
    """Create a tool from a query and a set of parameters.

    Args:
        name: The name of the tool.
        description: The description of the tool.
        params: The parameters of the tool.
        query: The query to execute.
        result_limit: The number of rows to return in the result set.

    Returns:
        A Tool object.

    Raises:
        PlanError: If the the logical plan contains unresolved parameters that are not in the tool parameters.
    """
    unresolved_parameters = {expr.parameter_name: expr for expr in walker.find_expressions(query, lambda expr: isinstance(expr, UnresolvedLiteralExpr))}
    tool_params = {param.name: param for param in params.params}
    missing_params = unresolved_parameters.keys() - tool_params.keys()
    if missing_params:
        raise PlanError(f"Missing parameters: {missing_params}")
    extra_params = tool_params.keys() - unresolved_parameters.keys()
    if extra_params:
        logger.warning(f"Extra parameters: {extra_params}")

    for param_name, param in tool_params.items():
        unresolved_param = unresolved_parameters[param_name]
        tool_param_dtype = infer_dtype_from_pyobj(param.type)
        if unresolved_param.data_type != tool_param_dtype:
            raise TypeMismatchError.from_message(f"Parameter {param_name} has incompatible type. Expected {unresolved_param.data_type}, got {param.type}")
    return ValidatedTool(name, description, params, query, result_limit)
    
def create_pydantic_model_for_tool(tool: ValidatedTool) -> type[BaseModel]:
    """Create a Pydantic model for a tool.

    Args:
        tool: The tool to create a Pydantic model for.

    Returns:
        A Pydantic model.
    """
    model_name = f"{tool.name}_Params"
    model_fields = {}
    for param in tool.params.params:
        if param.default is None:
            model_fields[param.name] = (param.type, Field(..., description=param.description))
        else:
            model_fields[param.name] = (param.type, Field(param.default, description=param.description))

    return create_model(model_name, **model_fields)


