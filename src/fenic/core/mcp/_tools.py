from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

from pydantic import AfterValidator, BaseModel, Field, create_model
from typing_extensions import Annotated as TypingAnnotated
from typing_extensions import Literal

from fenic.core._logical_plan import walker
from fenic.core._logical_plan.expressions.basic import UnresolvedLiteralExpr
from fenic.core._logical_plan.plans.base import LogicalPlan
from fenic.core._utils.type_inference import infer_pytype_from_dtype
from fenic.core.error import PlanError
from fenic.core.mcp._validators import get_param_validator
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
        validators = []
        if tool_param_model.validator_names:
            missing_validators = []
            for validator_name in tool_param_model.validator_names:
                try:
                    validator = get_param_validator(validator_name)
                    if unresolved_expr.data_type not in validator.data_types():
                        supported_data_types = ", ".join([str(dt) for dt in validator.data_types()])
                        raise PlanError(
                            f"Param Validator `{validator_name}` supports data types ({supported_data_types}), "
                            f"but the parameter `{unresolved_expr_name}` has data type {unresolved_expr.data_type}."
                        )
                    validators.append(validator)
                except KeyError:
                    missing_validators.append(validator_name)
            if missing_validators:
                raise PlanError(
                    f"Could not find a ParamValidator for the following validator names: {missing_validators}"
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
                constraints=tool_param_model.constraints,
                validators=validators,
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

    def _infer_base_type(p: BoundToolParam):
        if p.allowed_values:
            literal_values = tuple(p.allowed_values)
            literal_type = Literal[literal_values]  # type: ignore[valid-type]
            if isinstance(p.data_type, ArrayType):
                return list[literal_type]  # type: ignore[valid-type]
            return literal_type
        if isinstance(p.data_type, ArrayType):
            inner_type = infer_pytype_from_dtype(p.data_type.element_type)
            return list[inner_type]  # type: ignore[valid-type]
        return infer_pytype_from_dtype(p.data_type)

    def _field_kwargs(p: BoundToolParam, include_default: bool) -> dict:
        kwargs: dict = {"description": p.description}
        constraints = p.constraints
        if constraints is not None:
            if constraints.gt is not None:
                kwargs["gt"] = constraints.gt
            if constraints.ge is not None:
                kwargs["ge"] = constraints.ge
            if constraints.lt is not None:
                kwargs["lt"] = constraints.lt
            if constraints.le is not None:
                kwargs["le"] = constraints.le
            if constraints.multiple_of is not None:
                kwargs["multiple_of"] = constraints.multiple_of
            if constraints.min_length is not None:
                kwargs["min_length"] = constraints.min_length
            if constraints.max_length is not None:
                kwargs["max_length"] = constraints.max_length
            if constraints.pattern is not None:
                kwargs["pattern"] = constraints.pattern
        if include_default:
            kwargs["default"] = p.default_value
        return kwargs

    for param in tool.params:

        def validate_param(input, param=param):
            for validator in param.validators:
                validator.validate(input)
            return input

        base_type = _infer_base_type(param)
        annotated_type = TypingAnnotated[base_type, AfterValidator(validate_param)]
        if param.has_default:
            model_fields[param.name] = (
                Optional[annotated_type],
                Field(**_field_kwargs(param, include_default=True)),
            )
        else:
            model_fields[param.name] = (
                annotated_type,
                Field(..., **_field_kwargs(param, include_default=False)),
            )

    model_fields["table_format"] = (
        TableFormat,
        Field(default="markdown", description=TABLE_FORMAT_DESCRIPTION),
    )
    model_fields["limit"] = (
        int,
        Field(default=tool.max_result_limit, description=LIMIT_DESCRIPTION),
    )

    return create_model(model_name, **model_fields)
