from __future__ import annotations

import copy
from typing import Any, Dict, List

from fenic.core._logical_plan.expressions.base import LogicalExpr
from fenic.core._logical_plan.expressions.basic import (
    LiteralExpr,
    UnresolvedLiteralExpr,
)
from fenic.core._logical_plan.plans.base import LogicalPlan
from fenic.core._logical_plan.walker import find_expressions
from fenic.core._utils.type_inference import TypeInferenceError, infer_dtype_from_pyobj
from fenic.core.error import PlanError, TypeMismatchError
from fenic.core.mcp.types import BoundToolParam


def collect_unresolved_parameters(plan: LogicalPlan) -> dict[str, UnresolvedLiteralExpr]:
    """Return the set of parameter names referenced by UnresolvedLiteralExpr in the plan."""
    expressions = find_expressions(plan, lambda expr: isinstance(expr, UnresolvedLiteralExpr))
    return {expr.parameter_name: expr for expr in expressions}


def bind_parameters(plan: LogicalPlan, params: Dict[str, Any], tool_params: List[BoundToolParam]) -> LogicalPlan:
    """Bind provided params into a LogicalPlan by replacing placeholder literals.

    - Replaces every UnresolvedLiteralExpr with a LiteralExpr(value, data_type)
    - Raises PlanError if any placeholder parameter is missing in params
    - Raises TypeMismatchError if provided value does not match placeholder data_type
    - Operates in-place on the plan/expressions and returns the same plan reference

    Note: This only substitutes literal values. It does not change node schemas
    and assumes placeholder data types match the provided values semantically.
    """
    plan_copy = copy.deepcopy(plan)
    unresolved_parameters = collect_unresolved_parameters(plan_copy)
    tool_configs_by_name = {tool_config.name: tool_config for tool_config in tool_params}
    missing = unresolved_parameters.keys() - set(params.keys())
    missing_with_no_defaults = []
    if missing:
        for param_name in missing:
            tool_param = tool_configs_by_name.get(param_name)
            # If we have no tool metadata or it has no default, it's an error
            if tool_param is None or not tool_param.has_default:
                missing_with_no_defaults.append(param_name)

    if missing_with_no_defaults:
        raise PlanError(
            "Missing parameter values for placeholders: " + ", ".join(sorted(missing_with_no_defaults))
        )

    def _transform_expr(expr: LogicalExpr) -> LogicalExpr:
        # Replace this node if it is a parameter placeholder
        if isinstance(expr, UnresolvedLiteralExpr):
            param_name = expr.parameter_name
            if param_name not in params:
                tool_param = tool_configs_by_name.get(param_name)
                if tool_param is None:
                    # Defensive: metadata missing and value not provided
                    raise PlanError(
                        "Missing parameter values for placeholders: " + param_name
                    )
                if not tool_param.has_default:
                    raise PlanError(
                        f"Parameter '{param_name}' was not provided and has no default value."
                        "Either provide a value for the parameter or add a default value to the tool parameter."
                    )
                return LiteralExpr(literal=tool_param.default_value, data_type=expr.data_type)
            value = params[param_name]
            try:
                inferred = infer_dtype_from_pyobj(value, path=param_name)
            except TypeInferenceError as e:
                raise PlanError(
                    f"Failed to infer type for parameter '{param_name}': {e}"
                ) from e
            if inferred != expr.data_type:
                raise TypeMismatchError.from_message(
                    f"Parameter '{param_name}' has incompatible type. "
                    f"Expected {expr.data_type}, got {inferred}."
                )
            return LiteralExpr(literal=value, data_type=expr.data_type)
        # Otherwise, recurse into child attributes generically
        for attr, val in list(vars(expr).items()):
            new_val = _transform_value(val)
            if new_val is not val:
                setattr(expr, attr, new_val)
        return expr

    def _transform_value(value: Any) -> Any:
        if isinstance(value, LogicalExpr):
            return _transform_expr(value)
        if isinstance(value, dict):
            changed = False
            new_dict: Dict[Any, Any] = {}
            for k, v in value.items():
                new_v = _transform_value(v)
                changed = changed or (new_v is not v)
                new_dict[k] = new_v
            return new_dict if changed else value
        if isinstance(value, (list, tuple, set)):
            container_type = type(value)
            new_items: List[Any] = []
            changed = False
            for v in value:
                new_v = _transform_value(v)
                changed = changed or (new_v is not v)
                new_items.append(new_v)
            if not changed:
                return value
            if container_type is tuple:
                return tuple(new_items)
            if container_type is set:
                return set(new_items)
            return new_items
        return value

    def _transform_plan(node: LogicalPlan) -> None:
        # Transform expressions attached to this plan node
        for attr, val in list(vars(node).items()):
            new_val = _transform_value(val)
            if new_val is not val:
                setattr(node, attr, new_val)
        # Recurse into children
        for child in node.children():
            _transform_plan(child)

    _transform_plan(plan_copy)
    return plan_copy
