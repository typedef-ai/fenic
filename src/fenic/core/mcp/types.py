"""Exported Types related to Parameterized View/MCP Tool Generation."""
from __future__ import annotations

from typing import List, Optional, Union

from pydantic import BaseModel, ConfigDict, model_validator
from pydantic.dataclasses import dataclass
from typing_extensions import Literal

from fenic.core._logical_plan.plans.base import LogicalPlan
from fenic.core.types.datatypes import DataType

ToolParameterType = Union[str, int, float, bool, list, dict]
TableFormat = Literal["structured", "markdown"]


class ToolParam(BaseModel):
    """A parameter for a parameterized view tool.

    A parameter is a named value that can be passed to a tool. These are matched to the
    parameter names of the `tool_param` UnresolvedLiteralExpr expressions captured in the Logical Plan.

    Attributes:
        name: The name of the parameter.
        description: The description of the parameter.
        allowed_values: The allowed values for the parameter.
        has_default: Whether the parameter has a default value.
        default_value: The default value for the parameter.
    """
    name: str
    description: str
    allowed_values: Optional[List[ToolParameterType]] = None
    has_default: bool = False
    default_value: Optional[ToolParameterType] = None

    @model_validator(mode='after')
    def _check_default_value(self):
        # If a default value is provided, mark has_default True
        if self.default_value is not None and not self.has_default:
            self.has_default = True
        return self

    @property
    def required(self) -> bool:
        """Whether the parameter is required.

        Returns:
            True if the parameter is required, False otherwise.
        """
        return not self.has_default


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class BoundToolParam:
    """A bound tool parameter.

    A bound tool parameter is a parameter that has been bound to a specific, typed,
    `tool_param` usage within a Dataframe.
    """
    name: str
    description: str
    data_type: DataType
    required: bool
    has_default: bool
    default_value: Optional[ToolParameterType]
    allowed_values: Optional[List[ToolParameterType]]


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ParameterizedToolDefinition:
    """A tool that has been bound to a specific Parameterized View."""
    name: str
    description: str
    params: list[BoundToolParam]
    result_limit: int
    _parameterized_view: LogicalPlan
