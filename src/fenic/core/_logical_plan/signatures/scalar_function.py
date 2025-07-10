"""ScalarFunction base class for functions with centralized signature validation.

This module provides the ScalarFunction class that uses the registry system
for type validation and return type inference.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from fenic.core._logical_plan.plans.base import LogicalPlan

from fenic.core._logical_plan.expressions.base import LogicalExpr
from fenic.core._logical_plan.signatures.registry import FunctionRegistry
from fenic.core.types.datatypes import DataType
from fenic.core.types.schema import ColumnField


class ScalarFunction(LogicalExpr):
    """Base class for scalar functions with signatures.

    Concrete subclasses store their own parameters as attributes and only pass LogicalExpr
    arguments to super().__init__() for type validation.
    """

    function_name: str  # Each subclass must specify its function name

    def __init__(self, *children: LogicalExpr):
        """Initialize ScalarFunction with LogicalExpr children for type validation.

        Args:
            *children: LogicalExpr arguments that will be validated and form the expression tree
        """
        self._children = list(children)

    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
        """Use signature to validate and get return type."""
        signature = FunctionRegistry.get_signature(self.function_name)
        return_type = signature.validate_and_infer_type(
            self._children, plan, self._infer_dynamic_return_type
        )
        return ColumnField(name=str(self), data_type=return_type)

    def _infer_dynamic_return_type(self, arg_types: List[DataType]) -> DataType:
        """Override in subclasses that use DYNAMIC return type strategy."""
        raise NotImplementedError(f"{self.function_name} must implement _infer_dynamic_return_type")

    def children(self) -> List[LogicalExpr]:
        """Return child expressions (automatically managed by base class)."""
        return self._children

    def __str__(self) -> str:
        args_str = ", ".join(str(arg) for arg in self._children)
        return f"{self.function_name}({args_str})"
