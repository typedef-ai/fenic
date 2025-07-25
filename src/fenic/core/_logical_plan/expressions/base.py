"""Base classes for expression implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from fenic.core._logical_plan import LogicalPlan
    from fenic.core._logical_plan.signatures import SignatureValidator
    from fenic.core.types.datatypes import DataType

from fenic.core._interfaces.session_state import BaseSessionState
from fenic.core.types import ColumnField


class Operator(Enum):
    """Enumeration of supported operators for expressions."""

    EQ = "="
    NOT_EQ = "!="
    LT = "<"
    LTEQ = "<="
    GT = ">"
    GTEQ = ">="
    PLUS = "+"
    MINUS = "-"
    MULTIPLY = "*"
    DIVIDE = "/"
    AND = "AND"
    OR = "OR"


class LogicalExpr(ABC):
    """Abstract base class for logical expressions."""

    @abstractmethod
    def __str__(self):
        """String representation of the expression."""
        pass

    @abstractmethod
    def to_column_field(self, plan: LogicalPlan, session_state: BaseSessionState) -> ColumnField:
        """Returns the schema field for the expression within the given plan."""
        pass

    @abstractmethod
    def children(self) -> List[LogicalExpr]:
        """Returns the children of the expression. Returns an empty list if the expression has no children."""
        pass

    def __eq__(self, other: LogicalExpr) -> bool:
        if not isinstance(other, LogicalExpr):
            return False
        if type(self) is not type(other):
            return False
        if not self._eq_specific(other):
            return False
        self_children = self.children()
        other_children = other.children()
        if len(self_children) != len(other_children):
            return False
        return all(
            child1 == child2
            for child1, child2 in zip(self_children, other_children, strict=True)
        )

    @abstractmethod
    def _eq_specific(self, other: LogicalExpr) -> bool:
        """Returns True if the expression has equal non expression attributes to the other expression.

        Args:
            other: The other expression to compare to.

        Returns:
            bool: True if the expression has equal non expression attributes to the other expression, False otherwise.
        """
        pass

class AggregateExpr(LogicalExpr):
    """Marker class for aggregate expressions used by optimizer and validation."""
    pass


class SemanticExpr(LogicalExpr):
    """Marker class for semantic expressions that use LLM models."""

    @abstractmethod
    def _validate_completion_parameters(self, session_state: BaseSessionState):
        """Common validation for semantic functions."""
        pass


class ValidatedSignature:
    """Mixin for expressions with simple signature validation.

    This mixin provides standard to_column_field() implementation
    for expressions that use SignatureValidator for type validation without
    dynamic return type inference.

    Expressions using this mixin must implement the validator property and children() method.
    """

    function_name: str  # Type hint to indicate this must be provided by subclass

    @property
    @abstractmethod
    def validator(self) -> SignatureValidator:
        """Must be implemented by subclass to provide validator instance.

        Returns:
            SignatureValidator: The validator instance for this expression
        """
        pass

    @abstractmethod
    def children(self) -> List[LogicalExpr]:
        pass

    def to_column_field(self, plan: LogicalPlan, session_state: BaseSessionState) -> ColumnField:
        """Default implementation using validator property."""
        return_type = self.validator.validate_and_infer_type(self.children(), plan, session_state)
        return ColumnField(name=str(self), data_type=return_type)

    def __str__(self) -> str:
        """Default string representation for function expressions."""
        args_str = ", ".join(str(arg) for arg in self.children())
        return f"{self.function_name}({args_str})"


class ValidatedDynamicSignature:
    """Mixin for expressions requiring dynamic return type inference.

    This mixin provides standard to_column_field() implementation
    for expressions that use SignatureValidator with custom return type logic.

    Expressions using this mixin must implement the validator property,
    children() method, and _infer_dynamic_return_type method.
    """

    function_name: str  # Type hint to indicate this must be provided by subclass

    @property
    @abstractmethod
    def validator(self) -> SignatureValidator:
        """Must be implemented by subclass to provide validator instance.

        Returns:
            SignatureValidator: The validator instance for this expression
        """
        pass

    @abstractmethod
    def children(self) -> List[LogicalExpr]:
        pass

    @abstractmethod
    def _infer_dynamic_return_type(self, arg_types: List[DataType], plan: LogicalPlan, session_state: BaseSessionState) -> DataType:
        """Must be implemented by subclass for dynamic return type inference.

        Args:
            arg_types: List of argument data types after validation
            plan: LogicalPlan for schema context
            session_state: BaseSessionState for session context

        Returns:
            DataType: The dynamically inferred return type
        """
        pass

    def to_column_field(self, plan: LogicalPlan, session_state: BaseSessionState) -> ColumnField:
        """Default implementation using validator property with dynamic return type."""
        return_type = self.validator.validate_and_infer_type(
            self.children(), plan, session_state, self._infer_dynamic_return_type
        )
        return ColumnField(name=str(self), data_type=return_type)

    def __str__(self) -> str:
        """Default string representation for function expressions."""
        args_str = ", ".join(str(arg) for arg in self.children())
        return f"{self.function_name}({args_str})"


class BinaryExpr(LogicalExpr):
    """Base class for binary expressions (comparison, boolean, arithmetic)."""

    def __init__(self, left: LogicalExpr, right: LogicalExpr, op: Operator):
        self.left = left
        self.right = right
        self.op = op

    def __str__(self):
        return f"({self.left} {self.op.value} {self.right})"

    def children(self) -> List[LogicalExpr]:
        return [self.left, self.right]

    def _eq_specific(self, other: BinaryExpr) -> bool:
        return self.op == other.op

class UnparameterizedExpr:
    """Mixin for expressions that are not parameterized that implements _eq_specific."""
    def _eq_specific(self, other: UnparameterizedExpr) -> bool:
        return True