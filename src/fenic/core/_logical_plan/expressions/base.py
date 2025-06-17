"""Base classes for expression implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from fenic.core._logical_plan import LogicalPlan

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
    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
        """Returns the schema field for the expression within the given plan."""
        pass

    @abstractmethod
    def children(self) -> List[LogicalExpr]:
        """Returns the children of the expression. Returns an empty list if the expression has no children."""
        pass


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
