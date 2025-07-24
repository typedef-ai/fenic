from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from fenic._constants import PRETTY_PRINT_INDENT
from fenic.core._interfaces.session_state import BaseSessionState
from fenic.core.error import InternalError, PlanError
from fenic.core.types.schema import Schema

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

@dataclass
class CacheInfo:
    duckdb_table_name: Optional[str] = None

class LogicalPlanNode(ABC):

    def __init__(self):
        self._input: Optional[LogicalPlanNode] = None
        self._schema: Optional[Schema] = None
        self.cache_info: Optional[CacheInfo] = None

    def set_input(self, input: LogicalPlanNode):
        """Sets the default input node, this can be overriden by the subclass if not needed."""
        self._input = input

    @abstractmethod
    def children(self) -> List[LogicalPlanNode]:
        pass

    @abstractmethod
    def _build_schema(self, session_state: BaseSessionState) -> Schema:
        pass

    def _build_schema_with_validation(self, session_state: BaseSessionState):
        self._schema = self._build_schema(session_state)
        self.validate_schema()

    @abstractmethod
    def _repr(self) -> str:
        pass

    def _repr_with_indent(self, _level: int) -> str:
        """Default: just call __repr(). Override this method to build an indentation aware string plan representation."""
        return self._repr()

    def __str__(self) -> str:
        """Recursively pretty-print with indentation."""

        def pretty_print(node: LogicalPlanNode, level: int) -> str:
            indent = PRETTY_PRINT_INDENT * level
            cache_info = " (cached=true)" if node.cache_info is not None else ""
            result = f"{indent}{node._repr_with_indent(level)}{cache_info}\n"
            for child in node.children():
                result += pretty_print(child, level + 1)
            return result

        return pretty_print(self, 0)
    
    def schema(self) -> Schema:
        return self._schema
    
    def validate_schema(self) -> None:
        """Validate the schema for the node."""
        if not self._schema:
            raise PlanError("Schema is not set, must call _build_schema before validation")
        
        column_names = [field.name for field in self._schema.column_fields]
        seen = set()
        duplicates = {name for name in column_names if name in seen or seen.add(name)}
        if duplicates:
            example_duplicate = next(iter(duplicates))
            duplicate_list = ", ".join(f"'{name}'" for name in duplicates)
            raise PlanError(
                f"Duplicate column names found: {duplicate_list}. "
                "Column names must be unique. "
                f"Use aliases to rename columns, e.g., col('{example_duplicate}').alias('{example_duplicate}_2')."
            )
    
    @abstractmethod
    def with_children(self, children: List[LogicalPlanNode]) -> LogicalPlanNode:
        """Creates and returns a new instance of the logical node plan with the given children.

        This method acts as a factory method that preserves the current node's properties
        while replacing its child nodes.

        Args:
            children: The new child nodes to use in the created logical plan node

        Returns:
            A new logical plan node instance of the same type with updated children
        """
        pass

    def set_cache_info(self, cache_info: CacheInfo):
        self.cache_info = cache_info

    def set_schema(self, schema: Schema):
        self._schema = schema

    @classmethod
    def copy(cls, node: LogicalPlanNode, children: List[LogicalPlanNode]) -> LogicalPlanNode:
        """Creates a copy of the node."""
        if cls is not type(node):
            raise InternalError(f"Expected {cls}, got {type(node)}")
        new_node = cls._create_new_node(node=node, children=children)
        new_node.set_schema(node.schema())
        new_node.set_cache_info(node.cache_info)
        return new_node

    @classmethod
    @abstractmethod
    def _create_new_node(cls, node: LogicalPlanNode, children: List[LogicalPlanNode]) -> LogicalPlanNode:
        pass
