from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from fenic._constants import PRETTY_PRINT_INDENT
from fenic.core._interfaces.session_state import BaseSessionState
from fenic.core.error import InternalError, PlanError
from fenic.core.types.schema import Schema


@dataclass
class CacheInfo:
    duckdb_table_name: Optional[str] = None

class LogicalPlan(ABC):
    def __init__(self,
            session_state: Optional[BaseSessionState] = None,
            schema: Optional[Schema] = None,
            ):
        self.cache_info = None
        if schema is None and session_state is None:
            raise InternalError("Either a session state or a schema must be provided")

        if schema:
            self._schema = schema
        else:
            self._schema = self._build_schema(session_state)
        self._validate_schema()

    def _validate_schema(self):
        if self._schema is None:
            raise InternalError("schema is required")

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
    def children(self) -> List[LogicalPlan]:
        """Returns the child nodes of this logical plan operator.

        Returns:
            List[LogicalPlan]: A list of child logical plan nodes. For leaf nodes
                like Source, this will be an empty list.
        """
        pass

    @abstractmethod
    def _build_schema(self, session_state: BaseSessionState) -> Schema:
        """Constructs the output schema for this logical plan operator.

        This method is called during initialization to determine the schema of the
        data that will be produced by this operator when executed.

        session_state: The session state to use for the schema construction.

        Returns:
            Schema: The schema describing the structure and types of the output columns
                that this operator will produce.

        Raises:
            ValueError: If the operation would produce an invalid schema, for example
                calling a semantic map on a non-string column.
        """
        pass

    @abstractmethod
    def _repr(self) -> str:
        """Return the string representation for this logical plan."""
        pass

    def _repr_with_indent(self, _level: int) -> str:
        """Default: just call __repr(). Override this method to build an indentation aware string plan representation."""
        return self._repr()

    def __str__(self) -> str:
        """Recursively pretty-print with indentation."""

        def pretty_print(plan: LogicalPlan, level: int) -> str:
            indent = PRETTY_PRINT_INDENT * level
            cache_info = " (cached=true)" if plan.cache_info is not None else ""
            result = f"{indent}{plan._repr_with_indent(level)}{cache_info}\n"
            for child in plan.children():
                result += pretty_print(child, level + 1)
            return result

        return pretty_print(self, 0)

    def set_cache_info(self, cache_info: CacheInfo):
        """Set the cache metadata for this plan."""
        self.cache_info = cache_info

    def schema(self) -> Schema:
        return self._schema

    @abstractmethod
    def with_children(self, children: List[LogicalPlan], session_state: Optional[BaseSessionState] = None) -> LogicalPlan:
        """Creates and returns a new instance of the logical plan with the given children.

        This method acts as a factory method that preserves the current node's properties
        while replacing its child nodes.

        Args:
            children: The new child nodes to use in the created logical plan
            session_state: The session state to use for the created logical plan

        Returns:
            A new logical plan instance of the same type with updated children
        """
        pass
