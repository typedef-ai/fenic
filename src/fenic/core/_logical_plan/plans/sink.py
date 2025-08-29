from __future__ import annotations

from typing import TYPE_CHECKING, List, Literal, Optional

from fenic.core._interfaces.session_state import BaseSessionState
from fenic.core._logical_plan.plans.base import LogicalPlan
from fenic.core.error import InternalError
from fenic.core.types import Schema

if TYPE_CHECKING:
    from fenic.core._logical_plan.expressions.base import LogicalExpr


class FileSink(LogicalPlan):
    """Logical plan node that represents a file writing operation."""

    def __init__(
            self,
            child: LogicalPlan,
            sink_type: Literal["csv", "parquet"],
            path: str,
            mode: Literal["error", "overwrite", "ignore"] = "error",
            session_state: Optional[BaseSessionState] = None,
            schema: Optional[Schema] = None):
        """Initialize a file sink node.

        Args:
            child: The logical plan that produces data to be written
            sink_type: The type of file sink (CSV, Parquet)
            path: File path to write to
            mode: Write mode - "error" or "overwrite". Default: "overwrite"
                 - error: Raises an error if file exists
                 - overwrite: Overwrites the file if it exists
                 - ignore: Silently ignores operation if file exists
            session_state: The session state to use for the new node.
            schema: The schema to use for the new node.
        """
        self.child = child
        self.sink_type = sink_type
        self.path = path
        self.mode = mode
        super().__init__(session_state, schema)

    @classmethod
    def from_session_state(
        cls,
        child: LogicalPlan,
        sink_type: Literal["csv", "parquet"],
        path: str,
        mode: Literal["error", "overwrite", "ignore"] = "error",
        session_state: Optional[BaseSessionState] = None,
    ) -> FileSink:
        return FileSink(child, sink_type, path, mode, session_state, None)

    @classmethod
    def from_schema(
        cls,
        child: LogicalPlan,
        sink_type: Literal["csv", "parquet"],
        path: str,
        mode: Literal["error", "overwrite", "ignore"] = "error",
        schema: Optional[Schema] = None,
    ) -> FileSink:
        return FileSink(child, sink_type, path, mode, None, schema)

    def children(self) -> List[LogicalPlan]:
        """Returns the child node of this sink operator."""
        return [self.child]

    def exprs(self) -> List[LogicalExpr]:
        return []

    def _build_schema(self, session_state: BaseSessionState) -> Schema:
        """The schema of a sink node is the same as its child's schema."""
        return self.child.schema()

    def _repr(self) -> str:
        """Return the string representation for this file sink plan."""
        return (
            f"FileSink(type={self.sink_type}, path='{self.path}', mode='{self.mode}')"
        )

    def with_children(
        self,
        children: List[LogicalPlan],
        session_state: Optional[BaseSessionState] = None,
    ) -> LogicalPlan:
        """Create a new file sink with the same properties but different children.

        Args:
            children: The new children for this node. Must contain exactly one child.
            session_state: The session state to use for the new node.

        Returns:
            A new FileSink instance with the updated child.

        Raises:
            ValueError: If children doesn't contain exactly one LogicalPlan.
        """
        if len(children) != 1:
            raise InternalError(
                f"FileSink expects exactly one child but got {len(children)}"
            )
        return FileSink.from_session_state(
            child=children[0],
            sink_type=self.sink_type,
            path=self.path,
            mode=self.mode,
            session_state=session_state,
        )

    def _eq_specific(self, other: FileSink) -> bool:
        return (
            self.sink_type == other.sink_type
            and self.path == other.path
            and self.mode == other.mode
        )

class TableSink(LogicalPlan):
    """Logical plan node that represents a table writing operation."""

    def __init__(
            self,
            child: LogicalPlan,
            table_name: str,
            mode: Literal["error", "append", "overwrite", "ignore"] = "error",
            session_state: Optional[BaseSessionState] = None,
            schema: Optional[Schema] = None,
        ):
        """Initialize a table sink node.

        Args:
            child: The logical plan that produces data to be written
            table_name: Name of the table to write to
            mode: Write mode. Default: "error"
                 - error: Raises an error if table exists
                 - append: Appends data to table if it exists
                 - overwrite: Overwrites existing table
                 - ignore: Silently ignores operation if table exists
            session_state: The session state to use for the new node.
            schema: The schema to use for the new node.
        """
        self.child = child
        self.table_name = table_name
        self.mode = mode
        super().__init__(session_state, schema)

    @classmethod
    def from_session_state(
        cls,
        child: LogicalPlan,
        table_name: str,
        mode: Literal["error", "append", "overwrite", "ignore"] = "error",
        session_state: Optional[BaseSessionState] = None,
    ) -> TableSink:
        return TableSink(child, table_name, mode, session_state, None)

    @classmethod
    def from_schema(
        cls,
        child: LogicalPlan,
        table_name: str,
        mode: Literal["error", "append", "overwrite", "ignore"] = "error",
        schema: Optional[Schema] = None,
    ) -> TableSink:
        return TableSink(child, table_name, mode, None, schema)

    def children(self) -> List[LogicalPlan]:
        """Returns the child node of this sink operator."""
        return [self.child]

    def exprs(self) -> List[LogicalExpr]:
        return []

    def _build_schema(self, session_state: BaseSessionState) -> Schema:
        """The schema of a sink node is the same as its child's schema."""
        return self.child.schema()

    def _repr(self) -> str:
        """Return the string representation for this table sink plan."""
        return f"TableSink(table_name='{self.table_name}', mode='{self.mode}')"

    def with_children(
        self,
        children: List[LogicalPlan],
        session_state: Optional[BaseSessionState] = None,
    ) -> LogicalPlan:
        """Create a new table sink with the same properties but different children.

        Args:
            children: The new children for this node. Must contain exactly one child.
            session_state: The session state to use for the new node.

        Returns:
            A new TableSink instance with the updated child.

        Raises:
            ValueError: If children doesn't contain exactly one LogicalPlan.
        """
        if len(children) != 1:
            raise InternalError(
                f"TableSink expects exactly one child but got {len(children)}"
            )
        return TableSink.from_session_state(
            child=children[0],
            table_name=self.table_name,
            mode=self.mode,
            session_state=session_state,
        )

    def _eq_specific(self, other: TableSink) -> bool:
        return (
            self.table_name == other.table_name
            and self.mode == other.mode
        )
