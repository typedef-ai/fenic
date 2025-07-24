from typing import List, Literal

from fenic.core._interfaces.session_state import BaseSessionState
from fenic.core._logical_plan.plans.node import LogicalPlanNode
from fenic.core.error import InternalError
from fenic.core.types import Schema


class FileSink(LogicalPlanNode):
    """Logical plan node that represents a file writing operation."""

    def __init__(
        self,
        child: LogicalPlanNode,
        sink_type: Literal["csv", "parquet"],
        path: str,
        mode: Literal["error", "overwrite", "ignore"] = "error",
    ):
        """Initialize a file sink node.

        Args:
            child: The logical plan that produces data to be written
            sink_type: The type of file sink (CSV, Parquet)
            path: File path to write to
            mode: Write mode - "error" or "overwrite". Default: "overwrite"
                 - error: Raises an error if file exists
                 - overwrite: Overwrites the file if it exists
                 - ignore: Silently ignores operation if file exists
        """
        super().__init__()
        self.child = child
        self.sink_type = sink_type
        self.path = path
        self.mode = mode

    def children(self) -> List[LogicalPlanNode]:
        """Returns the child node of this sink operator."""
        return [self.child]

    def _build_schema(self, session_state: BaseSessionState) -> Schema:
        """The schema of a sink node is the same as its child's schema."""
        return self.child.schema()

    def _repr(self) -> str:
        """Return the string representation for this file sink plan."""
        return (
            f"FileSink(type={self.sink_type}, "
            f"path='{self.path}', "
            f"mode='{self.mode}')"
        )

    def with_children(self, children: List[LogicalPlanNode]) -> LogicalPlanNode:
        """Create a new file sink with the same properties but different children.

        Args:
            children: The new children for this node. Must contain exactly one child.

        Returns:
            A new FileSink instance with the updated child.

        Raises:
            ValueError: If children doesn't contain exactly one LogicalPlan.
        """
        if len(children) != 1:
            raise InternalError(
                f"FileSink expects exactly one child but got {len(children)}"
            )
        return self.copy(self, children)

    @classmethod
    def _create_new_node(cls, node: LogicalPlanNode, children: List[LogicalPlanNode]) -> LogicalPlanNode:
        return FileSink(children[0], node.sink_type, node.path, node.mode)

class TableSink(LogicalPlanNode):
    """Logical plan node that represents a table writing operation."""
    def __init__(
        self,
        child: LogicalPlanNode,
        table_name: str,
        mode: Literal["error", "append", "overwrite", "ignore"] = "error",
    ):
        """Initialize a table sink node.

        Args:
            child: The logical plan node that produces data to be written
            table_name: Name of the table to write to
            mode: Write mode. Default: "error"
                 - error: Raises an error if table exists
                 - append: Appends data to table if it exists
                 - overwrite: Overwrites existing table
                 - ignore: Silently ignores operation if table exists
        """
        super().__init__()
        self.child = child
        self.table_name = table_name
        self.mode = mode

    def children(self) -> List[LogicalPlanNode]:
        """Returns the child node of this sink operator."""
        return [self.child]

    def _build_schema(self, session_state: BaseSessionState) -> Schema:
        """The schema of a sink node is the same as its child's schema."""
        return self.child.schema()

    def _repr(self) -> str:
        """Return the string representation for this table sink plan."""
        return f"TableSink(table_name='{self.table_name}', mode='{self.mode}')"

    def with_children(self, children: List[LogicalPlanNode]) -> LogicalPlanNode:
        """Create a new table sink with the same properties but different children.

        Args:
            children: The new children for this node. Must contain exactly one child.

        Returns:
            A new TableSink instance with the updated child.

        Raises:
            ValueError: If children doesn't contain exactly one LogicalPlan.
        """
        if len(children) != 1:
            raise InternalError(
                f"TableSink expects exactly one child but got {len(children)}"
            )
        return self.copy(self, children)

    @classmethod
    def _create_new_node(cls, node: LogicalPlanNode, children: List[LogicalPlanNode]) -> LogicalPlanNode:
        return TableSink(
            child=children[0],
            table_name=node.table_name,
            mode=node.mode,
        )
