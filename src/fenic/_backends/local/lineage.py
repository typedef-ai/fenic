from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

import polars as pl

if TYPE_CHECKING:
    from fenic._backends.local.session_state import LocalSessionState

from fenic.api.lineage import BaseLineage
from fenic.core.types.enums import BranchSide

"""
Data types for representing query execution lineage.
"""


@dataclass
class ChildEdge:
    """Represents a connection between an operator and its child operator.

    This class maintains the relationship between parent and child operators
    in the query execution lineage, including the mapping table that connects them.
    """

    mapping_table: str
    child_operator: OperatorLineage


@dataclass
class OperatorLineage:
    """Represents a single operator in the query execution lineage.

    This class tracks an operator's relationships with its parent and children,
    as well as the table where its results are materialized.
    """

    operator_name: str
    children: List[ChildEdge]
    materialize_table: str
    parent: Optional[OperatorLineage] = None

    def __str__(self, indent_level: int = 0) -> str:
        indent = "  " * indent_level
        result = f"{indent}{self.operator_name}"
        for child_link in self.children:
            result += f"\n{child_link.child_operator.__str__(indent_level + 1)}"
        return result


class LineageGraph:
    """Represents the complete lineage of a query execution.

    This class maintains the full tree structure of operators involved in
    executing a query, from source data to final results.
    """

    def __init__(self, root_node: OperatorLineage, leaf_nodes: List[OperatorLineage]):
        self.root_node = root_node
        self.leaf_nodes = leaf_nodes

    def get_leaf_nodes(self) -> Dict[str, OperatorLineage]:
        return {node.operator_name: node for node in self.leaf_nodes}

    def get_root_node(self) -> OperatorLineage:
        return self.root_node


class LocalLineage(BaseLineage):
    """A class for traversing a lineage graph in a local session using intermediate lineage tables backed by DuckDB."""

    def __init__(self, lineage_graph: LineageGraph, session_state: LocalSessionState):
        self.lineage_graph = lineage_graph
        self.session_state = session_state
        self.curr_operator = self.lineage_graph.get_root_node()

    def get_source_names(self) -> List[str]:
        """Get the names of all sources in the query plan."""
        return list(self.lineage_graph.get_leaf_nodes().keys())

    def stringify_graph(self) -> str:
        """Print the operator tree of the query."""
        return str(self.lineage_graph.get_root_node())

    def start_from_source(self, source_name: str) -> None:
        """Set the current position to a specific source in the query plan."""
        self.curr_operator = self.lineage_graph.get_leaf_nodes()[source_name]

    def forwards(self, row_ids: List[str]) -> pl.DataFrame:
        """Trace rows forward to see how they are transformed by the next operation."""
        if self.curr_operator.parent is None:
            raise ValueError("Cannot step forward from the root operator.")
        if not isinstance(row_ids, list) or not all(
            isinstance(id, str) for id in row_ids
        ):
            raise ValueError("The row_ids must be a list of strings.")
        parent_operator = self.curr_operator.parent
        parent_backwards_df = self.session_state.intermediate_df_client.read_df(
            parent_operator.children[0].mapping_table
        )
        parent_backwards_df = parent_backwards_df.filter(
            pl.col("_backwards_uuid").is_in(row_ids)
        ).drop("_backwards_uuid")

        forwards_uuid_list = (
            parent_backwards_df.unique(subset="_uuid").to_series(0).to_list()
        )
        parent_df = self.session_state.intermediate_df_client.read_df(
            parent_operator.materialize_table
        )
        result = parent_df.filter(pl.col("_uuid").is_in(forwards_uuid_list))
        self.curr_operator = parent_operator
        return result

    def backwards(
        self, ids: List[str], branch_side: Optional[BranchSide] = None
    ) -> pl.DataFrame:
        """Trace rows backwards to see which input rows produced them."""
        self._validate_backwards_trace_inputs(ids, branch_side)
        if not branch_side or branch_side == "left":
            res = self._backwards(ids, self.curr_operator.children[0])
            return res
        else:
            res = self._backwards(ids, self.curr_operator.children[1])
            return res

    def _validate_backwards_trace_inputs(
        self, ids: List[str], branch_side: Optional[str]
    ) -> None:
        """Validate inputs for backwards tracing."""
        # Validate ids format
        if not isinstance(ids, list):
            raise ValueError("ids must be a list")
        if not all(isinstance(id, str) for id in ids):
            raise ValueError("all ids must be strings")

        # Validate operator has children
        if self.curr_operator.children is None:
            raise ValueError("current operator has no children")

        # Validate branch_side specification
        has_multiple_children = len(self.curr_operator.children) > 1
        if has_multiple_children:
            if not branch_side:
                raise ValueError(
                    "branch_side required for operators with multiple children"
                )
            if branch_side not in ["left", "right"]:
                raise ValueError("branch_side must be 'left' or 'right'")
        else:
            if branch_side:
                raise ValueError(
                    "branch_side not allowed for operators with single child"
                )

    def _backwards(self, ids: List[str], child_link: ChildEdge) -> pl.DataFrame:
        """Trace backwards through a single child operator."""
        self.curr_operator = child_link.child_operator

        backwards_df = self.session_state.intermediate_df_client.read_df(
            child_link.mapping_table
        )
        backwards_df = backwards_df.filter(pl.col("_uuid").is_in(ids)).drop("_uuid")
        backwards_uuid_list = (
            backwards_df.unique(subset="_backwards_uuid").to_series(0).to_list()
        )
        child_df = self.session_state.intermediate_df_client.read_df(
            self.curr_operator.materialize_table
        )
        return child_df.filter(pl.col("_uuid").is_in(backwards_uuid_list))

    def get_result_df(self) -> pl.DataFrame:
        """Get the result of the query as a Polars DataFrame."""
        return self.session_state.intermediate_df_client.read_df(
            self.curr_operator.materialize_table
        )

    def get_source_df(self, source_name: str) -> pl.DataFrame:
        """Get a query source by name as a Polars DataFrame."""
        return self.session_state.intermediate_df_client.read_df(
            f"materialize_{source_name}"
        )
