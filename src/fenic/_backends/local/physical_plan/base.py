from __future__ import annotations

import logging
import time
import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import polars as pl

from fenic._backends.local.lineage import ChildEdge, LineageGraph, OperatorLineage
from fenic.core._logical_plan.plans import CacheInfo
from fenic.core.metrics import (
    LMMetrics,
    OperatorMetrics,
    PhysicalPlanRepr,
    QueryMetrics,
    RMMetrics,
)

if TYPE_CHECKING:
    from fenic._backends.local.session_state import LocalSessionState

logger = logging.getLogger(__name__)

# TODO(rohitrastogi): Consider using a visitor pattern to traverse logical and physical plans. This can help
# with separating the traversal logic from the node processing logic.
class PhysicalPlan(ABC):
    def __init__(
        self,
        children: List[PhysicalPlan],
        cache_info: Optional[CacheInfo],
        session_state: LocalSessionState,
    ):
        self.children = children if children is not None else []
        self.cache_info = cache_info
        self.session_state = session_state

        short_uuid = str(uuid.uuid4().hex)[:8]
        self.operator_id = f"{self.__class__.__name__}_{short_uuid}"

    def execute(self, execution_id: str) -> Tuple[pl.DataFrame, QueryMetrics]:
        """Execute the physical plan and return the result DataFrame along with execution metrics.

        This method handles:
        1. Checking and retrieving from cache if available
        2. Recursively executing child operators
        3. Processing the current operator's logic
        4. Collecting and aggregating metrics from all operations

        Returns:
            Tuple[pl.DataFrame, QueryMetrics]: A tuple containing:
                - The resulting Polars DataFrame
                - QueryMetrics object with execution statistics and LM usage metrics
        """
        # Step 1: Initialize query metrics state.
        curr_operator_metrics = OperatorMetrics(operator_id=self.operator_id)
        plan_repr = PhysicalPlanRepr(operator_id=self.operator_id)
        all_operator_metrics: Dict[str, OperatorMetrics] = {
            self.operator_id: curr_operator_metrics
        }
        total_lm_metrics = LMMetrics()
        total_rm_metrics = RMMetrics()

        # Step 2: Execute child operators and collect their metrics
        child_dfs = []
        child_execution_time = 0
        for child in self.children:
            child_df, child_metrics = child.execute(execution_id)
            child_dfs.append(child_df)
            plan_repr.children.append(child_metrics._plan_repr)
            all_operator_metrics.update(child_metrics._operator_metrics)
            total_lm_metrics += child_metrics.total_lm_metrics
            total_rm_metrics += child_metrics.total_rm_metrics
            child_execution_time += child_metrics.execution_time_ms

        # Step 3: Execute the current operator - measure only the time spent in this operator
        operator_start_time = time.time()
        result_df = self.execute_node(child_dfs)
        operator_execution_time = (time.time() - operator_start_time) * 1000

        curr_operator_metrics.num_output_rows = result_df.height
        curr_operator_metrics.execution_time_ms = operator_execution_time

        lm_metrics, rm_metrics = self.session_state.get_model_metrics()
        curr_operator_metrics.lm_metrics = lm_metrics
        curr_operator_metrics.rm_metrics = rm_metrics
        self.session_state.reset_model_metrics()

        # Step 4: Write to cache if applicable.
        if self.cache_info and not self.session_state.intermediate_df_client.is_df_cached(self.cache_info.cache_key):
            self.session_state.intermediate_df_client.write_df(
                result_df, self.cache_info.cache_key
            )

        # Calculate total execution time for the query metrics
        total_execution_time = child_execution_time + operator_execution_time

        # Create query metrics with session information
        query_metrics = QueryMetrics(
            execution_id=execution_id,
            session_id=self.session_state.session_id,
            execution_time_ms=total_execution_time,
            num_output_rows=curr_operator_metrics.num_output_rows,
            total_lm_metrics=total_lm_metrics + curr_operator_metrics.lm_metrics,
            total_rm_metrics=total_rm_metrics + curr_operator_metrics.rm_metrics,
            _plan_repr=plan_repr,
            _operator_metrics=all_operator_metrics,
        )

        return result_df, query_metrics

    @abstractmethod
    def execute_node(self, child_dfs: List[pl.DataFrame]) -> pl.DataFrame:
        """Execute the specific operation for this physical plan node.

        This method contains the core execution logic for each operator in the physical plan.
        Each subclass must implement this method to perform its specific operation
        (e.g., filter, join, project, etc.) on the input dataframes.

        Args:
            child_dfs: A list of Polars DataFrames from child operators.
                       Typically contains one DataFrame for unary operators like Filter,
                       or two DataFrames for binary operators like Join.

        Returns:
            pl.DataFrame: The resulting DataFrame after applying this operator's logic.
        """
        pass

    @abstractmethod
    def with_children(self, children: List[PhysicalPlan]) -> PhysicalPlan:
        """Construct a new physical plan, replacing the current children with the new children."""
        pass

    def build_lineage(self) -> LineageGraph:
        """Return the lineage of the physical plan."""
        leaf_nodes = []
        root_node, _ = self.build_node_lineage(leaf_nodes)
        return LineageGraph(root_node, leaf_nodes)

    @abstractmethod
    def build_node_lineage(
        self,
        leaf_nodes: List[OperatorLineage],
    ) -> Tuple[OperatorLineage, pl.DataFrame]:
        """Build lineage for this operator and its children.

        Each operator should:
        1. Call _build_lineage on its children first
        2. Create its own materialized dataframe
        3. Create backwards mapping dataframes
        4. Create and return its own OperatorLineage

        Args:
            leaf_nodes: List to collect leaf nodes

        Returns:
            Tuple of (OperatorLineage for this operator, materialized DataFrame)
        """
        pass

    def _build_source_operator_lineage(
        self,
        materialize_df: pl.DataFrame,
    ) -> OperatorLineage:
        materialize_table_name = f"materialize_{self.operator_id}"
        self.session_state.intermediate_df_client.write_df(
            materialize_df, materialize_table_name
        )

        source_operator = OperatorLineage(
            operator_name=f"{self.operator_id}",
            children=[],
            materialize_table=materialize_table_name,
        )
        return source_operator

    def _build_unary_operator_lineage(
        self,
        materialize_df: pl.DataFrame,
        child: Tuple[OperatorLineage, pl.DataFrame],
    ) -> OperatorLineage:
        materialize_table_name = f"materialize_{self.operator_id}"
        self.session_state.intermediate_df_client.write_df(
            materialize_df, materialize_table_name
        )
        child_operator, backwards_df = child

        backwards_table_name = f"backwards_{self.operator_id}"
        self.session_state.intermediate_df_client.write_df(
            backwards_df, backwards_table_name
        )

        operator = OperatorLineage(
            operator_name=f"{self.operator_id}",
            children=[
                ChildEdge(
                    mapping_table=backwards_table_name, child_operator=child_operator
                )
            ],
            materialize_table=materialize_table_name,
        )
        child_operator.parent = operator
        return operator

    def _build_binary_operator_lineage(
        self,
        materialize_df: pl.DataFrame,
        left_child: Tuple[OperatorLineage, pl.DataFrame],
        right_child: Tuple[OperatorLineage, pl.DataFrame],
    ) -> OperatorLineage:
        materialize_table_name = f"materialize_{self.operator_id}"
        self.session_state.intermediate_df_client.write_df(
            materialize_df, materialize_table_name
        )
        left_operator, left_backwards_df = left_child
        right_operator, right_backwards_df = right_child

        left_backwards_table_name = f"backwards_{self.operator_id}_left"
        right_backwards_table_name = f"backwards_{self.operator_id}_right"
        self.session_state.intermediate_df_client.write_df(
            left_backwards_df, left_backwards_table_name
        )
        self.session_state.intermediate_df_client.write_df(
            right_backwards_df, right_backwards_table_name
        )

        operator = OperatorLineage(
            operator_name=f"{self.operator_id}",
            children=[
                ChildEdge(
                    mapping_table=left_backwards_table_name,
                    child_operator=left_operator,
                ),
                ChildEdge(
                    mapping_table=right_backwards_table_name,
                    child_operator=right_operator,
                ),
            ],
            materialize_table=materialize_table_name,
        )
        left_operator.parent = operator
        right_operator.parent = operator
        return operator

    def _build_row_subset_lineage(
        self,
        leaf_nodes: List[OperatorLineage],
    ) -> Tuple[OperatorLineage, pl.DataFrame]:
        """Implementation of _build_lineage for operators that preserve row identity.

        This method is designed for operators where:
        1. There is a single child operator
        2. The operator returns all rows or a subset of input rows (never adds new rows)
        3. Examples include filtering, limits, and dropping duplicates

        Args:
            leaf_nodes: List to collect leaf nodes

        Returns:
            Tuple of (OperatorLineage for this operator, materialized DataFrame)
        """
        if len(self.children) != 1:
            raise ValueError(f"Unreachable: {self.__class__.__name__} expects 1 child")

        # Get lineage from child
        child_operator, child_df = self.children[0].build_node_lineage(leaf_nodes)

        # Apply the operator-specific transformation
        materialize_df = self.execute_node([child_df])

        # Create the trivial backwards mapping dataframe
        backwards_df = materialize_df.select(["_uuid"])
        backwards_df = backwards_df.with_columns(
            pl.col("_uuid").alias("_backwards_uuid")
        )

        # Build and return the operator lineage
        operator = self._build_unary_operator_lineage(
            materialize_df=materialize_df,
            child=(child_operator, backwards_df),
        )
        return operator, materialize_df


def _with_lineage_uuid(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.Series("_uuid", [str(uuid.uuid4().hex) for _ in range(df.height)])
    )
