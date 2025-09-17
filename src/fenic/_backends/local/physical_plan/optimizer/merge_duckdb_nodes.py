"""Optimization rule for merging adjacent DuckDB operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fenic._backends.local.physical_plan.sink import DuckDBTableSinkExec, FileSinkExec
from fenic._backends.local.physical_plan.source import (
    DuckDBTableSourceExec,
    FileSourceExec,
)

if TYPE_CHECKING:
    from fenic._backends.local.session_state import LocalSessionState

from fenic._backends.local.physical_plan.base import PhysicalPlan
from fenic._backends.local.physical_plan.optimizer.base import (
    PhysicalPlanOptimizationResult,
    PhysicalPlanRule,
)
from fenic._backends.local.physical_plan.transform import MergedDuckDBExec, SQLExec


class MergeDuckDBNodesRule(PhysicalPlanRule):
    """Rule that merges adjacent DuckDB operations into single execution units.

    The resulting `MergedDuckDBExec` nodes behave like a boundary between the
    outer execution engine and DuckDB. During execution:

    - The *outer engine* only sees the flattened `children`, which connect this
      merged subtree to the rest of the plan.
    - Inside the merged node, the `merge_root` subtree is traversed bottom-up
      to register SQL views for every original operator. This allows DuckDB's
      planner to perform its own fusion optimizations across what were
      previously separate plan nodes.
    """

    def apply(
        self, plan: PhysicalPlan, session_state: LocalSessionState
    ) -> PhysicalPlanOptimizationResult:
        """Apply DuckDB merging to the entire plan using bottom-up traversal."""
        optimized_plan = self._optimize(plan, session_state, original=plan)
        optimized = optimized_plan is not plan
        return PhysicalPlanOptimizationResult(plan=optimized_plan, optimized=optimized)

    def _optimize(
        self, node: PhysicalPlan, session_state: LocalSessionState, original: PhysicalPlan
    ) -> PhysicalPlan:
        """Recursively optimize the plan tree bottom-up."""
        # Optimize children first
        optimized_children = [
            self._optimize(child, session_state, original=child) for child in node.children
        ]

        # If children changed, make a new node with updated children
        if optimized_children != node.children:
            current_node = node.with_children(optimized_children)
        else:
            current_node = node

        # Try merging using the *original* node as merge_root
        return self._try_merge_with_children(current_node, original, session_state)

    def _try_merge_with_children(
        self, node: PhysicalPlan, original: PhysicalPlan, session_state: LocalSessionState
    ) -> PhysicalPlan:
        """Try to merge the current node with its DuckDB children."""
        if (not _is_duckdb_node(node) or not node.children or node.cache_info):
            return node

        fusable_children = [
            (i, child)
            for i, child in enumerate(node.children)
            if not child.cache_info and _is_duckdb_node(child)
        ]
        if not fusable_children:
            return node

        # Flatten children
        new_children = []
        fusable_indices = {i for i, _ in fusable_children}
        for i, child in enumerate(node.children):
            if i in fusable_indices:
                new_children.extend(child.children)
            else:
                new_children.append(child)

        # Create merged node.
        # - `merge_root` preserves the *entire original subtree*. This is what
        #   execution will traverse bottom-up to build SQL views. By keeping
        #   the full tree intact, DuckDB's own planner can see all operators
        #   and apply its fusion rules.
        # - `children` are flattened, so from the perspective of the outer
        #   query engine, this subtree is a single execution unit with direct
        #   connections to only the non-DuckDB children.
        return MergedDuckDBExec(
            merge_root=original,       # original subtree (traversed at runtime
            children=new_children,     # flattened edges for outer engine
            cache_info=node.cache_info,
            session_state=session_state,
        )

def _is_duckdb_node(node: PhysicalPlan) -> bool:
    return isinstance(
        node,
        (
            DuckDBTableSinkExec,
            DuckDBTableSourceExec,
            MergedDuckDBExec,
            SQLExec,
            FileSinkExec,
            FileSourceExec,
        ),
    )
