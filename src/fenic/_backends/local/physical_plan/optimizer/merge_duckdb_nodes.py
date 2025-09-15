"""Optimization rule for merging adjacent DuckDB operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from fenic._backends.local.session_state import LocalSessionState

from fenic._backends.local.physical_plan import DuckDBTableSinkExec
from fenic._backends.local.physical_plan.transform import SQLExec
from fenic._backends.local.physical_plan.base import DuckDBNodeMixin, PhysicalPlan
from fenic._backends.local.physical_plan.optimizer.base import (
    PhysicalPlanOptimizationResult,
    PhysicalPlanRule,
)
from fenic._backends.local.physical_plan.transform import MergedDuckDBExec, SQLExec


class MergeDuckDBNodesRule(PhysicalPlanRule):
    """Rule that merges adjacent DuckDB operations into single execution units."""

    def apply(self, plan: PhysicalPlan, session_state: LocalSessionState) -> PhysicalPlanOptimizationResult:
        """Apply DuckDB merging to the entire plan using bottom-up traversal."""
        optimized_plan = self._optimize(plan, session_state)
        # Check if the plan was actually changed
        optimized = optimized_plan is not plan
        return PhysicalPlanOptimizationResult(plan=optimized_plan, optimized=optimized)

    def _optimize(self, node: PhysicalPlan, session_state: LocalSessionState) -> PhysicalPlan:
        """Recursively optimize the plan tree bottom-up."""
        # First, optimize all children (bottom-up traversal)
        optimized_children = []
        for child in node.children:
            optimized_child = self._optimize(child, session_state)
            optimized_children.append(optimized_child)

        # Update node with optimized children if they changed
        if optimized_children != node.children:
            current_node = node.with_children(optimized_children)
        else:
            current_node = node

        # Try to merge current node with its children
        return self._try_merge_with_children(current_node, session_state)

    def _try_merge_with_children(self, node: PhysicalPlan, session_state: LocalSessionState) -> PhysicalPlan:
        """Try to merge the current node with its DuckDB children."""
        # Early returns for non-mergeable cases
        if (not isinstance(node, DuckDBNodeMixin) or
            not node.children or
            node.cache_info):
            return node

        # Find fusable children (cached DuckDB nodes)
        fusable_children = [
            (i, child) for i, child in enumerate(node.children)
            if not child.cache_info and isinstance(child, DuckDBNodeMixin)
        ]

        if not fusable_children:
            return node

        # Build new children list by replacing fusable children with their children
        new_children = []
        fusable_indices = {i for i, _ in fusable_children}

        for i, child in enumerate(node.children):
            if i in fusable_indices:
                new_children.extend(child.children)
            else:
                new_children.append(child)

        # Create merged node
        return MergedDuckDBExec(
            merge_root=node,
            children=new_children,
            cache_info=node.cache_info,
            session_state=session_state,
        )
