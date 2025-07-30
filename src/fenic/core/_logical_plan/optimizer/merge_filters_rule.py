from fenic.core._interfaces.session_state import BaseSessionState
from fenic.core._logical_plan.expressions import BooleanExpr, Operator
from fenic.core._logical_plan.optimizer.base import (
    LogicalPlanOptimizerRule,
    OptimizationResult,
)
from fenic.core._logical_plan.plans.base import LogicalPlan
from fenic.core._logical_plan.plans.transform import Filter


class MergeFiltersRule(LogicalPlanOptimizerRule):
    """Optimization rule that merges consecutive filter operations into a single filter.

    This rule identifies consecutive filter operations and combines their predicates
    into a single filter operation so the combined predicates can be better optimized by SemanticPredicateReorderRule.
    """

    def apply(self, logical_plan: LogicalPlan, session_state: BaseSessionState) -> OptimizationResult:
        result = self.optimize_node(logical_plan, session_state)
        return result

    def optimize_node(self, node: LogicalPlan, session_state: BaseSessionState) -> OptimizationResult:
        any_child_modified = False
        optimized_children = []

        for child in node.children():
            child_result = self.optimize_node(child, session_state)
            optimized_children.append(child_result.plan)
            any_child_modified = any_child_modified or child_result.was_modified

        new_node = node.with_children(optimized_children, session_state)

        if isinstance(node, Filter):
            merge_result = self.merge_filter(new_node, session_state)
            return OptimizationResult(
                merge_result.plan, any_child_modified or merge_result.was_modified
            )

        return OptimizationResult(new_node, any_child_modified)

    def merge_filter(self, node: LogicalPlan, session_state: BaseSessionState) -> OptimizationResult:
        if isinstance(node._input, Filter) and node._input.cache_info is None:
            merged_filter = Filter.from_session_state(
                node._input._input,
                BooleanExpr(node.predicate(), node._input.predicate(), Operator.AND),
                session_state
            )
            merged_filter.cache_info = node.cache_info
            # Return with was_modified=True since we merged filters
            return OptimizationResult(merged_filter, True)

        return OptimizationResult(node, False)
