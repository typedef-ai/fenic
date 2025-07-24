from fenic.core._interfaces.session_state import BaseSessionState
from fenic.core._logical_plan.expressions import BooleanExpr, Operator
from fenic.core._logical_plan.optimizer.base import (
    LogicalPlanOptimizerRule,
    OptimizationResult,
    OptimizerNodeResult,
)
from fenic.core._logical_plan.plans.base import LogicalPlan
from fenic.core._logical_plan.plans.node import LogicalPlanNode
from fenic.core._logical_plan.plans.transform import Filter


class MergeFiltersRule(LogicalPlanOptimizerRule):
    session_state: BaseSessionState

    """Optimization rule that merges consecutive filter operations into a single filter.

    This rule identifies consecutive filter operations and combines their predicates
    into a single filter operation so the combined predicates can be better optimized by SemanticPredicateReorderRule.
    """

    def apply(self, logical_plan: LogicalPlan) -> OptimizationResult:
        self.session_state = logical_plan.session_state
        return self.optimize_plan(logical_plan)

    def optimize_plan(self, plan: LogicalPlan) -> OptimizationResult:
        optimizer_node_result = self.optimize_node(plan.logical_plan_node)
        logical_plan = LogicalPlan(plan.session_state)
        logical_plan.logical_plan_node = optimizer_node_result.node
        return OptimizationResult(
            logical_plan,
            optimizer_node_result.was_modified
        )

    def optimize_node(self, node: LogicalPlanNode) -> OptimizerNodeResult:
        any_child_modified = False
        optimized_children: list[LogicalPlanNode] = []

        for child in node.children():
            child_result = self.optimize_node(child)
            optimized_children.append(child_result.node)
            any_child_modified = any_child_modified or child_result.was_modified

        new_node = node.with_children(optimized_children)
        new_node._schema = node._schema

        if isinstance(node, Filter):
            merge_result = self.merge_filter(new_node)
            return OptimizerNodeResult(
                merge_result.node, any_child_modified or merge_result.was_modified
            )

        return OptimizerNodeResult(new_node, any_child_modified)

    def merge_filter(self, node: LogicalPlanNode) -> OptimizerNodeResult:
        if isinstance(node._input, Filter) and node._input.cache_info is None:
            merged_filter = Filter(
                BooleanExpr(node.predicate(), node._input.predicate(), Operator.AND),
            )
            merged_filter.set_input(node._input._input)
            merged_filter.set_cache_info(node.cache_info)
            merged_filter._build_schema_with_validation(self.session_state)

            # Return with was_modified=True since we merged filters
            return OptimizerNodeResult(merged_filter, True)

        return OptimizerNodeResult(node, False)
