from typing import List, Tuple

from fenic.core._logical_plan.expressions import (
    AggregateExpr,
    BooleanExpr,
    LogicalExpr,
    Operator,
    SemanticExpr,
    SortExpr,
)
from fenic.core._logical_plan.optimizer.base import (
    LogicalPlanOptimizerRule,
    OptimizationResult,
    OptimizerNodeResult,
)
from fenic.core._logical_plan.plans.base import LogicalPlan
from fenic.core._logical_plan.plans.node import LogicalPlanNode
from fenic.core._logical_plan.plans.transform import Filter


class SemanticFilterRewriteRule(LogicalPlanOptimizerRule):
    """Optimization rule to reorder filter predicates by evaluation cost to minimize expensive LLM calls.

    This rule decomposes `AND` conditions and reorders predicates so that cheaper ones run first.
    It ensures that standard (non-LLM) filters are applied before filters involving LLM function calls,
    which helps avoid unnecessary LLM evaluations on rows that would be filtered out anyway.

    Predicates are categorized into two groups:

    1. **Standard predicates** ~ regular filter conditions with no LLM calls.
    2. **Semantic predicates** ~ contain one or more LLM function calls, sorted by number of calls (least to most complex).

    `OR` expressions involving LLM calls are treated as atomic units and are not decomposed to short circuit evaluation.
    For example:
        (col_a > 10) AND (semantic.predicate("...{col_b}") OR semantic.predicate("...{col_c}"))

    In this case, `(col_a > 10)` is applied first, and the entire `OR` expression is evaluated as a single unit.

    Short circuiting `OR` conditions is more involved than `AND` due to limitations in Polars' evaluation model.
    Properly supporting them would require either:
    - Generating row-wise UDFs that respect conditional logic, (since case statements are evaluating sequentially) or
    - Decomposing filters into `UNION ALL` operations with intermediate result caching to avoid recomputation.
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

        if isinstance(new_node, Filter):
            filter_result = self.optimize_filter(new_node)
            return OptimizerNodeResult(
                filter_result.node, any_child_modified or filter_result.was_modified
            )

        return OptimizerNodeResult(new_node, any_child_modified)

    def optimize_filter(self, node: Filter) -> OptimizerNodeResult:
        predicate = node.predicate()

        # Skip optimization if not an AND expression or doesn't contain semantic predicates
        if not self.is_and_expr(
            predicate
        ) or not self.count_semantic_predicate_expressions(predicate):
            return OptimizerNodeResult(node, False)

        standard_predicates, semantic_predicates = self.partition_predicates(predicate)

        # Skip optimization if there's only one predicate total
        total_predicates = len(standard_predicates) + len(semantic_predicates)
        if total_predicates <= 1:
            return OptimizerNodeResult(node, False)

        # Apply predicates in order of increasing cost
        input = node._input

        # Apply standard predicates first (if any)
        if standard_predicates:
            standard_predicate_expr = self._make_and_expr(standard_predicates)
            result = Filter(standard_predicate_expr)
            result.set_input(input)
            result._build_schema_with_validation(self.session_state)
        else:
            result = input

        # Apply semantic predicates last
        for pred in semantic_predicates:
            sem_predicate_result = Filter(pred)
            sem_predicate_result.set_input(result)
            sem_predicate_result._build_schema_with_validation(self.session_state)
            result = sem_predicate_result

        result.set_cache_info(node.cache_info)
        return OptimizerNodeResult(result, True)

    # Returns a tuple of two lists:
    # - The first list contains all the standard predicates.
    # - The second list contains all the semantic predicates, sorted by the number of semantic expressions.
    def partition_predicates(
        self, predicate: LogicalExpr
    ) -> Tuple[List[LogicalExpr], List[LogicalExpr]]:
        standard_predicates: List[LogicalExpr] = []
        semantic_predicates: List[Tuple[LogicalExpr, int]] = []

        def _partition_predicates(expr: LogicalExpr):
            num_semantic_expressions = self.count_semantic_predicate_expressions(expr)
            if self.is_and_expr(expr):
                _partition_predicates(expr.left)
                _partition_predicates(expr.right)
            elif num_semantic_expressions > 0:
                semantic_predicates.append((expr, num_semantic_expressions))
            else:
                standard_predicates.append(expr)

        _partition_predicates(predicate)

        semantic_predicates.sort(key=lambda pair: pair[1])
        return standard_predicates, [expr for expr, _ in semantic_predicates]

    @staticmethod
    def _make_and_expr(predicates: List[LogicalExpr]) -> LogicalExpr:
        """Combine multiple predicates with AND operators."""
        if not predicates:
            raise ValueError("Cannot make AND expression from empty predicates")
        if len(predicates) == 1:
            return predicates[0]

        result = predicates[0]
        for predicate in predicates[1:]:
            result = BooleanExpr(result, predicate, Operator.AND)
        return result

    @staticmethod
    def count_semantic_predicate_expressions(expr: LogicalExpr) -> int:
        """Count the number of semantic predicate expressions in the expression tree."""
        if isinstance(expr, AggregateExpr):
            raise ValueError("AggregateExpr cannot be used in filter predicates.")
        if isinstance(expr, SortExpr):
            raise ValueError("SortExpr cannot be used in filter predicates.")
        return int(isinstance(expr, SemanticExpr)) + sum(
            SemanticFilterRewriteRule.count_semantic_predicate_expressions(child)
            for child in expr.children()
        )

    @staticmethod
    def is_and_expr(expr: LogicalExpr) -> bool:
        return isinstance(expr, BooleanExpr) and expr.op == Operator.AND
