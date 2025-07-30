from fenic.core._interfaces.session_state import BaseSessionState
from fenic.core._logical_plan.expressions import (
    BooleanExpr,
    LogicalExpr,
    NotExpr,
    Operator,
)
from fenic.core._logical_plan.optimizer.base import (
    LogicalPlanOptimizerRule,
    OptimizationResult,
)
from fenic.core._logical_plan.plans.base import LogicalPlan
from fenic.core._logical_plan.plans.transform import Filter


class NotFilterPushdownRule(LogicalPlanOptimizerRule):
    """Optimization rule that selectively pushes NOT operators inward using De Morgan's laws.

    This transformation only applies optimizations that increase AND expressions:
    - NOT(A OR B) becomes NOT(A) AND NOT(B)
    - NOT(NOT(A)) becomes A

    The rule deliberately avoids converting NOT(A AND B) to OR expressions since our
    optimization pipeline is focused on maximizing and optimizing AND expressions.

    This rule should be applied before semantic predicate reordering to increase
    the number of AND expressions that can be effectively reordered.
    """

    def apply(self, logical_plan: LogicalPlan, session_state: BaseSessionState) -> OptimizationResult:
        return self.optimize_node(logical_plan, session_state)

    def optimize_node(self, node: LogicalPlan, session_state: BaseSessionState) -> OptimizationResult:
        any_child_modified = False
        optimized_children = []

        # First, recursively optimize all children
        for child in node.children():
            child_result = self.optimize_node(child, session_state)
            optimized_children.append(child_result.plan)
            any_child_modified = any_child_modified or child_result.was_modified

        # Update node with optimized children
        new_node = node.with_children(optimized_children, session_state)

        # If this is a filter node, apply NOT pushdown to its predicate
        if isinstance(new_node, Filter):
            filter_result = self.optimize_filter(new_node, session_state)
            return OptimizationResult(
                filter_result.plan, any_child_modified or filter_result.was_modified
            )

        return OptimizationResult(new_node, any_child_modified)

    def optimize_filter(self, node: Filter, session_state: BaseSessionState) -> OptimizationResult:
        predicate = node.predicate()

        # Apply selective NOT pushdown transformation to the predicate
        transformed_predicate = self.push_not_inward(predicate)

        # If the predicate was changed, create a new filter with the transformed predicate
        if transformed_predicate != predicate:
            new_filter = Filter.from_session_state(node._input, transformed_predicate, session_state)
            new_filter.cache_info = node.cache_info
            return OptimizationResult(new_filter, True)

        # No change needed
        return OptimizationResult(node, False)

    def push_not_inward(self, expr: LogicalExpr) -> LogicalExpr:
        """Recursively push NOT operators inward but only in ways that increase AND expressions.

        Specifically, converts NOT(OR) to AND but leaves NOT(AND) intact.
        """
        # Base case: if expression is a leaf node or not a NOT expression
        if not isinstance(expr, NotExpr):
            # If it's a Boolean expression, recursively transform its children
            if isinstance(expr, BooleanExpr):
                left = self.push_not_inward(expr.left)
                right = self.push_not_inward(expr.right)

                # If either child changed, create a new Boolean expression
                if left != expr.left or right != expr.right:
                    return BooleanExpr(left, right, expr.operator)
                return expr

            # Not a NOT or Boolean expression, return as is
            return expr

        # Handle NOT expression
        inner_expr = expr.expr

        # Case 1: Double negation - NOT(NOT(A)) becomes A
        if isinstance(inner_expr, NotExpr):
            return self.push_not_inward(inner_expr.expr)

        # Case 2: De Morgan for OR - NOT(A OR B) becomes NOT(A) AND NOT(B)
        if self.is_or_expr(inner_expr):
            return BooleanExpr(
                self.push_not_inward(NotExpr(inner_expr.left)),
                self.push_not_inward(NotExpr(inner_expr.right)),
                Operator.AND,
            )

        # Handle NOT over other expressions (including AND and leaf nodes)
        # Just recursively process the inner expression without distributing the NOT
        inner_transformed = self.push_not_inward(inner_expr)
        if inner_transformed != inner_expr:
            return NotExpr(inner_transformed)
        return expr

    @staticmethod
    def is_or_expr(expr: LogicalExpr) -> bool:
        return isinstance(expr, BooleanExpr) and expr.op == Operator.OR
