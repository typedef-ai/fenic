from fenic.core._logical_plan.expressions import (
    BooleanExpr,
    LogicalExpr,
    NotExpr,
    Operator,
)
from fenic.core._logical_plan.optimizer.base import (
    LogicalPlanOptimizerRule,
    OptimizationResult,
    OptimizerNodeResult,
)
from fenic.core._logical_plan.plans.base import LogicalPlan
from fenic.core._logical_plan.plans.node import LogicalPlanNode
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

    def apply(self, logical_plan: LogicalPlan) -> OptimizationResult:
        return self.optimize_plan(logical_plan)

    def optimize_plan(self, plan: LogicalPlan) -> OptimizationResult:
        # Optimizes the plan by traversing the logical plan node tree.
        optimizer_node_result = self.optimize_node(plan.logical_plan_node)
        logical_plan = LogicalPlan(plan.session_state)
        logical_plan.logical_plan_node = optimizer_node_result.node
        return OptimizationResult(
            logical_plan,
            optimizer_node_result.was_modified,
        )


    def optimize_node(self, node: LogicalPlanNode) -> OptimizerNodeResult:
        any_child_modified = False
        optimized_children: list[LogicalPlanNode] = []
                # First, recursively optimize all children
        for child in node.children():
            child_result = self.optimize_node(child)
            optimized_children.append(child_result.node)
            any_child_modified = any_child_modified or child_result.was_modified

        # Update node with optimized children
        new_node = node.with_children(optimized_children)
        new_node._schema = node._schema

        # If this is a filter node, apply NOT pushdown to its predicate
        if isinstance(new_node, Filter):
            filter_result = self.optimize_filter(new_node)
            return OptimizerNodeResult(
                filter_result.node, any_child_modified or filter_result.was_modified
            )

        return OptimizerNodeResult(new_node, any_child_modified)

    def optimize_filter(self, node: Filter) -> OptimizerNodeResult:
        predicate = node.predicate()

        # Apply selective NOT pushdown transformation to the predicate
        transformed_predicate = self.push_not_inward(predicate)

        # If the predicate was changed, create a new filter with the transformed predicate
        if transformed_predicate != predicate:
            new_filter = Filter(transformed_predicate)
            new_filter.set_input(node._input)
            new_filter._schema = node._schema
            new_filter.cache_info = node.cache_info
            return OptimizerNodeResult(new_filter, True)

        # No change needed
        return OptimizerNodeResult(node, False)

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
