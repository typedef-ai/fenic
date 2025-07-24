from typing import List

from fenic.core._interfaces.session_state import BaseSessionState
from fenic.core._logical_plan.expressions import (
    AliasExpr,
    LogicalExpr,
    SortExpr,
)
from fenic.core._logical_plan.expressions.base import AggregateExpr
from fenic.core._logical_plan.plans.node import LogicalPlanNode
from fenic.core.types import Schema


class Aggregate(LogicalPlanNode):
    def __init__(
        self,
        group_exprs: List[LogicalExpr],
        agg_exprs: List[AliasExpr],
    ):
        super().__init__()
        self._group_exprs = group_exprs
        self._agg_exprs = agg_exprs

    def children(self) -> List[LogicalPlanNode]:
        return [self._input]

    def _validate_expressions(self):
        for expr in self._agg_exprs:
            if not isinstance(expr.expr, AggregateExpr):
                raise ValueError(f"Expression {expr} is not an aggregation")
            _validate_agg_expr(expr.expr, self._group_exprs)
        for expr in self._group_exprs:
            _validate_groupby_expr(expr)


    def _build_schema(self, session_state: BaseSessionState) -> Schema:
        self._validate_expressions()

        group_fields = [expr.to_column_field(self._input, session_state) for expr in self._group_exprs]
        agg_fields = [expr.to_column_field(self._input, session_state) for expr in self._agg_exprs]
        return Schema(column_fields=group_fields + agg_fields)

    def _repr(self) -> str:
        return f"Aggregate(group_exprs=[{', '.join(str(expr) for expr in self._group_exprs)}], agg_exprs=[{', '.join(str(expr) for expr in self._agg_exprs)}])"

    def group_exprs(self) -> List[LogicalExpr]:
        return self._group_exprs

    def agg_exprs(self) -> List[LogicalExpr]:
        return self._agg_exprs

    def with_children(self, children: List[LogicalPlanNode]) -> LogicalPlanNode:
        if len(children) != 1:
            raise ValueError("Aggregate must have exactly one child")
        return self.copy(self, children)

    @classmethod
    def _create_new_node(cls, node: LogicalPlanNode, children: List[LogicalPlanNode]) -> LogicalPlanNode:
        new_node = Aggregate(node._group_exprs, node._agg_exprs)
        new_node.set_input(children[0])
        return new_node

def _validate_agg_expr(
    expr: LogicalExpr,
    by_exprs: List[LogicalExpr],
    in_agg_function: bool = False,
):
    """Validate aggregation expressions."""
    if isinstance(expr, AggregateExpr):
        if in_agg_function:
            raise ValueError(
                f"Nested aggregation functions are not allowed. Found inner aggregation '{expr.children()[0]}' inside outer aggregation '{expr}'. "
                f"Each column can only be aggregated once within a single aggregation operation. "
                f"If you need to perform multiple levels of aggregation, please do so in separate operations."
            )
        for child in expr.children():
            _validate_agg_expr(child, by_exprs, in_agg_function=True)
        return
    for child in expr.children():
        _validate_agg_expr(child, by_exprs, in_agg_function)


def _validate_groupby_expr(expr: LogicalExpr):
    """Validate groupby expressions."""
    if isinstance(expr, AggregateExpr):
        raise ValueError(
            f"Aggregate function: {expr} cannot be used in the group by clause."
        )
    if isinstance(expr, SortExpr):
        raise ValueError(
            f"Sort expression: {expr} cannot be used in the group by clause."
        )
    for child in expr.children():
        _validate_groupby_expr(child)
