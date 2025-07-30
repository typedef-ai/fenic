from __future__ import annotations

from typing import List, Optional

from fenic.core._interfaces.session_state import BaseSessionState
from fenic.core._logical_plan.expressions import (
    AliasExpr,
    LogicalExpr,
    SortExpr,
)
from fenic.core._logical_plan.expressions.base import AggregateExpr
from fenic.core._logical_plan.plans.base import LogicalPlan
from fenic.core.types import Schema


class Aggregate(LogicalPlan):
    def __init__(
            self,
            input: LogicalPlan,
            group_exprs: List[LogicalExpr],
            agg_exprs: List[AliasExpr],
            session_state: Optional[BaseSessionState] = None,
            schema: Optional[Schema] = None):
        self._input = input
        self._group_exprs = group_exprs
        self._agg_exprs = agg_exprs
        for expr in agg_exprs:
            if not isinstance(expr.expr, AggregateExpr):
                raise ValueError(f"Expression {expr} is not an aggregation")
            _validate_agg_expr(expr.expr, group_exprs)
        for expr in group_exprs:
            _validate_groupby_expr(expr)
        super().__init__(session_state, schema)

    @classmethod
    def from_session_state(
        cls,
        input: LogicalPlan,
        group_exprs: List[LogicalExpr],
        agg_exprs: List[AliasExpr],
        session_state: BaseSessionState,
    ) -> Aggregate:
        return Aggregate(input, group_exprs, agg_exprs, session_state)

    @classmethod
    def from_schema(
        cls,
        input: LogicalPlan,
        group_exprs: List[LogicalExpr],
        agg_exprs: List[AliasExpr],
        schema: Schema,
    ) -> Aggregate:
        return Aggregate(input, group_exprs, agg_exprs, None, schema)

    def children(self) -> List[LogicalPlan]:
        return [self._input]

    def _build_schema(self, session_state: BaseSessionState) -> Schema:
        group_fields = [expr.to_column_field(self._input, session_state) for expr in self._group_exprs]
        agg_fields = [expr.to_column_field(self._input, session_state) for expr in self._agg_exprs]
        return Schema(column_fields=group_fields + agg_fields)

    def _repr(self) -> str:
        return f"Aggregate(group_exprs=[{', '.join(str(expr) for expr in self._group_exprs)}], agg_exprs=[{', '.join(str(expr) for expr in self._agg_exprs)}])"

    def group_exprs(self) -> List[LogicalExpr]:
        return self._group_exprs

    def agg_exprs(self) -> List[LogicalExpr]:
        return self._agg_exprs

    def with_children(self, children: List[LogicalPlan], session_state: Optional[BaseSessionState] = None) -> LogicalPlan:
        if len(children) != 1:
            raise ValueError("Aggregate must have exactly one child")
        result = Aggregate.from_session_state(children[0], self._group_exprs, self._agg_exprs, session_state)
        result.set_cache_info(self.cache_info)
        return result

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
