from __future__ import annotations

from typing import List, Optional

from fenic.core._interfaces.session_state import BaseSessionState
from fenic.core._logical_plan.expressions import (
    AliasExpr,
    LogicalExpr,
    SemanticReduceExpr,
    SortExpr,
)
from fenic.core._logical_plan.expressions.base import AggregateExpr
from fenic.core._logical_plan.plans.base import LogicalPlan
from fenic.core.error import InternalError, PlanError
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
                raise PlanError(f"Expression {expr} is not an aggregation")
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
        agg_fields = []

        for expr in self._agg_exprs:
            field = expr.to_column_field(self._input, session_state)
            agg_fields.append(field)

            agg_expr = expr.expr
            if isinstance(agg_expr, SemanticReduceExpr) and agg_expr.group_context_exprs:
                # Validate each context expression is in group by
                for context_key, context_expr in agg_expr.group_context_exprs.items():
                    if context_expr not in self._group_exprs:
                        raise PlanError(
                            f"semantic.reduce context expression '{context_key}' not found in group by. "
                            f"Available group by expressions: {', '.join(str(e) for e in self._group_exprs) if self._group_exprs else 'none'}."
                        )

        return Schema(column_fields=group_fields + agg_fields)

    def _repr(self) -> str:
        return f"Aggregate(group_exprs=[{', '.join(str(expr) for expr in self._group_exprs)}], agg_exprs=[{', '.join(str(expr) for expr in self._agg_exprs)}])"

    def group_exprs(self) -> List[LogicalExpr]:
        return self._group_exprs

    def agg_exprs(self) -> List[LogicalExpr]:
        return self._agg_exprs

    def exprs(self) -> List[LogicalExpr]:
        plan_exprs = list(self._group_exprs)
        plan_exprs.extend(self._agg_exprs)
        return plan_exprs

    def with_children(self, children: List[LogicalPlan], session_state: Optional[BaseSessionState] = None) -> LogicalPlan:
        if len(children) != 1:
            raise InternalError("Aggregate must have exactly one child")
        result = Aggregate.from_session_state(children[0], self._group_exprs, self._agg_exprs, session_state)
        result.set_cache_info(self.cache_info)
        return result

    def _eq_specific(self, other: Aggregate) -> bool:
        return (
            # Compare group expressions
            len(self._group_exprs) == len(other._group_exprs)
            and all(expr1 == expr2 for expr1, expr2 in zip(self._group_exprs, other._group_exprs, strict=True))
            # Compare aggregate expressions
            and len(self._agg_exprs) == len(other._agg_exprs)
            and all(expr1 == expr2 for expr1, expr2 in zip(self._agg_exprs, other._agg_exprs, strict=True))
        )

def _validate_groupby_expr(expr: LogicalExpr):
    """Validate groupby expressions."""
    if isinstance(expr, AggregateExpr):
        raise PlanError(
            f"Aggregate function: {expr} cannot be used in the group by clause."
        )
    if isinstance(expr, SortExpr):
        raise PlanError(
            f"Sort expression: {expr} cannot be used in the group by clause."
        )
    for child in expr.children():
        _validate_groupby_expr(child)
