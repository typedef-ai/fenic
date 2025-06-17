from typing import List

from fenic.core._logical_plan.expressions import (
    AggregateExpr,
    AliasExpr,
    LogicalExpr,
    SortExpr,
)
from fenic.core._logical_plan.plans.base import LogicalPlan
from fenic.core.types import ColumnField, IntegerType, Schema


class Aggregate(LogicalPlan):
    def __init__(
        self,
        input: LogicalPlan,
        group_exprs: List[LogicalExpr],
        agg_exprs: List[AliasExpr],
    ):
        self._input = input
        self._group_exprs = group_exprs
        self._agg_exprs = agg_exprs
        for expr in agg_exprs:
            if not isinstance(expr.expr, AggregateExpr):
                raise ValueError(f"Expression {expr} is not an aggregation")
            _validate_agg_expr(expr.expr, group_exprs)
        for expr in group_exprs:
            _validate_groupby_expr(expr)
        super().__init__(self._input.session_state)

    def children(self) -> List[LogicalPlan]:
        return [self._input]

    def _build_schema(self) -> Schema:
        group_fields = [expr.to_column_field(self._input) for expr in self._group_exprs]
        agg_fields = [expr.to_column_field(self._input) for expr in self._agg_exprs]
        return Schema(column_fields=group_fields + agg_fields)

    def _repr(self) -> str:
        return f"Aggregate(group_exprs=[{', '.join(str(expr) for expr in self._group_exprs)}], agg_exprs=[{', '.join(str(expr) for expr in self._agg_exprs)}])"

    def group_exprs(self) -> List[LogicalExpr]:
        return self._group_exprs

    def agg_exprs(self) -> List[LogicalExpr]:
        return self._agg_exprs

    def with_children(self, children: List[LogicalPlan]) -> LogicalPlan:
        if len(children) != 1:
            raise ValueError("Aggregate must have exactly one child")
        result = Aggregate(children[0], self._group_exprs, self._agg_exprs)
        result.set_cache_info(self.cache_info)
        return result


class SemanticAggregate(LogicalPlan):
    def __init__(
        self,
        input: LogicalPlan,
        group_expr: LogicalExpr,
        agg_exprs: List[AliasExpr],
        num_clusters: int,
    ):
        self._input = input
        self._group_expr = group_expr
        self._agg_exprs = agg_exprs
        self._num_clusters = num_clusters
        for expr in agg_exprs:
            if not isinstance(expr.expr, AggregateExpr):
                raise ValueError(f"Expression {expr} is not an aggregation")
            _validate_agg_expr(expr.expr, [group_expr])
        _validate_groupby_expr(group_expr)
        super().__init__(self._input.session_state)

    def children(self) -> List[LogicalPlan]:
        return [self._input]

    def _build_schema(self) -> Schema:
        cluster_id_field = ColumnField("_cluster_id", IntegerType)
        agg_fields = [expr.to_column_field(self._input) for expr in self._agg_exprs]
        return Schema(column_fields=[cluster_id_field] + agg_fields)

    def _repr(self) -> str:
        return f"Aggregate(group_expr={str(self._group_expr)}, agg_exprs=[{', '.join(str(expr) for expr in self._agg_exprs)}], num_clusters={self._num_clusters})"

    def num_clusters(self) -> int:
        return self._num_clusters

    def group_expr(self) -> LogicalExpr:
        return self._group_expr

    def agg_exprs(self) -> List[LogicalExpr]:
        return self._agg_exprs

    def with_children(self, children: List[LogicalPlan]) -> LogicalPlan:
        if len(children) != 1:
            raise ValueError("SemanticAggregate must have exactly one child")
        result = SemanticAggregate(
            children[0], self._group_expr, self._agg_exprs, self._num_clusters
        )
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
