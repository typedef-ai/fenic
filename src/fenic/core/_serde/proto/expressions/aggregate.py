"""Aggregate expression serialization/deserialization."""

from fenic.core._logical_plan.expressions.aggregate import (
    ApproxCountDistinctExpr,
    AvgExpr,
    CountDistinctExpr,
    CountExpr,
    FirstExpr,
    ListExpr,
    MaxExpr,
    MinExpr,
    StdDevExpr,
    SumDistinctExpr,
    SumExpr,
)
from fenic.core._serde.proto.expression_serde import (
    _deserialize_logical_expr_helper,
    serialize_logical_expr,
)
from fenic.core._serde.proto.serde_context import SerdeContext
from fenic.core._serde.proto.types import (
    ApproxCountDistinctExprProto,
    AvgExprProto,
    CountDistinctExprProto,
    CountExprProto,
    FirstExprProto,
    ListExprProto,
    LogicalExprProto,
    MaxExprProto,
    MinExprProto,
    StdDevExprProto,
    SumDistinctExprProto,
    SumExprProto,
)

# =============================================================================
# SumExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_sum_expr(logical: SumExpr, context: SerdeContext) -> LogicalExprProto:
    return LogicalExprProto(
        sum=SumExprProto(
            expr=context.serialize_logical_expr(SerdeContext.EXPR, logical.expr)
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_sum_expr(
    logical_proto: SumExprProto, context: SerdeContext
) -> SumExpr:
    return SumExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr)
    )


# =============================================================================
# AvgExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_avg_expr(logical: AvgExpr, context: SerdeContext) -> LogicalExprProto:
    return LogicalExprProto(
        avg=AvgExprProto(
            expr=context.serialize_logical_expr(SerdeContext.EXPR, logical.expr)
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_avg_expr(
    logical_proto: AvgExprProto, context: SerdeContext
) -> AvgExpr:
    return AvgExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr)
    )


# =============================================================================
# MinExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_min_expr(logical: MinExpr, context: SerdeContext) -> LogicalExprProto:
    return LogicalExprProto(
        min=MinExprProto(
            expr=context.serialize_logical_expr(SerdeContext.EXPR, logical.expr)
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_min_expr(
    logical_proto: MinExprProto, context: SerdeContext
) -> MinExpr:
    return MinExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr)
    )


# =============================================================================
# MaxExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_max_expr(logical: MaxExpr, context: SerdeContext) -> LogicalExprProto:
    return LogicalExprProto(
        max=MaxExprProto(
            expr=context.serialize_logical_expr(SerdeContext.EXPR, logical.expr)
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_max_expr(
    logical_proto: MaxExprProto, context: SerdeContext
) -> MaxExpr:
    return MaxExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr)
    )


# =============================================================================
# CountExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_count_expr(
    logical: CountExpr, context: SerdeContext
) -> LogicalExprProto:
    return LogicalExprProto(
        count=CountExprProto(
            expr=context.serialize_logical_expr(SerdeContext.EXPR, logical.expr)
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_count_expr(
    logical_proto: CountExprProto, context: SerdeContext
) -> CountExpr:
    return CountExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr)
    )


# =============================================================================
# ListExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_list_expr(logical: ListExpr, context: SerdeContext) -> LogicalExprProto:
    return LogicalExprProto(
        list=ListExprProto(
            expr=context.serialize_logical_expr(SerdeContext.EXPR, logical.expr)
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_list_expr(
    logical_proto: ListExprProto, context: SerdeContext
) -> ListExpr:
    return ListExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr)
    )


# =============================================================================
# FirstExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_first_expr(
    logical: FirstExpr, context: SerdeContext
) -> LogicalExprProto:
    return LogicalExprProto(
        first=FirstExprProto(
            expr=context.serialize_logical_expr(SerdeContext.EXPR, logical.expr)
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_first_expr(
    logical_proto: FirstExprProto, context: SerdeContext
) -> FirstExpr:
    return FirstExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr)
    )


# =============================================================================
# StdDevExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_std_dev_expr(
    logical: StdDevExpr, context: SerdeContext
) -> LogicalExprProto:
    return LogicalExprProto(
        std_dev=StdDevExprProto(
            expr=context.serialize_logical_expr(SerdeContext.EXPR, logical.expr)
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_std_dev_expr(
    logical_proto: StdDevExprProto, context: SerdeContext
) -> StdDevExpr:
    return StdDevExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr)
    )


# =============================================================================
# CountDistinctExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_count_distinct_expr(
    logical: CountDistinctExpr, context: SerdeContext
) -> LogicalExprProto:
    return LogicalExprProto(
        count_distinct=CountDistinctExprProto(
            exprs=context.serialize_logical_expr_list(SerdeContext.EXPRS, logical.exprs)
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_count_distinct_expr(
    logical_proto: CountDistinctExprProto, context: SerdeContext
) -> CountDistinctExpr:
    return CountDistinctExpr(
        exprs=context.deserialize_logical_expr_list(SerdeContext.EXPRS, logical_proto.exprs)
    )


# =============================================================================
# ApproxCountDistinctExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_approx_count_distinct_expr(
    logical: ApproxCountDistinctExpr, context: SerdeContext
) -> LogicalExprProto:
    return LogicalExprProto(
        approx_count_distinct=ApproxCountDistinctExprProto(
            expr=context.serialize_logical_expr(SerdeContext.EXPR, logical.expr)
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_approx_count_distinct_expr(
    logical_proto: ApproxCountDistinctExprProto, context: SerdeContext
) -> ApproxCountDistinctExpr:
    return ApproxCountDistinctExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr)
    )


# =============================================================================
# SumDistinctExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_sum_distinct_expr(
    logical: SumDistinctExpr, context: SerdeContext
) -> LogicalExprProto:
    return LogicalExprProto(
        sum_distinct=SumDistinctExprProto(
            expr=context.serialize_logical_expr(SerdeContext.EXPR, logical.expr)
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_sum_distinct_expr(
    logical_proto: SumDistinctExprProto, context: SerdeContext
) -> SumDistinctExpr:
    return SumDistinctExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr)
    )
