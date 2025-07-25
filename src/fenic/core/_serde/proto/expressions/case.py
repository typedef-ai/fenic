"""Case expression serialization/deserialization."""

from fenic.core._logical_plan.expressions.case import OtherwiseExpr, WhenExpr

# Import the main serialize/deserialize functions from parent
from fenic.core._serde.proto.expression_serde import (
    _deserialize_logical_expr_helper,
    serialize_logical_expr,
)
from fenic.core._serde.proto.serde_context import SerdeContext
from fenic.core._serde.proto.types import (
    LogicalExprProto,
    OtherwiseExprProto,
    WhenExprProto,
)

# =============================================================================
# WhenExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_when_expr(logical: WhenExpr, context: SerdeContext) -> LogicalExprProto:
    return LogicalExprProto(
        when=WhenExprProto(
            condition=context.serialize_logical_expr(SerdeContext.CONDITION, logical.condition),
            then=context.serialize_logical_expr(SerdeContext.THEN, logical.then),
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_when_expr(
    logical_proto: WhenExprProto, context: SerdeContext
) -> WhenExpr:
    return WhenExpr(
        condition=context.deserialize_logical_expr(SerdeContext.CONDITION, logical_proto.condition),
        then=context.deserialize_logical_expr(SerdeContext.THEN, logical_proto.then),
    )


# =============================================================================
# OtherwiseExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_otherwise_expr(
    logical: OtherwiseExpr, context: SerdeContext
) -> LogicalExprProto:
    return LogicalExprProto(
        otherwise=OtherwiseExprProto(
            expr=context.serialize_logical_expr(SerdeContext.EXPR, logical.expr)
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_otherwise_expr(
    logical_proto: OtherwiseExprProto, context: SerdeContext
) -> OtherwiseExpr:
    return OtherwiseExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr)
    )
