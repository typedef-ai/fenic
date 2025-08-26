"""JSON expression serialization/deserialization."""

from fenic.core._logical_plan.expressions.json import (
    JqExpr,
    JsonContainsExpr,
    JsonTypeExpr,
)

# Import the main serialize/deserialize functions from parent
from fenic.core._serde.proto.expression_serde import (
    _deserialize_logical_expr_helper,
    serialize_logical_expr,
)
from fenic.core._serde.proto.serde_context import SerdeContext
from fenic.core._serde.proto.types import (
    JqExprProto,
    JsonContainsExprProto,
    JsonTypeExprProto,
    LogicalExprProto,
)

# =============================================================================
# JqExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_jq_expr(logical: JqExpr, context: SerdeContext) -> LogicalExprProto:
    return LogicalExprProto(
        jq=JqExprProto(
            expr=context.serialize_logical_expr(SerdeContext.EXPR, logical.expr),
            query=logical.query,
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_jq_expr(logical_proto: JqExprProto, context: SerdeContext) -> JqExpr:
    return JqExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        query=logical_proto.query,
    )


# =============================================================================
# JsonTypeExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_json_type_expr(
    logical: JsonTypeExpr, context: SerdeContext
) -> LogicalExprProto:
    return LogicalExprProto(
        json_type=JsonTypeExprProto(
            expr=context.serialize_logical_expr(SerdeContext.EXPR, logical.expr)
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_json_type_expr(
    logical_proto: JsonTypeExprProto, context: SerdeContext
) -> JsonTypeExpr:
    return JsonTypeExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr)
    )


# =============================================================================
# JsonContainsExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_json_contains_expr(
    logical: JsonContainsExpr, context: SerdeContext
) -> LogicalExprProto:
    return LogicalExprProto(
        json_contains=JsonContainsExprProto(
            expr=context.serialize_logical_expr(SerdeContext.EXPR, logical.expr),
            value=logical.value,
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_json_contains_expr(
    logical_proto: JsonContainsExprProto, context: SerdeContext
) -> JsonContainsExpr:
    return JsonContainsExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        value=logical_proto.value,
    )
