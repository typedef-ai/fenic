"""Embedding expression serialization/deserialization."""

from fenic.core._logical_plan.expressions.base import LogicalExpr
from fenic.core._logical_plan.expressions.embedding import (
    EmbeddingNormalizeExpr,
    EmbeddingSimilarityExpr,
)

# Import the main serialize/deserialize functions from parent
from fenic.core._serde.proto.expression_serde import (
    _deserialize_logical_expr_helper,
    serialize_logical_expr,
)
from fenic.core._serde.proto.serde_context import SerdeContext
from fenic.core._serde.proto.types import (
    EmbeddingNormalizeExprProto,
    EmbeddingSimilarityExprProto,
    LogicalExprProto,
)

# =============================================================================
# EmbeddingNormalizeExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_embedding_normalize_expr(
    logical: EmbeddingNormalizeExpr, context: SerdeContext
) -> LogicalExprProto:
    return LogicalExprProto(
        embedding_normalize=EmbeddingNormalizeExprProto(
            expr=context.serialize_logical_expr(SerdeContext.EXPR, logical.expr)
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_embedding_normalize_expr(
    logical_proto: EmbeddingNormalizeExprProto, context: SerdeContext
) -> EmbeddingNormalizeExpr:
    return EmbeddingNormalizeExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr)
    )


# =============================================================================
# EmbeddingSimilarityExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_embedding_similarity_expr(
    logical: EmbeddingSimilarityExpr, context: SerdeContext
) -> LogicalExprProto:
    proto = EmbeddingSimilarityExprProto(expr=context.serialize_logical_expr(SerdeContext.EXPR, logical.expr), metric=logical.metric)
    if isinstance(logical.other, LogicalExpr):
        proto.other_expr.CopyFrom(context.serialize_logical_expr("other_expr", logical.other))
    else:
        proto.query_vector.CopyFrom(context.serialize_numpy_array("query_vector", logical.other))
    return LogicalExprProto(embedding_similarity=proto)


@_deserialize_logical_expr_helper.register
def _deserialize_embedding_similarity_expr(
    logical_proto: EmbeddingSimilarityExprProto, context: SerdeContext
) -> EmbeddingSimilarityExpr:
    if logical_proto.other_expr:
        other = context.deserialize_logical_expr("other_expr", logical_proto.other_expr)
    elif logical_proto.query_vector:
        other = context.deserialize_numpy_array("query_vector", logical_proto.query_vector)
    else:
        raise ValueError(f"Invalid expression type: {logical_proto.WhichOneof('other_type')}")
    return EmbeddingSimilarityExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        other=other,
        metric=logical_proto.metric,
    )
