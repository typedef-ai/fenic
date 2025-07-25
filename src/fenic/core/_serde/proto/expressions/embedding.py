"""Embedding expression serialization/deserialization."""

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
    return LogicalExprProto(
        embedding_similarity=EmbeddingSimilarityExprProto(
            left=context.serialize_logical_expr(SerdeContext.LEFT, logical.left),
            right=context.serialize_logical_expr(SerdeContext.RIGHT, logical.right),
            metric=logical.metric,
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_embedding_similarity_expr(
    logical_proto: EmbeddingSimilarityExprProto, context: SerdeContext
) -> EmbeddingSimilarityExpr:
    return EmbeddingSimilarityExpr(
        left=context.deserialize_logical_expr(SerdeContext.LEFT, logical_proto.left),
        right=context.deserialize_logical_expr(SerdeContext.RIGHT, logical_proto.right),
        metric=logical_proto.metric,
    )
