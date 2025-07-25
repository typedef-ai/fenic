"""Markdown expression serialization/deserialization."""

from fenic.core._logical_plan.expressions.markdown import (
    MdGenerateTocExpr,
    MdToJsonExpr,
)

# Import the main serialize/deserialize functions from parent
from fenic.core._serde.proto.expression_serde import (
    _deserialize_logical_expr_helper,
    serialize_logical_expr,
)
from fenic.core._serde.proto.serde_context import SerdeContext
from fenic.core._serde.proto.types import (
    LogicalExprProto,
    MdGenerateTocExprProto,
    MdToJsonExprProto,
)

# =============================================================================
# MdToJsonExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_md_to_json_expr(
    logical: MdToJsonExpr, context: SerdeContext
) -> LogicalExprProto:
    return LogicalExprProto(
        md_to_json=MdToJsonExprProto(
            expr=context.serialize_logical_expr(SerdeContext.EXPR, logical.expr),
            schema=logical.schema,
            include_metadata=logical.include_metadata,
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_md_to_json_expr(
    logical_proto: MdToJsonExprProto, context: SerdeContext
) -> MdToJsonExpr:
    return MdToJsonExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        schema=logical_proto.schema,
        include_metadata=logical_proto.include_metadata,
    )


# =============================================================================
# MdGenerateTocExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_md_generate_toc_expr(
    logical: MdGenerateTocExpr, context: SerdeContext
) -> LogicalExprProto:
    return LogicalExprProto(
        md_generate_toc=MdGenerateTocExprProto(
            expr=context.serialize_logical_expr(SerdeContext.EXPR, logical.expr),
            max_depth=logical.max_depth,
            include_links=logical.include_links,
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_md_generate_toc_expr(
    logical_proto: MdGenerateTocExprProto, context: SerdeContext
) -> MdGenerateTocExpr:
    return MdGenerateTocExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        max_depth=logical_proto.max_depth,
        include_links=logical_proto.include_links,
    )
