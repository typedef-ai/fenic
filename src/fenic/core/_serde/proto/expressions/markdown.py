"""Markdown expression serialization/deserialization."""

from fenic.core._logical_plan.expressions.markdown import (
    MdExtractHeaderChunks,
    MdGenerateTocExpr,
    MdGetCodeBlocksExpr,
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
    MdExtractHeaderChunksProto,
    MdGenerateTocExprProto,
    MdGetCodeBlocksExprProto,
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
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_md_to_json_expr(
    logical_proto: MdToJsonExprProto, context: SerdeContext
) -> MdToJsonExpr:
    return MdToJsonExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
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
            max_level=logical.max_level,
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_md_generate_toc_expr(
    logical_proto: MdGenerateTocExprProto, context: SerdeContext
) -> MdGenerateTocExpr:
    return MdGenerateTocExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        max_level=logical_proto.max_level,
    )


# =============================================================================
# MdExtractHeaderChunks
# =============================================================================

@serialize_logical_expr.register
def _serialize_md_extract_header_chunks(
    logical: MdExtractHeaderChunks, context: SerdeContext
) -> LogicalExprProto:
    return LogicalExprProto(
        md_extract_header_chunks=MdExtractHeaderChunksProto(
            expr=context.serialize_logical_expr(SerdeContext.EXPR, logical.expr),
            header_level=logical.header_level,
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_md_extract_header_chunks(
    logical_proto: MdExtractHeaderChunksProto, context: SerdeContext
) -> MdExtractHeaderChunks:
    return MdExtractHeaderChunks(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        header_level=logical_proto.header_level,
    )


# =============================================================================
# MdGetCodeBlocksExpr
# =============================================================================

@serialize_logical_expr.register
def _serialize_md_get_code_blocks_expr(
    logical: MdGetCodeBlocksExpr, context: SerdeContext
) -> LogicalExprProto:
    return LogicalExprProto(
        md_get_code_blocks=MdGetCodeBlocksExprProto(
            expr=context.serialize_logical_expr(SerdeContext.EXPR, logical.expr),
            language_filter=logical.language_filter,
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_md_get_code_blocks_expr(
    logical_proto: MdGetCodeBlocksExprProto, context: SerdeContext
) -> MdGetCodeBlocksExpr:
    return MdGetCodeBlocksExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        language_filter=logical_proto.language_filter if logical_proto.HasField("language_filter") else None,
    )
