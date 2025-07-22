"""Basic expression serialization/deserialization."""

from fenic.core._logical_plan.expressions.basic import (
    AliasExpr,
    ArrayContainsExpr,
    ArrayExpr,
    ArrayLengthExpr,
    CastExpr,
    CoalesceExpr,
    ColumnExpr,
    IndexExpr,
    InExpr,
    IsNullExpr,
    LiteralExpr,
    NotExpr,
    SortExpr,
    StructExpr,
)
from fenic.core._serde.proto.expression_serde import (
    _deserialize_logical_expr_helper,
    serialize_logical_expr,
)
from fenic.core._serde.proto.serde_context import SerdeContext
from fenic.core._serde.proto.types import (
    AliasExprProto,
    ArrayContainsExprProto,
    ArrayExprProto,
    ArrayLengthExprProto,
    CastExprProto,
    CoalesceExprProto,
    ColumnExprProto,
    IndexExprProto,
    InExprProto,
    IsNullExprProto,
    LiteralExprProto,
    LogicalExprProto,
    NotExprProto,
    SortExprProto,
    StructExprProto,
)

# =============================================================================
# ColumnExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_column_expr(
    logical: ColumnExpr, context: SerdeContext
) -> LogicalExprProto:
    return LogicalExprProto(column=ColumnExprProto(name=logical.name))


@_deserialize_logical_expr_helper.register
def _deserialize_column_expr(
    logical_proto: ColumnExprProto, context: SerdeContext
) -> ColumnExpr:
    return ColumnExpr(name=logical_proto.name)


# =============================================================================
# LiteralExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_literal_expr(
    logical: LiteralExpr, context: SerdeContext
) -> LogicalExprProto:
    return LogicalExprProto(
        literal=LiteralExprProto(
            value=context.serialize_scalar_value(SerdeContext.VALUE, logical.literal),
            data_type=context.serialize_data_type(SerdeContext.DATA_TYPE, logical.data_type),
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_literal_expr(
    logical_proto: LiteralExprProto, context: SerdeContext
) -> LiteralExpr:
    from fenic.core._logical_plan.expressions.basic import LiteralExpr

    return LiteralExpr(
        literal=context.deserialize_scalar_value(SerdeContext.VALUE, logical_proto.value),
        data_type=context.deserialize_data_type(SerdeContext.DATA_TYPE, logical_proto.data_type),
    )


# =============================================================================
# AliasExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_alias_expr(
    logical: AliasExpr, context: SerdeContext
) -> LogicalExprProto:
    return LogicalExprProto(
        alias=AliasExprProto(
            expr=context.serialize_logical_expr(SerdeContext.EXPR, logical.expr),
            name=logical.name,
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_alias_expr(
    logical_proto: AliasExprProto, context: SerdeContext
) -> AliasExpr:
    return AliasExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        name=logical_proto.name,
    )


# =============================================================================
# ArrayExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_array_expr(
    logical: ArrayExpr, context: SerdeContext
) -> LogicalExprProto:
    return LogicalExprProto(
        array=ArrayExprProto(
            exprs=context.serialize_logical_expr_list(SerdeContext.EXPRS, logical.exprs)
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_array_expr(
    logical_proto: ArrayExprProto, context: SerdeContext
) -> ArrayExpr:
    return ArrayExpr(
        exprs=context.deserialize_logical_expr_list(SerdeContext.EXPRS, logical_proto.exprs)
    )


# =============================================================================
# NotExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_not_expr(logical: NotExpr, context: SerdeContext) -> LogicalExprProto:
    return LogicalExprProto(
        not_expr=NotExprProto(
            expr=context.serialize_logical_expr(SerdeContext.EXPR, logical.expr)
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_not_expr(
    logical_proto: NotExprProto, context: SerdeContext
) -> NotExpr:
    return NotExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr)
    )


# =============================================================================
# SortExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_sort_expr(logical: SortExpr, context: SerdeContext) -> LogicalExprProto:
    return LogicalExprProto(
        sort=SortExprProto(
            expr=context.serialize_logical_expr(SerdeContext.EXPR, logical.expr),
            ascending=logical.ascending,
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_sort_expr(
    logical_proto: SortExprProto, context: SerdeContext
) -> SortExpr:
    return SortExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        ascending=logical_proto.ascending,
    )


# =============================================================================
# IndexExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_index_expr(
    logical: IndexExpr, context: SerdeContext
) -> LogicalExprProto:
    if isinstance(logical.index, str):
        proto = IndexExprProto(
            expr=context.serialize_logical_expr(SerdeContext.EXPR, logical.expr),
            string_index=logical.index,
        )
    else:
        proto = IndexExprProto(
            expr=context.serialize_logical_expr(SerdeContext.EXPR, logical.expr),
            int_index=logical.index,
        )
    return LogicalExprProto(index=proto)


@_deserialize_logical_expr_helper.register
def _deserialize_index_expr(
    logical_proto: IndexExprProto, context: SerdeContext
) -> IndexExpr:
    if logical_proto.HasField("string_index"):
        index = logical_proto.string_index
    else:
        index = logical_proto.int_index
    return IndexExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        index=index,
    )


# =============================================================================
# StructExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_struct_expr(
    logical: StructExpr, context: SerdeContext
) -> LogicalExprProto:
    return LogicalExprProto(
        struct=StructExprProto(
            exprs=context.serialize_logical_expr_list(SerdeContext.EXPRS, logical.exprs)
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_struct_expr(
    logical_proto: StructExprProto, context: SerdeContext
) -> StructExpr:
    return StructExpr(
        exprs=context.deserialize_logical_expr_list(SerdeContext.EXPRS, logical_proto.exprs)
    )


# =============================================================================
# CastExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_cast_expr(logical: CastExpr, context: SerdeContext) -> LogicalExprProto:
    return LogicalExprProto(
        cast=CastExprProto(
            expr=context.serialize_logical_expr(SerdeContext.EXPR, logical.expr),
            data_type=context.serialize_data_type(SerdeContext.DATA_TYPE, logical.data_type),
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_cast_expr(
    logical_proto: CastExprProto, context: SerdeContext
) -> CastExpr:
    return CastExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        data_type=context.deserialize_data_type(SerdeContext.DATA_TYPE, logical_proto.data_type),
    )


# =============================================================================
# CoalesceExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_coalesce_expr(
    logical: CoalesceExpr, context: SerdeContext
) -> LogicalExprProto:
    return LogicalExprProto(
        coalesce=CoalesceExprProto(
            exprs=context.serialize_logical_expr_list(SerdeContext.EXPRS, logical.exprs)
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_coalesce_expr(
    logical_proto: CoalesceExprProto, context: SerdeContext
) -> CoalesceExpr:
    return CoalesceExpr(
        exprs=context.deserialize_logical_expr_list(SerdeContext.EXPRS, logical_proto.exprs)
    )


# =============================================================================
# InExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_in_expr(logical: InExpr, context: SerdeContext) -> LogicalExprProto:
    return LogicalExprProto(
        in_expr=InExprProto(
            expr=context.serialize_logical_expr(SerdeContext.EXPR, logical.expr),
            values=context.serialize_logical_expr_list(SerdeContext.VALUES, logical.values),
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_in_expr(logical_proto: InExprProto, context: SerdeContext) -> InExpr:
    return InExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        values=context.deserialize_logical_expr_list(SerdeContext.VALUES, logical_proto.values),
    )


# =============================================================================
# IsNullExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_is_null_expr(
    logical: IsNullExpr, context: SerdeContext
) -> LogicalExprProto:
    return LogicalExprProto(
        is_null=IsNullExprProto(
            expr=context.serialize_logical_expr(SerdeContext.EXPR, logical.expr)
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_is_null_expr(
    logical_proto: IsNullExprProto, context: SerdeContext
) -> IsNullExpr:
    return IsNullExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr)
    )


# =============================================================================
# ArrayLengthExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_array_length_expr(
    logical: ArrayLengthExpr, context: SerdeContext
) -> LogicalExprProto:
    return LogicalExprProto(
        array_length=ArrayLengthExprProto(
            expr=context.serialize_logical_expr(SerdeContext.EXPR, logical.expr)
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_array_length_expr(
    logical_proto: ArrayLengthExprProto, context: SerdeContext
) -> ArrayLengthExpr:
    return ArrayLengthExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr)
    )


# =============================================================================
# ArrayContainsExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_array_contains_expr(
    logical: ArrayContainsExpr, context: SerdeContext
) -> LogicalExprProto:
    return LogicalExprProto(
        array_contains=ArrayContainsExprProto(
            expr=context.serialize_logical_expr(SerdeContext.EXPR, logical.expr),
            value=context.serialize_logical_expr(SerdeContext.VALUE, logical.value),
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_array_contains_expr(
    logical_proto: ArrayContainsExprProto, context: SerdeContext
) -> ArrayContainsExpr:
    return ArrayContainsExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        value=context.deserialize_logical_expr(SerdeContext.VALUE, logical_proto.value),
    )
