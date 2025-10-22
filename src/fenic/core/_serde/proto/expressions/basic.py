"""Basic expression serialization/deserialization."""

from io import BytesIO

import polars as pl

from fenic.core._logical_plan.expressions.basic import (
    AliasExpr,
    ArrayExpr,
    CastExpr,
    CoalesceExpr,
    ColumnExpr,
    FlattenExpr,
    GreatestExpr,
    IndexExpr,
    InExpr,
    IsNullExpr,
    LeastExpr,
    LiteralExpr,
    NotExpr,
    SeriesLiteralExpr,
    SortExpr,
    StructExpr,
    UnresolvedLiteralExpr,
)
from fenic.core._serde.proto.expression_serde import (
    _deserialize_logical_expr_helper,
    serialize_logical_expr,
)
from fenic.core._serde.proto.serde_context import SerdeContext
from fenic.core._serde.proto.types import (
    AliasExprProto,
    ArrayExprProto,
    CastExprProto,
    CoalesceExprProto,
    ColumnExprProto,
    FlattenExprProto,
    GreatestExprProto,
    IndexExprProto,
    InExprProto,
    IsNullExprProto,
    LeastExprProto,
    LiteralExprProto,
    LogicalExprProto,
    NotExprProto,
    SeriesLiteralExprProto,
    SortExprProto,
    StructExprProto,
    UnresolvedLiteralExprProto,
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
    return LiteralExpr(
        literal=context.deserialize_scalar_value(SerdeContext.VALUE, logical_proto.value),
        data_type=context.deserialize_data_type(SerdeContext.DATA_TYPE, logical_proto.data_type),
    )


# =============================================================================
# SeriesLiteralExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_series_literal_expr(
    logical: SeriesLiteralExpr, context: SerdeContext
) -> LogicalExprProto:
    """Serialize SeriesLiteralExpr by converting Polars Series to bytes via Arrow IPC."""
    # Serialize the Series to Arrow IPC binary format
    # We convert to DataFrame first since write_ipc is a DataFrame method
    buffer = BytesIO()
    logical.series.to_frame().write_ipc(buffer)

    return LogicalExprProto(
        series_literal=SeriesLiteralExprProto(
            series_data=buffer.getvalue(),
            data_type=context.serialize_data_type(SerdeContext.DATA_TYPE, logical.data_type),
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_series_literal_expr(
    logical_proto: SeriesLiteralExprProto, context: SerdeContext
) -> SeriesLiteralExpr:
    """Deserialize SeriesLiteralExpr from bytes back to Polars Series."""
    # Deserialize the bytes back to a DataFrame, then extract the first column as Series
    buffer = BytesIO(logical_proto.series_data)
    df = pl.read_ipc(buffer)
    series = df.to_series()

    return SeriesLiteralExpr(series=series)


# =============================================================================
# UnresolvedLiteralExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_unresolved_literal_expr(
    logical: UnresolvedLiteralExpr, context: SerdeContext
) -> LogicalExprProto:
    return LogicalExprProto(
        unresolved_literal=UnresolvedLiteralExprProto(
            parameter_name=logical.parameter_name,
            data_type=context.serialize_data_type(SerdeContext.DATA_TYPE, logical.data_type),
        )
    )

@_deserialize_logical_expr_helper.register
def _deserialize_unresolved_literal_expr(
    logical_proto: UnresolvedLiteralExprProto, context: SerdeContext
) -> UnresolvedLiteralExpr:
    return UnresolvedLiteralExpr(
        parameter_name=logical_proto.parameter_name,
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
            nulls_last=logical.nulls_last,
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_sort_expr(
    logical_proto: SortExprProto, context: SerdeContext
) -> SortExpr:
    return SortExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        ascending=logical_proto.ascending,
        nulls_last=logical_proto.nulls_last,
    )


# =============================================================================
# IndexExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_index_expr(
    logical: IndexExpr, context: SerdeContext
) -> LogicalExprProto:
    return LogicalExprProto(index=IndexExprProto(
        expr=context.serialize_logical_expr(SerdeContext.EXPR, logical.expr),
        index=context.serialize_logical_expr("index", logical.index),
    ))


@_deserialize_logical_expr_helper.register
def _deserialize_index_expr(
    logical_proto: IndexExprProto, context: SerdeContext
) -> IndexExpr:
    return IndexExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        index=context.deserialize_logical_expr("index", logical_proto.index),
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
            dest_type=context.serialize_data_type("dest_type", logical.dest_type),
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_cast_expr(
    logical_proto: CastExprProto, context: SerdeContext
) -> CastExpr:
    return CastExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        dest_type=context.deserialize_data_type("dest_type", logical_proto.dest_type),
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
            other=context.serialize_logical_expr(SerdeContext.OTHER, logical.other),
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_in_expr(logical_proto: InExprProto, context: SerdeContext) -> InExpr:
    return InExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        other=context.deserialize_logical_expr(SerdeContext.OTHER, logical_proto.other),
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
            expr=context.serialize_logical_expr(SerdeContext.EXPR, logical.expr),
            is_null=logical.is_null,
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_is_null_expr(
    logical_proto: IsNullExprProto, context: SerdeContext
) -> IsNullExpr:
    return IsNullExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        is_null=logical_proto.is_null,
    )


# =============================================================================
# FlattenExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_flatten_expr(
    logical: FlattenExpr, context: SerdeContext
) -> LogicalExprProto:
    return LogicalExprProto(
        flatten=FlattenExprProto(
            expr=context.serialize_logical_expr(SerdeContext.EXPR, logical.expr)
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_flatten_expr(
    logical_proto: FlattenExprProto, context: SerdeContext
) -> FlattenExpr:
    return FlattenExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr)
    )


# =============================================================================
# GreatestExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_greatest_expr(logical: GreatestExpr, context: SerdeContext) -> LogicalExprProto:
    return LogicalExprProto(
        greatest=GreatestExprProto(
            exprs=context.serialize_logical_expr_list(SerdeContext.EXPRS, logical.exprs)
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_greatest_expr(
    logical_proto: GreatestExprProto, context: SerdeContext
) -> GreatestExpr:
    return GreatestExpr(
        exprs=context.deserialize_logical_expr_list(SerdeContext.EXPRS, logical_proto.exprs)
    )

# =============================================================================
# LeastExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_least_expr(logical: LeastExpr, context: SerdeContext) -> LogicalExprProto:
    return LogicalExprProto(
        least=LeastExprProto(
            exprs=context.serialize_logical_expr_list(SerdeContext.EXPRS, logical.exprs)
        )
    )

@_deserialize_logical_expr_helper.register
def _deserialize_least_expr(
    logical_proto: LeastExprProto, context: SerdeContext
) -> LeastExpr:
    return LeastExpr(
        exprs=context.deserialize_logical_expr_list(SerdeContext.EXPRS, logical_proto.exprs)
    )


# Import array serde module to trigger registration
import fenic.core._serde.proto.expressions.array  # noqa: E402, F401
