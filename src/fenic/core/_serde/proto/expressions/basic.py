"""Basic expression serialization/deserialization."""

from fenic.core._logical_plan.expressions.basic import (
    AliasExpr,
    ArrayCompactExpr,
    ArrayContainsExpr,
    ArrayDistinctExpr,
    ArrayExceptExpr,
    ArrayExpr,
    ArrayIntersectExpr,
    ArrayLengthExpr,
    ArrayMaxExpr,
    ArrayMinExpr,
    ArrayRemoveExpr,
    ArrayRepeatExpr,
    ArrayReverseExpr,
    ArraySliceExpr,
    ArraySortExpr,
    ArraysOverlapExpr,
    ArrayUnionExpr,
    CastExpr,
    CoalesceExpr,
    ColumnExpr,
    ElementAtExpr,
    FlattenExpr,
    GreatestExpr,
    IndexExpr,
    InExpr,
    IsNullExpr,
    LeastExpr,
    LiteralExpr,
    NotExpr,
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
    ArrayCompactExprProto,
    ArrayContainsExprProto,
    ArrayDistinctExprProto,
    ArrayExceptExprProto,
    ArrayExprProto,
    ArrayIntersectExprProto,
    ArrayLengthExprProto,
    ArrayMaxExprProto,
    ArrayMinExprProto,
    ArrayRemoveExprProto,
    ArrayRepeatExprProto,
    ArrayReverseExprProto,
    ArraySliceExprProto,
    ArraySortExprProto,
    ArraysOverlapExprProto,
    ArrayUnionExprProto,
    CastExprProto,
    CoalesceExprProto,
    ColumnExprProto,
    ElementAtExprProto,
    FlattenExprProto,
    GreatestExprProto,
    IndexExprProto,
    InExprProto,
    IsNullExprProto,
    LeastExprProto,
    LiteralExprProto,
    LogicalExprProto,
    NotExprProto,
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
    from fenic.core._logical_plan.expressions.basic import LiteralExpr

    return LiteralExpr(
        literal=context.deserialize_scalar_value(SerdeContext.VALUE, logical_proto.value),
        data_type=context.deserialize_data_type(SerdeContext.DATA_TYPE, logical_proto.data_type),
    )

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
# ArrayDistinctExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_array_distinct_expr(
    logical: ArrayDistinctExpr, context: SerdeContext
) -> LogicalExprProto:
    return LogicalExprProto(
        array_distinct=ArrayDistinctExprProto(
            expr=context.serialize_logical_expr(SerdeContext.EXPR, logical.expr)
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_array_distinct_expr(
    logical_proto: ArrayDistinctExprProto, context: SerdeContext
) -> ArrayDistinctExpr:
    return ArrayDistinctExpr(
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
            other=context.serialize_logical_expr(SerdeContext.OTHER, logical.other),
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_array_contains_expr(
    logical_proto: ArrayContainsExprProto, context: SerdeContext
) -> ArrayContainsExpr:
    return ArrayContainsExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        other=context.deserialize_logical_expr(SerdeContext.OTHER, logical_proto.other),
    )


# =============================================================================
# ArrayMaxExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_array_max_expr(
    logical: ArrayMaxExpr, context: SerdeContext
) -> LogicalExprProto:
    return LogicalExprProto(
        array_max=ArrayMaxExprProto(
            expr=context.serialize_logical_expr(SerdeContext.EXPR, logical.expr)
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_array_max_expr(
    logical_proto: ArrayMaxExprProto, context: SerdeContext
) -> ArrayMaxExpr:
    return ArrayMaxExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr)
    )


# =============================================================================
# ArrayMinExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_array_min_expr(
    logical: ArrayMinExpr, context: SerdeContext
) -> LogicalExprProto:
    return LogicalExprProto(
        array_min=ArrayMinExprProto(
            expr=context.serialize_logical_expr(SerdeContext.EXPR, logical.expr)
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_array_min_expr(
    logical_proto: ArrayMinExprProto, context: SerdeContext
) -> ArrayMinExpr:
    return ArrayMinExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr)
    )


# =============================================================================
# ArraySortExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_array_sort_expr(
    logical: ArraySortExpr, context: SerdeContext
) -> LogicalExprProto:
    return LogicalExprProto(
        array_sort=ArraySortExprProto(
            expr=context.serialize_logical_expr(SerdeContext.EXPR, logical.expr)
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_array_sort_expr(
    logical_proto: ArraySortExprProto, context: SerdeContext
) -> ArraySortExpr:
    return ArraySortExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr)
    )


# =============================================================================
# ArrayReverseExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_array_reverse_expr(
    logical: ArrayReverseExpr, context: SerdeContext
) -> LogicalExprProto:
    return LogicalExprProto(
        array_reverse=ArrayReverseExprProto(
            expr=context.serialize_logical_expr(SerdeContext.EXPR, logical.expr)
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_array_reverse_expr(
    logical_proto: ArrayReverseExprProto, context: SerdeContext
) -> ArrayReverseExpr:
    return ArrayReverseExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr)
    )


# =============================================================================
# ArrayRemoveExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_array_remove_expr(
    logical: ArrayRemoveExpr, context: SerdeContext
) -> LogicalExprProto:
    return LogicalExprProto(
        array_remove=ArrayRemoveExprProto(
            expr=context.serialize_logical_expr(SerdeContext.EXPR, logical.expr),
            element=context.serialize_logical_expr(SerdeContext.EXPR, logical.element),
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_array_remove_expr(
    logical_proto: ArrayRemoveExprProto, context: SerdeContext
) -> ArrayRemoveExpr:
    return ArrayRemoveExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        element=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.element),
    )


# =============================================================================
# ArrayUnionExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_array_union_expr(
    logical: ArrayUnionExpr, context: SerdeContext
) -> LogicalExprProto:
    return LogicalExprProto(
        array_union=ArrayUnionExprProto(
            left=context.serialize_logical_expr(SerdeContext.LEFT, logical.left),
            right=context.serialize_logical_expr(SerdeContext.RIGHT, logical.right),
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_array_union_expr(
    logical_proto: ArrayUnionExprProto, context: SerdeContext
) -> ArrayUnionExpr:
    return ArrayUnionExpr(
        left=context.deserialize_logical_expr(SerdeContext.LEFT, logical_proto.left),
        right=context.deserialize_logical_expr(SerdeContext.RIGHT, logical_proto.right),
    )


# =============================================================================
# ArrayIntersectExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_array_intersect_expr(
    logical: ArrayIntersectExpr, context: SerdeContext
) -> LogicalExprProto:
    return LogicalExprProto(
        array_intersect=ArrayIntersectExprProto(
            left=context.serialize_logical_expr(SerdeContext.LEFT, logical.left),
            right=context.serialize_logical_expr(SerdeContext.RIGHT, logical.right),
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_array_intersect_expr(
    logical_proto: ArrayIntersectExprProto, context: SerdeContext
) -> ArrayIntersectExpr:
    return ArrayIntersectExpr(
        left=context.deserialize_logical_expr(SerdeContext.LEFT, logical_proto.left),
        right=context.deserialize_logical_expr(SerdeContext.RIGHT, logical_proto.right),
    )


# =============================================================================
# ArrayExceptExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_array_except_expr(
    logical: ArrayExceptExpr, context: SerdeContext
) -> LogicalExprProto:
    return LogicalExprProto(
        array_except=ArrayExceptExprProto(
            left=context.serialize_logical_expr(SerdeContext.LEFT, logical.left),
            right=context.serialize_logical_expr(SerdeContext.RIGHT, logical.right),
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_array_except_expr(
    logical_proto: ArrayExceptExprProto, context: SerdeContext
) -> ArrayExceptExpr:
    return ArrayExceptExpr(
        left=context.deserialize_logical_expr(SerdeContext.LEFT, logical_proto.left),
        right=context.deserialize_logical_expr(SerdeContext.RIGHT, logical_proto.right),
    )


# =============================================================================
# ArrayCompactExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_array_compact_expr(
    logical: ArrayCompactExpr, context: SerdeContext
) -> LogicalExprProto:
    return LogicalExprProto(
        array_compact=ArrayCompactExprProto(
            expr=context.serialize_logical_expr(SerdeContext.EXPR, logical.expr)
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_array_compact_expr(
    logical_proto: ArrayCompactExprProto, context: SerdeContext
) -> ArrayCompactExpr:
    return ArrayCompactExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr)
    )


# =============================================================================
# ArrayRepeatExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_array_repeat_expr(
    logical: ArrayRepeatExpr, context: SerdeContext
) -> LogicalExprProto:
    return LogicalExprProto(
        array_repeat=ArrayRepeatExprProto(
            element=context.serialize_logical_expr(SerdeContext.EXPR, logical.element),
            count=context.serialize_logical_expr(SerdeContext.EXPR, logical.count),
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_array_repeat_expr(
    logical_proto: ArrayRepeatExprProto, context: SerdeContext
) -> ArrayRepeatExpr:
    return ArrayRepeatExpr(
        element=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.element),
        count=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.count),
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
# ArraySliceExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_array_slice_expr(
    logical: ArraySliceExpr, context: SerdeContext
) -> LogicalExprProto:
    return LogicalExprProto(
        array_slice=ArraySliceExprProto(
            expr=context.serialize_logical_expr(SerdeContext.EXPR, logical.expr),
            start=context.serialize_logical_expr(SerdeContext.EXPR, logical.start),
            length=context.serialize_logical_expr(SerdeContext.EXPR, logical.length),
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_array_slice_expr(
    logical_proto: ArraySliceExprProto, context: SerdeContext
) -> ArraySliceExpr:
    return ArraySliceExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        start=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.start),
        length=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.length),
    )


# =============================================================================
# ElementAtExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_element_at_expr(
    logical: ElementAtExpr, context: SerdeContext
) -> LogicalExprProto:
    return LogicalExprProto(
        element_at=ElementAtExprProto(
            expr=context.serialize_logical_expr(SerdeContext.EXPR, logical.expr),
            index=context.serialize_logical_expr(SerdeContext.EXPR, logical.index),
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_element_at_expr(
    logical_proto: ElementAtExprProto, context: SerdeContext
) -> ElementAtExpr:
    return ElementAtExpr(
        expr=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.expr),
        index=context.deserialize_logical_expr(SerdeContext.EXPR, logical_proto.index),
    )


# =============================================================================
# ArraysOverlapExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_arrays_overlap_expr(
    logical: ArraysOverlapExpr, context: SerdeContext
) -> LogicalExprProto:
    return LogicalExprProto(
        arrays_overlap=ArraysOverlapExprProto(
            left=context.serialize_logical_expr(SerdeContext.LEFT, logical.left),
            right=context.serialize_logical_expr(SerdeContext.RIGHT, logical.right),
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_arrays_overlap_expr(
    logical_proto: ArraysOverlapExprProto, context: SerdeContext
) -> ArraysOverlapExpr:
    return ArraysOverlapExpr(
        left=context.deserialize_logical_expr(SerdeContext.LEFT, logical_proto.left),
        right=context.deserialize_logical_expr(SerdeContext.RIGHT, logical_proto.right),
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
