"""Array expression serialization/deserialization."""

from fenic.core._logical_plan.expressions.array import (
    ArrayCompactExpr,
    ArrayContainsExpr,
    ArrayDistinctExpr,
    ArrayExceptExpr,
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
    ElementAtExpr,
)
from fenic.core._serde.proto.expression_serde import (
    _deserialize_logical_expr_helper,
    serialize_logical_expr,
)
from fenic.core._serde.proto.serde_context import SerdeContext
from fenic.core._serde.proto.types import (
    ArrayCompactExprProto,
    ArrayContainsExprProto,
    ArrayDistinctExprProto,
    ArrayExceptExprProto,
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
    ElementAtExprProto,
    LogicalExprProto,
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
