"""Binary expression serialization/deserialization."""

from fenic._gen.protos.logical_plan.v1.enums_pb2 import Operator as OperatorProto
from fenic.core._logical_plan.expressions.arithmetic import ArithmeticExpr
from fenic.core._logical_plan.expressions.base import Operator
from fenic.core._logical_plan.expressions.comparison import (
    BooleanExpr,
    EqualityComparisonExpr,
    NumericComparisonExpr,
)
from fenic.core._serde.proto.expression_serde import (
    _deserialize_logical_expr_helper,
    serialize_logical_expr,
)
from fenic.core._serde.proto.serde_context import SerdeContext
from fenic.core._serde.proto.types import (
    ArithmeticExprProto,
    BooleanExprProto,
    EqualityComparisonExprProto,
    LogicalExprProto,
    NumericComparisonExprProto,
)

# =============================================================================
# ArithmeticExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_arithmetic_expr(
    logical: ArithmeticExpr, context: SerdeContext
) -> LogicalExprProto:
    return LogicalExprProto(
        arithmetic=ArithmeticExprProto(
            left=context.serialize_logical_expr(SerdeContext.LEFT, logical.left),
            right=context.serialize_logical_expr(SerdeContext.RIGHT, logical.right),
            operator=context.serialize_enum_value("operator", logical.op, OperatorProto),
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_arithmetic_expr(
    logical_proto: ArithmeticExprProto, context: SerdeContext
) -> ArithmeticExpr:
    """Deserialize an arithmetic expression."""
    return ArithmeticExpr(
        left=context.deserialize_logical_expr(SerdeContext.LEFT, logical_proto.left),
        right=context.deserialize_logical_expr(SerdeContext.RIGHT, logical_proto.right),
        op=context.deserialize_enum_value("operator", Operator, OperatorProto, logical_proto.operator),
    )


# =============================================================================
# BooleanExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_boolean_expr(
    logical: BooleanExpr, context: SerdeContext
) -> LogicalExprProto:
    return LogicalExprProto(
        boolean=BooleanExprProto(
            left=context.serialize_logical_expr(SerdeContext.LEFT, logical.left),
            right=context.serialize_logical_expr(SerdeContext.RIGHT, logical.right),
            operator=context.serialize_enum_value(SerdeContext.OPERATOR, logical.op, OperatorProto),
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_boolean_expr(
    logical_proto: BooleanExprProto, context: SerdeContext
) -> BooleanExpr:
    """Deserialize a boolean expression."""
    return BooleanExpr(
        left=context.deserialize_logical_expr(SerdeContext.LEFT, logical_proto.left),
        right=context.deserialize_logical_expr(SerdeContext.RIGHT, logical_proto.right),
        op=context.deserialize_enum_value(SerdeContext.OPERATOR, Operator, OperatorProto, logical_proto.operator),
    )


# =============================================================================
# NumericComparisonExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_numeric_comparison_expr(
    logical: NumericComparisonExpr, context: SerdeContext
) -> LogicalExprProto:
    return LogicalExprProto(
        numeric_comparison=NumericComparisonExprProto(
            left=context.serialize_logical_expr(SerdeContext.LEFT, logical.left),
            right=context.serialize_logical_expr(SerdeContext.RIGHT, logical.right),
            operator=context.serialize_enum_value("operator", logical.op, OperatorProto),
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_numeric_comparison_expr(
    logical_proto: NumericComparisonExprProto, context: SerdeContext
) -> NumericComparisonExpr:
    """Deserialize a numeric comparison expression."""
    return NumericComparisonExpr(
        left=context.deserialize_logical_expr(SerdeContext.LEFT, logical_proto.left),
        right=context.deserialize_logical_expr(SerdeContext.RIGHT, logical_proto.right),
        op=context.deserialize_enum_value("operator", Operator, OperatorProto, logical_proto.operator),
    )


# =============================================================================
# EqualityComparisonExpr
# =============================================================================


@serialize_logical_expr.register
def _serialize_equality_comparison_expr(
    logical: EqualityComparisonExpr, context: SerdeContext
) -> LogicalExprProto:
    """Serialize an equality comparison expression."""
    return LogicalExprProto(
        equality_comparison=EqualityComparisonExprProto(
            left=context.serialize_logical_expr(SerdeContext.LEFT, logical.left),
            right=context.serialize_logical_expr(SerdeContext.RIGHT, logical.right),
            operator=context.serialize_enum_value("operator", logical.op, OperatorProto),
        )
    )


@_deserialize_logical_expr_helper.register
def _deserialize_equality_comparison_expr(
    logical_proto: EqualityComparisonExprProto, context: SerdeContext
) -> EqualityComparisonExpr:
    """Deserialize an equality comparison expression."""
    return EqualityComparisonExpr(
        left=context.deserialize_logical_expr(SerdeContext.LEFT, logical_proto.left),
        right=context.deserialize_logical_expr(SerdeContext.RIGHT, logical_proto.right),
        op=context.deserialize_enum_value("operator", Operator, OperatorProto, logical_proto.operator),
    )
