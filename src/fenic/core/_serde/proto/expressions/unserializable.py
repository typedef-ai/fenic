from fenic.core._logical_plan.expressions.basic import AsyncUDFExpr, UDFExpr
from fenic.core._serde.proto.errors import UnsupportedTypeError
from fenic.core._serde.proto.expression_serde import (
    _deserialize_logical_expr_helper,
    serialize_logical_expr,
)
from fenic.core._serde.proto.serde_context import SerdeContext
from fenic.core._serde.proto.types import LogicalExprProto

# =============================================================================
# UDFExpr
# =============================================================================

@serialize_logical_expr.register
def _serialize_udf_expr(logical: UDFExpr, context: SerdeContext) -> LogicalExprProto:
    raise context.create_serde_error(UnsupportedTypeError, "UDFExpr cannot be serialized.", UDFExpr)

@_deserialize_logical_expr_helper.register
def _deserialize_udf_expr(logical_proto: LogicalExprProto, context: SerdeContext) -> UDFExpr:
    # Technically should be unreachable, but we'll raise an error just in case
    raise context.create_serde_error(UnsupportedTypeError, "UDFExpr cannot be deserialized.", UDFExpr)

@serialize_logical_expr.register
def _serialize_async_udf_expr(logical: AsyncUDFExpr, context: SerdeContext) -> LogicalExprProto:
    raise context.create_serde_error(UnsupportedTypeError, "AsyncUDFExpr cannot be serialized.", AsyncUDFExpr)

@_deserialize_logical_expr_helper.register
def _deserialize_async_udf_expr(logical_proto: LogicalExprProto, context: SerdeContext) -> AsyncUDFExpr:
    # Technically should be unreachable, but we'll raise an error just in case
    raise context.create_serde_error(UnsupportedTypeError, "AsyncUDFExpr cannot be deserialized.", AsyncUDFExpr)
