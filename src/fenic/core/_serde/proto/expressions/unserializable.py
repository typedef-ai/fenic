from fenic.core._logical_plan.expressions.basic import UDFExpr
from fenic.core._serde.proto.errors import UnsupportedTypeError
from fenic.core._serde.proto.expression_serde import serialize_logical_expr
from fenic.core._serde.proto.serde_context import SerdeContext
from fenic.core._serde.proto.types import LogicalExprProto


@serialize_logical_expr.register
def _serialize_udf_expr(logical: UDFExpr, context: SerdeContext) -> LogicalExprProto:
    """Serialize a UDF expression."""
    raise UnsupportedTypeError(UDFExpr, "UDFExpr is not currently supported for serde")
