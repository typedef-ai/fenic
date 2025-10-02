"""Expression serialization/deserialization using singledispatch.

This module provides the main dispatch functions for expression serialization.
The actual serialization implementations are organized in the expressions/ subdirectory.
"""

from functools import singledispatch
from typing import Optional

from google.protobuf.message import Message

from fenic.core._logical_plan.expressions.base import LogicalExpr
from fenic.core._serde.proto.errors import (
    DeserializationError,
    SerializationError,
)
from fenic.core._serde.proto.serde_context import SerdeContext
from fenic.core._serde.proto.types import LogicalExprProto


@singledispatch
def serialize_logical_expr(
    logical: LogicalExpr, context: SerdeContext
) -> LogicalExprProto:
    """Serialize a logical expression to protobuf format.

    This function uses singledispatch to handle different logical expression types.
    Each expression type should have a corresponding register function that implements
    the specific serialization logic.

    Args:
        logical: The logical expression to serialize.
        context: The serde context for error reporting and path tracking.

    Returns:
        LogicalExprProto: The serialized protobuf representation.

    Raises:
        SerializationError: If the expression type is not registered or serialization fails.
    """
    if logical is None:
        return LogicalExprProto()
    raise context.create_serde_error(
        SerializationError,
        f"Serialization not implemented for LogicalExpr: {type(logical)}",
        type(logical),
    )


def deserialize_logical_expr(
    logical_proto: LogicalExprProto,
    context: SerdeContext,
) -> Optional[LogicalExpr]:
    """Deserialize a logical expression from protobuf format.

    This function determines which oneof field is set in the LogicalExprProto
    and delegates to the appropriate deserialization helper function.

    Args:
        logical_proto: The protobuf representation to deserialize.
        context: The serde context for error reporting and path tracking.

    Returns:
        LogicalExpr: The deserialized logical expression, or None if empty.

    Raises:
        DeserializationError: If the protobuf is invalid or deserialization fails.
    """
    which_oneof = logical_proto.WhichOneof("expr_type")
    if not which_oneof:  # Optional LogicalExpr argument not populated
        return None
    underlying_proto = getattr(logical_proto, which_oneof)
    return _deserialize_logical_expr_helper(underlying_proto, context)


@singledispatch
def _deserialize_logical_expr_helper(
    logical_proto: Message, context: SerdeContext
) -> LogicalExpr:
    """Deserialize a logical expression helper.

    The actual deserialization implementations are registered in the expressions/ submodules.
    """
    raise context.create_serde_error(
        DeserializationError,
        f"Deserialization not implemented for LogicalExpr: {type(logical_proto)}",
    )


# Import all expression modules to register their serialization functions
# This must be done after the main functions are defined
from fenic.core._serde.proto.expressions import (  # noqa: F401, E402
    aggregate,
    basic,
    binary,
    case,
    dt,
    embedding,
    json,
    markdown,
    semantic,
    text,
    unserializable,
)