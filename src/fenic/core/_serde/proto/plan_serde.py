"""Logical plan serialization/deserialization using singledispatch.

This module provides the main dispatch functions for plan serialization.
The actual serialization implementations are organized in the plans/ subdirectory.
"""

from functools import singledispatch
from typing import Optional

from google.protobuf.message import Message

from fenic.core._logical_plan.plans.base import LogicalPlan
from fenic.core._serde.proto.errors import (
    DeserializationError,
    SerializationError,
)
from fenic.core._serde.proto.serde_context import SerdeContext
from fenic.core._serde.proto.types import LogicalPlanProto


@singledispatch
def serialize_logical_plan(
    logical_plan: LogicalPlan, context: SerdeContext
) -> LogicalPlanProto:
    """Serialize a logical plan to protobuf format.

    This function uses singledispatch to handle different logical plan types.
    Each plan type should have a corresponding register function that implements
    the specific serialization logic.

    Args:
        logical_plan: The logical plan to serialize.
        context: The serde context for error reporting and path tracking.

    Returns:
        LogicalPlanProto: The serialized protobuf representation.

    Raises:
        SerializationError: If the plan type is not registered or serialization fails.
    """
    raise context.create_serde_error(
        SerializationError,
        f"Serialization not implemented for {type(logical_plan)}",
        type(logical_plan),
    )


def deserialize_logical_plan(
    logical_plan_proto: LogicalPlanProto,
    context: SerdeContext,
) -> Optional[LogicalPlan]:
    """Deserialize a logical plan from protobuf format.

    This function determines which oneof field is set in the LogicalPlanProto
    and delegates to the appropriate deserialization helper function.

    Args:
        logical_plan_proto: The protobuf representation to deserialize.
        context: The serde context for error reporting and path tracking.
        session_state: Optional session state to include in the plan.

    Returns:
        LogicalPlan: The deserialized logical plan, or None if empty.

    Raises:
        DeserializationError: If the protobuf is invalid or deserialization fails.
    """
    which_oneof = logical_plan_proto.WhichOneof("plan_type")
    if not which_oneof:  # Optional LogicalPlan arg
        return None
    underlying_proto = getattr(logical_plan_proto, which_oneof)
    return _deserialize_logical_plan_helper(underlying_proto, context)


@singledispatch
def _deserialize_logical_plan_helper(
    underlying_proto: Message,
    context: SerdeContext,
) -> Optional[LogicalPlan]:
    """Deserialize a logical plan."""
    raise context.create_serde_error(
        DeserializationError,
        f"Deserialization not implemented for {type(underlying_proto)}",
        type(underlying_proto),
    )


# Import all plan modules to register their serialization functions
# This must be done after the main functions are defined
from fenic.core._serde.proto.plans import (  # noqa: F401 E402
    aggregate,
    join,
    sink,
    source,
    transform,
)
