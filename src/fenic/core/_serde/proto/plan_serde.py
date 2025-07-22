"""Logical plan serialization/deserialization using singledispatch.

This module provides the main dispatch functions for plan serialization.
The actual serialization implementations are organized in the plans/ subdirectory.
"""

from functools import singledispatch
from typing import Optional, Type, TypeVar

from google.protobuf.message import Message

from fenic.core._interfaces.session_state import BaseSessionState
from fenic.core._logical_plan.plans.base import LogicalPlan
from fenic.core._serde.proto.errors import (
    DeserializationError,
    SerializationError,
)
from fenic.core._serde.proto.serde_context import SerdeContext
from fenic.core._serde.proto.types import LogicalPlanProto

LogicalPlanType = TypeVar("LogicalPlanType", bound=LogicalPlan)


@singledispatch
def serialize_logical_plan(
    logical_plan: LogicalPlan, context: SerdeContext
) -> LogicalPlanProto:
    """Serialize a logical plan."""
    raise SerializationError(f"Serialization not implemented for {type(logical_plan)}")


def deserialize_logical_plan(
    logical_plan_proto: LogicalPlanProto,
    context: SerdeContext,
    session_state: Optional[BaseSessionState] = None,
    _target_type: Type[LogicalPlanType] = LogicalPlan,
) -> Optional[LogicalPlanType]:
    """Deserialize a logical plan."""
    which_oneof = logical_plan_proto.WhichOneof("plan_type")
    if not which_oneof:  # Optional LogicalPlan arg
        return None
    underlying_proto = getattr(logical_plan_proto, which_oneof)
    return _deserialize_logical_plan_helper(underlying_proto, context, session_state)


@singledispatch
def _deserialize_logical_plan_helper(
    underlying_proto: Message, context: SerdeContext, session_state: Optional[BaseSessionState] = None
) -> LogicalPlan:
    """Deserialize a logical plan."""
    raise DeserializationError(
        f"{context.current_path}: Deserialization not implemented for {type(underlying_proto)}"
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
