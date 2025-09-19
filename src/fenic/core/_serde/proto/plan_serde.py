"""Logical plan serialization/deserialization using singledispatch.

This module provides the main dispatch functions for plan serialization.
The actual serialization implementations are organized in the plans/ subdirectory.
"""

from functools import singledispatch
from typing import Optional

from google.protobuf.message import Message

from fenic.core._logical_plan.plans.base import CacheInfo, LogicalPlan
from fenic.core._serde.proto.errors import (
    DeserializationError,
    SerializationError,
)
from fenic.core._serde.proto.serde_context import SerdeContext
from fenic.core._serde.proto.types import CacheInfoProto, LogicalPlanProto


@singledispatch
def _serialize_logical_plan_helper(
    logical_plan: LogicalPlan, context: SerdeContext
) -> LogicalPlanProto:
    """Serialize a logical plan to a wrapper LogicalPlanProto (plan oneof set).

    Plan-specific modules register implementations for this helper to return the
    LogicalPlanProto with the appropriate oneof field set. Common fields (schema/cache_info)
    are added by serialize_logical_plan().
    """
    raise context.create_serde_error(
        SerializationError,
        f"Serialization not implemented for Logical Plan: {type(logical_plan)}",
        type(logical_plan),
    )


def serialize_logical_plan(
    logical_plan: LogicalPlan, context: SerdeContext
) -> LogicalPlanProto:
    """Serialize a logical plan to the wrapper LogicalPlanProto.

    This calls the helper to get a wrapper with the correct oneof set, then adds
    common fields like schema and cache_info.
    """
    wrapper: LogicalPlanProto = _serialize_logical_plan_helper(logical_plan, context)

    # Add/overwrite schema
    wrapper.schema.CopyFrom(context.serialize_fenic_schema(logical_plan.schema()))

    # Add cache_info if present
    if logical_plan.cache_info is not None:
        cache_info_proto = CacheInfoProto()
        if logical_plan.cache_info.cache_key is not None:
            cache_info_proto.cache_key = logical_plan.cache_info.cache_key
        wrapper.cache_info.CopyFrom(cache_info_proto)

    return wrapper


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

    Returns:
        LogicalPlan: The deserialized logical plan, or None if empty.

    Raises:
        DeserializationError: If the protobuf is invalid or deserialization fails.
    """
    which_oneof = logical_plan_proto.WhichOneof("plan_type")
    if not which_oneof:  # Optional LogicalPlan arg
        return None
    underlying_proto = getattr(logical_plan_proto, which_oneof)

    # Deserialize the specific plan type with schema from base level
    plan = _deserialize_logical_plan_helper(
        underlying_proto,
        context,
        context.deserialize_fenic_schema(logical_plan_proto.schema)
    )

    # Set cache_info if present
    if logical_plan_proto.HasField("cache_info"):
        cache_info = CacheInfo(
            cache_key=logical_plan_proto.cache_info.cache_key if logical_plan_proto.cache_info.HasField("cache_key") else None
        )
        plan.set_cache_info(cache_info)

    return plan


@singledispatch
def _deserialize_logical_plan_helper(
    underlying_proto: Message,
    context: SerdeContext,
    schema,
) -> Optional[LogicalPlan]:
    """Deserialize a logical plan."""
    raise context.create_serde_error(
        DeserializationError,
        f"Deserialization not implemented for Logical Plan: {type(underlying_proto)}",
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
