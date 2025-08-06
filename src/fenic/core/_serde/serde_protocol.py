"""Protocol for serialization/deserialization of LogicalPlan objects."""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from fenic.core._logical_plan.plans.base import LogicalPlan


@runtime_checkable
class SupportsLogicalPlanSerde(Protocol):
    """Protocol for LogicalPlan serialization."""

    @staticmethod
    def serialize(plan: LogicalPlan) -> bytes:
        """Serialize a LogicalPlan to bytes.

        Removes any local session state refs from the plan.

        Args:
            plan: The LogicalPlan to serialize

        Returns:
            bytes: The serialized plan
        """
        ...

    @staticmethod
    def deserialize(data: bytes) -> LogicalPlan:
        """Deserialize bytes back into a LogicalPlan.

        Args:
            data: The serialized plan data

        Returns:
            The deserialized plan
        """
        ...