"""CloudPickle-based implementation of LogicalPlan serialization."""
from __future__ import annotations

from typing import TYPE_CHECKING

import cloudpickle  # nosec: B403

if TYPE_CHECKING:
    from fenic.core._logical_plan.plans.base import LogicalPlan
from fenic.core._serde.serde_protocol import SupportsLogicalPlanSerde


class CloudPickleSerde(SupportsLogicalPlanSerde):
    """CloudPickle-based LogicalPlan serialization implementation."""

    @staticmethod
    def serialize(plan: LogicalPlan) -> bytes:
        """Serialize a LogicalPlan to bytes using cloudpickle.

        Args:
            plan: The LogicalPlan to serialize

        Returns:
            bytes: The serialized plan
        """
        return cloudpickle.dumps(plan)

    @staticmethod
    def deserialize(data: bytes) -> LogicalPlan:
        """Deserialize bytes back into a LogicalPlan using cloudpickle.

        Args:
            data: The serialized plan data

        Returns:
            The deserialized plan
        """
        deserialized: LogicalPlan = cloudpickle.loads(data)  # nosec: B301
        return deserialized
