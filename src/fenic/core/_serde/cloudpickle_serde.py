"""CloudPickle-based implementation of LogicalPlan serialization."""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import cloudpickle  # nosec: B403

from fenic.core._interfaces.session_state import BaseSessionState

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
    def deserialize(data: bytes, _: Optional[BaseSessionState] = None) -> LogicalPlan:
        """Deserialize bytes back into a LogicalPlan using cloudpickle.

        Args:
            data: The serialized plan data
            session: The session data with which to rehydrate the LogicalPlan

        Returns:
            The deserialized plan
        """
        deserialized: LogicalPlan = cloudpickle.loads(data)  # nosec: B301
        return deserialized

    @staticmethod
    def build_logical_plan_with_session_state(
        plan: LogicalPlan, session: BaseSessionState
    ) -> LogicalPlan:
        """Build a LogicalPlan with the session state.

        Args:
            plan: The LogicalPlan to build
            session: The session state

        Returns:
            LogicalPlan with session state restored
        """
        new_children = []
        for child in plan.children():
            new_children.append(
                CloudPickleSerde.build_logical_plan_with_session_state(child, session)
            )
        plan.session_state = session
        return plan.with_children(new_children)