"""CloudPickle-based implementation of LogicalPlan serialization."""
from __future__ import annotations

import cloudpickle  # nosec: B403
from typing import Optional, TYPE_CHECKING

from fenic.core._interfaces.session_state import BaseSessionState
if TYPE_CHECKING:
    from fenic.core._logical_plan.plans.base import LogicalPlan
from fenic.core._logical_plan.serde.serde_protocol import SupportsLogicalPlanSerde

class CloudPickleSerde(SupportsLogicalPlanSerde):
    """CloudPickle-based LogicalPlan serialization implementation."""

    @staticmethod
    def serialize(plan: LogicalPlan) -> bytes:
        """Serialize a LogicalPlan to bytes using cloudpickle.

        Removes any local session state refs from the plan.

        Args:
            plan: The LogicalPlan to serialize

        Returns:
            bytes: The serialized plan
        """
        # For now, we need to copy the plan in a bottom-up manner, and then walk it again top-down to remove the session state.
        # We can't nullify the session state during the bottom-up traversal because some plan nodes rely on their children's
        # session state during initialization. Clearing it too early can break this initialization logic.
        # TODO(rohitrastogi): Decouple plan construction logic from plan validation logic.
        def copy_plan(plan: LogicalPlan) -> LogicalPlan:
            new_children = []
            for child in plan.children():
                new_children.append(copy_plan(child))
            return plan.with_children(new_children)

        def remove_session_state(plan: LogicalPlan) -> LogicalPlan:
            plan.session_state = None
            for child in plan.children():
                remove_session_state(child)
            return plan

        copied_plan = copy_plan(plan)
        remove_session_state(copied_plan)
        return cloudpickle.dumps(copied_plan)

    @staticmethod
    def deserialize(data: bytes) -> LogicalPlan:
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