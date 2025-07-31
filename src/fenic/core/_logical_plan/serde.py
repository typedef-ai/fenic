import cloudpickle  # nosec: B403

from fenic.core._interfaces.session_state import BaseSessionState
from fenic.core._logical_plan.plans.base import LogicalPlan


class LogicalPlanSerde:
    @staticmethod
    def serialize(plan: LogicalPlan) -> bytes:
        """Serialize a LogicalPlan to bytes using pickle.

        Args:
            plan: The LogicalPlan to serialize

        Returns:
            bytes: The serialized plan
        """
        return cloudpickle.dumps(plan)

    @staticmethod
    def deserialize(data: bytes) -> LogicalPlan:
        """Deserialize bytes back into a LogicalPlan using pickle.

        Args:
            data: The serialized plan data

        Returns:
            The deserialized plan
        """
        return cloudpickle.loads(data)  # nosec: B301

    @staticmethod
    def build_logical_plan_with_session_state(
        plan: LogicalPlan, session: BaseSessionState
    ) -> LogicalPlan:
        """Build a LogicalPlan with the session state.

        Args:
            plan: The LogicalPlan to build
            session: The session state
        """
        # TODO(DY): replace pickle with substrait so we don't need this step
        new_children = []
        for child in plan.children():
            new_children.append(
                LogicalPlanSerde.build_logical_plan_with_session_state(child, session)
            )
        plan.session_state = session
        return plan.with_children(new_children)
