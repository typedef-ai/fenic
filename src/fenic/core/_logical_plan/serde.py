import cloudpickle  # nosec: B403

from fenic.core._interfaces.session_state import BaseSessionState
from fenic.core._logical_plan.plans.base import LogicalPlan
from fenic.core._logical_plan.plans.node import LogicalPlanNode


class LogicalPlanSerde:
    @staticmethod
    def serialize(plan: LogicalPlan) -> bytes:
        """Serialize a LogicalPlan to bytes using pickle.

        Removes any local session state refs from the plan.

        Args:
            plan: The LogicalPlan to serialize

        Returns:
            bytes: The serialized plan
        """
        # For now, we need to copy the plan in a bottom-up manner.
        def copy_plan(node: LogicalPlanNode) -> LogicalPlanNode:
            new_children = []
            for child in node.children():
                new_children.append(copy_plan(child))
            return node.with_children(new_children)

        # we only want to serialize the logical plan node tree, not the session state.
        copied_logical_plan_node = copy_plan(plan.logical_plan_node)
        return cloudpickle.dumps(copied_logical_plan_node)

    @staticmethod
    def deserialize_into_logical_plan(data: bytes, session_state: BaseSessionState) -> LogicalPlan:
        print("Deserializing into logical plan.")
        logical_plan_node = LogicalPlanSerde.deserialize(data)
        logical_plan = LogicalPlan(session_state)
        logical_plan.logical_plan_node = logical_plan_node
        #logical_plan.update_schema()
        return logical_plan

    @staticmethod
    def deserialize(data: bytes) -> LogicalPlanNode:
        """Deserialize bytes back into a LogicalPlan using pickle.

        Args:
            data: The serialized plan data

        Returns:
            The deserialized plan
        """
        return cloudpickle.loads(data)  # nosec: B301

    @staticmethod
    def build_logical_plan_with_session_state(
        node: LogicalPlanNode, session: BaseSessionState
    ) -> LogicalPlan:
        """Build a LogicalPlan with the session state.

        Args:
            node: The LogicalPlanNode tree to source the plan
            session: The session state
        """
        plan = LogicalPlan(session)
        plan.logical_plan_node = node
        return plan
