"""Main API for logical plan and expression serialization/deserialization."""

from typing import Optional

from fenic.core._interfaces.session_state import BaseSessionState
from fenic.core._logical_plan.plans.base import LogicalPlan
from fenic.core._serde.proto.serde_context import create_serde_context
from fenic.core._serde.proto.types import LogicalPlanProto
from fenic.core._serde.serde_protocol import SupportsLogicalPlanSerde


class ProtoSerde(SupportsLogicalPlanSerde):
    """Proto Serde implementation.

    This implementation uses the Protobuf specs defined in the `protos` package to serialize
    and deserialize logical plans. Provides the main API for converting between LogicalPlan
    objects and their binary protobuf representation.
    """

    @staticmethod
    def serialize(logical_plan: LogicalPlan) -> bytes:
        """Serialize a logical plan to binary protobuf format.

        Args:
            logical_plan: The logical plan to serialize.

        Returns:
            Binary protobuf representation of the logical plan.
        """
        context = create_serde_context()
        logical_plan_proto = context.serialize_logical_plan("root", logical_plan)
        return logical_plan_proto.SerializeToString()

    @staticmethod
    def deserialize(
        data: bytes, session_state: Optional[BaseSessionState] = None
    ) -> LogicalPlan:
        """Deserialize a logical plan from binary protobuf format.

        Args:
            data: Binary protobuf data to deserialize.
            session_state: Optional session state to include in the plan.

        Returns:
            The deserialized logical plan.
        """
        context = create_serde_context()
        logical_plan_proto = LogicalPlanProto.FromString(data)
        logical_plan = context.deserialize_logical_plan(
            "root", logical_plan_proto, session_state
        )
        return logical_plan

    @staticmethod
    def build_logical_plan_with_session_state(
        plan: LogicalPlan, session: BaseSessionState
    ) -> LogicalPlan:
        """Deserialize bytes back into a LogicalPlan.

        Args:
            session: Session to add into the plan
            plan: LogicalPlan to add session state to.

        Returns:
            The deserialized plan
        """
        return plan  # no-op for proto serde, which builds the session state into the plan as it is deserialized
