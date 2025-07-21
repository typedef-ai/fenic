"""LogicalPlan serialization with pluggable backends."""
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from fenic.core._interfaces.session_state import BaseSessionState
from fenic.core._logical_plan.serde import CloudPickleSerde
from fenic.core._logical_plan.serde.serde_protocol import SupportsLogicalPlanSerde

if TYPE_CHECKING:
    from fenic.core._logical_plan.plans.base import LogicalPlan

_default_serde_type = CloudPickleSerde

# temporary facade-lite until we have the additional serde backends implemented.
class LogicalPlanSerde(SupportsLogicalPlanSerde):
    """Facade for LogicalPlan serialization with pluggable backends."""

    _serde: ClassVar[SupportsLogicalPlanSerde] = _default_serde_type

    @classmethod
    def serialize(cls, plan: LogicalPlan) -> bytes:
        """Serialize a LogicalPlan to bytes."""
        return cls._serde.serialize(plan)

    @classmethod
    def deserialize(
        cls,
        serialized_plan: bytes,
    ) -> LogicalPlan:
        """Deserialize a LogicalPlan from bytes."""
        return cls._serde.deserialize(serialized_plan)


    @classmethod
    def build_logical_plan_with_session_state(
        cls,
        plan: LogicalPlan,
        session: BaseSessionState
    ) -> LogicalPlan:
        """Deserialize a LogicalPlan from bytes."""
        return cls._serde.build_logical_plan_with_session_state(plan, session)