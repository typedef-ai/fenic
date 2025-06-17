from __future__ import annotations

from typing import TYPE_CHECKING

from fenic._backends.local.transpiler.base import BaseTranspiler
from fenic._backends.local.transpiler.plan_converter import (
    PlanConverter,
)

if TYPE_CHECKING:
    from fenic._backends.local.physical_plan import PhysicalPlan
    from fenic._backends.local.session_state import LocalSessionState
    from fenic.core._logical_plan import (
        LogicalPlan,
    )

class LocalTranspiler(BaseTranspiler):

    def __init__(self, session_state: LocalSessionState):
        self.plan_converter = PlanConverter(session_state)

    def transpile(self, logical_plan: LogicalPlan) -> PhysicalPlan:
        return self.plan_converter.convert(logical_plan)
