from __future__ import annotations

from fenic.core._interfaces.session_state import BaseSessionState
from fenic.core._logical_plan.plans.node import CacheInfo, LogicalPlanNode
from fenic.core.error import SessionError
from fenic.core.types.schema import Schema


class LogicalPlan:
    def __init__(self, session_state: BaseSessionState):
        self.logical_plan_node: LogicalPlanNode = None
        self.session_state: BaseSessionState = session_state

    def add_node(self, node: LogicalPlanNode) -> LogicalPlan:
        new_logical_plan = LogicalPlan(self.session_state)

        if self.logical_plan_node:
            node.set_input(self.logical_plan_node)

        node._build_schema_with_validation(self.session_state)
        new_logical_plan.logical_plan_node = node
        return new_logical_plan

    def __str__(self) -> str:
        return str(self.logical_plan_node)

    def set_cache_info(self, cache_info: CacheInfo):
        """Set the cache metadata for this plan."""
        self.logical_plan_node.set_cache_info(cache_info)

    def schema(self) -> Schema:
        return self.logical_plan_node.schema()

def ensure_same_session(lhs: BaseSessionState, rhs: BaseSessionState):
    """Ensure that two LogicalPlans belong to the same session context.

    This check prevents accidental combinations of DataFrames created in different
    sessions, which can lead to inconsistent behavior due to differing configurations,
    catalogs, or function registries.
    """
    if lhs is not rhs:
        raise SessionError(
            "Cannot combine DataFrames created in different sessions. "
            "This operation requires all inputs to belong to the same session context. "
            "Make sure that you're not mixing DataFrames from different interactive environments, notebooks, or clients."
        )
