
"""Physical plan optimization framework base classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

from fenic._backends.local.physical_plan.base import PhysicalPlan

if TYPE_CHECKING:
    from fenic._backends.local.session_state import LocalSessionState


@dataclass
class PhysicalPlanOptimizationResult:
    """Result of applying a physical plan optimization rule."""
    plan: PhysicalPlan
    optimized: bool


class PhysicalPlanRule(ABC):
    """Abstract base class for physical plan optimization rules."""

    @abstractmethod
    def apply(self, plan: PhysicalPlan, session_state: LocalSessionState) -> PhysicalPlanOptimizationResult:
        """Apply the optimization rule to the entire plan and return the result."""
        pass


class PhysicalPlanOptimizer:
    """Optimizer for physical execution plans."""

    def __init__(self, session_state: LocalSessionState, rules: List[PhysicalPlanRule]):
        self.session_state = session_state
        self.rules = rules

    def optimize(self, plan: PhysicalPlan) -> PhysicalPlanOptimizationResult:
        """Apply optimization rules to the physical plan."""
        current_plan = plan
        overall_optimized = False

        for rule in self.rules:
            result = rule.apply(current_plan, self.session_state)
            current_plan = result.plan
            if result.optimized:
                overall_optimized = True

        return PhysicalPlanOptimizationResult(
            plan=current_plan,
            optimized=overall_optimized
        )
