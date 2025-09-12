"""Base classes for physical plan optimization."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from fenic._backends.local.physical_plan.base import PhysicalPlan
from fenic.core._interfaces.session_state import BaseSessionState


@dataclass
class OptimizationResult:
    """Holds the result of an optimization pass.

    Includes both the optimized plan and whether any changes were made.
    """

    plan: PhysicalPlan
    was_modified: bool


class PhysicalPlanOptimizerRule(ABC):
    """Base class for physical plan optimization rules."""

    @abstractmethod
    def apply(self, physical_plan: PhysicalPlan, session_state: BaseSessionState) -> OptimizationResult:
        """Apply the optimization rule to the physical plan.

        Args:
            physical_plan: The physical plan to optimize
            session_state: The session state to use for the optimization

        Returns:
            OptimizationResult: The optimized plan and whether any changes were made
        """
        pass


class PhysicalPlanOptimizer:
    """Optimizer for physical plans using a list of optimization rules."""

    def __init__(self, session_state: BaseSessionState, rules: List[PhysicalPlanOptimizerRule] = None):
        """Initialize the optimizer.
        
        Args:
            session_state: The session state to use for optimization
            rules: List of optimization rules to apply (defaults to empty list)
        """
        self.session_state = session_state
        self.rules = rules or []

    def optimize(self, physical_plan: PhysicalPlan, session_state: BaseSessionState) -> PhysicalPlan:
        """Optimize the physical plan using all rules.

        Args:
            physical_plan: The physical plan to optimize
            session_state: The session state to use for the optimization

        Returns:
            PhysicalPlan: The optimized plan
        """
        optimized_plan = physical_plan

        for rule in self.rules:
            result = rule.apply(optimized_plan, session_state)
            optimized_plan = result.plan

        return optimized_plan