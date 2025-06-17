from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from fenic.core._logical_plan.plans.base import LogicalPlan


@dataclass
class OptimizationResult:
    """Holds the result of an optimization pass.

    Includes both the optimized plan and whether any changes were made.
    """

    plan: LogicalPlan
    was_modified: bool


class LogicalPlanOptimizerRule(ABC):
    @abstractmethod
    def apply(self, logical_plan: LogicalPlan) -> OptimizationResult:
        """Apply the optimization rule to the logical plan.

        Args:
            logical_plan: The logical plan to optimize

        Returns:
            OptimizationResult: The optimized plan and whether any changes were made
        """
        pass


class LogicalPlanOptimizer:
    def __init__(self, rules: List[LogicalPlanOptimizerRule] = None):
        self.rules = rules

    def optimize(self, logical_plan: LogicalPlan) -> OptimizationResult:
        """Optimize the logical plan using all rules.

        Args:
            logical_plan: The logical plan to optimize

        Returns:
            OptimizationResult: The optimized plan and whether any changes were made
        """
        any_changes = False
        optimized_plan = logical_plan

        for rule in self.rules:
            result = rule.apply(optimized_plan)
            optimized_plan = result.plan
            any_changes = any_changes or result.was_modified

        return OptimizationResult(optimized_plan, any_changes)
