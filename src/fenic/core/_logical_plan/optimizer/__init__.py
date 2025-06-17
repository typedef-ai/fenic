"""Optimizer module for optimizing logical plans.

This module contains the logic for optimizing logical query plans. These classes are not
part of the public API and should not be used directly.
"""

from fenic.core._logical_plan.optimizer.base import (
    LogicalPlanOptimizer as LogicalPlanOptimizer,
)
from fenic.core._logical_plan.optimizer.base import (
    LogicalPlanOptimizerRule as LogicalPlanOptimizerRule,
)
from fenic.core._logical_plan.optimizer.base import (
    OptimizationResult as OptimizationResult,
)
from fenic.core._logical_plan.optimizer.merge_filters_rule import (
    MergeFiltersRule as MergeFiltersRule,
)
from fenic.core._logical_plan.optimizer.not_filter_pushdown_rule import (
    NotFilterPushdownRule as NotFilterPushdownRule,
)
from fenic.core._logical_plan.optimizer.semantic_filter_rewrite_rule import (
    SemanticFilterRewriteRule as SemanticFilterRewriteRule,
)

all = [
    "LogicalPlanOptimizerRule",
    "MergeFiltersRule",
    "NotFilterPushdownRule",
    "SemanticFilterRewriteRule",
    "OptimizationResult",
    "LogicalPlanOptimizer",
]
