"""Physical plan optimization module."""

from fenic._backends.local.physical_plan.optimizer.base import (
    PhysicalPlanOptimizationResult,
    PhysicalPlanOptimizer,
    PhysicalPlanRule,
)
from fenic._backends.local.physical_plan.optimizer.merge_duckdb_nodes import (
    MergeDuckDBNodesRule,
)

__all__ = [
    "PhysicalPlanOptimizer",
    "PhysicalPlanOptimizationResult",
    "PhysicalPlanRule",
    "MergeDuckDBNodesRule",
]
