"""Physical plan optimization module."""

from fenic._backends.local.physical_plan.optimizer.base import (
    PhysicalPlanOptimizationResult,
    PhysicalPlanOptimizer,
    PhysicalPlanRule,
)
from fenic._backends.local.physical_plan.optimizer.merge_duckdb_nodes import (
    MergeDuckDBNodesRule,
)
from fenic._backends.local.physical_plan.transform import DuckDBSubtree

__all__ = [
    "PhysicalPlanOptimizer",
    "PhysicalPlanOptimizationResult",
    "PhysicalPlanRule",
    "DuckDBSubtree",
    "MergeDuckDBNodesRule",
]
