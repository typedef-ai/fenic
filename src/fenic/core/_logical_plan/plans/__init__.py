"""Logical plan module for query representation.

Note: These classes are not part of the public API and should not be used directly.
"""

from fenic.core._logical_plan.plans.aggregate import Aggregate, SemanticAggregate
from fenic.core._logical_plan.plans.base import CacheInfo, LogicalPlan
from fenic.core._logical_plan.plans.join import (
    Join,
    SemanticJoin,
    SemanticSimilarityJoin,
)
from fenic.core._logical_plan.plans.sink import FileSink, TableSink
from fenic.core._logical_plan.plans.source import (
    FileSource,
    InMemorySource,
    TableSource,
)
from fenic.core._logical_plan.plans.transform import (
    SQL,
    DropDuplicates,
    Explode,
    Filter,
    Limit,
    Projection,
    Sort,
    Union,
)
from fenic.core._logical_plan.plans.transform import Unnest as Unnest

__all__ = [
    "Aggregate",
    "SemanticAggregate",
    "CacheInfo",
    "LogicalPlan",
    "Join",
    "SemanticJoin",
    "SemanticSimilarityJoin",
    "FileSink",
    "TableSink",
    "FileSource",
    "InMemorySource",
    "TableSource",
    "SQL",
    "DropDuplicates",
    "Explode",
    "Filter",
    "Limit",
    "Projection",
    "Sort",
    "Union",
    "Unnest",
]
