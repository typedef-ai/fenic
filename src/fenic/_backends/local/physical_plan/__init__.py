"""Physical plan module for internal query execution.

This module contains the physical execution plan classes that implement
the actual execution logic for query operations. These classes are not
part of the public API and should not be used directly.
"""

from fenic._backends.local.physical_plan.aggregate import (
    AggregateExec as AggregateExec,
)
from fenic._backends.local.physical_plan.base import PhysicalPlan as PhysicalPlan
from fenic._backends.local.physical_plan.join import JoinExec as JoinExec
from fenic._backends.local.physical_plan.join import (
    SemanticJoinExec as SemanticJoinExec,
)
from fenic._backends.local.physical_plan.join import (
    SemanticSimilarityJoinExec as SemanticSimilarityJoinExec,
)
from fenic._backends.local.physical_plan.sink import (
    DuckDBTableSinkExec as DuckDBTableSinkExec,
)
from fenic._backends.local.physical_plan.sink import FileSinkExec as FileSinkExec
from fenic._backends.local.physical_plan.source import CacheReadExec as CacheReadExec
from fenic._backends.local.physical_plan.source import (
    DocSourceExec as DocSourceExec,
)
from fenic._backends.local.physical_plan.source import (
    DuckDBTableSourceExec as DuckDBTableSourceExec,
)
from fenic._backends.local.physical_plan.source import (
    FileSourceExec as FileSourceExec,
)
from fenic._backends.local.physical_plan.source import (
    InMemorySourceExec as InMemorySourceExec,
)
from fenic._backends.local.physical_plan.transform import (
    DropDuplicatesExec as DropDuplicatesExec,
)
from fenic._backends.local.physical_plan.transform import ExplodeExec as ExplodeExec
from fenic._backends.local.physical_plan.transform import (
    ExplodeWithIndexExec as ExplodeWithIndexExec,
)
from fenic._backends.local.physical_plan.transform import FilterExec as FilterExec
from fenic._backends.local.physical_plan.transform import LimitExec as LimitExec
from fenic._backends.local.physical_plan.transform import (
    ProjectionExec as ProjectionExec,
)
from fenic._backends.local.physical_plan.transform import (
    SemanticClusterExec as SemanticClusterExec,
)
from fenic._backends.local.physical_plan.transform import (
    SeriesLiteralCheck as SeriesLiteralCheck,
)
from fenic._backends.local.physical_plan.transform import SortExec as SortExec
from fenic._backends.local.physical_plan.transform import SQLExec as SQLExec
from fenic._backends.local.physical_plan.transform import UnionExec as UnionExec
from fenic._backends.local.physical_plan.transform import UnnestExec as UnnestExec

__all__ = [
    "AggregateExec",
    "SemanticClusterExec",
    "PhysicalPlan",
    "JoinExec",
    "SemanticJoinExec",
    "SemanticSimilarityJoinExec",
    "DuckDBTableSinkExec",
    "FileSinkExec",
    "DuckDBTableSourceExec",
    "FileSourceExec",
    "DocSourceExec",
    "InMemorySourceExec",
    "DropDuplicatesExec",
    "ExplodeExec",
    "ExplodeWithIndexExec",
    "FilterExec",
    "LimitExec",
    "ProjectionExec",
    "SeriesLiteralCheck",
    "SortExec",
    "UnionExec",
    "UnnestExec",
    "SQLExec",
    "CacheReadExec",
]
