from abc import ABC, abstractmethod
from typing import List, Optional

import polars as pl

from fenic.core.types.enums import BranchSide


class BaseLineage(ABC):
    """Base class for lineage traversal implementations."""

    @abstractmethod
    def get_source_names(self) -> List[str]:
        """Get the names of all sources in the query plan. Used to determine where to start the lineage traversal."""
        pass

    @abstractmethod
    def stringify_graph(self) -> str:
        """Print the operator tree of the query."""
        pass

    @abstractmethod
    def start_from_source(self, source_name: str) -> None:
        """Set the current position to a specific source in the query plan."""
        pass

    @abstractmethod
    def forwards(self, ids: List[str]) -> pl.DataFrame:
        """Trace rows forward to see how they are transformed by the next operation."""
        pass

    @abstractmethod
    def backwards(
        self, ids: List[str], branch_side: Optional[BranchSide] = None
    ) -> pl.DataFrame:
        """Trace rows backwards to see which input rows produced them."""
        pass

    @abstractmethod
    def get_result_df(self) -> pl.DataFrame:
        """Get the result of the query as a Polars DataFrame."""
        pass

    @abstractmethod
    def get_source_df(self, source_name: str) -> pl.DataFrame:
        """Get a query source by name as a Polars DataFrame."""
        pass
