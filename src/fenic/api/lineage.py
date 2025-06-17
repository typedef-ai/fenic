"""Query interface for tracing data lineage through a query plan."""

from typing import Dict, List, Optional

import polars as pl
from pydantic import ConfigDict, validate_call

from fenic.core._interfaces import BaseLineage
from fenic.core.types.enums import BranchSide


class Lineage:
    """Query interface for tracing data lineage through a query plan.

    This class allows you to navigate through the query plan both forwards and backwards,
    tracing how specific rows are transformed through each operation.

    Example:
        ```python
        # Create a lineage query starting from the root
        query = LineageQuery(lineage, session.execution)

        # Or start from a specific source
        query.start_from_source("my_table")

        # Trace rows backwards through a transformation
        result = query.backward(["uuid1", "uuid2"])

        # Trace rows forward to see their outputs
        result = query.forward(["uuid3", "uuid4"])
        ```
    """

    def __init__(self, lineage: BaseLineage):
        """Initialize a Lineage instance.

        Args:
            lineage: The underlying lineage implementation.
        """
        self.lineage = lineage

    @validate_call(config=ConfigDict(strict=True))
    def get_source_names(self) -> List[str]:
        """Get the names of all sources in the query plan. Used to determine where to start the lineage traversal."""
        return self.lineage.get_source_names()

    def show(self) -> None:
        """Print the operator tree of the query."""
        print(self.lineage.stringify_graph())

    @validate_call(config=ConfigDict(strict=True))
    def start_from_source(self, source_name: str) -> None:
        """Set the current position to a specific source in the query plan.

        Args:
            source_name: Name of the source table to start from

        Example:
            ```python
            query.start_from_source("customers")
            # Now you can trace forward from the customers table
            ```
        """
        self.lineage.start_from_source(source_name)

    @validate_call(config=ConfigDict(strict=True))
    def forwards(self, row_ids: List[str]) -> pl.DataFrame:
        """Trace rows forward to see how they are transformed by the next operation.

        Args:
            row_ids: List of UUIDs identifying the rows to trace

        Returns:
            DataFrame containing the transformed rows in the next operation

        Raises:
            ValueError: If at root node or if row_ids format is invalid

        Example:
            ```python
            # Trace how specific customer rows are transformed
            transformed = query.forward(["customer_uuid1", "customer_uuid2"])
            ```
        """
        return self.lineage.forwards(row_ids)

    @validate_call(config=ConfigDict(strict=True))
    def backwards(
        self, ids: List[str], branch_side: Optional[BranchSide] = None
    ) -> pl.DataFrame:
        """Trace rows backwards to see which input rows produced them.

        Args:
            ids: List of UUIDs identifying the rows to trace back
            branch_side: For operators with multiple inputs (like joins), specify which
                input to trace ("left" or "right"). Not needed for single-input operations.

        Returns:
            DataFrame containing the source rows that produced the specified outputs

        Raises:
            ValueError: If invalid ids format or incorrect branch_side specification

        Example:
            ```python
            # Simple backward trace
            source_rows = query.backward(["result_uuid1"])

            # Trace back through a join
            left_rows = query.backward(["join_uuid1"], branch_side="left")
            right_rows = query.backward(["join_uuid1"], branch_side="right")
            ```
        """
        return self.lineage.backwards(ids, branch_side)

    def skip_forwards(self, row_ids: List[str]) -> pl.DataFrame:
        """[Not Implemented] Trace rows forward through multiple operations at once.

        This method will allow efficient tracing through multiple operations without
        intermediate results.

        Args:
            row_ids: List of UUIDs identifying the rows to trace

        Returns:
            DataFrame containing the final transformed rows

        Raises:
            NotImplementedError: This method is not yet implemented
        """
        raise NotImplementedError("Skip forwards not yet implemented")

    def skip_backwards(self, ids: List[str]) -> Dict[str, pl.DataFrame]:
        """[Not Implemented] Trace rows backwards through multiple operations at once.

        This method will allow efficient tracing through multiple operations without
        intermediate results.

        Args:
            ids: List of UUIDs identifying the rows to trace back

        Returns:
            Dictionary mapping operation names to their source DataFrames

        Raises:
            NotImplementedError: This method is not yet implemented
        """
        raise NotImplementedError("Skip backwards not yet implemented")

    def get_result_df(self) -> pl.DataFrame:
        """Get the result of the query as a Polars DataFrame."""
        return self.lineage.get_result_df()

    @validate_call(config=ConfigDict(strict=True))
    def get_source_df(self, source_name: str) -> pl.DataFrame:
        """Get a query source by name as a Polars DataFrame."""
        return self.lineage.get_source_df(source_name)
