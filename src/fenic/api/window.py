"""Window functions for DataFrame operations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Literal, Optional, Union

from fenic.api.column import Column, ColumnOrName
from fenic.core._logical_plan.expressions import LogicalExpr


class WindowFrameBoundary(Enum):
    """Enumeration of special boundary types for window frames.

    These define the limits of a window frame relative to the current row.
    """
    UNBOUNDED_PRECEDING = "unbounded_preceding"
    UNBOUNDED_FOLLOWING = "unbounded_following"
    CURRENT_ROW = "current_row"


FrameBound = Union[int, WindowFrameBoundary]
"""
Type alias for window frame boundaries.

Can be either:
- An integer offset (positive for following rows, negative for preceding rows)
- A WindowFrameBoundary enum value for special boundaries
"""

class Window:
    """Represents the specification for a window in window functions.
    
    Including partitioning, ordering, and frame boundaries.
    """

    def __init__(self):
        """Creates a new Window."""
        self._partition_by: List[LogicalExpr] = []
        self._order_by: List[LogicalExpr] = []
        self._frame: Optional[_WindowFrame] = None

    def partition_by(self, *cols: ColumnOrName) -> Window:
        """Returns this Window with the given partitioning columns.

        Args:
            *cols: Column names or Column expressions to partition by.

        Returns:
            This Window with updated partitioning (for chaining).
        """
        self._partition_by = [Column._from_col_or_name(c)._logical_expr for c in cols]
        return self

    def order_by(self, *cols: ColumnOrName) -> Window:
        """Returns this Window with the given ordering columns.

        Args:
            *cols: Column names or Column expressions to order by.

        Returns:
            This Window with updated ordering (for chaining).
        """
        self._order_by = [Column._from_col_or_name(c)._logical_expr for c in cols]
        return self

    def rows_between(self, start: FrameBound, end: FrameBound) -> Window:
        """Specifies a row-based frame between start (inclusive) and end (exclusive) rows.

        Args:
            start: Start offset (can be negative).
            end: End offset.

        Returns:
            This Window with row frame boundaries (for chaining).
        """
        self._frame = _WindowFrame(start=start, end=end, frame_type="rows")
        return self

    def range_between(self, start: FrameBound, end: FrameBound) -> Window:
        """Specifies a range-based frame between start (inclusive) and end (exclusive) values.

        Args:
            start: Start value.
            end: End value.

        Returns:
            This Window with range frame boundaries (for chaining).
        """
        self._frame = _WindowFrame(start=start, end=end, frame_type="range")
        return self

    def _finalize(self) -> None:
        """Internal: Finalizes the window by assigning a default frame if none is set.
        
        This is meant to be called during logical plan construction.
        """
        if self._frame:
            return

        if self._order_by:
            self._frame = _WindowFrame(
                start=WindowFrameBoundary.UNBOUNDED_PRECEDING,
                end=WindowFrameBoundary.CURRENT_ROW,
                frame_type="range",
            )
        else:
            self._frame = _WindowFrame(
                start=WindowFrameBoundary.UNBOUNDED_PRECEDING,
                end=WindowFrameBoundary.UNBOUNDED_FOLLOWING,
                frame_type="rows",
            )

    # Spark style CamelCase aliases
    partitionBy = partition_by
    orderBy = order_by
    rowsBetween = rows_between
    rangeBetween = range_between


@dataclass(frozen=True)
class _WindowFrame:
    start: FrameBound
    end: FrameBound
    frame_type: Literal["rows", "range"]

    def __post_init__(self):
        def to_numeric(bound: FrameBound) -> float:
            if isinstance(bound, int):
                return float(bound)
            if bound == WindowFrameBoundary.UNBOUNDED_PRECEDING:
                return float("-inf")
            elif bound == WindowFrameBoundary.UNBOUNDED_FOLLOWING:
                return float("inf")
            elif bound == WindowFrameBoundary.CURRENT_ROW:
                return 0.0
            else:
                raise ValueError(f"Invalid frame boundary: {bound}")

        start_val = to_numeric(self.start)
        end_val = to_numeric(self.end)

        if start_val > end_val:
            raise ValueError(
                f"Invalid window frame: start={self.start} must be <= end={self.end}"
            )
