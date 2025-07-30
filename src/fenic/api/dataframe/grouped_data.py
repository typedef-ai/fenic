"""GroupedData class for aggregations on grouped DataFrames."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fenic.api.dataframe import DataFrame

from typing import Dict, List, Optional, Union

from fenic.api.column import Column, ColumnOrName
from fenic.api.dataframe._base_grouped_data import BaseGroupedData
from fenic.api.functions import col
from fenic.core._logical_plan.expressions import LiteralExpr
from fenic.core._logical_plan.plans import Aggregate


class GroupedData(BaseGroupedData):
    """Methods for aggregations on a grouped DataFrame."""

    def __init__(self, df: DataFrame, by: Optional[List[ColumnOrName]] = None):
        """Initialize grouped data.

        Args:
            df: The DataFrame to group.
            by: Optional list of columns to group by.
        """
        super().__init__(df)
        self._by: List[Column] = []
        for c in by or []:
            if isinstance(c, str):
                self._by.append(col(c))
            elif isinstance(c, Column):
                # Allow any expression except literals
                if isinstance(c._logical_expr, LiteralExpr):
                    raise ValueError(f"Cannot group by literal value: {c}")
                self._by.append(c)
            else:
                raise TypeError(
                    f"Group by expressions must be string or Column, got {type(c)}"
                )
        self._by_exprs = [c._logical_expr for c in self._by]

    def agg(self, *exprs: Union[Column, Dict[str, str]]) -> DataFrame:
        """Compute aggregations on grouped data and return the result as a DataFrame.

        This method applies aggregate functions to the grouped data.

        Args:
            *exprs: Aggregation expressions. Can be:

                - Column expressions with aggregate functions (e.g., `count("*")`, `sum("amount")`)
                - A dictionary mapping column names to aggregate function names (e.g., `{"amount": "sum", "age": "avg"}`)

        Returns:
            DataFrame: A new DataFrame with one row per group and columns for group keys and aggregated values

        Raises:
            ValueError: If arguments are not Column expressions or a dictionary
            ValueError: If dictionary values are not valid aggregate function names

        Example: Count employees by department
            ```python
            # Group by department and count employees
            df.group_by("department").agg(count("*").alias("employee_count"))
            ```

        Example: Multiple aggregations
            ```python
            # Multiple aggregations
            df.group_by("department").agg(
                count("*").alias("employee_count"),
                avg("salary").alias("avg_salary"),
                max("age").alias("max_age")
            )
            ```

        Example: Dictionary style aggregations
            ```python
            # Dictionary style for simple aggregations
            df.group_by("department", "location").agg({"salary": "avg", "age": "max"})
            ```
        """
        self._validate_agg_exprs(*exprs)
        if len(exprs) == 1 and isinstance(exprs[0], dict):
            agg_dict = exprs[0]
            return self.agg(*self._process_agg_dict(agg_dict))

        agg_exprs = self._process_agg_exprs(exprs)
        return self._df._from_logical_plan(
            Aggregate.from_session_state(self._df._logical_plan, self._by_exprs, agg_exprs, self._df._session_state),
            self._df._session_state,
        )
