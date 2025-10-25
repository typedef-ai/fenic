"""DataFrame class providing PySpark-inspired API for data manipulation."""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

from fenic.core._logical_plan.plans.transform import ExplodeWithIndex

if TYPE_CHECKING:
    from fenic.api.io.writer import DataFrameWriter

import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import polars as pl
import pyarrow as pa
from pydantic import ConfigDict, validate_call

from fenic.api.column import Column, ColumnOrName
from fenic.api.dataframe._join_utils import (
    build_join_conditions,
    validate_join_parameters,
)
from fenic.api.dataframe.grouped_data import GroupedData
from fenic.api.dataframe.semantic_extensions import SemanticExtensions
from fenic.api.functions import col, lit
from fenic.api.io.writer import DataFrameWriter
from fenic.api.lineage import Lineage
from fenic.core._interfaces.session_state import BaseSessionState
from fenic.core._logical_plan.expressions import (
    SeriesLiteralExpr,
    SortExpr,
)
from fenic.core._logical_plan.plans import (
    CacheInfo,
    DropDuplicates,
    Explode,
    Filter,
    Join,
    Limit,
    LogicalPlan,
    Projection,
    Sort,
    Unnest,
)
from fenic.core._logical_plan.plans import (
    Union as UnionLogicalPlan,
)
from fenic.core.error import (
    SessionError,
    ValidationError,
)
from fenic.core.metrics import QueryMetrics
from fenic.core.types import Schema
from fenic.core.types.enums import JoinType
from fenic.core.types.query_result import DataLikeType, QueryResult

logger = logging.getLogger(__name__)


class DataFrame:
    """A data collection organized into named columns.

    The DataFrame class represents a lazily evaluated computation on data. Operations on
    DataFrame build up a logical query plan that is only executed when an action like
    show(), to_polars(), to_pandas(), to_arrow(), to_pydict(), to_pylist(), or count() is called.

    The DataFrame supports method chaining for building complex transformations.

    Example: Create and transform a DataFrame
        ```python
        # Create a DataFrame from a dictionary
        df = session.create_dataframe({"id": [1, 2, 3], "value": ["a", "b", "c"]})

        # Chain transformations
        result = df.filter(col("id") > 1).select("id", "value")

        # Show results
        result.show()
        # Output:
        # +---+-----+
        # | id|value|
        # +---+-----+
        # |  2|    b|
        # |  3|    c|
        # +---+-----+
        ```
    """

    _logical_plan: LogicalPlan
    _session_state: BaseSessionState

    def __new__(cls):
        """Prevent direct DataFrame construction.

        DataFrames must be created through Session.create_dataframe().
        """
        if cls is DataFrame:
            raise TypeError(
                "Direct construction of DataFrame is not allowed. Use Session.create_dataframe() to create a DataFrame."
            )
        return super().__new__(cls)

    @classmethod
    def _from_logical_plan(
        cls,
        logical_plan: LogicalPlan,
        session_state: BaseSessionState,
    ) -> DataFrame:
        """Factory method to create DataFrame instances.

        This method is intended for internal use by the Session class and other
        DataFrame methods that need to create new DataFrame instances.

        Args:
            logical_plan: The logical plan for this DataFrame
            session_state: The session state for this DataFrame

        Returns:
            A new DataFrame instance
        """
        if not isinstance(logical_plan, LogicalPlan):
            raise TypeError(f"Expected LogicalPlan, got {type(logical_plan)}")
        df = super().__new__(cls)
        df._session_state = session_state
        df._logical_plan = logical_plan
        return df

    @property
    def semantic(self) -> SemanticExtensions:
        """Interface for semantic operations on the DataFrame."""
        return SemanticExtensions(self)

    @property
    def write(self) -> DataFrameWriter:
        """Interface for saving the content of the DataFrame.

        Returns:
            DataFrameWriter: Writer interface to write DataFrame.
        """
        return DataFrameWriter(self)

    @property
    def schema(self) -> Schema:
        """Get the schema of this DataFrame.

        Returns:
            Schema: Schema containing field names and data types

        Examples:
            >>> df.schema
            Schema([
                ColumnField('name', StringType),
                ColumnField('age', IntegerType)
            ])
        """
        return self._logical_plan.schema()

    @property
    def columns(self) -> List[str]:
        """Get list of column names.

        Returns:
            List[str]: List of all column names in the DataFrame

        Examples:
            >>> df.columns
            ['name', 'age', 'city']
        """
        return self.schema.column_names()

    def __getitem__(self, col_name: str) -> Column:
        """Enable DataFrame[column_name] syntax for column access.

        Args:
            col_name: Name of the column to access

        Returns:
            Column: Column object for the specified column

        Raises:
            TypeError: If item is not a string

        Examples:
            >>> df[col("age")]  # Returns Column object for "age"
            >>> df.filter(df[col("age")] > 25)  # Use in expressions
        """
        if not isinstance(col_name, str):
            raise TypeError(
                f"Column name must be a string, got {type(col_name).__name__}. "
                "Example: df['column_name']"
            )
        if col_name not in self.columns:
            raise KeyError(
                f"Column '{col_name}' does not exist in DataFrame. "
                f"Available columns: {', '.join(self.columns)}\n"
                "Check for typos or case sensitivity issues."
            )

        return col(col_name)

    def __getattr__(self, col_name: str) -> Column:
        """Enable DataFrame.column_name syntax for column access.

        Args:
            col_name: Name of the column to access

        Returns:
            Column: Column object for the specified column

        Raises:
            TypeError: If col_name is not a string

        Examples:
            >>> df.age  # Returns Column object for "age"
            >>> df.filter(col("age") > 25)  # Use in expressions
        """
        if not isinstance(col_name, str):
            raise TypeError(
                f"Column name must be a string, got {type(col_name).__name__}. "
                "Example: df.column_name"
            )
        if col_name not in self.columns:
            raise KeyError(
                f"Column '{col_name}' does not exist in DataFrame. "
                f"Available columns: {', '.join(self.columns)}\n"
                "Check for typos or case sensitivity issues."
            )
        return col(col_name)

    def explain(self) -> None:
        """Display the logical plan of the DataFrame."""
        print(str(self._logical_plan))

    def show(self, n: int = 10, explain_analyze: bool = False) -> None:
        """Display the DataFrame content in a tabular form.

        This is an action that triggers computation of the DataFrame.
        The output is printed to stdout in a formatted table.

        Args:
            n: Number of rows to display
            explain_analyze: Whether to print the explain analyze plan
        """
        output, metrics = self._session_state.execution.show(self._logical_plan, n)
        logger.info(metrics.get_summary())
        print(output)
        if explain_analyze:
            print(metrics.get_execution_plan_details())

    def collect(self, data_type: DataLikeType = "polars") -> QueryResult:
        """Execute the DataFrame computation and return the result as a QueryResult.

        This is an action that triggers computation of the DataFrame query plan.
        All transformations and operations are executed, and the results are
        materialized into a QueryResult, which contains both the result data and the query metrics.

        Args:
            data_type: The type of data to return

        Returns:
            QueryResult: A QueryResult with materialized data and query metrics
        """
        result: Tuple[pl.DataFrame, QueryMetrics] = self._session_state.execution.collect(self._logical_plan)
        df, metrics = result
        logger.info(metrics.get_summary())

        if data_type == "polars":
            return QueryResult(df, metrics)
        elif data_type == "pandas":
            return QueryResult(df.to_pandas(use_pyarrow_extension_array=True), metrics)
        elif data_type == "arrow":
            return QueryResult(df.to_arrow(), metrics)
        elif data_type == "pydict":
            return QueryResult(df.to_dict(as_series=False), metrics)
        elif data_type == "pylist":
            return QueryResult(df.to_dicts(), metrics)
        else:
            raise ValidationError(f"Invalid data type: {data_type} in collect(). Valid data types are: polars, pandas, arrow, pydict, pylist")

    def to_polars(self) -> pl.DataFrame:
        """Execute the DataFrame computation and return the result as a Polars DataFrame.

        This is an action that triggers computation of the DataFrame query plan.
        All transformations and operations are executed, and the results are
        materialized into a Polars DataFrame.

        Returns:
            pl.DataFrame: A Polars DataFrame with materialized results
        """
        return self.collect("polars").data

    def to_pandas(self) -> pd.DataFrame:
        """Execute the DataFrame computation and return a Pandas DataFrame.

        This is an action that triggers computation of the DataFrame query plan.
        All transformations and operations are executed, and the results are
        materialized into a Pandas DataFrame.

        Returns:
            pd.DataFrame: A Pandas DataFrame containing the computed results with
        """
        return self.collect("pandas").data

    def to_arrow(self) -> pa.Table:
        """Execute the DataFrame computation and return an Apache Arrow Table.

        This is an action that triggers computation of the DataFrame query plan.
        All transformations and operations are executed, and the results are
        materialized into an Apache Arrow Table with columnar memory layout
        optimized for analytics and zero-copy data exchange.

        Returns:
            pa.Table: An Apache Arrow Table containing the computed results
        """
        return self.collect("arrow").data

    def to_pydict(self) -> Dict[str, List[Any]]:
        """Execute the DataFrame computation and return a dictionary of column arrays.

        This is an action that triggers computation of the DataFrame query plan.
        All transformations and operations are executed, and the results are
        materialized into a Python dictionary where each column becomes a list of values.

        Returns:
            Dict[str, List[Any]]: A dictionary containing the computed results with:
                - Keys: Column names as strings
                - Values: Lists containing all values for each column
        """
        return self.collect("pydict").data

    def to_pylist(self) -> List[Dict[str, Any]]:
        """Execute the DataFrame computation and return a list of row dictionaries.

        This is an action that triggers computation of the DataFrame query plan.
        All transformations and operations are executed, and the results are
        materialized into a Python list where each element is a dictionary
        representing one row with column names as keys.

        Returns:
            List[Dict[str, Any]]: A list containing the computed results with:
                - Each element: A dictionary representing one row
                - Dictionary keys: Column names as strings
                - Dictionary values: Cell values in Python native types
                - List length equals number of rows in the result
        """
        return self.collect("pylist").data

    def count(self) -> int:
        """Count the number of rows in the DataFrame.

        This is an action that triggers computation of the DataFrame.
        The output is an integer representing the number of rows.

        Returns:
            int: The number of rows in the DataFrame
        """
        return self._session_state.execution.count(self._logical_plan)[0]

    def lineage(self) -> Lineage:
        """Create a Lineage object to trace data through transformations.

        The Lineage interface allows you to trace how specific rows are transformed
        through your DataFrame operations, both forwards and backwards through the
        computation graph.

        Returns:
            Lineage: Interface for querying data lineage

        Example:
            ```python
            # Create lineage query
            lineage = df.lineage()

            # Trace specific rows backwards through transformations
            source_rows = lineage.backward(["result_uuid1", "result_uuid2"])

            # Or trace forwards to see outputs
            result_rows = lineage.forward(["source_uuid1"])
            ```

        See Also:
            LineageQuery: Full documentation of lineage querying capabilities
        """
        return Lineage(self._session_state.execution.build_lineage(self._logical_plan))

    def persist(self) -> DataFrame:
        """Mark this DataFrame to be persisted after first computation.

        The persisted DataFrame will be cached after its first computation,
        avoiding recomputation in subsequent operations. This is useful for:
            - DataFrames that are created once and reused multiple times in your workflow
            - DataFrames that are computationally expensive (large joins, aggregations, etc.)

        Returns:
            DataFrame: Same DataFrame, but marked for persistence

        Example:
            ```python
            # Cache intermediate results for reuse
            filtered_df = (df
                .filter(col("age") > 25)
                .persist()  # Cache these results
            )

            # Both operations will use cached results
            result1 = filtered_df.group_by("department").count()
            result2 = filtered_df.select("name", "salary")
            ```
        """
        cache_info = CacheInfo(cache_key=f"cache_{uuid.uuid4().hex}")
        self._logical_plan.set_cache_info(cache_info)
        return self._from_logical_plan(
            self._logical_plan,
            self._session_state)

    def cache(self) -> DataFrame:
        """Alias for persist(). Mark DataFrame for caching after first computation.

        Returns:
            DataFrame: Same DataFrame, but marked for caching

        See Also:
            persist(): Full documentation of caching behavior
        """
        return self.persist()

    def select(self, *cols: ColumnOrName) -> DataFrame:
        """Projects a set of Column expressions or column names.

        Args:
            *cols: Column expressions to select. Can be:
                - String column names (e.g., "id", "name")
                - Column objects (e.g., col("id"), col("age") + 1)

        Returns:
            DataFrame: A new DataFrame with selected columns

        Example: Select by column names
            ```python
            # Create a DataFrame
            df = session.create_dataframe({"name": ["Alice", "Bob"], "age": [25, 30]})

            # Select by column names
            df.select(col("name"), col("age")).show()
            # Output:
            # +-----+---+
            # | name|age|
            # +-----+---+
            # |Alice| 25|
            # |  Bob| 30|
            # +-----+---+
            ```

        Example: Select with expressions
            ```python
            # Select with expressions
            df.select(col("name"), col("age") + 1).show()
            # Output:
            # +-----+-------+
            # | name|age + 1|
            # +-----+-------+
            # |Alice|     26|
            # |  Bob|     31|
            # +-----+-------+
            ```

        Example: Mix strings and expressions
            ```python
            # Mix strings and expressions
            df.select(col("name"), col("age") * 2).show()
            # Output:
            # +-----+-------+
            # | name|age * 2|
            # +-----+-------+
            # |Alice|     50|
            # |  Bob|     60|
            # +-----+-------+
            ```
        """
        exprs = []
        if not cols:
            return self
        for c in cols:
            if isinstance(c, str):
                if c == "*":
                    exprs.extend(col(field)._logical_expr for field in self.columns)
                else:
                    exprs.append(col(c)._logical_expr)
            else:
                exprs.append(c._logical_expr)

        return self._from_logical_plan(
            Projection.from_session_state(self._logical_plan, exprs, self._session_state),
            self._session_state,
        )

    def where(self, condition: Column) -> DataFrame:
        """Filters rows using the given condition (alias for filter()).

        Args:
            condition: A Column expression that evaluates to a boolean

        Returns:
            DataFrame: Filtered DataFrame

        See Also:
            filter(): Full documentation of filtering behavior
        """
        return self.filter(condition)

    def filter(self, condition: Column) -> DataFrame:
        """Filters rows using the given condition.

        Args:
            condition: A Column expression that evaluates to a boolean

        Returns:
            DataFrame: Filtered DataFrame

        Example: Filter with numeric comparison
            ```python
            # Create a DataFrame
            df = session.create_dataframe({"age": [25, 30, 35], "name": ["Alice", "Bob", "Charlie"]})

            # Filter with numeric comparison
            df.filter(col("age") > 25).show()
            # Output:
            # +---+-------+
            # |age|   name|
            # +---+-------+
            # | 30|    Bob|
            # | 35|Charlie|
            # +---+-------+
            ```

        Example: Filter with semantic predicate
            ```python
            # Filter with semantic predicate
            df.filter((col("age") > 25) & semantic.predicate("This {feedback} mentions problems with the user interface or navigation")).show()
            # Output:
            # +---+-------+
            # |age|   name|
            # +---+-------+
            # | 30|    Bob|
            # | 35|Charlie|
            # +---+-------+
            ```

        Example: Filter with multiple conditions
            ```python
            # Filter with multiple conditions
            df.filter((col("age") > 25) & (col("age") <= 35)).show()
            # Output:
            # +---+-------+
            # |age|   name|
            # +---+-------+
            # | 30|    Bob|
            # | 35|Charlie|
            # +---+-------+
            ```
        """
        return self._from_logical_plan(
            Filter.from_session_state(self._logical_plan, condition._logical_expr, self._session_state),
            self._session_state,
        )

    def with_column(self, col_name: str, col: Union[Any, Column, pl.Series, pd.Series]) -> DataFrame:
        """Add a new column or replace an existing column.

        Args:
            col_name: Name of the new column
            col: Column expression, Series, or value to assign to the column:

                - Column: A Column expression (e.g., `col("age") + 1`)
                - `pl.Series` or `pd.Series`: A Polars or pandas Series with data
                    - **Note: Series length MUST match the DataFrame height**
                - Any other value: Treated as a literal value (broadcast to all rows)

        Returns:
            DataFrame: New DataFrame with added/replaced column

        Raises:
            ExecutionError:
                - If a Series length does not match the DataFrame height
            ValidationError:
                - If the Series contains all null values and no dtype is specified
                - If the Series has length 0
        Notes:
            - The name of the created column will be the name defined in col_name, even if input is a Series with a different name.

        Example: Add literal column
            ```python
            # Create a DataFrame
            df = session.create_dataframe({"name": ["Alice", "Bob"], "age": [25, 30]})

            # Add literal column
            df.with_column("constant", lit(1)).show()
            # Output:
            # +-----+---+--------+
            # | name|age|constant|
            # +-----+---+--------+
            # |Alice| 25|       1|
            # |  Bob| 30|       1|
            # +-----+---+--------+
            ```

        Example: Add computed column
            ```python
            # Add computed column
            df.with_column("double_age", col("age") * 2).show()
            # Output:
            # +-----+---+----------+
            # | name|age|double_age|
            # +-----+---+----------+
            # |Alice| 25|        50|
            # |  Bob| 30|        60|
            # +-----+---+----------+
            ```

        Example: Replace existing column
            ```python
            # Replace existing column
            df.with_column("age", col("age") + 1).show()
            # Output:
            # +-----+---+
            # | name|age|
            # +-----+---+
            # |Alice| 26|
            # |  Bob| 31|
            # +-----+---+
            ```

        Example: Add column with complex expression
            ```python
            # Add column with complex expression
            df.with_column(
                "age_category",
                when(col("age") < 30, "young")
                .when(col("age") < 50, "middle")
                .otherwise("senior")
            ).show()
            # Output:
            # +-----+---+------------+
            # | name|age|age_category|
            # +-----+---+------------+
            # |Alice| 25|       young|
            # |  Bob| 30|     middle|
            # +-----+---+------------+
            ```

        Example: Add column from Polars Series
            ```python
            import polars as pl

            # Create a DataFrame
            df = session.create_dataframe({"name": ["Alice", "Bob"], "age": [25, 30]})

            # Add column from Polars Series
            bonus = pl.Series([100, 200])
            df.with_column("bonus", bonus).show()
            # Output:
            # +-----+---+-----+
            # | name|age|bonus|
            # +-----+---+-----+
            # |Alice| 25|  100|
            # |  Bob| 30|  200|
            # +-----+---+-----+
            ```

        Example: Add column from pandas Series
            ```python
            import pandas as pd

            # Create a DataFrame
            df = session.create_dataframe({"name": ["Alice", "Bob"], "age": [25, 30]})

            # Add column from pandas Series
            score = pd.Series([85.5, 92.0])
            df.with_column("score", score).show()
            # Output:
            # +-----+---+-----+
            # | name|age|score|
            # +-----+---+-----+
            # |Alice| 25| 85.5|
            # |  Bob| 30| 92.0|
            # +-----+---+-----+
            ```
        """
        exprs = []

        # Handle different input types: Column, Series, or literal value
        if isinstance(col, (pl.Series, pd.Series)):
            # Wrap Series in SeriesLiteralExpr and then in Column
            col = Column._from_logical_expr(SeriesLiteralExpr(col))
        elif not isinstance(col, Column):
            # Wrap other values as literals
            col = lit(col)

        for field in self.columns:
            if field != col_name:
                exprs.append(Column._from_column_name(field)._logical_expr)

        # Add the new column with alias
        exprs.append(col.alias(col_name)._logical_expr)

        return self._from_logical_plan(
            Projection.from_session_state(self._logical_plan, exprs, self._session_state),
            self._session_state,
        )

    def with_columns(self, cols_map: Dict[str, Union[Any, Column, pl.Series, pd.Series]]) -> DataFrame:
        """Add multiple new columns or replace existing columns.

        Args:
            cols_map: A dictionary where keys are column names and values are:

                - Column: Column expressions (e.g., col("age") + 1)
                - pl.Series or pd.Series: Series with data
                    - **Note: Series length MUST match the DataFrame height**
                - Any other value: Treated as literal values (broadcast to all rows)

        Returns:
            DataFrame: New DataFrame with added/replaced columns

        Raises:
            ValueError:
                - If two columns being created in the same `with_columns` call depend on each other
            ExecutionError:
                - If any Series length does not match the DataFrame height
            ValidationError:
                - If any Series contains all null values and no dtype is specified
                - If any Series has length 0
        Notes:
            - All columns are created at once, so new columns cannot depend on each other.
            - The name of the created column will be the name defined in cols_map, even if input is a Series with a different name.

        Example: Add multiple columns
            ```python
            # Create a DataFrame
            df = session.create_dataframe({"name": ["Alice", "Bob"], "age": [25, 30]})

            # Add multiple columns at once
            df.with_columns({
                "double_age": col("age") * 2,
                "constant": lit(1),
                "age_plus_10": col("age") + 10
            }).show()
            # Output:
            # +-----+---+----------+--------+-----------+
            # | name|age|double_age|constant|age_plus_10|
            # +-----+---+----------+--------+-----------+
            # |Alice| 25|        50|       1|         35|
            # |  Bob| 30|        60|       1|         40|
            # +-----+---+----------+--------+-----------+
            ```

        Example: Replace and add columns
            ```python
            # Replace existing column and add new ones
            df.with_columns({
                "age": col("age") + 1,
                "is_adult": col("age") >= 18
            }).show()
            # Output:
            # +-----+---+--------+
            # | name|age|is_adult|
            # +-----+---+--------+
            # |Alice| 26|    true|
            # |  Bob| 31|    true|
            # +-----+---+--------+
            ```

        Example: Complex expressions
            ```python
            # Add multiple columns with complex expressions
            df.with_columns({
                "age_category": when(col("age") < 30, "young")
                    .when(col("age") < 50, "middle")
                    .otherwise("senior"),
                "name_length": length(col("name")),
                "name_upper": upper(col("name"))
            }).show()
            # Output:
            # +-----+---+------------+-----------+----------+
            # | name|age|age_category|name_length|name_upper|
            # +-----+---+------------+-----------+----------+
            # |Alice| 25|       young|          5|     ALICE|
            # |  Bob| 30|      middle|          3|       BOB|
            # +-----+---+------------+-----------+----------+
            ```

        Example: Add columns from Series
            ```python
            import polars as pl

            # Create a DataFrame
            df = session.create_dataframe({"name": ["Alice", "Bob"], "age": [25, 30]})

            # Add multiple columns from Series
            df.with_columns({
                "bonus": pl.Series([100, 200]),
                "score": pl.Series([85.5, 92.0])
            }).show()
            # Output:
            # +-----+---+-----+-----+
            # | name|age|bonus|score|
            # +-----+---+-----+-----+
            # |Alice| 25|  100| 85.5|
            # |  Bob| 30|  200| 92.0|
            # +-----+---+-----+-----+
            ```

        Example: Mix Series with Column expressions
            ```python
            import polars as pl

            # Mix Series with Column expressions
            df.with_columns({
                "bonus": pl.Series([100, 200]),
                "double_age": col("age") * 2,
                "constant": 1
            }).show()
            # Output:
            # +-----+---+-----+----------+--------+
            # | name|age|bonus|double_age|constant|
            # +-----+---+-----+----------+--------+
            # |Alice| 25|  100|        50|       1|
            # |  Bob| 30|  200|        60|       1|
            # +-----+---+-----+----------+--------+
            ```

        Example: Error when adding columns that depend on each other
            ```python
            df.with_columns({
                "age_plus_1": col("age") + 1,
                "age_plus_2": col("age_plus_1") + 1
            })
            # ValueError: Column 'age_plus_1' not found in schema

            # Instead, use a single with_column call
            df = df.with_column(
                "age_plus_1", col("age") + 1
            ).with_column(
                "age_plus_2", col("age_plus_1") + 1
            )
            df.show()
            # Output:
            # +-----+---+----------+----------+
            # | name|age|age_plus_1|age_plus_2|
            # +-----+---+----------+----------+
            # |Alice| 25|        26|        27|
            # |  Bob| 30|        31|        32|
            # +-----+---+----------+----------+
            ```
        """
        if not cols_map:
            return self

        exprs = []
        new_col_names = set(cols_map.keys())

        # Add existing columns that are not being replaced
        for field in self.columns:
            if field not in new_col_names:
                exprs.append(Column._from_column_name(field)._logical_expr)

        # Add all new columns with aliases
        for col_name, col_expr in cols_map.items():
            # Handle different input types: Column, Series, or literal value
            if isinstance(col_expr, (pl.Series, pd.Series)):
                # Wrap Series in SeriesLiteralExpr and then in Column
                col_expr = Column._from_logical_expr(SeriesLiteralExpr(col_expr))
            elif not isinstance(col_expr, Column):
                # Automatically wrap non-Column values (literals) with lit() for convenience
                # This allows users to pass raw Python values like: {"constant": 100, "status": "active"}
                # instead of requiring: {"constant": lit(100), "status": lit("active")}
                col_expr = lit(col_expr)
            exprs.append(col_expr.alias(col_name)._logical_expr)

        return self._from_logical_plan(
            Projection.from_session_state(self._logical_plan, exprs, self._session_state),
            self._session_state,
        )

    def with_column_renamed(self, col_name: str, new_col_name: str) -> DataFrame:
        """Rename a column. No-op if the column does not exist.

        Args:
            col_name: Name of the column to rename.
            new_col_name: New name for the column.

        Returns:
            DataFrame: New DataFrame with the column renamed.

        Example: Rename a column
            ```python
            # Create sample DataFrame
            df = session.create_dataframe({
                "age": [25, 30, 35],
                "name": ["Alice", "Bob", "Charlie"]
            })

            # Rename a column
            df.with_column_renamed("age", "age_in_years").show()
            # Output:
            # +------------+-------+
            # |age_in_years|   name|
            # +------------+-------+
            # |         25|  Alice|
            # |         30|    Bob|
            # |         35|Charlie|
            # +------------+-------+
            ```

        Example: Rename multiple columns
            ```python
            # Rename multiple columns
            df = (df
                .with_column_renamed("age", "age_in_years")
                .with_column_renamed("name", "full_name")
            ).show()
            # Output:
            # +------------+----------+
            # |age_in_years|full_name |
            # +------------+----------+
            # |         25|     Alice|
            # |         30|       Bob|
            # |         35|   Charlie|
            # +------------+----------+
            ```
        """
        exprs = []
        renamed = False

        for field in self.schema.column_fields:
            name = field.name
            if name == col_name:
                exprs.append(col(name).alias(new_col_name)._logical_expr)
                renamed = True
            else:
                exprs.append(col(name)._logical_expr)

        if not renamed:
            return self

        return self._from_logical_plan(
            Projection.from_session_state(self._logical_plan, exprs, self._session_state),
            self._session_state,
        )

    def drop(self, *col_names: str) -> DataFrame:
        """Remove one or more columns from this DataFrame.

        Args:
            *col_names: Names of columns to drop.

        Returns:
            DataFrame: New DataFrame without specified columns.

        Raises:
            ValueError: If any specified column doesn't exist in the DataFrame.
            ValueError: If dropping the columns would result in an empty DataFrame.

        Example: Drop single column
            ```python
            # Create sample DataFrame
            df = session.create_dataframe({
                "id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
                "age": [25, 30, 35]
            })

            # Drop single column
            df.drop("age").show()
            # Output:
            # +---+-------+
            # | id|   name|
            # +---+-------+
            # |  1|  Alice|
            # |  2|    Bob|
            # |  3|Charlie|
            # +---+-------+
            ```

        Example: Drop multiple columns
            ```python
            # Drop multiple columns
            df.drop(col("id"), "age").show()
            # Output:
            # +-------+
            # |   name|
            # +-------+
            # |  Alice|
            # |    Bob|
            # |Charlie|
            # +-------+
            ```

        Example: Error when dropping non-existent column
            ```python
            # This will raise a ValueError
            df.drop("non_existent_column")
            # ValueError: Column 'non_existent_column' not found in DataFrame
            ```
        """
        if not col_names:
            return self

        current_cols = set(self.columns)
        to_drop = set(col_names)
        missing = to_drop - current_cols

        if missing:
            missing_str = (
                f"Column '{next(iter(missing))}'"
                if len(missing) == 1
                else f"Columns {sorted(missing)}"
            )
            raise ValueError(f"{missing_str} not found in DataFrame")

        remaining_cols = [
            col(c)._logical_expr for c in self.columns if c not in to_drop
        ]

        if not remaining_cols:
            raise ValueError("Cannot drop all columns from DataFrame")

        return self._from_logical_plan(
            Projection.from_session_state(self._logical_plan, remaining_cols, self._session_state),
            self._session_state,
        )

    def union(self, other: DataFrame) -> DataFrame:
        """Return a new DataFrame containing the union of rows in this and another DataFrame.

        This is equivalent to UNION ALL in SQL. To remove duplicates, use drop_duplicates() after union().

        Args:
            other: Another DataFrame with the same schema.

        Returns:
            DataFrame: A new DataFrame containing rows from both DataFrames.

        Raises:
            ValueError: If the DataFrames have different schemas.
            TypeError: If other is not a DataFrame.

        Example: Union two DataFrames
            ```python
            # Create two DataFrames
            df1 = session.create_dataframe({
                "id": [1, 2],
                "value": ["a", "b"]
            })
            df2 = session.create_dataframe({
                "id": [3, 4],
                "value": ["c", "d"]
            })

            # Union the DataFrames
            df1.union(df2).show()
            # Output:
            # +---+-----+
            # | id|value|
            # +---+-----+
            # |  1|    a|
            # |  2|    b|
            # |  3|    c|
            # |  4|    d|
            # +---+-----+
            ```

        Example: Union with duplicates
            ```python
            # Create DataFrames with overlapping data
            df1 = session.create_dataframe({
                "id": [1, 2],
                "value": ["a", "b"]
            })
            df2 = session.create_dataframe({
                "id": [2, 3],
                "value": ["b", "c"]
            })

            # Union with duplicates
            df1.union(df2).show()
            # Output:
            # +---+-----+
            # | id|value|
            # +---+-----+
            # |  1|    a|
            # |  2|    b|
            # |  2|    b|
            # |  3|    c|
            # +---+-----+

            # Remove duplicates after union
            df1.union(df2).drop_duplicates().show()
            # Output:
            # +---+-----+
            # | id|value|
            # +---+-----+
            # |  1|    a|
            # |  2|    b|
            # |  3|    c|
            # +---+-----+
            ```
        """
        self._ensure_same_session(self._session_state, [other._session_state])
        return self._from_logical_plan(
            UnionLogicalPlan.from_session_state([self._logical_plan, other._logical_plan], self._session_state),
            self._session_state,
        )

    def limit(self, n: int) -> DataFrame:
        """Limits the number of rows to the specified number.

        Args:
            n: Maximum number of rows to return.

        Returns:
            DataFrame: DataFrame with at most n rows.

        Raises:
            TypeError: If n is not an integer.

        Example: Limit rows
            ```python
            # Create sample DataFrame
            df = session.create_dataframe({
                "id": [1, 2, 3, 4, 5],
                "name": ["Alice", "Bob", "Charlie", "Dave", "Eve"]
            })

            # Get first 3 rows
            df.limit(3).show()
            # Output:
            # +---+-------+
            # | id|   name|
            # +---+-------+
            # |  1|  Alice|
            # |  2|    Bob|
            # |  3|Charlie|
            # +---+-------+
            ```

        Example: Limit with other operations
            ```python
            # Limit after filtering
            df.filter(col("id") > 2).limit(2).show()
            # Output:
            # +---+-------+
            # | id|   name|
            # +---+-------+
            # |  3|Charlie|
            # |  4|   Dave|
            # +---+-------+
            ```
        """
        return self._from_logical_plan(
            Limit.from_session_state(self._logical_plan, n, self._session_state),
            self._session_state)

    @overload
    def join(
        self,
        other: DataFrame,
        on: Union[str, List[str]],
        *,
        how: JoinType = "inner",
    ) -> DataFrame: ...

    @overload
    def join(
        self,
        other: DataFrame,
        *,
        left_on: Union[ColumnOrName, List[ColumnOrName]],
        right_on: Union[ColumnOrName, List[ColumnOrName]],
        how: JoinType = "inner",
    ) -> DataFrame: ...

    def join(
        self,
        other: DataFrame,
        on: Optional[Union[str, List[str]]] = None,
        *,
        left_on: Optional[Union[ColumnOrName, List[ColumnOrName]]] = None,
        right_on: Optional[Union[ColumnOrName, List[ColumnOrName]]] = None,
        how: JoinType = "inner",
    ) -> DataFrame:
        """Joins this DataFrame with another DataFrame.

        The Dataframes must have no duplicate column names between them. This API only supports equi-joins.
        For non-equi-joins, use session.sql().

        Args:
            other: DataFrame to join with.
            on: Join condition(s). Can be:
                - A column name (str)
                - A list of column names (List[str])
                - A Column expression (e.g., col('a'))
                - A list of Column expressions
                - `None` for cross joins
            left_on: Column(s) from the left DataFrame to join on. Can be:
                - A column name (str)
                - A Column expression (e.g., col('a'), col('a') + 1)
                - A list of column names or expressions
            right_on: Column(s) from the right DataFrame to join on. Can be:
                - A column name (str)
                - A Column expression (e.g., col('b'), upper(col('b')))
                - A list of column names or expressions
            how: Type of join to perform.

        Returns:
            Joined DataFrame.

        Raises:
            ValidationError: If cross join is used with an ON clause.
            ValidationError: If join condition is invalid.
            ValidationError: If both 'on' and 'left_on'/'right_on' parameters are provided.
            ValidationError: If only one of 'left_on' or 'right_on' is provided.
            ValidationError: If 'left_on' and 'right_on' have different lengths

        Example: Inner join on column name
            ```python
            # Create sample DataFrames
            df1 = session.create_dataframe({
                "id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"]
            })
            df2 = session.create_dataframe({
                "id": [1, 2, 4],
                "age": [25, 30, 35]
            })

            # Join on single column
            df1.join(df2, on=col("id")).show()
            # Output:
            # +---+-----+---+
            # | id| name|age|
            # +---+-----+---+
            # |  1|Alice| 25|
            # |  2|  Bob| 30|
            # +---+-----+---+
            ```

        Example: Join with expression
            ```python
            # Join with Column expressions
            df1.join(
                df2,
                left_on=col("id"),
                right_on=col("id"),
            ).show()
            # Output:
            # +---+-----+---+
            # | id| name|age|
            # +---+-----+---+
            # |  1|Alice| 25|
            # |  2|  Bob| 30|
            # +---+-----+---+
            ```

        Example: Cross join
            ```python
            # Cross join (cartesian product)
            df1.join(df2, how="cross").show()
            # Output:
            # +---+-----+---+---+
            # | id| name| id|age|
            # +---+-----+---+---+
            # |  1|Alice|  1| 25|
            # |  1|Alice|  2| 30|
            # |  1|Alice|  4| 35|
            # |  2|  Bob|  1| 25|
            # |  2|  Bob|  2| 30|
            # |  2|  Bob|  4| 35|
            # |  3|Charlie| 1| 25|
            # |  3|Charlie| 2| 30|
            # |  3|Charlie| 4| 35|
            # +---+-----+---+---+
            ```
        """
        validate_join_parameters(self, on, left_on, right_on, how)

        # Build join conditions
        left_conditions, right_conditions = build_join_conditions(on, left_on, right_on)

        self._ensure_same_session(self._session_state, [other._session_state])
        return self._from_logical_plan(
            Join.from_session_state(
                self._logical_plan,
                other._logical_plan,
                left_conditions,
                right_conditions,
                how,
                self._session_state),
            self._session_state,
        )

    def explode(self, column: ColumnOrName) -> DataFrame:
        """Create a new row for each element in an array column.

        This operation is useful for flattening nested data structures. For each row in the
        input DataFrame that contains an array/list in the specified column, this method will:
        1. Create N new rows, where N is the length of the array
        2. Each new row will be identical to the original row, except the array column will
           contain just a single element from the original array
        3. Rows with NULL values or empty arrays in the specified column are filtered out

        Args:
            column: Name of array column to explode (as string) or Column expression.

        Returns:
            DataFrame: New DataFrame with the array column exploded into multiple rows.

        Raises:
            TypeError: If column argument is not a string or Column.

        Example: Explode array column
            ```python
            # Create sample DataFrame
            df = session.create_dataframe({
                "id": [1, 2, 3, 4],
                "tags": [["red", "blue"], ["green"], [], None],
                "name": ["Alice", "Bob", "Carol", "Dave"]
            })

            # Explode the tags column
            df.explode("tags").show()
            # Output:
            # +---+-----+-----+
            # | id| tags| name|
            # +---+-----+-----+
            # |  1|  red|Alice|
            # |  1| blue|Alice|
            # |  2|green|  Bob|
            # +---+-----+-----+
            ```

        Example: Using column expression
            ```python
            # Explode using column expression
            df.explode(col("tags")).show()
            # Output:
            # +---+-----+-----+
            # | id| tags| name|
            # +---+-----+-----+
            # |  1|  red|Alice|
            # |  1| blue|Alice|
            # |  2|green|  Bob|
            # +---+-----+-----+
            ```
        """
        return self._from_logical_plan(
            Explode.from_session_state(self._logical_plan, Column._from_col_or_name(column)._logical_expr, self._session_state),
            self._session_state,
        )

    def explode_with_index(
        self,
        column: ColumnOrName,
        index_col_name: str = "pos",
        value_col_name: str = "col",
        keep_null_and_empty: bool = False,
    ) -> DataFrame:
        """Create a new row for each element in an array column, with the element's position in the array and its value.

        This operation is similar to explode(), but also adds a column containing the 0-based
        position of each element within its original array. By default, the position column is named "pos".
        and the value column is named "col". These columns replace the original column in the output DataFrame.
        If keep_null_and_empty is True, the position column will be null for rows where the array is null or empty.

        Args:
            column: Name of array column to explode (as string) or Column expression.
            index_col_name: Name for the column containing 0-based array positions (default: "pos").
            value_col_name: Name for the exploded value column (default: "col").
            keep_null_and_empty: If True, preserves rows where the array is null or empty (default: False).
                Mimicks the behavior of posexplode (false) vs posexplode_outer (true).

        Returns:
            DataFrame: New DataFrame with:
                - An integer column (named `index_col_name`) containing 0-based positions
                - The exploded array column (named `value_col_name`)
                - All other columns from the original DataFrame

        Example: Explode with index
            ```python
            df = session.create_dataframe({
                "id": [1, 2, 3],
                "tags": [["red", "blue"], ["green"], []],
            })

            df.explode_with_index("tags").show()
            # Output:
            # +-----+---+-----+
            # | pos| id| tags|
            # +-----+---+-----+
            # |    0|  1|  red|
            # |    1|  1| blue|
            # |    0|  2|green|
            # +-----+---+-----+
            ```

        Example: Custom column names
            ```python
            df.explode_with_index("tags", index_col_name="index", value_name="tag").show()
            # Output:
            # +-----+---+-----+
            # |index| id|  tag|
            # +-----+---+-----+
            # |    0|  1|  red|
            # |    1|  1| blue|
            # |    0|  2|green|
            # +-----+---+-----+
            ```
        """
        return self._from_logical_plan(
            ExplodeWithIndex.from_session_state(
                self._logical_plan,
                Column._from_col_or_name(column)._logical_expr,
                index_col_name,
                value_col_name,
                self._session_state,
                keep_null_and_empty,
            ),
            self._session_state,
        )

    def posexplode(self, column: ColumnOrName) -> DataFrame:
        """Create a new row for each element in an array column, with the element's position in the array and its value.

        This is a PySpark-compatible alias for explode_with_index.
        Creates two columns: 'pos' (0-based position) and 'col' (the array element value).
        These columns replace the original column in the output DataFrame.

        Args:
            column: Name of array column to explode (as string) or Column expression.

        Returns:
            DataFrame: New DataFrame with 'pos' and 'col' columns, plus all other original columns.

        Example: PySpark-style posexplode
            ```python
            df = session.create_dataframe({
                "id": [1, 2],
                "tags": [["red", "blue"], ["green"]],
            })

            df.posexplode("tags").show()
            # Output:
            # +---+---+-----+
            # |pos| id|  col|
            # +---+---+-----+
            # |  0|  1|  red|
            # |  1|  1| blue|
            # |  0|  2|green|
            # +---+---+-----+
            ```
        """
        return self.explode_with_index(column)

    def explode_outer(self, column: ColumnOrName) -> DataFrame:
        """Create a new row for each element in an array column, containing the element's position in the array and its value, and preserving null/empty arrays.

        This operation is similar to explode(), but keeps rows where the array column
        is null or empty, producing a row with null in the exploded column.

        Args:
            column: Name of array column to explode (as string) or Column expression.

        Returns:
            DataFrame: New DataFrame with the array column exploded into multiple rows.
            Rows with null or empty arrays are preserved with null in the exploded column.

        Example: Explode with outer join behavior
            ```python
            df = session.create_dataframe({
                "id": [1, 2, 3],
                "tags": [["red", "blue"], [], None],
            })

            df.explode_outer("tags").show()
            # Output:
            # +---+-----+
            # | id| tags|
            # +---+-----+
            # |  1|  red|
            # |  1| blue|
            # |  2| NULL|  # empty array preserved as null
            # |  3| NULL|  # null array preserved as null
            # +---+-----+
            ```
        """
        return self._from_logical_plan(
            Explode.from_session_state(
                self._logical_plan,
                Column._from_col_or_name(column)._logical_expr,
                self._session_state,
                keep_null_and_empty=True
            ),
            self._session_state,
        )

    def posexplode_outer(self, column: ColumnOrName) -> DataFrame:
        """Create a new row for each element in an array column with position and value, preserving null/empty arrays.

        This is a PySpark-compatible alias for explode_with_index with keep_null_and_empty=True.
        Creates two columns: 'pos' (0-based position) and 'col' (the array element value).
        Rows with null or empty arrays produce (null, null).

        Args:
            column: Name of array column to explode (as string) or Column expression.

        Returns:
            DataFrame: New DataFrame with 'pos' and 'col' columns, plus all other original columns.
            Rows with null or empty arrays are preserved with (null, null).

        Example: PySpark-style posexplode_outer
            ```python
            df = session.create_dataframe({
                "id": [1, 2, 3],
                "tags": [["red", "blue"], [], None],
            })

            df.posexplode_outer("tags").show()
            # Output:
            # +---+---+-----+
            # |pos| id|  col|
            # +---+---+-----+
            # |  0|  1|  red|
            # |  1|  1| blue|
            # |NULL|  2| NULL|  # empty array preserved
            # |NULL|  3| NULL|  # null array preserved
            # +---+---+-----+
            ```
        """
        return self.explode_with_index(
            column, keep_null_and_empty=True
        )

    def group_by(self, *cols: ColumnOrName) -> GroupedData:
        """Groups the DataFrame using the specified columns.

        Args:
            *cols: Columns to group by. Can be column names as strings or Column expressions.

        Returns:
            GroupedData: Object for performing aggregations on the grouped data.

        Example: Group by single column
            ```python
            # Create sample DataFrame
            df = session.create_dataframe({
                "department": ["IT", "HR", "IT", "HR", "IT"],
                "salary": [80000, 70000, 90000, 75000, 85000]
            })

            # Group by single column
            df.group_by(col("department")).agg(count("*")).show()
            # Output:
            # +----------+-----+
            # |department|count|
            # +----------+-----+
            # |        IT|    3|
            # |        HR|    2|
            # +----------+-----+
            ```

        Example: Group by multiple columns
            ```python
            # Group by multiple columns
            df.group_by(col("department"), col("location")).agg({"salary": "avg"}).show()
            # Output:
            # +----------+--------+-----------+
            # |department|location|avg(salary)|
            # +----------+--------+-----------+
            # |        IT|    NYC|    85000.0|
            # |        HR|    NYC|    72500.0|
            # +----------+--------+-----------+
            ```

        Example: Group by expression
            ```python
            # Group by expression
            df.group_by(lower(col("department")).alias("department")).agg(count("*")).show()
            # Output:
            # +---------+-----+
            # |department|count|
            # +----------+-----+
            # |        it|    3|
            # |        hr|    2|
            # +---------+-----+
            ```
        """
        return GroupedData(self, list(cols) if cols else None)

    def agg(self, *exprs: Union[Column, Dict[str, str]]) -> DataFrame:
        """Aggregate on the entire DataFrame without groups.

        This is equivalent to group_by() without any grouping columns.

        Args:
            *exprs: Aggregation expressions or dictionary of aggregations.

        Returns:
            DataFrame: Aggregation results.

        Example: Multiple aggregations
            ```python
            # Create sample DataFrame
            df = session.create_dataframe({
                "salary": [80000, 70000, 90000, 75000, 85000],
                "age": [25, 30, 35, 28, 32]
            })

            # Multiple aggregations
            df.agg(
                count().alias("total_rows"),
                avg(col("salary")).alias("avg_salary")
            ).show()
            # Output:
            # +----------+-----------+
            # |total_rows|avg_salary|
            # +----------+-----------+
            # |         5|   80000.0|
            # +----------+-----------+
            ```

        Example: Dictionary style
            ```python
            # Dictionary style
            df.agg({col("salary"): "avg", col("age"): "max"}).show()
            # Output:
            # +-----------+--------+
            # |avg(salary)|max(age)|
            # +-----------+--------+
            # |    80000.0|      35|
            # +-----------+--------+
            ```
        """
        return self.group_by().agg(*exprs)

    def distinct(self) -> DataFrame:
        """Return a DataFrame with duplicate rows removed. Alias for drop_duplicates(subset=None).

        Returns:
            DataFrame: A new DataFrame with duplicate rows removed.

        Example:
            ```python
            # Create sample DataFrame
            df = session.create_dataframe({
                "c1": [1, 2, 3, 1],
                "c2": ["a", "a", "a", "a"],
                "c3": ["b", "b", "b", "b"]
            })

            # Remove duplicates considering all columns
            df.distinct().show()
            # Output:
            # +---+---+---+
            # | c1| c2| c3|
            # +---+---+---+
            # |  1|  a|  b|
            # |  2|  a|  b|
            # |  3|  a|  b|
            # +---+---+---+
            ```
        """
        return self.drop_duplicates()

    def drop_duplicates(
        self,
        subset: Optional[List[str]] = None,
    ) -> DataFrame:
        """Return a DataFrame with duplicate rows removed.

        Args:
            subset: Column names to consider when identifying duplicates. If not provided, all columns are considered.

        Returns:
            DataFrame: A new DataFrame with duplicate rows removed.

        Raises:
            ValueError: If a specified column is not present in the current DataFrame schema.

        Example: Remove duplicates considering specific columns
            ```python
            # Create sample DataFrame
            df = session.create_dataframe({
                "c1": [1, 2, 3, 1],
                "c2": ["a", "a", "a", "a"],
                "c3": ["b", "b", "b", "b"]
            })

            # Remove duplicates considering all columns
            df.drop_duplicates([col("c1"), col("c2"), col("c3")]).show()
            # Output:
            # +---+---+---+
            # | c1| c2| c3|
            # +---+---+---+
            # |  1|  a|  b|
            # |  2|  a|  b|
            # |  3|  a|  b|
            # +---+---+---+

            # Remove duplicates considering only c1
            df.drop_duplicates([col("c1")]).show()
            # Output:
            # +---+---+---+
            # | c1| c2| c3|
            # +---+---+---+
            # |  1|  a|  b|
            # |  2|  a|  b|
            # |  3|  a|  b|
            # +---+---+---+
            ```
        """
        exprs = []
        if subset:
            for c in subset:
                if c not in self.columns:
                    raise TypeError(f"Column {c} not found in DataFrame.")
                exprs.append(col(c)._logical_expr)

        return self._from_logical_plan(
            DropDuplicates.from_session_state(self._logical_plan, exprs, self._session_state),
            self._session_state,
        )

    def sort(
        self,
        cols: Union[ColumnOrName, List[ColumnOrName], None] = None,
        ascending: Optional[Union[bool, List[bool]]] = None,
    ) -> DataFrame:
        """Sort the DataFrame by the specified columns.

        Args:
            cols: Columns to sort by. This can be:
                - A single column name (str)
                - A Column expression (e.g., `col("name")`)
                - A list of column names or Column expressions
                - Column expressions may include sorting directives such as `asc("col")`, `desc("col")`,
                `asc_nulls_last("col")`, etc.
                - If no columns are provided, the operation is a no-op.

            ascending: A boolean or list of booleans indicating sort order.
                - If `True`, sorts in ascending order; if `False`, descending.
                - If a list is provided, its length must match the number of columns.
                - Cannot be used if any of the columns use `asc()`/`desc()` expressions.
                - If not specified and no sort expressions are used, columns will be sorted in ascending order by default.

        Returns:
            DataFrame: A new DataFrame sorted by the specified columns.

        Raises:
            ValueError:
                - If `ascending` is provided and its length does not match `cols`
                - If both `ascending` and column expressions like `asc()`/`desc()` are used
            TypeError:
                - If `cols` is not a column name, Column, or list of column names/Columns
                - If `ascending` is not a boolean or list of booleans

        Example: Sort in ascending order
            ```python
            # Create sample DataFrame
            df = session.create_dataframe([(2, "Alice"), (5, "Bob")], schema=["age", "name"])

            # Sort by age in ascending order
            df.sort(asc(col("age"))).show()
            # Output:
            # +---+-----+
            # |age| name|
            # +---+-----+
            # |  2|Alice|
            # |  5|  Bob|
            # +---+-----+
            ```

        Example: Sort in descending order
            ```python
            # Sort by age in descending order
            df.sort(col("age").desc()).show()
            # Output:
            # +---+-----+
            # |age| name|
            # +---+-----+
            # |  5|  Bob|
            # |  2|Alice|
            # +---+-----+
            ```

        Example: Sort with boolean ascending parameter
            ```python
            # Sort by age in descending order using boolean
            df.sort(col("age"), ascending=False).show()
            # Output:
            # +---+-----+
            # |age| name|
            # +---+-----+
            # |  5|  Bob|
            # |  2|Alice|
            # +---+-----+
            ```

        Example: Multiple columns with different sort orders
            ```python
            # Create sample DataFrame
            df = session.create_dataframe([(2, "Alice"), (2, "Bob"), (5, "Bob")], schema=["age", "name"])

            # Sort by age descending, then name ascending
            df.sort(desc(col("age")), col("name")).show()
            # Output:
            # +---+-----+
            # |age| name|
            # +---+-----+
            # |  5|  Bob|
            # |  2|Alice|
            # |  2|  Bob|
            # +---+-----+
            ```

        Example: Multiple columns with list of ascending strategies
            ```python
            # Sort both columns in descending order
            df.sort([col("age"), col("name")], ascending=[False, False]).show()
            # Output:
            # +---+-----+
            # |age| name|
            # +---+-----+
            # |  5|  Bob|
            # |  2|  Bob|
            # |  2|Alice|
            # +---+-----+
            ```
        """
        col_args = cols
        if cols is None:
            return self._from_logical_plan(
                Sort.from_session_state(self._logical_plan, [], self._session_state),
                self._session_state,
            )
        elif not isinstance(cols, List):
            col_args = [cols]

        # parse the ascending arguments
        bool_ascending = []
        using_default_ascending = False
        if ascending is None:
            using_default_ascending = True
            bool_ascending = [True] * len(col_args)
        elif isinstance(ascending, bool):
            bool_ascending = [ascending] * len(col_args)
        elif isinstance(ascending, List):
            bool_ascending = ascending
            if len(bool_ascending) != len(cols):
                raise ValueError(
                    f"the list length of ascending sort strategies must match the specified sort columns"
                    f"Got {len(cols)} column expressions and {len(bool_ascending)} ascending strategies. "
                )
        else:
            raise TypeError(
                f"Invalid ascending strategy type: {type(ascending)}.  Must be a boolean or list of booleans."
            )

        # create our list of sort expressions, for each column expression
        # that isn't already provided as a asc()/desc() SortExpr
        sort_exprs = []
        for c, asc_bool in zip(col_args, bool_ascending, strict=True):
            if isinstance(c, ColumnOrName):
                c_expr = Column._from_col_or_name(c)._logical_expr
            else:
                raise TypeError(
                    f"Invalid column type: {type(c).__name__}.  Must be a string or Column Expression."
                )
            if not isinstance(asc_bool, bool):
                raise TypeError(
                    f"Invalid ascending strategy type: {type(asc_bool).__name__}.  Must be a boolean."
                )
            if isinstance(c_expr, SortExpr):
                if not using_default_ascending:
                    raise TypeError(
                        "Cannot specify both asc()/desc() expressions and boolean ascending strategies."
                        f"Got expression: {c_expr} and ascending argument: {bool_ascending}"
                    )
                sort_exprs.append(c_expr)
            else:
                sort_exprs.append(SortExpr(c_expr, ascending=asc_bool))

        return self._from_logical_plan(
            Sort.from_session_state(self._logical_plan, sort_exprs, self._session_state),
            self._session_state,
        )

    def order_by(
        self,
        cols: Union[ColumnOrName, List[ColumnOrName], None] = None,
        ascending: Optional[Union[bool, List[bool]]] = None,
    ) -> DataFrame:
        """Sort the DataFrame by the specified columns. Alias for sort().

        Returns:
            DataFrame: sorted Dataframe.

        See Also:
            sort(): Full documentation of sorting behavior and parameters.
        """
        return self.sort(cols, ascending)

    def unnest(self, *col_names: str) -> DataFrame:
        """Unnest the specified struct columns into separate columns.

        This operation flattens nested struct data by expanding each field of a struct
        into its own top-level column.

        For each specified column containing a struct:
        1. Each field in the struct becomes a separate column.
        2. New columns are named after the corresponding struct fields.
        3. The new columns are inserted into the DataFrame in place of the original struct column.
        4. The overall column order is preserved.

        Args:
            *col_names: One or more struct columns to unnest. Each can be a string (column name)
                or a Column expression.

        Returns:
            DataFrame: A new DataFrame with the specified struct columns expanded.

        Raises:
            TypeError: If any argument is not a string or Column.
            ValueError: If a specified column does not contain struct data.

        Example: Unnest struct column
            ```python
            # Create sample DataFrame
            df = session.create_dataframe({
                "id": [1, 2],
                "tags": [{"red": 1, "blue": 2}, {"red": 3}],
                "name": ["Alice", "Bob"]
            })

            # Unnest the tags column
            df.unnest(col("tags")).show()
            # Output:
            # +---+---+----+-----+
            # | id| red|blue| name|
            # +---+---+----+-----+
            # |  1|  1|   2|Alice|
            # |  2|  3|null|  Bob|
            # +---+---+----+-----+
            ```

        Example: Unnest multiple struct columns
            ```python
            # Create sample DataFrame with multiple struct columns
            df = session.create_dataframe({
                "id": [1, 2],
                "tags": [{"red": 1, "blue": 2}, {"red": 3}],
                "info": [{"age": 25, "city": "NY"}, {"age": 30, "city": "LA"}],
                "name": ["Alice", "Bob"]
            })

            # Unnest multiple struct columns
            df.unnest(col("tags"), col("info")).show()
            # Output:
            # +---+---+----+---+----+-----+
            # | id| red|blue|age|city| name|
            # +---+---+----+---+----+-----+
            # |  1|  1|   2| 25|  NY|Alice|
            # |  2|  3|null| 30|  LA|  Bob|
            # +---+---+----+---+----+-----+
            ```
        """
        if not col_names:
            return self
        exprs = []
        for c in col_names:
            if c not in self.columns:
                raise TypeError(f"Column {c} not found in DataFrame.")
            exprs.append(col(c)._logical_expr)
        return self._from_logical_plan(
            Unnest.from_session_state(self._logical_plan, exprs, self._session_state),
            self._session_state,
        )

    @classmethod
    def _ensure_same_session(
            cls,
            session_state: BaseSessionState,
            other_session_states: List[BaseSessionState]):
        """Ensure that any session context passed are from the same session context.

        This check prevents accidental combinations of DataFrames created in different
        sessions, which can lead to inconsistent behavior due to differing configurations,
        catalogs, or function registries.
        """
        for other_session_state in other_session_states:
            if session_state is not other_session_state:
                raise SessionError(
                    "Cannot combine DataFrames created in different sessions. "
                    "This operation requires all inputs to belong to the same session context. "
                    "Make sure that you're not mixing DataFrames from different interactive environments, notebooks, or clients."
                )


DataFrame.show = validate_call(config=ConfigDict(strict=True))(DataFrame.show)
DataFrame.select = validate_call(
    config=ConfigDict(arbitrary_types_allowed=True, strict=True)
)(DataFrame.select)
DataFrame.where = validate_call(
    config=ConfigDict(arbitrary_types_allowed=True, strict=True)
)(DataFrame.where)
DataFrame.filter = validate_call(
    config=ConfigDict(arbitrary_types_allowed=True, strict=True)
)(DataFrame.filter)
DataFrame.with_column = validate_call(
    config=ConfigDict(arbitrary_types_allowed=True, strict=True)
)(DataFrame.with_column)
DataFrame.with_columns = validate_call(
    config=ConfigDict(arbitrary_types_allowed=True, strict=True)
)(DataFrame.with_columns)
DataFrame.with_column_renamed = validate_call(
    config=ConfigDict(strict=True)
)(DataFrame.with_column_renamed)
DataFrame.withColumnRenamed = DataFrame.with_column_renamed
DataFrame.withColumn = DataFrame.with_column
DataFrame.withColumns = DataFrame.with_columns
DataFrame.drop = validate_call(
    config=ConfigDict(strict=True)
)(DataFrame.drop)
DataFrame.union = validate_call(
    config=ConfigDict(arbitrary_types_allowed=True, strict=True)
)(DataFrame.union)
DataFrame.unionAll = DataFrame.union
DataFrame.limit = validate_call(
    config=ConfigDict(arbitrary_types_allowed=True, strict=True)
)(DataFrame.limit)
DataFrame.join = validate_call(
    config=ConfigDict(arbitrary_types_allowed=True, strict=True)
)(DataFrame.join)
DataFrame.explode = validate_call(
    config=ConfigDict(arbitrary_types_allowed=True, strict=True)
)(DataFrame.explode)
DataFrame.group_by = validate_call(
    config=ConfigDict(arbitrary_types_allowed=True, strict=True)
)(DataFrame.group_by)
DataFrame.groupBy = DataFrame.group_by
DataFrame.groupby = DataFrame.group_by
DataFrame.agg = validate_call(
    config=ConfigDict(arbitrary_types_allowed=True, strict=True)
)(DataFrame.agg)
DataFrame.drop_duplicates = validate_call(
    config=ConfigDict(arbitrary_types_allowed=True, strict=True)
)(DataFrame.drop_duplicates)
DataFrame.dropDuplicates = DataFrame.drop_duplicates
DataFrame.sort = validate_call(
    config=ConfigDict(arbitrary_types_allowed=True, strict=True)
)(DataFrame.sort)
DataFrame.orderBy = DataFrame.order_by
DataFrame.orderBy = DataFrame.order_by
DataFrame.unnest = validate_call(
    config=ConfigDict(strict=True)
)(DataFrame.unnest)
