"""QueryResult class and related types."""

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Union

import pandas as pd
import polars as pl
import pyarrow as pa

from fenic.core.metrics import QueryMetrics

# Type literal defining supported data format names
DataLikeType = Literal["polars", "pandas", "pydict", "pylist", "arrow"]
"""String literal type for specifying data output formats.

Valid values:
    - "polars": Native Polars DataFrame format
    - "pandas": Pandas DataFrame with PyArrow extension arrays
    - "pydict": Python dictionary with column names as keys, lists as values
    - "pylist": Python list of dictionaries, each representing one row
    - "arrow": Apache Arrow Table format

Used as input parameter for methods that can return data in multiple formats.
"""


# Union type representing all supported data formats for input and output
DataLike = Union[
    pl.DataFrame,
    pd.DataFrame,
    Dict[str, List[Any]],
    List[Dict[str, Any]],
    pa.Table,
]
"""Union type representing any supported data format for both input and output operations.

This type encompasses all possible data structures that can be:
1. Used as input when creating DataFrames
2. Returned as output from query results

Supported formats:
    - pl.DataFrame: Native Polars DataFrame with efficient columnar storage
    - pd.DataFrame: Pandas DataFrame, optionally with PyArrow extension arrays
    - Dict[str, List[Any]]: Column-oriented dictionary where:
        * Keys are column names (str)
        * Values are lists containing all values for that column
    - List[Dict[str, Any]]: Row-oriented list where:
        * Each element is a dictionary representing one row
        * Dictionary keys are column names, values are cell values
    - pa.Table: Apache Arrow Table with columnar memory layout

Usage:
    - Input: Used in create_dataframe() to accept data in various formats
    - Output: Used in QueryResult.data to return results in requested format

The specific type returned depends on the DataLikeType format specified
when collecting query results.
"""


@dataclass
class QueryResult:
    """Container for query execution results and associated metadata.

    This dataclass bundles together the materialized data from a query execution
    along with metrics about the execution process. It provides a unified interface
    for accessing both the computed results and performance information.

    Attributes:
        data (DataLike): The materialized query results in the requested format.
            Can be any of the supported data types (Polars/Pandas DataFrame,
            Arrow Table, or Python dict/list structures).

        metrics (QueryMetrics): Execution metadata including timing information,
            memory usage, rows processed, and other performance metrics collected
            during query execution.

    Example: Access query results and metrics
        ```python
        # Execute query and get results with metrics
        result = df.filter(col("age") > 25).collect("pandas")
        pandas_df = result.data  # Access the Pandas DataFrame
        print(result.metrics.execution_time)  # Access execution metrics
        print(result.metrics.rows_processed)  # Access row count
        ```

    Example: Work with different data formats
        ```python
        # Get results in different formats
        polars_result = df.collect("polars")
        arrow_result = df.collect("arrow")
        dict_result = df.collect("pydict")

        # All contain the same data, different formats
        print(type(polars_result.data))  # <class 'polars.DataFrame'>
        print(type(arrow_result.data))   # <class 'pyarrow.lib.Table'>
        print(type(dict_result.data))    # <class 'dict'>
        ```

    Note:
        The actual type of the `data` attribute depends on the format requested
        during collection. Use type checking or isinstance() if you need to
        handle the data differently based on its format.
    """
    data: DataLike
    metrics: QueryMetrics
