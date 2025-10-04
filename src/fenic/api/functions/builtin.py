"""Built-in functions for Fenic DataFrames."""

import inspect
from functools import wraps
from typing import Any, Awaitable, Callable, List, Optional, Tuple, Union

from pydantic import ConfigDict, validate_call

from fenic.api.column import Column, ColumnOrName
from fenic.api.functions.core import lit
from fenic.core._logical_plan.expressions import (
    ApproxCountDistinctExpr,
    ArrayCompactExpr,
    ArrayContainsExpr,
    ArrayDistinctExpr,
    ArrayExceptExpr,
    ArrayExpr,
    ArrayIntersectExpr,
    ArrayLengthExpr,
    ArrayMaxExpr,
    ArrayMinExpr,
    ArrayRemoveExpr,
    ArrayRepeatExpr,
    ArrayReverseExpr,
    ArraySliceExpr,
    ArraySortExpr,
    ArraysOverlapExpr,
    ArrayUnionExpr,
    AsyncUDFExpr,
    AvgExpr,
    CoalesceExpr,
    CountDistinctExpr,
    CountExpr,
    ElementAtExpr,
    FirstExpr,
    FlattenExpr,
    GreatestExpr,
    LeastExpr,
    ListExpr,
    MaxExpr,
    MinExpr,
    StdDevExpr,
    StructExpr,
    SumDistinctExpr,
    SumExpr,
    UDFExpr,
    WhenExpr,
)
from fenic.core.error import ValidationError
from fenic.core.types import DataType
from fenic.core.types.datatypes import _is_logical_type

"""Built-in functions."""


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def sum(column: ColumnOrName) -> Column:
    """Aggregate function: returns the sum of all values in the specified column.

    Args:
        column: Column or column name to compute the sum of

    Returns:
        A Column expression representing the sum aggregation

    Raises:
        TypeError: If column is not a Column or string
    """
    return Column._from_logical_expr(
        SumExpr(Column._from_col_or_name(column)._logical_expr)
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def sum_distinct(column: ColumnOrName) -> Column:
    """Aggregate function: returns the sum of distinct numeric values in the specified column.

    Args:
        column: Column or column name to compute the sum of distinct values

    Returns:
        A Column expression representing the sum-distinct aggregation

    Example: Sum of distinct values per group
        ```python
        # Sample input
        df = session.create_dataframe({
            "k": ["a", "a", "b", "b"],
            "v": [1, None, 2, 2],
        })

        # Sum distinct values of column `v` within each group `k`
        df.group_by(fc.col("k")).agg(
            fc.sum_distinct(fc.col("v")).alias("sum_distinct_v")
        ).show()
        # Output:
        # +---+----------------+
        # | k | sum_distinct_v |
        # +---+----------------+
        # | a |              1 |
        # | b |              2 |
        # +---+----------------+
        ```

    Example: Nulls are ignored when summing distinct values
        ```python
        df = session.create_dataframe({"k": ["x", "x"], "v": [None, 3]})
        df.group_by(fc.col("k")).agg(fc.sum_distinct(fc.col("v")).alias("sd")).show()
        # Output:
        # +---+----+
        # | k | sd |
        # +---+----+
        # | x |  3 |
        # +---+----+
        ```

    Raises:
        TypeMismatchError: If column is not a numeric or boolean type
    """
    return Column._from_logical_expr(
        SumDistinctExpr(Column._from_col_or_name(column)._logical_expr)
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def avg(column: ColumnOrName) -> Column:
    """Aggregate function: returns the average (mean) of all values in the specified column. Applies to numeric and embedding types.

    Args:
        column: Column or column name to compute the average of

    Returns:
        A Column expression representing the average aggregation

    Raises:
        TypeError: If column is not a Column or string
    """
    return Column._from_logical_expr(
        AvgExpr(Column._from_col_or_name(column)._logical_expr)
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def mean(column: ColumnOrName) -> Column:
    """Aggregate function: returns the mean (average) of all values in the specified column.

    Alias for avg().

    Args:
        column: Column or column name to compute the mean of

    Returns:
        A Column expression representing the mean aggregation

    Raises:
        TypeError: If column is not a Column or string
    """
    return Column._from_logical_expr(
        AvgExpr(Column._from_col_or_name(column)._logical_expr)
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def min(column: ColumnOrName) -> Column:
    """Aggregate function: returns the minimum value in the specified column.

    Args:
        column: Column or column name to compute the minimum of

    Returns:
        A Column expression representing the minimum aggregation

    Raises:
        TypeError: If column is not a Column or string
    """
    return Column._from_logical_expr(
        MinExpr(Column._from_col_or_name(column)._logical_expr)
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def max(column: ColumnOrName) -> Column:
    """Aggregate function: returns the maximum value in the specified column.

    Args:
        column: Column or column name to compute the maximum of

    Returns:
        A Column expression representing the maximum aggregation

    Raises:
        TypeError: If column is not a Column or string
    """
    return Column._from_logical_expr(
        MaxExpr(Column._from_col_or_name(column)._logical_expr)
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def count(column: ColumnOrName) -> Column:
    """Aggregate function: returns the count of non-null values in the specified column.

    Args:
        column: Column or column name to count values in

    Returns:
        A Column expression representing the count aggregation

    Raises:
        TypeError: If column is not a Column or string
    """
    if isinstance(column, str) and column == "*":
        return Column._from_logical_expr(CountExpr(lit("*")._logical_expr))
    return Column._from_logical_expr(
        CountExpr(Column._from_col_or_name(column)._logical_expr)
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def count_distinct(*cols: ColumnOrName) -> Column:
    """Aggregate function: returns the number of distinct non-null rows across one or more columns.

    Behavior: Any row where one or more inputs is null is ignored.

    Args:
        *cols: One or more columns or column names to include in the distinct count.

    Returns:
        A Column expression representing the count-distinct aggregation over the provided columns.

    Example: Distinct count per group (single column)
        ```python
        # Sample input
        df = session.create_dataframe({
            "k": ["a", "a", "b", "b"],
            "v": [1, None, 2, 2],
        })

        df.group_by(fc.col("k")).agg(
            fc.count_distinct(fc.col("v")).alias("num_unique_v")
        ).show()
        # Output:
        # +---+--------------+
        # | k | num_unique_v |
        # +---+--------------+
        # | a |            1 |
        # | b |            1 |
        # +---+--------------+
        ```

    Example: Distinct count across multiple columns (whole DataFrame)
        ```python
        # Sample input
        df = session.create_dataframe({
            "a": [1, 1, 2, 2, None],
            "b": ["x", "x", "y", "y", "z"],
        })

        df.agg(
            fc.count_distinct(fc.col("a"), fc.col("b")).alias("num_unique_pairs")
        ).show()
        # Output:
        # +------------------+
        # | num_unique_pairs |
        # +------------------+
        # |                2 |
        # +------------------+
        ```

    Example: Nulls in any input column are ignored for multi-column distinct
        ```python
        df = session.create_dataframe({"a": [1, 1, None], "b": [1, 2, 1]})
        df.agg(fc.count_distinct(fc.col("a"), fc.col("b")).alias("cd")).show()
        # Output:
        # +----+
        # | cd |
        # +----+
        # |  2 |
        # +----+
        ```

    Raises:
        ValidationError: If no columns are provided.
        TypeMismatchError: If a column has an unsupported type
    """
    if not cols:
        raise ValidationError("count_distinct requires at least one column")
    exprs = [Column._from_col_or_name(c)._logical_expr for c in cols]
    return Column._from_logical_expr(CountDistinctExpr(exprs))


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def collect_list(column: ColumnOrName) -> Column:
    """Aggregate function: collects all values from the specified column into a list.

    Args:
        column: Column or column name to collect values from

    Returns:
        A Column expression representing the list aggregation

    Raises:
        TypeError: If column is not a Column or string
    """
    return Column._from_logical_expr(
        ListExpr(Column._from_col_or_name(column)._logical_expr)
    )

@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def approx_count_distinct(column: ColumnOrName) -> Column:
    """Aggregate function: returns an approximate count (HyperLogLog++) of distinct non-null values.

    Args:
        column: Column or column name to approximately count distinct values in. Cannot be a StructType column.

    Returns:
        A Column expression representing the approximate count-distinct aggregation

    Note:
        Differs from the pyspark implementation in that the relative standard deviation is not configurable.

    Example: Approximate distinct count per group
        ```python
        # Sample input
        df = session.create_dataframe({
            "k": ["a", "a", "b", "b", "b"],
            "v": [1, None, 1, 2, 3],
        })

        df.group_by(fc.col("k")).agg(
            fc.approx_count_distinct(fc.col("v")).alias("approx_unique_v")
        ).show()
        # Output:
        # +---+------------------+
        # | k | approx_unique_v  |
        # +---+------------------+
        # | a |                1 |
        # | b |                3 |
        # +---+------------------+
        ```

    Example: Nulls are ignored in approximate distinct counts
        ```python
        df = session.create_dataframe({"k": ["x", "x"], "v": [None, 3]})
        df.group_by(fc.col("k")).agg(fc.approx_count_distinct(fc.col("v")).alias("acd")).show()
        # Output:
        # +---+-----+
        # | k | acd |
        # +---+-----+
        # | x |   1 |
        # +---+-----+
        ```

    Raises:
        TypeMismatchError: If column is a StructType or ArrayType<StructType> column.
    """
    return Column._from_logical_expr(
        ApproxCountDistinctExpr(Column._from_col_or_name(column)._logical_expr)
    )

@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def array_agg(column: ColumnOrName) -> Column:
    """Alias for collect_list()."""
    return collect_list(column)

@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def first(column: ColumnOrName) -> Column:
    """Aggregate function: returns the first non-null value in the specified column.

    Typically used in aggregations to select the first observed value per group.

    Args:
        column: Column or column name.

    Returns:
        Column expression for the first value.
    """
    return Column._from_logical_expr(
        FirstExpr(Column._from_col_or_name(column)._logical_expr)
    )

@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def stddev(column: ColumnOrName) -> Column:
    """Aggregate function: returns the sample standard deviation of the specified column.

    Args:
        column: Column or column name.

    Returns:
        Column expression for sample standard deviation.
    """
    return Column._from_logical_expr(
        StdDevExpr(Column._from_col_or_name(column)._logical_expr)
    )

@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def struct(
    *args: Union[ColumnOrName, List[ColumnOrName], Tuple[ColumnOrName, ...]]
) -> Column:
    """Creates a new struct column from multiple input columns.

    Args:
        *args: Columns or column names to combine into a struct. Can be:

            - Individual arguments
            - Lists of columns/column names
            - Tuples of columns/column names

    Returns:
        A Column expression representing a struct containing the input columns

    Raises:
        TypeError: If any argument is not a Column, string, or collection of
            Columns/strings
    """
    flattened_args = []
    for arg in args:
        if isinstance(arg, (list, tuple)):
            flattened_args.extend(arg)
        else:
            flattened_args.append(arg)

    expr_columns = [Column._from_col_or_name(c)._logical_expr for c in flattened_args]

    return Column._from_logical_expr(StructExpr(expr_columns))


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def array(
    *args: Union[ColumnOrName, List[ColumnOrName], Tuple[ColumnOrName, ...]]
) -> Column:
    """Creates a new array column from multiple input columns.

    Args:
        *args: Columns or column names to combine into an array. Can be:

            - Individual arguments
            - Lists of columns/column names
            - Tuples of columns/column names

    Returns:
        A Column expression representing an array containing values from the input columns

    Raises:
        TypeError: If any argument is not a Column, string, or collection of
            Columns/strings
    """
    flattened_args = []
    for arg in args:
        if isinstance(arg, (list, tuple)):
            flattened_args.extend(arg)
        else:
            flattened_args.append(arg)

    expr_columns = [Column._from_col_or_name(c)._logical_expr for c in flattened_args]

    return Column._from_logical_expr(ArrayExpr(expr_columns))


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def udf(f: Optional[Callable] = None, *, return_type: DataType):
    """A decorator or function for creating user-defined functions (UDFs) that can be applied to DataFrame rows.

    Warning:
        UDFs cannot be serialized and are not supported in cloud execution.
        User-defined functions contain arbitrary Python code that cannot be transmitted
        to remote workers. For cloud compatibility, use built-in fenic functions instead.

    When applied, UDFs will:
    - Access `StructType` columns as Python dictionaries (`dict[str, Any]`).
    - Access `ArrayType` columns as Python lists (`list[Any]`).
    - Access primitive types (e.g., `int`, `float`, `str`) as their respective Python types.

    Args:
        f: Python function to convert to UDF

        return_type: Expected return type of the UDF. Required parameter.

    Example: UDF with primitive types
        ```python
        # UDF with primitive types
        @udf(return_type=IntegerType)
        def add_one(x: int):
            return x + 1

        # Or
        add_one = udf(lambda x: x + 1, return_type=IntegerType)
        ```

    Example: UDF with nested types
        ```python
        # UDF with nested types
        @udf(return_type=StructType([StructField("value1", IntegerType), StructField("value2", IntegerType)]))
        def example_udf(x: dict[str, int], y: list[int]):
            return {
                "value1": x["value1"] + x["value2"] + y[0],
                "value2": x["value1"] + x["value2"] + y[1],
            }
        ```
    """

    def _create_udf(func: Callable) -> Callable:
        @wraps(func)
        def _udf_wrapper(*cols: ColumnOrName) -> Column:
            col_exprs = [Column._from_col_or_name(c)._logical_expr for c in cols]
            return Column._from_logical_expr(UDFExpr(func, col_exprs, return_type))

        return _udf_wrapper

    if _is_logical_type(return_type):
        raise NotImplementedError(f"return_type {return_type} is not supported for UDFs")

    if f is not None:
        return _create_udf(f)
    return _create_udf

@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def async_udf(
    f: Optional[Callable[..., Awaitable[Any]]] = None,
    *,
    return_type: DataType,
    max_concurrency: int = 10,
    timeout_seconds: float = 30,
    num_retries: int = 0,
):
    """A decorator for creating async user-defined functions (UDFs) with configurable concurrency and retries.

    Async UDFs allow IO-bound operations (API calls, database queries, MCP tool calls)
    to be executed concurrently while maintaining DataFrame semantics.

    Args:
        f: Async function to convert to UDF
        return_type: Expected return type of the UDF. Required parameter.
        max_concurrency: Maximum number of concurrent executions (default: 10)
        timeout_seconds: Per-item timeout in seconds (default: 30)
        num_retries: Number of retries for failed items (default: 0)

    Example: Basic async UDF
        ```python
        @async_udf(return_type=IntegerType)
        async def slow_add(x: int, y: int) -> int:
            await asyncio.sleep(1)
            return x + y

        df = df.select(slow_add(fc.col("x"), fc.col("y")).alias("slow_sum"))

        # Or
        async def slow_add_fn(x: int, y: int) -> int:
            await asyncio.sleep(1)
            return x + y

        slow_add = async_udf(
            slow_add_fn,
            return_type=IntegerType
        )
    ```

    Example: API call with custom concurrency and retries
        ```python
        @async_udf(
            return_type=StructType([
                StructField("status", IntegerType),
                StructField("data", StringType)
            ]),
            max_concurrency=20,
            timeout_seconds=5,
            num_retries=2
        )
        async def fetch_data(id: str) -> dict:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"https://api.example.com/{id}") as resp:
                    return {
                        "status": resp.status,
                        "data": await resp.text()
                    }
        ```

    Note:
        - Individual failures return None instead of raising exceptions
        - Async UDFs should not block or do CPU-intensive work, as they
          will block execution of other instances of the function call.
    """

    def _create_async_udf(func: Callable[..., Awaitable[Any]]) -> Callable:
        if not inspect.iscoroutinefunction(func):
            raise ValidationError(
                f"@async_udf requires an async function, but found a synchronous "
                f"function {func.__name__!r} of type {type(func)}"
            )

        @wraps(func)
        def _async_udf_wrapper(*cols: ColumnOrName) -> Column:
            col_exprs = [Column._from_col_or_name(c)._logical_expr for c in cols]
            return Column._from_logical_expr(
                AsyncUDFExpr(
                    func,
                    col_exprs,
                    return_type,
                    max_concurrency=max_concurrency,
                    timeout_seconds=timeout_seconds,
                    num_retries=num_retries
                )
            )
        return _async_udf_wrapper

    if _is_logical_type(return_type):
        raise NotImplementedError(f"return_type {return_type} is not supported for async UDFs")

    # Support both @async_udf and async_udf(...) syntax
    if f is None:
        return _create_async_udf
    else:
        return _create_async_udf(f)


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def asc(column: ColumnOrName) -> Column:
    """Mark this column for ascending sort order with nulls first.

    Args:
        column: The column to apply the ascending ordering to.

    Returns:
        A sort expression with ascending order and nulls first.
    """
    return Column._from_col_or_name(column).asc()


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def asc_nulls_first(column: ColumnOrName) -> Column:
    """Alias for asc().

    Args:
        column: The column to apply the ascending ordering to.

    Returns:
        A sort expression with ascending order and nulls first.
    """
    return Column._from_col_or_name(column).asc_nulls_first()


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def asc_nulls_last(column: ColumnOrName) -> Column:
    """Mark this column for ascending sort order with nulls last.

    Args:
        column: The column to apply the ascending ordering to.

    Returns:
        A Column expression representing the column and the ascending sort order with nulls last.
    """
    return Column._from_col_or_name(column).asc_nulls_last()


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def desc(column: ColumnOrName) -> Column:
    """Mark this column for descending sort order with nulls first.

    Args:
        column: The column to apply the descending ordering to.

    Returns:
        A sort expression with descending order and nulls first.
    """
    return Column._from_col_or_name(column).desc()


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def desc_nulls_first(column: ColumnOrName) -> Column:
    """Alias for desc().

    Args:
        column: The column to apply the descending ordering to.

    Returns:
        A sort expression with descending order and nulls first.
    """
    return Column._from_col_or_name(column).desc_nulls_first()


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def desc_nulls_last(column: ColumnOrName) -> Column:
    """Mark this column for descending sort order with nulls last.

    Args:
        column: The column to apply the descending ordering to.

    Returns:
        A sort expression with descending order and nulls last.
    """
    return Column._from_col_or_name(column).desc_nulls_last()


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def array_size(column: ColumnOrName) -> Column:
    """Returns the number of elements in an array column.

    This function computes the length of arrays stored in the specified column.
    Returns None for None arrays.

    Args:
        column: Column or column name containing arrays whose length to compute.

    Returns:
        A Column expression representing the array length.

    Raises:
        TypeError: If the column does not contain array data.

    Example: Get array sizes
        ```python
        # Get the size of arrays in 'tags' column
        df.select(array_size("tags"))

        # Use with column reference
        df.select(array_size(col("tags")))
        ```
    """
    return Column._from_logical_expr(
        ArrayLengthExpr(Column._from_col_or_name(column)._logical_expr)
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def array_distinct(column: ColumnOrName) -> Column:
    """Removes duplicate values from an array column.

    Args:
        column: Column or column name containing arrays.

    Returns:
        A new column that is an array of unique values from the input column.

    Notes:
        - Will attempt to preserve order of first appearances, but order is not guaranteed.

    Example:
        ```python
        # create a dataframe with an array column with duplicates
        df = session.create_dataframe({
            "array_col": [[1, 2, 2, 3], [4, 4, 4], None]
        })

        # remove duplicates
        df.select(array_distinct("array_col").alias("distinct_array"))
        # Output:
        # +--------------------+
        # | distinct_array     |
        # +--------------------+
        # | [1, 2, 3]          |
        # | [4]                |
        # | None               |
        # +--------------------+
        ```
    """
    return Column._from_logical_expr(
        ArrayDistinctExpr(Column._from_col_or_name(column)._logical_expr)
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def array_contains(
    column: ColumnOrName, value: Union[str, int, float, bool, Column]
) -> Column:
    """Checks if array column contains a specific value.

    This function returns True if the array in the specified column contains the given value,
    and False otherwise. Returns False if the array is None.

    Args:
        column: Column or column name containing the arrays to check.

        value: Value to search for in the arrays. Can be:
            - A literal value (string, number, boolean)
            - A Column expression

    Returns:
        A boolean Column expression (True if value is found, False otherwise).

    Raises:
        TypeError: If value type is incompatible with the array element type.
        TypeError: If the column does not contain array data.

    Example: Check for values in arrays
        ```python
        # Check if 'python' exists in arrays in the 'tags' column
        df.select(array_contains("tags", "python"))

        # Check using a value from another column
        df.select(array_contains("tags", col("search_term")))
        ```
    """
    value_column = None
    if isinstance(value, Column):
        value_column = value
    else:
        value_column = lit(value)
    return Column._from_logical_expr(
        ArrayContainsExpr(
            Column._from_col_or_name(column)._logical_expr, value_column._logical_expr
        )
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def array_max(column: ColumnOrName) -> Column:
    """Returns the maximum value in an array.

    Only works on arrays of comparable types (numeric, string, date, boolean).
    Returns null if the array is null or empty.

    Args:
        column: Column or column name containing arrays of comparable types
            (numeric, string, date, boolean). Does not work on arrays of structs.

    Returns:
        A Column containing the maximum value from each array. Returns the element
        type of the array (e.g., int for array of ints).

    Raises:
        TypeMismatchError: If array contains non-comparable element types (e.g., structs).

    Example: Finding maximum in numeric arrays
        ```python
        import fenic as fc

        df = fc.Session.local().create_dataframe({
            "numbers": [[3, 1, 5, 2], [10, 20], None, []]
        })

        result = df.select(fc.array_max("numbers").alias("max_value"))
        # Output:
        # ┌───────────┐
        # │ max_value │
        # ├───────────┤
        # │ 5         │
        # │ 20        │
        # │ null      │
        # │ null      │
        # └───────────┘
        ```

    Example: Finding maximum in string arrays
        ```python
        df = fc.Session.local().create_dataframe({
            "words": [["cat", "apple", "zebra"], ["dog", "bat"]]
        })

        result = df.select(fc.array_max("words").alias("max_word"))
        # Output: ["zebra", "dog"]
        ```
    """
    return Column._from_logical_expr(
        ArrayMaxExpr(Column._from_col_or_name(column)._logical_expr)
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def array_min(column: ColumnOrName) -> Column:
    """Returns the minimum value in an array.

    Only works on arrays of comparable types (numeric, string, date, boolean).
    Returns null if the array is null or empty.

    Args:
        column: Column or column name containing arrays of comparable types
            (numeric, string, date, boolean). Does not work on arrays of structs.

    Returns:
        A Column containing the minimum value from each array. Returns the element
        type of the array (e.g., int for array of ints).

    Raises:
        TypeMismatchError: If array contains non-comparable element types (e.g., structs).

    Example: Finding minimum in numeric arrays
        ```python
        import fenic as fc

        df = fc.Session.local().create_dataframe({
            "numbers": [[3, 1, 5, 2], [10, 20], None, []]
        })

        result = df.select(fc.array_min("numbers").alias("min_value"))
        # Output:
        # ┌───────────┐
        # │ min_value │
        # ├───────────┤
        # │ 1         │
        # │ 10        │
        # │ null      │
        # │ null      │
        # └───────────┘
        ```

    Example: Finding minimum in string arrays
        ```python
        df = fc.Session.local().create_dataframe({
            "words": [["cat", "apple", "zebra"], ["dog", "bat"]]
        })

        result = df.select(fc.array_min("words").alias("min_word"))
        # Output: ["apple", "bat"]
        ```
    """
    return Column._from_logical_expr(
        ArrayMinExpr(Column._from_col_or_name(column)._logical_expr)
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def array_sort(column: ColumnOrName) -> Column:
    """Sorts the array in ascending order.

    Only works on arrays of comparable types (numeric, string, date, boolean).
    Null values are placed at the end of the array.

    Args:
        column: Column or column name containing arrays of comparable types
            (numeric, string, date, boolean). Does not work on arrays of structs.

    Returns:
        A Column with sorted arrays in ascending order. Returns null if the input
        array is null.

    Raises:
        TypeMismatchError: If array contains non-comparable element types (e.g., structs).

    Note:
        Unlike PySpark's array_sort, this does not support a custom comparator function.
        For custom sorting logic on complex types, consider using other transformations.

    Example: Sorting numeric arrays
        ```python
        import fenic as fc

        df = fc.Session.local().create_dataframe({
            "numbers": [[3, 1, 5, 2], [10, 30, 20], None]
        })

        result = df.select(fc.array_sort("numbers").alias("sorted"))
        # Output:
        # ┌────────────────┐
        # │ sorted         │
        # ├────────────────┤
        # │ [1, 2, 3, 5]   │
        # │ [10, 20, 30]   │
        # │ null           │
        # └────────────────┘
        ```

    Example: Sorting string arrays
        ```python
        df = fc.Session.local().create_dataframe({
            "words": [["cat", "apple", "bat"], ["zebra", "apple"]]
        })

        result = df.select(fc.array_sort("words").alias("sorted"))
        # Output: [["apple", "bat", "cat"], ["apple", "zebra"]]
        ```
    """
    return Column._from_logical_expr(
        ArraySortExpr(Column._from_col_or_name(column)._logical_expr)
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def reverse(column: ColumnOrName) -> Column:
    """Reverses the elements of an array.

    Returns a new array with elements in reverse order. Returns null if the input
    array is null.

    Args:
        column: Column or column name containing arrays.

    Returns:
        A Column with reversed arrays.

    Example: Reversing arrays
        ```python
        import fenic as fc

        df = fc.Session.local().create_dataframe({
            "numbers": [[1, 2, 3, 4], [10, 20]],
            "words": [["a", "b", "c"], ["x", "y"]]
        })

        result = df.select(
            fc.reverse("numbers").alias("reversed_nums"),
            fc.reverse("words").alias("reversed_words")
        )
        # Output:
        # ┌────────────────┬─────────────────┐
        # │ reversed_nums  │ reversed_words  │
        # ├────────────────┼─────────────────┤
        # │ [4, 3, 2, 1]   │ ["c", "b", "a"] │
        # │ [20, 10]       │ ["y", "x"]      │
        # └────────────────┴─────────────────┘
        ```
    """
    return Column._from_logical_expr(
        ArrayReverseExpr(Column._from_col_or_name(column)._logical_expr)
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def array_remove(column: ColumnOrName, element: Union[str, int, float, bool, Column]) -> Column:
    """Removes all occurrences of an element from an array.

    Returns a new array with all instances of the specified element removed.
    Returns null if the input array is null.

    Args:
        column: Column or column name containing arrays.
        element: Element to remove from the arrays. Can be a literal value or a Column expression.

    Returns:
        A Column with arrays having all occurrences of the element removed.

    Example: Removing literals
        ```python
        import fenic as fc

        df = fc.Session.local().create_dataframe({
            "tags": [["a", "b", "a", "c"], ["x", "y", "x"]],
            "numbers": [[1, 2, 1, 3], [5, 5, 5]]
        })

        result = df.select(
            fc.array_remove("tags", "a").alias("no_a"),
            fc.array_remove("numbers", 5).alias("no_five")
        )
        # Output:
        # ┌─────────────┬──────────┐
        # │ no_a        │ no_five  │
        # ├─────────────┼──────────┤
        # │ ["b", "c"]  │ [1, 2, 1, 3] │
        # │ ["x", "y"]  │ []       │
        # └─────────────┴──────────┘
        ```

    Example: Removing with column expression
        ```python
        df = fc.Session.local().create_dataframe({
            "values": [[1, 2, 3], [4, 5, 6]],
            "to_remove": [2, 5]
        })

        result = df.select(fc.array_remove("values", fc.col("to_remove")))
        # Output: [[1, 3], [4, 6]]
        ```
    """
    element_column = element if isinstance(element, Column) else lit(element)
    return Column._from_logical_expr(
        ArrayRemoveExpr(
            Column._from_col_or_name(column)._logical_expr,
            element_column._logical_expr
        )
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def array_union(col1: ColumnOrName, col2: ColumnOrName) -> Column:
    """Returns the union of two arrays without duplicates.

    Returns all distinct elements from both arrays. The order of elements is not
    guaranteed. Returns null if either input array is null.

    Args:
        col1: First array column or column name.
        col2: Second array column or column name.

    Returns:
        A Column containing the distinct union of both arrays.

    Example: Union of tag arrays
        ```python
        import fenic as fc

        df = fc.Session.local().create_dataframe({
            "tags1": [["a", "b", "c"], ["x", "y"]],
            "tags2": [["b", "c", "d"], ["y", "z"]]
        })

        result = df.select(fc.array_union("tags1", "tags2").alias("all_tags"))
        # Output:
        # ┌──────────────────────┐
        # │ all_tags             │
        # ├──────────────────────┤
        # │ ["a", "b", "c", "d"] │
        # │ ["x", "y", "z"]      │
        # └──────────────────────┘
        ```

    Example: Union with numeric arrays
        ```python
        df = fc.Session.local().create_dataframe({
            "nums1": [[1, 2, 3], [5, 6]],
            "nums2": [[2, 3, 4], [6, 7]]
        })

        result = df.select(fc.array_union("nums1", "nums2"))
        # Output: [[1, 2, 3, 4], [5, 6, 7]]
        ```
    """
    return Column._from_logical_expr(
        ArrayUnionExpr(
            Column._from_col_or_name(col1)._logical_expr,
            Column._from_col_or_name(col2)._logical_expr
        )
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def array_intersect(col1: ColumnOrName, col2: ColumnOrName) -> Column:
    """Returns the intersection of two arrays.

    Returns distinct elements that appear in both arrays. The order of elements
    is not guaranteed. Returns null if either input array is null.

    Args:
        col1: First array column or column name.
        col2: Second array column or column name.

    Returns:
        A Column containing distinct elements present in both arrays.

    Example: Intersection of arrays
        ```python
        import fenic as fc

        df = fc.Session.local().create_dataframe({
            "arr1": [["a", "b", "c"], ["x", "y", "z"]],
            "arr2": [["b", "c", "d"], ["y", "z", "w"]]
        })

        result = df.select(fc.array_intersect("arr1", "arr2").alias("common"))
        # Output:
        # ┌────────────┐
        # │ common     │
        # ├────────────┤
        # │ ["b", "c"] │
        # │ ["y", "z"] │
        # └────────────┘
        ```

    Example: No intersection
        ```python
        df = fc.Session.local().create_dataframe({
            "arr1": [[1, 2, 3]],
            "arr2": [[4, 5, 6]]
        })

        result = df.select(fc.array_intersect("arr1", "arr2"))
        # Output: [[]]  # Empty array when no common elements
        ```
    """
    return Column._from_logical_expr(
        ArrayIntersectExpr(
            Column._from_col_or_name(col1)._logical_expr,
            Column._from_col_or_name(col2)._logical_expr
        )
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def array_except(col1: ColumnOrName, col2: ColumnOrName) -> Column:
    """Returns elements in the first array but not in the second.

    Returns distinct elements from the first array that are not present in the
    second array (set difference). Returns null if either input array is null.

    Args:
        col1: First array column or column name.
        col2: Second array column or column name.

    Returns:
        A Column containing distinct elements in col1 but not in col2.

    Example: Filtering out deprecated tags
        ```python
        import fenic as fc

        df = fc.Session.local().create_dataframe({
            "all_tags": [["a", "b", "c", "d"], ["x", "y", "z"]],
            "deprecated": [["b", "d"], ["y"]]
        })

        result = df.select(fc.array_except("all_tags", "deprecated").alias("active"))
        # Output:
        # ┌────────────┐
        # │ active     │
        # ├────────────┤
        # │ ["a", "c"] │
        # │ ["x", "z"] │
        # └────────────┘
        ```

    Example: No common elements
        ```python
        df = fc.Session.local().create_dataframe({
            "arr1": [[1, 2, 3]],
            "arr2": [[4, 5, 6]]
        })

        result = df.select(fc.array_except("arr1", "arr2"))
        # Output: [[1, 2, 3]]  # All elements retained
        ```
    """
    return Column._from_logical_expr(
        ArrayExceptExpr(
            Column._from_col_or_name(col1)._logical_expr,
            Column._from_col_or_name(col2)._logical_expr
        )
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def array_compact(column: ColumnOrName) -> Column:
    """Removes null values from an array.

    Returns a new array with all null values removed. Returns null if the input
    array itself is null.

    Args:
        column: Column or column name containing arrays.

    Returns:
        A Column with arrays having null values removed.

    Example: Removing nulls from arrays
        ```python
        import fenic as fc

        df = fc.Session.local().create_dataframe({
            "values": [[1, None, 2, None, 3], ["a", None, "b"], None]
        })

        result = df.select(fc.array_compact("values").alias("compact"))
        # Output:
        # ┌───────────┐
        # │ compact   │
        # ├───────────┤
        # │ [1, 2, 3] │
        # │ ["a", "b"]│
        # │ null      │
        # └───────────┘
        ```

    Example: All nulls removed
        ```python
        df = fc.Session.local().create_dataframe({
            "sparse": [[None, None, 1], [None]]
        })

        result = df.select(fc.array_compact("sparse"))
        # Output: [[1], []]
        ```
    """
    return Column._from_logical_expr(
        ArrayCompactExpr(Column._from_col_or_name(column)._logical_expr)
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def array_repeat(col: ColumnOrName, count: Union[int, ColumnOrName]) -> Column:
    """Creates an array containing the element repeated count times.

    Returns a new array where the element is repeated the specified number of times.
    Returns null if count is null or negative.

    Args:
        col: Column, column name, or literal value to repeat.
        count: Number of times to repeat the element. Can be an integer literal
            or a Column expression.

    Returns:
        A Column containing an array with the element repeated count times.

    Example: Repeating literals
        ```python
        import fenic as fc

        df = fc.Session.local().create_dataframe({
            "id": [1, 2, 3]
        })

        result = df.select(
            fc.array_repeat(fc.lit("x"), 3).alias("repeated"),
            fc.array_repeat(fc.lit(0), 5).alias("zeros")
        )
        # Output:
        # ┌─────────────────┬──────────────────────┐
        # │ repeated        │ zeros                │
        # ├─────────────────┼──────────────────────┤
        # │ ["x", "x", "x"] │ [0, 0, 0, 0, 0]      │
        # │ ["x", "x", "x"] │ [0, 0, 0, 0, 0]      │
        # │ ["x", "x", "x"] │ [0, 0, 0, 0, 0]      │
        # └─────────────────┴──────────────────────┘
        ```

    Example: Repeating column values
        ```python
        df = fc.Session.local().create_dataframe({
            "value": ["a", "b", "c"],
            "count": [2, 3, 1]
        })

        result = df.select(fc.array_repeat(fc.col("value"), fc.col("count")))
        # Output: [["a", "a"], ["b", "b", "b"], ["c"]]
        ```
    """
    count_column = count if isinstance(count, Column) else lit(count)
    return Column._from_logical_expr(
        ArrayRepeatExpr(
            Column._from_col_or_name(col)._logical_expr,
            count_column._logical_expr
        )
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def flatten(column: ColumnOrName) -> Column:
    """Flattens an array of arrays into a single array (one level deep).

    Flattens nested arrays by concatenating all inner arrays into a single array.
    Only flattens one level of nesting. Returns null if the input is null.

    Args:
        column: Column or column name containing arrays of arrays.

    Returns:
        A Column with flattened arrays (one level deep).

    Example: Flattening nested arrays
        ```python
        import fenic as fc

        df = fc.Session.local().create_dataframe({
            "nested": [[[1, 2], [3, 4]], [[5], [6, 7, 8]], None]
        })

        result = df.select(fc.flatten("nested").alias("flat"))
        # Output:
        # ┌──────────────────┐
        # │ flat             │
        # ├──────────────────┤
        # │ [1, 2, 3, 4]     │
        # │ [5, 6, 7, 8]     │
        # │ null             │
        # └──────────────────┘
        ```

    Example: One level only
        ```python
        # Deeply nested arrays - only flattens one level
        df = fc.Session.local().create_dataframe({
            "deep": [[[[1]], [[2]]], [[[3]]]]
        })

        result = df.select(fc.flatten("deep"))
        # Output: [[[1], [2]], [[3]]]  # Still nested after one level
        ```
    """
    return Column._from_logical_expr(
        FlattenExpr(Column._from_col_or_name(column)._logical_expr)
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def slice(column: ColumnOrName, start: Union[int, ColumnOrName], length: Union[int, ColumnOrName]) -> Column:
    """Extracts a subarray from an array using 1-based indexing (PySpark compatible).

    Extracts a contiguous subarray starting from the given position. Uses 1-based
    indexing for compatibility with PySpark. Returns null if the input array is null.

    Args:
        column: Column or column name containing arrays.
        start: Starting position (1-based index). Positive indices count from the
            start (1 = first element), negative indices count from the end
            (-1 = last element).
        length: Number of elements to extract. Must be positive.

    Returns:
        A Column with subarrays extracted.

    Example: Extracting from the start
        ```python
        import fenic as fc

        df = fc.Session.local().create_dataframe({
            "numbers": [[1, 2, 3, 4, 5], [10, 20, 30]]
        })

        result = df.select(
            fc.slice("numbers", 1, 3).alias("first_three"),
            fc.slice("numbers", 2, 2).alias("middle_two")
        )
        # Output:
        # ┌───────────────┬────────────┐
        # │ first_three   │ middle_two │
        # ├───────────────┼────────────┤
        # │ [1, 2, 3]     │ [2, 3]     │
        # │ [10, 20, 30]  │ [20, 30]   │
        # └───────────────┴────────────┘
        ```

    Example: Using negative indices
        ```python
        df = fc.Session.local().create_dataframe({
            "arr": [[1, 2, 3, 4, 5]]
        })

        # Extract last 3 elements: start at -3, take 3
        result = df.select(fc.slice("arr", -3, 3))
        # Output: [[3, 4, 5]]
        ```

    Example: Dynamic slicing with columns
        ```python
        df = fc.Session.local().create_dataframe({
            "values": [[1, 2, 3, 4, 5], [10, 20, 30]],
            "start_idx": [2, 1],
            "num_elements": [2, 2]
        })

        result = df.select(
            fc.slice("values", fc.col("start_idx"), fc.col("num_elements"))
        )
        # Output: [[2, 3], [10, 20]]
        ```
    """
    start_column = start if isinstance(start, Column) else lit(start)
    length_column = length if isinstance(length, Column) else lit(length)
    return Column._from_logical_expr(
        ArraySliceExpr(
            Column._from_col_or_name(column)._logical_expr,
            start_column._logical_expr,
            length_column._logical_expr
        )
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def element_at(column: ColumnOrName, index: Union[int, ColumnOrName]) -> Column:
    """Returns the element at the given index in an array using 1-based indexing (PySpark compatible).

    Uses 1-based indexing for compatibility with PySpark. Returns null if the
    index is out of bounds or if the input array is null.

    Args:
        column: Column or column name containing arrays.
        index: Index of the element (1-based). Positive indices count from the
            start (1 = first element), negative indices count from the end
            (-1 = last element). Can be an integer literal or a Column expression.

    Returns:
        A Column containing the element at the specified index.

    Example: Accessing with positive indices
        ```python
        import fenic as fc

        df = fc.Session.local().create_dataframe({
            "numbers": [[10, 20, 30, 40], [100, 200]]
        })

        result = df.select(
            fc.element_at("numbers", 1).alias("first"),
            fc.element_at("numbers", 2).alias("second")
        )
        # Output:
        # ┌───────┬────────┐
        # │ first │ second │
        # ├───────┼────────┤
        # │ 10    │ 20     │
        # │ 100   │ 200    │
        # └───────┴────────┘
        ```

    Example: Accessing with negative indices
        ```python
        df = fc.Session.local().create_dataframe({
            "arr": [["a", "b", "c", "d"], ["x", "y", "z"]]
        })

        result = df.select(
            fc.element_at("arr", -1).alias("last"),
            fc.element_at("arr", -2).alias("second_last")
        )
        # Output:
        # ┌──────┬─────────────┐
        # │ last │ second_last │
        # ├──────┼─────────────┤
        # │ "d"  │ "c"         │
        # │ "z"  │ "y"         │
        # └──────┴─────────────┘
        ```

    Example: Dynamic indexing with columns
        ```python
        df = fc.Session.local().create_dataframe({
            "values": [[1, 2, 3], [10, 20, 30]],
            "position": [2, 3]
        })

        result = df.select(fc.element_at("values", fc.col("position")))
        # Output: [2, 30]
        ```
    """
    index_column = index if isinstance(index, Column) else lit(index)
    return Column._from_logical_expr(
        ElementAtExpr(
            Column._from_col_or_name(column)._logical_expr,
            index_column._logical_expr
        )
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def arrays_overlap(col1: ColumnOrName, col2: ColumnOrName) -> Column:
    """Checks if two arrays have at least one common element.

    Returns true if the two arrays share at least one common element, false if they
    have no common elements. Returns null if either input array is null.

    Args:
        col1: First array column or column name.
        col2: Second array column or column name.

    Returns:
        A boolean Column (True if arrays have common elements, False otherwise).

    Example: Detecting overlap
        ```python
        import fenic as fc

        df = fc.Session.local().create_dataframe({
            "arr1": [["a", "b", "c"], ["x", "y"], ["p", "q"]],
            "arr2": [["c", "d", "e"], ["w", "z"], ["q", "r"]]
        })

        result = df.select(fc.arrays_overlap("arr1", "arr2").alias("has_overlap"))
        # Output:
        # ┌─────────────┐
        # │ has_overlap │
        # ├─────────────┤
        # │ true        │  # "c" is common
        # │ false       │  # No common elements
        # │ true        │  # "q" is common
        # └─────────────┘
        ```

    Example: Using with filtering
        ```python
        df = fc.Session.local().create_dataframe({
            "user_tags": [["python", "ml"], ["java", "web"], ["python", "web"]],
            "required": [["python", "data"], ["python", "data"], ["python", "data"]]
        })

        # Filter users with at least one required tag
        result = df.filter(fc.arrays_overlap("user_tags", "required"))
        # Output: Rows with indices 0 and 2 (have "python" tag)
        ```

    Example: Numeric arrays
        ```python
        df = fc.Session.local().create_dataframe({
            "nums1": [[1, 2, 3], [4, 5, 6]],
            "nums2": [[3, 4, 5], [7, 8, 9]]
        })

        result = df.select(fc.arrays_overlap("nums1", "nums2"))
        # Output: [true, false]
        ```
    """
    return Column._from_logical_expr(
        ArraysOverlapExpr(
            Column._from_col_or_name(col1)._logical_expr,
            Column._from_col_or_name(col2)._logical_expr
        )
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def when(condition: Column, value: Column) -> Column:
    """Evaluates a conditional expression (like if-then).

    Evaluates a condition for each row and returns a value when true.
    Can be chained with more .when() calls or finished with .otherwise().
    All branches must return the same type.

    Args:
        condition: Boolean expression to test
        value: Value to return when condition is True

    Returns:
        Column: A when expression that can be chained with more conditions

    Raises:
        TypeMismatchError: If the condition is not a boolean Column expression.

    Example:
        ```python
        # Simple if-then (returns null when false)
        df.select(fc.when(col("age") >= 18, fc.lit("adult")))

        # If-then-else
        df.select(
            fc.when(col("age") >= 18, fc.lit("adult")).otherwise(fc.lit("minor"))
        )

        # Multiple conditions (if-elif-else)
        df.select(
            when(col("score") >= 90, "A")
            .when(col("score") >= 80, "B")
            .when(col("score") >= 70, "C")
            .otherwise("F")
        )
        ```

    Note: Without .otherwise(), unmatched rows return null
    """
    return Column._from_logical_expr(
        WhenExpr(None, condition._logical_expr, value._logical_expr)
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def coalesce(*cols: ColumnOrName) -> Column:
    """Returns the first non-null value from the given columns for each row.

    This function mimics the behavior of SQL's COALESCE function. It evaluates the input columns
    in order and returns the first non-null value encountered. If all values are null, returns null.

    Args:
        *cols: Column expressions or column names to evaluate. Each argument should be a single
            column expression or column name string.

    Returns:
        A Column expression containing the first non-null value from the input columns.

    Raises:
        ValidationError: If no columns are provided.

    Example: coalesce usage
        ```python
        df.select(coalesce("col1", "col2", "col3"))
        ```
    """
    if not cols:
        raise ValidationError("No columns were provided. Please specify at least one column to use with the coalesce method.")

    exprs = [
        Column._from_col_or_name(c)._logical_expr for c in cols
    ]
    return Column._from_logical_expr(CoalesceExpr(exprs))

@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def greatest(*cols: ColumnOrName) -> Column:
    """Returns the greatest value from the given columns for each row.

    This function mimics the behavior of SQL's GREATEST function. It evaluates the input columns
    in order and returns the greatest value encountered. If all values are null, returns null.

    All arguments must be of the same primitive type (e.g., StringType, BooleanType, FloatType, IntegerType, etc).

    Args:
        *cols: Column expressions or column names to evaluate. Each argument should be a single
            column expression or column name string.

    Returns:
        A Column expression containing the greatest value from the input columns.

    Raises:
        ValidationError: If fewer than two columns are provided.

    Example: greatest usage
        ```python
        df.select(fc.greatest("col1", "col2", "col3"))
        ```
    """
    if len(cols) < 2:
        raise ValidationError(f"greatest() requires at least 2 columns, got {len(cols)}")

    exprs = [
        Column._from_col_or_name(c)._logical_expr for c in cols
    ]
    return Column._from_logical_expr(GreatestExpr(exprs))


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def least(*cols: ColumnOrName) -> Column:
    """Returns the least value from the given columns for each row.

    This function mimics the behavior of SQL's LEAST function. It evaluates the input columns
    in order and returns the least value encountered. If all values are null, returns null.

    All arguments must be of the same primitive type (e.g., StringType, BooleanType, FloatType, IntegerType, etc).

    Args:
        *cols: Column expressions or column names to evaluate. Each argument should be a single
            column expression or column name string.

    Returns:
        A Column expression containing the least value from the input columns.

    Raises:
        ValidationError: If fewer than two columns are provided.

    Example: least usage
        ```python
        df.select(fc.least("col1", "col2", "col3"))
        ```
    """
    if len(cols) < 2:
        raise ValidationError(f"least() requires at least 2 columns, got {len(cols)}")

    exprs = [
        Column._from_col_or_name(c)._logical_expr for c in cols
    ]
    return Column._from_logical_expr(LeastExpr(exprs))
