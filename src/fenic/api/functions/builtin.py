"""Built-in functions for Fenic DataFrames."""

from functools import wraps
from typing import Callable, List, Optional, Tuple, Union

from pydantic import ConfigDict, validate_call

from fenic.api.column import Column, ColumnOrName
from fenic.api.functions.core import lit
from fenic.core._logical_plan.expressions import (
    ArrayContainsExpr,
    ArrayExpr,
    ArrayLengthExpr,
    AvgExpr,
    CoalesceExpr,
    CountExpr,
    FirstExpr,
    ListExpr,
    MaxExpr,
    MinExpr,
    StdDevExpr,
    StructExpr,
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
def avg(column: ColumnOrName) -> Column:
    """Aggregate function: returns the average (mean) of all values in the specified column.

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
def asc(column: ColumnOrName) -> Column:
    """Creates a Column expression representing an ascending sort order.

    Args:
        column: The column to apply the ascending ordering to.

    Returns:
        A Column expression representing the column and the ascending sort order.

    Raises:
        ValueError: If the type of the column cannot be inferred.
        Error: If this expression is passed to a dataframe operation besides sort() and order_by().
    """
    return Column._from_col_or_name(column).asc()


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def asc_nulls_first(column: ColumnOrName) -> Column:
    """Creates a Column expression representing an ascending sort order with nulls first.

    Args:
        column: The column to apply the ascending ordering to.

    Returns:
        A Column expression representing the column and the ascending sort order with nulls first.

    Raises:
        ValueError: If the type of the column cannot be inferred.
        Error: If this expression is passed to a dataframe operation besides sort() and order_by().
    """
    return Column._from_col_or_name(column).asc_nulls_first()


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def asc_nulls_last(column: ColumnOrName) -> Column:
    """Creates a Column expression representing an ascending sort order with nulls last.

    Args:
        column: The column to apply the ascending ordering to.

    Returns:
        A Column expression representing the column and the ascending sort order with nulls last.

    Raises:
        ValueError: If the type of the column cannot be inferred.
        Error: If this expression is passed to a dataframe operation besides sort() and order_by().
    """
    return Column._from_col_or_name(column).asc_nulls_last()


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def desc(column: ColumnOrName) -> Column:
    """Creates a Column expression representing a descending sort order.

    Args:
        column: The column to apply the descending ordering to.

    Returns:
        A Column expression representing the column and the descending sort order.

    Raises:
        ValueError: If the type of the column cannot be inferred.
        Error: If this expression is passed to a dataframe operation besides sort() and order_by().
    """
    return Column._from_col_or_name(column).desc()


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def desc_nulls_first(column: ColumnOrName) -> Column:
    """Creates a Column expression representing a descending sort order with nulls first.

    Args:
        column: The column to apply the descending ordering to.

    Returns:
        A Column expression representing the column and the descending sort order with nulls first.

    Raises:
        ValueError: If the type of the column cannot be inferred.
        Error: If this expression is passed to a dataframe operation besides sort() and order_by().
    """
    return Column._from_col_or_name(column).desc_nulls_first()


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def desc_nulls_last(column: ColumnOrName) -> Column:
    """Creates a Column expression representing a descending sort order with nulls last.

    Args:
        column: The column to apply the descending ordering to.

    Returns:
        A Column expression representing the column and the descending sort order with nulls last.

    Raises:
        ValueError: If the type of the column cannot be inferred.
        Error: If this expression is passed to a dataframe operation besides sort() and order_by().
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
def when(condition: Column, value: Column) -> Column:
    """Evaluates a condition and returns a value if true.

    This function is used to create conditional expressions. If Column.otherwise() is not invoked,
    None is returned for unmatched conditions.

    Args:
        condition: A boolean Column expression to evaluate.

        value: A Column expression to return if the condition is true.

    Returns:
        A Column expression that evaluates the condition and returns the specified value when true,
        and None otherwise.

    Raises:
        TypeError: If the condition is not a boolean Column expression.

    Example: Basic conditional expression
        ```python
        # Basic usage
        df.select(when(col("age") > 18, lit("adult")))

        # With otherwise
        df.select(when(col("age") > 18, lit("adult")).otherwise(lit("minor")))
        ```
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
        *cols: Column expressions or column names to evaluate. Can be:

            - Individual arguments
            - Lists of columns/column names
            - Tuples of columns/column names

    Returns:
        A Column expression containing the first non-null value from the input columns.

    Raises:
        ValueError: If no columns are provided.

    Example: Basic coalesce usage
        ```python
        # Basic usage
        df.select(coalesce("col1", "col2", "col3"))

        # With nested collections
        df.select(coalesce(["col1", "col2"], "col3"))
        ```
    """
    if not cols:
        raise ValidationError("No columns were provided. Please specify at least one column to use with the coalesce method.")

    flattened_args = []
    for arg in cols:
        if isinstance(arg, (list, tuple)):
            flattened_args.extend(arg)
        else:
            flattened_args.append(arg)

    flattened_exprs = [
        Column._from_col_or_name(c)._logical_expr for c in flattened_args
    ]
    return Column._from_logical_expr(CoalesceExpr(flattened_exprs))
