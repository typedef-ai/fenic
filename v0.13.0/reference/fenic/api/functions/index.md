# fenic.api.functions

Canonical HTML: https://docs.fenic.ai/latest/reference/fenic/api/functions/

Functions for working with DataFrame columns.

Note: Array functions are available via fc.arr.\* namespace (e.g., fc.arr.size()).

Functions:

- **`approx_count_distinct`**
  –

  Aggregate function: returns an approximate count (HyperLogLog++) of distinct non-null values.
- **`array_agg`**
  –

  Alias for collect_list().
- **`asc`**
  –

  Mark this column for ascending sort order with nulls first.
- **`asc_nulls_first`**
  –

  Alias for asc().
- **`asc_nulls_last`**
  –

  Mark this column for ascending sort order with nulls last.
- **`async_udf`**
  –

  A decorator for creating async user-defined functions (UDFs) with configurable concurrency and retries.
- **`avg`**
  –

  Aggregate function: returns the average (mean) of all values in the specified column. Applies to numeric and embedding types.
- **`coalesce`**
  –

  Returns the first non-null value from the given columns for each row.
- **`col`**
  –

  Creates a Column expression referencing a column in the DataFrame.
- **`collect_list`**
  –

  Aggregate function: collects all values from the specified column into a list.
- **`count`**
  –

  Aggregate function: returns the count of non-null values in the specified column.
- **`count_distinct`**
  –

  Aggregate function: returns the number of distinct non-null rows across one or more columns.
- **`desc`**
  –

  Mark this column for descending sort order with nulls first.
- **`desc_nulls_first`**
  –

  Alias for desc().
- **`desc_nulls_last`**
  –

  Mark this column for descending sort order with nulls last.
- **`empty`**
  –

  Creates a Column expression representing an empty value of the given type.
- **`first`**
  –

  Aggregate function: returns the first non-null value in the specified column.
- **`flatten`**
  –

  Flattens an array of arrays into a single array (one level deep).
- **`greatest`**
  –

  Returns the greatest value from the given columns for each row.
- **`least`**
  –

  Returns the least value from the given columns for each row.
- **`lit`**
  –

  Creates a Column expression representing a literal value.
- **`max`**
  –

  Aggregate function: returns the maximum value in the specified column.
- **`mean`**
  –

  Aggregate function: returns the mean (average) of all values in the specified column.
- **`min`**
  –

  Aggregate function: returns the minimum value in the specified column.
- **`null`**
  –

  Creates a Column expression representing a null value of the specified data type.
- **`stddev`**
  –

  Aggregate function: returns the sample standard deviation of the specified column.
- **`struct`**
  –

  Creates a new struct column from multiple input columns.
- **`sum`**
  –

  Aggregate function: returns the sum of all values in the specified column.
- **`sum_distinct`**
  –

  Aggregate function: returns the sum of distinct numeric values in the specified column.
- **`tool_param`**
  –

  Creates an unresolved literal placeholder column with a declared data type.
- **`udf`**
  –

  A decorator or function for creating user-defined functions (UDFs) that can be applied to DataFrame rows.
- **`when`**
  –

  Evaluates a conditional expression (like if-then).

## approx_count_distinct

```
approx_count_distinct(column: ColumnOrName) -> Column
```

Aggregate function: returns an approximate count (HyperLogLog++) of distinct non-null values.

Parameters:

- **`column`**
  (`ColumnOrName`)
  –

  Column or column name to approximately count distinct values in. Cannot be a StructType column.

Returns:

- `Column`
  –

  A Column expression representing the approximate count-distinct aggregation

Note

Differs from the pyspark implementation in that the relative standard deviation is not configurable.

Approximate distinct count per group

```
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

Nulls are ignored in approximate distinct counts

```
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

- `TypeMismatchError`
  –

  If column is a StructType or ArrayType column.

Source code in `src/fenic/api/functions/builtin.py`

```
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
```

## array_agg

```
array_agg(column: ColumnOrName) -> Column
```

Alias for collect_list().

Source code in `src/fenic/api/functions/builtin.py`

```
@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def array_agg(column: ColumnOrName) -> Column:
    """Alias for collect_list()."""
    return collect_list(column)
```

## asc

```
asc(column: ColumnOrName) -> Column
```

Mark this column for ascending sort order with nulls first.

Parameters:

- **`column`**
  (`ColumnOrName`)
  –

  The column to apply the ascending ordering to.

Returns:

- `Column`
  –

  A sort expression with ascending order and nulls first.

Source code in `src/fenic/api/functions/builtin.py`

```
@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def asc(column: ColumnOrName) -> Column:
    """Mark this column for ascending sort order with nulls first.

    Args:
        column: The column to apply the ascending ordering to.

    Returns:
        A sort expression with ascending order and nulls first.
    """
    return Column._from_col_or_name(column).asc()
```

## asc_nulls_first

```
asc_nulls_first(column: ColumnOrName) -> Column
```

Alias for asc().

Parameters:

- **`column`**
  (`ColumnOrName`)
  –

  The column to apply the ascending ordering to.

Returns:

- `Column`
  –

  A sort expression with ascending order and nulls first.

Source code in `src/fenic/api/functions/builtin.py`

```
@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def asc_nulls_first(column: ColumnOrName) -> Column:
    """Alias for asc().

    Args:
        column: The column to apply the ascending ordering to.

    Returns:
        A sort expression with ascending order and nulls first.
    """
    return Column._from_col_or_name(column).asc_nulls_first()
```

## asc_nulls_last

```
asc_nulls_last(column: ColumnOrName) -> Column
```

Mark this column for ascending sort order with nulls last.

Parameters:

- **`column`**
  (`ColumnOrName`)
  –

  The column to apply the ascending ordering to.

Returns:

- `Column`
  –

  A Column expression representing the column and the ascending sort order with nulls last.

Source code in `src/fenic/api/functions/builtin.py`

```
@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def asc_nulls_last(column: ColumnOrName) -> Column:
    """Mark this column for ascending sort order with nulls last.

    Args:
        column: The column to apply the ascending ordering to.

    Returns:
        A Column expression representing the column and the ascending sort order with nulls last.
    """
    return Column._from_col_or_name(column).asc_nulls_last()
```

## async_udf

```
async_udf(f: Optional[Callable[..., Awaitable[Any]]] = None, *, return_type: DataType, max_concurrency: int = 10, timeout_seconds: float = 30, num_retries: int = 0)
```

A decorator for creating async user-defined functions (UDFs) with configurable concurrency and retries.

Async UDFs allow IO-bound operations (API calls, database queries, MCP tool calls)
to be executed concurrently while maintaining DataFrame semantics.

Parameters:

- **`f`**
  (`Optional[Callable[..., Awaitable[Any]]]`, default:
  `None`
  )
  –

  Async function to convert to UDF
- **`return_type`**
  (`DataType`)
  –

  Expected return type of the UDF. Required parameter.
- **`max_concurrency`**
  (`int`, default:
  `10`
  )
  –

  Maximum number of concurrent executions (default: 10)
- **`timeout_seconds`**
  (`float`, default:
  `30`
  )
  –

  Per-item timeout in seconds (default: 30)
- **`num_retries`**
  (`int`, default:
  `0`
  )
  –

  Number of retries for failed items (default: 0)

Basic async UDF

```python
@async_udf(return_type=IntegerType)
async def slow_add(x: int, y: int) -> int:
await asyncio.sleep(1)
return x + y

df = df.select(slow_add(fc.col("x"), fc.col("y")).alias("slow_sum"))

### Or

async def slow_add_fn(x: int, y: int) -> int:
await asyncio.sleep(1)
return x + y

slow_add = async_udf(
slow_add_fn,
return_type=IntegerType
)

```

Example: API call with custom concurrency and retries
`python
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
}`

Note:
- Individual failures return None instead of raising exceptions
- Async UDFs should not block or do CPU-intensive work, as they
will block execution of other instances of the function call.

Source code in `src/fenic/api/functions/builtin.py`

```
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
```

## avg

```
avg(column: ColumnOrName) -> Column
```

Aggregate function: returns the average (mean) of all values in the specified column. Applies to numeric and embedding types.

Parameters:

- **`column`**
  (`ColumnOrName`)
  –

  Column or column name to compute the average of

Returns:

- `Column`
  –

  A Column expression representing the average aggregation

Raises:

- `TypeError`
  –

  If column is not a Column or string

Source code in `src/fenic/api/functions/builtin.py`

```
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
```

## coalesce

```
coalesce(*cols: ColumnOrName) -> Column
```

Returns the first non-null value from the given columns for each row.

This function mimics the behavior of SQL's COALESCE function. It evaluates the input columns
in order and returns the first non-null value encountered. If all values are null, returns null.

Parameters:

- **`*cols`**
  (`ColumnOrName`, default:
  `()`
  )
  –

  Column expressions or column names to evaluate. Each argument should be a single
  column expression or column name string.

Returns:

- `Column`
  –

  A Column expression containing the first non-null value from the input columns.

Raises:

- `ValidationError`
  –

  If no columns are provided.

coalesce usage

```
df.select(coalesce("col1", "col2", "col3"))
```

Source code in `src/fenic/api/functions/builtin.py`

```
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
```

## col

```
col(col_name: str) -> Column
```

Creates a Column expression referencing a column in the DataFrame.

Parameters:

- **`col_name`**
  (`str`)
  –

  Name of the column to reference

Returns:

- `Column`
  –

  A Column expression for the specified column

Raises:

- `TypeError`
  –

  If colName is not a string

Source code in `src/fenic/api/functions/core.py`

```
@validate_call(config=ConfigDict(strict=True))
def col(col_name: str) -> Column:
    """Creates a Column expression referencing a column in the DataFrame.

    Args:
        col_name: Name of the column to reference

    Returns:
        A Column expression for the specified column

    Raises:
        TypeError: If colName is not a string
    """
    return Column._from_column_name(col_name)
```

## collect_list

```
collect_list(column: ColumnOrName) -> Column
```

Aggregate function: collects all values from the specified column into a list.

Parameters:

- **`column`**
  (`ColumnOrName`)
  –

  Column or column name to collect values from

Returns:

- `Column`
  –

  A Column expression representing the list aggregation

Raises:

- `TypeError`
  –

  If column is not a Column or string

Source code in `src/fenic/api/functions/builtin.py`

```
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
```

## count

```
count(column: ColumnOrName) -> Column
```

Aggregate function: returns the count of non-null values in the specified column.

Parameters:

- **`column`**
  (`ColumnOrName`)
  –

  Column or column name to count values in

Returns:

- `Column`
  –

  A Column expression representing the count aggregation

Raises:

- `TypeError`
  –

  If column is not a Column or string

Source code in `src/fenic/api/functions/builtin.py`

```
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
```

## count_distinct

```
count_distinct(*cols: ColumnOrName) -> Column
```

Aggregate function: returns the number of distinct non-null rows across one or more columns.

Behavior: Any row where one or more inputs is null is ignored.

Parameters:

- **`*cols`**
  (`ColumnOrName`, default:
  `()`
  )
  –

  One or more columns or column names to include in the distinct count.

Returns:

- `Column`
  –

  A Column expression representing the count-distinct aggregation over the provided columns.

Distinct count per group (single column)

```
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

Distinct count across multiple columns (whole DataFrame)

```
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

Nulls in any input column are ignored for multi-column distinct

```
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

- `ValidationError`
  –

  If no columns are provided.
- `TypeMismatchError`
  –

  If a column has an unsupported type

Source code in `src/fenic/api/functions/builtin.py`

```
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
```

## desc

```
desc(column: ColumnOrName) -> Column
```

Mark this column for descending sort order with nulls first.

Parameters:

- **`column`**
  (`ColumnOrName`)
  –

  The column to apply the descending ordering to.

Returns:

- `Column`
  –

  A sort expression with descending order and nulls first.

Source code in `src/fenic/api/functions/builtin.py`

```
@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def desc(column: ColumnOrName) -> Column:
    """Mark this column for descending sort order with nulls first.

    Args:
        column: The column to apply the descending ordering to.

    Returns:
        A sort expression with descending order and nulls first.
    """
    return Column._from_col_or_name(column).desc()
```

## desc_nulls_first

```
desc_nulls_first(column: ColumnOrName) -> Column
```

Alias for desc().

Parameters:

- **`column`**
  (`ColumnOrName`)
  –

  The column to apply the descending ordering to.

Returns:

- `Column`
  –

  A sort expression with descending order and nulls first.

Source code in `src/fenic/api/functions/builtin.py`

```
@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def desc_nulls_first(column: ColumnOrName) -> Column:
    """Alias for desc().

    Args:
        column: The column to apply the descending ordering to.

    Returns:
        A sort expression with descending order and nulls first.
    """
    return Column._from_col_or_name(column).desc_nulls_first()
```

## desc_nulls_last

```
desc_nulls_last(column: ColumnOrName) -> Column
```

Mark this column for descending sort order with nulls last.

Parameters:

- **`column`**
  (`ColumnOrName`)
  –

  The column to apply the descending ordering to.

Returns:

- `Column`
  –

  A sort expression with descending order and nulls last.

Source code in `src/fenic/api/functions/builtin.py`

```
@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def desc_nulls_last(column: ColumnOrName) -> Column:
    """Mark this column for descending sort order with nulls last.

    Args:
        column: The column to apply the descending ordering to.

    Returns:
        A sort expression with descending order and nulls last.
    """
    return Column._from_col_or_name(column).desc_nulls_last()
```

## empty

```
empty(data_type: DataType) -> Column
```

Creates a Column expression representing an empty value of the given type.

- If the data type is `ArrayType(...)`, the empty value will be an empty array.
- If the data type is `StructType(...)`, the empty value will be an instance of the struct type with all fields set to `None`.
- For all other data types, the empty value is None (equivalent to calling `null(data_type)`)

This function is useful for creating columns with empty values of a particular type.

Parameters:

- **`data_type`**
  (`DataType`)
  –

  The data type of the empty value

Returns:

- `Column`
  –

  A Column expression representing the empty value

Raises:

- `ValidationError`
  –

  If the data type is not a valid data type

Creating a column with an empty array type

```
# The newly created `b` column will have a value of `[]` for all rows
df.select(fc.col("a"), fc.empty(fc.ArrayType(fc.IntegerType)).alias("b"))
```

Creating a column with an empty struct type

```
# The newly created `b` column will have a value of `{b: None}` for all rows
df.select(fc.col("a"), fc.empty(fc.StructType([fc.StructField("b", fc.IntegerType)])).alias("b"))
```

Creating a column with an empty primitive type

```
# The newly created `b` column will have a value of `None` for all rows
df.select(fc.col("a"), fc.empty(fc.IntegerType).alias("b"))
```

Source code in `src/fenic/api/functions/core.py`

```
def empty(data_type: DataType) -> Column:
    """Creates a Column expression representing an empty value of the given type.

    - If the data type is `ArrayType(...)`, the empty value will be an empty array.
    - If the data type is `StructType(...)`, the empty value will be an instance of the struct type with all fields set to `None`.
    - For all other data types, the empty value is None (equivalent to calling `null(data_type)`)

    This function is useful for creating columns with empty values of a particular type.

    Args:
        data_type: The data type of the empty value

    Returns:
        A Column expression representing the empty value

    Raises:
        ValidationError: If the data type is not a valid data type

    Example: Creating a column with an empty array type
        ```python
        # The newly created `b` column will have a value of `[]` for all rows
        df.select(fc.col("a"), fc.empty(fc.ArrayType(fc.IntegerType)).alias("b"))
        ```

    Example: Creating a column with an empty struct type
        ```python
        # The newly created `b` column will have a value of `{b: None}` for all rows
        df.select(fc.col("a"), fc.empty(fc.StructType([fc.StructField("b", fc.IntegerType)])).alias("b"))
        ```

    Example: Creating a column with an empty primitive type
        ```python
        # The newly created `b` column will have a value of `None` for all rows
        df.select(fc.col("a"), fc.empty(fc.IntegerType).alias("b"))
        ```
    """
    if isinstance(data_type, ArrayType):
        return Column._from_logical_expr(LiteralExpr([], data_type))
    elif isinstance(data_type, StructType):
        return Column._from_logical_expr(LiteralExpr({}, data_type))
    return null(data_type)
```

## first

```
first(column: ColumnOrName) -> Column
```

Aggregate function: returns the first non-null value in the specified column.

Typically used in aggregations to select the first observed value per group.

Parameters:

- **`column`**
  (`ColumnOrName`)
  –

  Column or column name.

Returns:

- `Column`
  –

  Column expression for the first value.

Source code in `src/fenic/api/functions/builtin.py`

```
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
```

## flatten

```
flatten(column: ColumnOrName) -> Column
```

Flattens an array of arrays into a single array (one level deep).

Flattens nested arrays by concatenating all inner arrays into a single array.
Only flattens one level of nesting. Returns null if the input is null.

Parameters:

- **`column`**
  (`ColumnOrName`)
  –

  Column or column name containing arrays of arrays.

Returns:

- `Column`
  –

  A Column with flattened arrays (one level deep).

Flattening nested arrays

```
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

One level only

```
# Deeply nested arrays - only flattens one level
df = fc.Session.local().create_dataframe({
    "deep": [[[[1]], [[2]]], [[[3]]]]
})

result = df.select(fc.flatten("deep"))
# Output: [[[1], [2]], [[3]]]  # Still nested after one level
```

Source code in `src/fenic/api/functions/builtin.py`

```
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
```

## greatest

```
greatest(*cols: ColumnOrName) -> Column
```

Returns the greatest value from the given columns for each row.

This function mimics the behavior of SQL's GREATEST function. It evaluates the input columns
in order and returns the greatest value encountered. If all values are null, returns null.

All arguments must be of the same primitive type (e.g., StringType, BooleanType, FloatType, IntegerType, etc).

Parameters:

- **`*cols`**
  (`ColumnOrName`, default:
  `()`
  )
  –

  Column expressions or column names to evaluate. Each argument should be a single
  column expression or column name string.

Returns:

- `Column`
  –

  A Column expression containing the greatest value from the input columns.

Raises:

- `ValidationError`
  –

  If fewer than two columns are provided.

greatest usage

```
df.select(fc.greatest("col1", "col2", "col3"))
```

Source code in `src/fenic/api/functions/builtin.py`

```
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
```

## least

```
least(*cols: ColumnOrName) -> Column
```

Returns the least value from the given columns for each row.

This function mimics the behavior of SQL's LEAST function. It evaluates the input columns
in order and returns the least value encountered. If all values are null, returns null.

All arguments must be of the same primitive type (e.g., StringType, BooleanType, FloatType, IntegerType, etc).

Parameters:

- **`*cols`**
  (`ColumnOrName`, default:
  `()`
  )
  –

  Column expressions or column names to evaluate. Each argument should be a single
  column expression or column name string.

Returns:

- `Column`
  –

  A Column expression containing the least value from the input columns.

Raises:

- `ValidationError`
  –

  If fewer than two columns are provided.

least usage

```
df.select(fc.least("col1", "col2", "col3"))
```

Source code in `src/fenic/api/functions/builtin.py`

```
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
```

## lit

```
lit(value: Any) -> Column
```

Creates a Column expression representing a literal value.

Column Data Type must be inferrable from the value

- Cannot be used to create a columm with the literal value `None`. Use `null(data_type)` instead.
- Cannot be used to create a columm with the literal value `[]`. Use `empty(ArrayType(...))` instead.
- Cannot be used to create a columm with the literal value `{}`. Use `empty(StructType(...))` instead.

Parameters:

- **`value`**
  (`Any`)
  –

  The literal value to create a column for

Returns:

- `Column`
  –

  A Column expression representing the literal value

Raises:
ValidationError: If the type of the value cannot be inferred

Source code in `src/fenic/api/functions/core.py`

```
def lit(value: Any) -> Column:
    """Creates a Column expression representing a literal value.

    Column Data Type must be inferrable from the value:
        - Cannot be used to create a columm with the literal value `None`. Use `null(data_type)` instead.
        - Cannot be used to create a columm with the literal value `[]`. Use `empty(ArrayType(...))` instead.
        - Cannot be used to create a columm with the literal value `{}`. Use `empty(StructType(...))` instead.

    Args:
        value: The literal value to create a column for

    Returns:
        A Column expression representing the literal value
    Raises:
        ValidationError: If the type of the value cannot be inferred
    """
    if value is None:
        raise ValidationError("Cannot create a literal with value `None`. Use `null(...)` instead.")
    elif value == []:
        raise ValidationError(f"Cannot create a literal with empty value `{value}` Use `empty(ArrayType(...))` instead.")
    elif value == {}:
        raise ValidationError(f"Cannot create a literal with empty value `{value}` Use `empty(StructType(...))` instead.")
    try:
        inferred_type = infer_dtype_from_pyobj(value)
    except TypeInferenceError as e:
        raise ValidationError(f"`lit` failed to infer type for value `{value}`") from e
    literal_expr = LiteralExpr(value, inferred_type)
    return Column._from_logical_expr(literal_expr)
```

## max

```
max(column: ColumnOrName) -> Column
```

Aggregate function: returns the maximum value in the specified column.

Parameters:

- **`column`**
  (`ColumnOrName`)
  –

  Column or column name to compute the maximum of

Returns:

- `Column`
  –

  A Column expression representing the maximum aggregation

Raises:

- `TypeError`
  –

  If column is not a Column or string

Source code in `src/fenic/api/functions/builtin.py`

```
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
```

## mean

```
mean(column: ColumnOrName) -> Column
```

Aggregate function: returns the mean (average) of all values in the specified column.

Alias for avg().

Parameters:

- **`column`**
  (`ColumnOrName`)
  –

  Column or column name to compute the mean of

Returns:

- `Column`
  –

  A Column expression representing the mean aggregation

Raises:

- `TypeError`
  –

  If column is not a Column or string

Source code in `src/fenic/api/functions/builtin.py`

```
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
```

## min

```
min(column: ColumnOrName) -> Column
```

Aggregate function: returns the minimum value in the specified column.

Parameters:

- **`column`**
  (`ColumnOrName`)
  –

  Column or column name to compute the minimum of

Returns:

- `Column`
  –

  A Column expression representing the minimum aggregation

Raises:

- `TypeError`
  –

  If column is not a Column or string

Source code in `src/fenic/api/functions/builtin.py`

```
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
```

## null

```
null(data_type: DataType) -> Column
```

Creates a Column expression representing a null value of the specified data type.

Regardless of the data type, the column will contain a null (None) value.
This function is useful for creating columns with null values of a particular type.

Parameters:

- **`data_type`**
  (`DataType`)
  –

  The data type of the null value

Returns:

- `Column`
  –

  A Column expression representing the null value

Raises:

- `ValidationError`
  –

  If the data type is not a valid data type

Creating a column with a null value of a primitive type

```
# The newly created `b` column will have a value of `None` for all rows
df.select(fc.col("a"), fc.null(fc.IntegerType).alias("b"))
```

Creating a column with a null value of an array/struct type

```
# The newly created `b` and `c` columns will have a value of `None` for all rows
df.select(
    fc.col("a"),
    fc.null(fc.ArrayType(fc.IntegerType)).alias("b"),
    fc.null(fc.StructType([fc.StructField("b", fc.IntegerType)])).alias("c"),
)
```

Source code in `src/fenic/api/functions/core.py`

```
def null(data_type: DataType) -> Column:
    """Creates a Column expression representing a null value of the specified data type.

    Regardless of the data type, the column will contain a null (None) value.
    This function is useful for creating columns with null values of a particular type.

    Args:
        data_type: The data type of the null value

    Returns:
        A Column expression representing the null value

    Raises:
        ValidationError: If the data type is not a valid data type

    Example: Creating a column with a null value of a primitive type
        ```python
        # The newly created `b` column will have a value of `None` for all rows
        df.select(fc.col("a"), fc.null(fc.IntegerType).alias("b"))
        ```

    Example: Creating a column with a null value of an array/struct type
        ```python
        # The newly created `b` and `c` columns will have a value of `None` for all rows
        df.select(
            fc.col("a"),
            fc.null(fc.ArrayType(fc.IntegerType)).alias("b"),
            fc.null(fc.StructType([fc.StructField("b", fc.IntegerType)])).alias("c"),
        )
        ```

    """
    return Column._from_logical_expr(LiteralExpr(None, data_type))
```

## stddev

```
stddev(column: ColumnOrName) -> Column
```

Aggregate function: returns the sample standard deviation of the specified column.

Parameters:

- **`column`**
  (`ColumnOrName`)
  –

  Column or column name.

Returns:

- `Column`
  –

  Column expression for sample standard deviation.

Source code in `src/fenic/api/functions/builtin.py`

```
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
```

## struct

```
struct(*args: Union[ColumnOrName, List[ColumnOrName], Tuple[ColumnOrName, ...]]) -> Column
```

Creates a new struct column from multiple input columns.

Parameters:

- **`*args`**
  (`Union[ColumnOrName, List[ColumnOrName], Tuple[ColumnOrName, ...]]`, default:
  `()`
  )
  –

  Columns or column names to combine into a struct. Can be:

  - Individual arguments
  - Lists of columns/column names
  - Tuples of columns/column names

Returns:

- `Column`
  –

  A Column expression representing a struct containing the input columns

Raises:

- `TypeError`
  –

  If any argument is not a Column, string, or collection of
  Columns/strings

Source code in `src/fenic/api/functions/builtin.py`

```
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
```

## sum

```
sum(column: ColumnOrName) -> Column
```

Aggregate function: returns the sum of all values in the specified column.

Parameters:

- **`column`**
  (`ColumnOrName`)
  –

  Column or column name to compute the sum of

Returns:

- `Column`
  –

  A Column expression representing the sum aggregation

Raises:

- `TypeError`
  –

  If column is not a Column or string

Source code in `src/fenic/api/functions/builtin.py`

```
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
```

## sum_distinct

```
sum_distinct(column: ColumnOrName) -> Column
```

Aggregate function: returns the sum of distinct numeric values in the specified column.

Parameters:

- **`column`**
  (`ColumnOrName`)
  –

  Column or column name to compute the sum of distinct values

Returns:

- `Column`
  –

  A Column expression representing the sum-distinct aggregation

Sum of distinct values per group

```
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

Nulls are ignored when summing distinct values

```
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

- `TypeMismatchError`
  –

  If column is not a numeric or boolean type

Source code in `src/fenic/api/functions/builtin.py`

```
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
```

## tool_param

```
tool_param(parameter_name: str, data_type: DataType) -> Column
```

Creates an unresolved literal placeholder column with a declared data type.

A placeholder argument for a DataFrame, representing a literal value to be provided at execution time.
If no value is supplied, it defaults to null. Enables parameterized views and macros over fenic DataFrames.

Notes

Supports only Primitive/Object/ArrayLike Types (StringType, IntegerType, FloatType, DoubleType, BooleanType, StructType, ArrayType)

Parameters:

- **`parameter_name`**
  (`str`)
  –

  The name of the parameter to reference.
- **`data_type`**
  (`DataType`)
  –

  The expected data type for the parameter value.

Returns:

- `Column`
  –

  A Column wrapping an UnresolvedLiteralExpr for the given parameter.

A simple tool with one parameter

```python

### Assume we are reading data with a `name` column.

df = session.read.csv(data.csv)
parameterized_df = df.filter(fc.col("name").contains(fc.tool_param('query', StringType)))
...
session.catalog.create_tool(
tool_name="my_tool",
tool_description="A tool that searches the name field",
tool_query=parameterized_df,
result_limit=100,
tool_params=[ToolParam(name="query", description="The name should contain the following value")]
)

A tool with multiple filters

```python

### Assume we are reading data with an `age` column.

df = session.read.csv(users.csv)

### create multiple filters that evaluate to true if a param is not passed.

optional_min = fc.coalesce(fc.col("age") >= tool_param("min_age", IntegerType), fc.lit(True))
optional_max = fc.coalesce(fc.col("age") <= tool_param("max_age", IntegerType), fc.lit(True))
core_filter = df.filter(optional_min & optional_max)
session.catalog.create_tool(
"users_filter",
"Filter users by age",
core_filter,
tool_params=[
ToolParam(name="min_age", description="Minimum age", has_default=True, default_value=None),
ToolParam(name="max_age", description="Maximum age", has_default=True, default_value=None),
]
)

Source code in `src/fenic/api/functions/core.py`

```
@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def tool_param(parameter_name: str, data_type: DataType) -> Column:
    """Creates an unresolved literal placeholder column with a declared data type.

    A placeholder argument for a DataFrame, representing a literal value to be provided at execution time. 
    If no value is supplied, it defaults to null. Enables parameterized views and macros over fenic DataFrames.

    Notes:
        Supports only Primitive/Object/ArrayLike Types (StringType, IntegerType, FloatType, DoubleType, BooleanType, StructType, ArrayType)

    Args:
        parameter_name: The name of the parameter to reference.
        data_type: The expected data type for the parameter value.

    Returns:
        A Column wrapping an UnresolvedLiteralExpr for the given parameter.

    Example: A simple tool with one parameter
        ```python
        # Assume we are reading data with a `name` column.
        df = session.read.csv(data.csv)
        parameterized_df = df.filter(fc.col("name").contains(fc.tool_param('query', StringType)))
        ...
        session.catalog.create_tool(
            tool_name="my_tool",
            tool_description="A tool that searches the name field",
            tool_query=parameterized_df,
            result_limit=100,
            tool_params=[ToolParam(name="query", description="The name should contain the following value")]
        )

    Example: A tool with multiple filters
        ```python
        # Assume we are reading data with an `age` column.
        df = session.read.csv(users.csv)
        # create multiple filters that evaluate to true if a param is not passed.
        optional_min = fc.coalesce(fc.col("age") >= tool_param("min_age", IntegerType), fc.lit(True))
        optional_max = fc.coalesce(fc.col("age") <= tool_param("max_age", IntegerType), fc.lit(True))
        core_filter = df.filter(optional_min & optional_max)
        session.catalog.create_tool(
            "users_filter",
            "Filter users by age",
            core_filter,
            tool_params=[
                ToolParam(name="min_age", description="Minimum age", has_default=True, default_value=None),
                ToolParam(name="max_age", description="Maximum age", has_default=True, default_value=None),
            ]
        )
    """
    if isinstance(data_type, _LogicalType):
        raise ValidationError(f"Cannot use a logical type as a parameter type: {data_type}")

    return Column._from_logical_expr(UnresolvedLiteralExpr(data_type=data_type, parameter_name=parameter_name))
```

## udf

```
udf(f: Optional[Callable] = None, *, return_type: DataType)
```

A decorator or function for creating user-defined functions (UDFs) that can be applied to DataFrame rows.

Warning

UDFs cannot be serialized and are not supported in cloud execution.
User-defined functions contain arbitrary Python code that cannot be transmitted
to remote workers. For cloud compatibility, use built-in fenic functions instead.

When applied, UDFs will:
- Access `StructType` columns as Python dictionaries (`dict[str, Any]`).
- Access `ArrayType` columns as Python lists (`list[Any]`).
- Access primitive types (e.g., `int`, `float`, `str`) as their respective Python types.

Parameters:

- **`f`**
  (`Optional[Callable]`, default:
  `None`
  )
  –

  Python function to convert to UDF
- **`return_type`**
  (`DataType`)
  –

  Expected return type of the UDF. Required parameter.

UDF with primitive types

```
# UDF with primitive types
@udf(return_type=IntegerType)
def add_one(x: int):
    return x + 1

# Or
add_one = udf(lambda x: x + 1, return_type=IntegerType)
```

UDF with nested types

```
# UDF with nested types
@udf(return_type=StructType([StructField("value1", IntegerType), StructField("value2", IntegerType)]))
def example_udf(x: dict[str, int], y: list[int]):
    return {
        "value1": x["value1"] + x["value2"] + y[0],
        "value2": x["value1"] + x["value2"] + y[1],
    }
```

Source code in `src/fenic/api/functions/builtin.py`

```
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
```

## when

```
when(condition: Column, value: Column) -> Column
```

Evaluates a conditional expression (like if-then).

Evaluates a condition for each row and returns a value when true.
Can be chained with more .when() calls or finished with .otherwise().
All branches must return the same type.

Parameters:

- **`condition`**
  (`Column`)
  –

  Boolean expression to test
- **`value`**
  (`Column`)
  –

  Value to return when condition is True

Returns:

- **`Column`** ( `Column`
  ) –

  A when expression that can be chained with more conditions

Raises:

- `TypeMismatchError`
  –

  If the condition is not a boolean Column expression.

Example

```
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

Source code in `src/fenic/api/functions/builtin.py`

```
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
```
