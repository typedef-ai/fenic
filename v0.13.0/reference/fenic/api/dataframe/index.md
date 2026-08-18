# fenic.api.dataframe

Canonical HTML: https://docs.fenic.ai/latest/reference/fenic/api/dataframe/

DataFrame API for Fenic - provides DataFrame and grouped data operations.

Classes:

- **`DataFrame`**
  –

  A data collection organized into named columns.
- **`GroupedData`**
  –

  Methods for aggregations on a grouped DataFrame.
- **`SemanticExtensions`**
  –

  A namespace for semantic dataframe operators.

## DataFrame

A data collection organized into named columns.

The DataFrame class represents a lazily evaluated computation on data. Operations on
DataFrame build up a logical query plan that is only executed when an action like
show(), to_polars(), to_pandas(), to_arrow(), to_pydict(), to_pylist(), or count() is called.

The DataFrame supports method chaining for building complex transformations.

Create and transform a DataFrame

```
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

Methods:

- **`agg`**
  –

  Aggregate on the entire DataFrame without groups.
- **`cache`**
  –

  Alias for persist(). Mark DataFrame for caching after first computation.
- **`collect`**
  –

  Execute the DataFrame computation and return the result as a QueryResult.
- **`count`**
  –

  Count the number of rows in the DataFrame.
- **`distinct`**
  –

  Return a DataFrame with duplicate rows removed. Alias for drop_duplicates(subset=None).
- **`drop`**
  –

  Remove one or more columns from this DataFrame.
- **`drop_duplicates`**
  –

  Return a DataFrame with duplicate rows removed.
- **`explain`**
  –

  Display the logical plan of the DataFrame.
- **`explode`**
  –

  Create a new row for each element in an array column.
- **`explode_outer`**
  –

  Create a new row for each element in an array column, containing the element's position in the array and its value, and preserving null/empty arrays.
- **`explode_with_index`**
  –

  Create a new row for each element in an array column, with the element's position in the array and its value.
- **`filter`**
  –

  Filters rows using the given condition.
- **`group_by`**
  –

  Groups the DataFrame using the specified columns.
- **`join`**
  –

  Joins this DataFrame with another DataFrame.
- **`limit`**
  –

  Limits the number of rows to the specified number.
- **`lineage`**
  –

  Create a Lineage object to trace data through transformations.
- **`order_by`**
  –

  Sort the DataFrame by the specified columns. Alias for sort().
- **`persist`**
  –

  Mark this DataFrame to be persisted after first computation.
- **`posexplode`**
  –

  Create a new row for each element in an array column, with the element's position in the array and its value.
- **`posexplode_outer`**
  –

  Create a new row for each element in an array column with position and value, preserving null/empty arrays.
- **`select`**
  –

  Projects a set of Column expressions or column names.
- **`show`**
  –

  Display the DataFrame content in a tabular form.
- **`sort`**
  –

  Sort the DataFrame by the specified columns.
- **`to_arrow`**
  –

  Execute the DataFrame computation and return an Apache Arrow Table.
- **`to_pandas`**
  –

  Execute the DataFrame computation and return a Pandas DataFrame.
- **`to_polars`**
  –

  Execute the DataFrame computation and return the result as a Polars DataFrame.
- **`to_pydict`**
  –

  Execute the DataFrame computation and return a dictionary of column arrays.
- **`to_pylist`**
  –

  Execute the DataFrame computation and return a list of row dictionaries.
- **`union`**
  –

  Return a new DataFrame containing the union of rows in this and another DataFrame.
- **`unnest`**
  –

  Unnest the specified struct columns into separate columns.
- **`where`**
  –

  Filters rows using the given condition (alias for filter()).
- **`with_column`**
  –

  Add a new column or replace an existing column.
- **`with_column_renamed`**
  –

  Rename a column. No-op if the column does not exist.
- **`with_columns`**
  –

  Add multiple new columns or replace existing columns.

Attributes:

- **`columns`**
  (`List[str]`)
  –

  Get list of column names.
- **`schema`**
  (`Schema`)
  –

  Get the schema of this DataFrame.
- **`semantic`**
  (`SemanticExtensions`)
  –

  Interface for semantic operations on the DataFrame.
- **`write`**
  (`DataFrameWriter`)
  –

  Interface for saving the content of the DataFrame.

### columns

```
columns: List[str]
```

Get list of column names.

Returns:

- `List[str]`
  –

  List[str]: List of all column names in the DataFrame

Examples:

```
>>> df.columns
['name', 'age', 'city']
```

### schema

```
schema: Schema
```

Get the schema of this DataFrame.

Returns:

- **`Schema`** ( `Schema`
  ) –

  Schema containing field names and data types

Examples:

```
>>> df.schema
Schema([
    ColumnField('name', StringType),
    ColumnField('age', IntegerType)
])
```

### semantic

```
semantic: SemanticExtensions
```

Interface for semantic operations on the DataFrame.

### write

```
write: DataFrameWriter
```

Interface for saving the content of the DataFrame.

Returns:

- **`DataFrameWriter`** ( `DataFrameWriter`
  ) –

  Writer interface to write DataFrame.

### agg

```
agg(*exprs: Union[Column, Dict[str, str]]) -> DataFrame
```

Aggregate on the entire DataFrame without groups.

This is equivalent to group_by() without any grouping columns.

Parameters:

- **`*exprs`**
  (`Union[Column, Dict[str, str]]`, default:
  `()`
  )
  –

  Aggregation expressions or dictionary of aggregations.

Returns:

- **`DataFrame`** ( `DataFrame`
  ) –

  Aggregation results.

Multiple aggregations

```
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

Dictionary style

```
# Dictionary style
df.agg({col("salary"): "avg", col("age"): "max"}).show()
# Output:
# +-----------+--------+
# |avg(salary)|max(age)|
# +-----------+--------+
# |    80000.0|      35|
# +-----------+--------+
```

Source code in `src/fenic/api/dataframe/dataframe.py`

```
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
```

### cache

```
cache() -> DataFrame
```

Alias for persist(). Mark DataFrame for caching after first computation.

Returns:

- **`DataFrame`** ( `DataFrame`
  ) –

  Same DataFrame, but marked for caching

See Also

persist(): Full documentation of caching behavior

Source code in `src/fenic/api/dataframe/dataframe.py`

```
def cache(self) -> DataFrame:
    """Alias for persist(). Mark DataFrame for caching after first computation.

    Returns:
        DataFrame: Same DataFrame, but marked for caching

    See Also:
        persist(): Full documentation of caching behavior
    """
    return self.persist()
```

### collect

```
collect(data_type: DataLikeType = 'polars') -> QueryResult
```

Execute the DataFrame computation and return the result as a QueryResult.

This is an action that triggers computation of the DataFrame query plan.
All transformations and operations are executed, and the results are
materialized into a QueryResult, which contains both the result data and the query metrics.

Parameters:

- **`data_type`**
  (`DataLikeType`, default:
  `'polars'`
  )
  –

  The type of data to return

Returns:

- **`QueryResult`** ( `QueryResult`
  ) –

  A QueryResult with materialized data and query metrics.
  For `data_type="polars"`, a no-op local plan may return a frame
  that aliases its in-memory Polars input rather than copying it.

Source code in `src/fenic/api/dataframe/dataframe.py`

```
def collect(self, data_type: DataLikeType = "polars") -> QueryResult:
    """Execute the DataFrame computation and return the result as a QueryResult.

    This is an action that triggers computation of the DataFrame query plan.
    All transformations and operations are executed, and the results are
    materialized into a QueryResult, which contains both the result data and the query metrics.

    Args:
        data_type: The type of data to return

    Returns:
        QueryResult: A QueryResult with materialized data and query metrics.
            For ``data_type="polars"``, a no-op local plan may return a frame
            that aliases its in-memory Polars input rather than copying it.
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
```

### count

```
count() -> int
```

Count the number of rows in the DataFrame.

This is an action that triggers computation of the DataFrame.
The output is an integer representing the number of rows.

Returns:

- **`int`** ( `int`
  ) –

  The number of rows in the DataFrame

Source code in `src/fenic/api/dataframe/dataframe.py`

```
def count(self) -> int:
    """Count the number of rows in the DataFrame.

    This is an action that triggers computation of the DataFrame.
    The output is an integer representing the number of rows.

    Returns:
        int: The number of rows in the DataFrame
    """
    return self._session_state.execution.count(self._logical_plan)[0]
```

### distinct

```
distinct() -> DataFrame
```

Return a DataFrame with duplicate rows removed. Alias for drop_duplicates(subset=None).

Returns:

- **`DataFrame`** ( `DataFrame`
  ) –

  A new DataFrame with duplicate rows removed.

Example

```
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

Source code in `src/fenic/api/dataframe/dataframe.py`

```
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
```

### drop

```
drop(*col_names: str) -> DataFrame
```

Remove one or more columns from this DataFrame.

Parameters:

- **`*col_names`**
  (`str`, default:
  `()`
  )
  –

  Names of columns to drop.

Returns:

- **`DataFrame`** ( `DataFrame`
  ) –

  New DataFrame without specified columns.

Raises:

- `ValueError`
  –

  If any specified column doesn't exist in the DataFrame.
- `ValueError`
  –

  If dropping the columns would result in an empty DataFrame.

Drop single column

```
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

Drop multiple columns

```
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

Error when dropping non-existent column

```
# This will raise a ValueError
df.drop("non_existent_column")
# ValueError: Column 'non_existent_column' not found in DataFrame
```

Source code in `src/fenic/api/dataframe/dataframe.py`

```
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
```

### drop_duplicates

```
drop_duplicates(subset: Optional[List[str]] = None) -> DataFrame
```

Return a DataFrame with duplicate rows removed.

Parameters:

- **`subset`**
  (`Optional[List[str]]`, default:
  `None`
  )
  –

  Column names to consider when identifying duplicates. If not provided, all columns are considered.

Returns:

- **`DataFrame`** ( `DataFrame`
  ) –

  A new DataFrame with duplicate rows removed.

Raises:

- `ValueError`
  –

  If a specified column is not present in the current DataFrame schema.

Remove duplicates considering specific columns

```
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

Source code in `src/fenic/api/dataframe/dataframe.py`

```
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
```

### explain

```
explain() -> None
```

Display the logical plan of the DataFrame.

Source code in `src/fenic/api/dataframe/dataframe.py`

```
def explain(self) -> None:
    """Display the logical plan of the DataFrame."""
    print(str(self._logical_plan))
```

### explode

```
explode(column: ColumnOrName) -> DataFrame
```

Create a new row for each element in an array column.

This operation is useful for flattening nested data structures. For each row in the
input DataFrame that contains an array/list in the specified column, this method will:
1. Create N new rows, where N is the length of the array
2. Each new row will be identical to the original row, except the array column will
contain just a single element from the original array
3. Rows with NULL values or empty arrays in the specified column are filtered out

Parameters:

- **`column`**
  (`ColumnOrName`)
  –

  Name of array column to explode (as string) or Column expression.

Returns:

- **`DataFrame`** ( `DataFrame`
  ) –

  New DataFrame with the array column exploded into multiple rows.

Raises:

- `TypeError`
  –

  If column argument is not a string or Column.

Explode array column

```
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

Using column expression

```
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

Source code in `src/fenic/api/dataframe/dataframe.py`

```
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
```

### explode_outer

```
explode_outer(column: ColumnOrName) -> DataFrame
```

Create a new row for each element in an array column, containing the element's position in the array and its value, and preserving null/empty arrays.

This operation is similar to explode(), but keeps rows where the array column
is null or empty, producing a row with null in the exploded column.

Parameters:

- **`column`**
  (`ColumnOrName`)
  –

  Name of array column to explode (as string) or Column expression.

Returns:

- **`DataFrame`** ( `DataFrame`
  ) –

  New DataFrame with the array column exploded into multiple rows.
- `DataFrame`
  –

  Rows with null or empty arrays are preserved with null in the exploded column.

Explode with outer join behavior

```
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

Source code in `src/fenic/api/dataframe/dataframe.py`

```
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
```

### explode_with_index

```
explode_with_index(column: ColumnOrName, index_col_name: str = 'pos', value_col_name: str = 'col', keep_null_and_empty: bool = False) -> DataFrame
```

Create a new row for each element in an array column, with the element's position in the array and its value.

This operation is similar to explode(), but also adds a column containing the 0-based
position of each element within its original array. By default, the position column is named "pos".
and the value column is named "col". These columns replace the original column in the output DataFrame.
If keep_null_and_empty is True, the position column will be null for rows where the array is null or empty.

Parameters:

- **`column`**
  (`ColumnOrName`)
  –

  Name of array column to explode (as string) or Column expression.
- **`index_col_name`**
  (`str`, default:
  `'pos'`
  )
  –

  Name for the column containing 0-based array positions (default: "pos").
- **`value_col_name`**
  (`str`, default:
  `'col'`
  )
  –

  Name for the exploded value column (default: "col").
- **`keep_null_and_empty`**
  (`bool`, default:
  `False`
  )
  –

  If True, preserves rows where the array is null or empty (default: False).
  Mimicks the behavior of posexplode (false) vs posexplode_outer (true).

Returns:

- **`DataFrame`** ( `DataFrame`
  ) –

  New DataFrame with:
  - An integer column (named `index_col_name`) containing 0-based positions
  - The exploded array column (named `value_col_name`)
  - All other columns from the original DataFrame

Explode with index

```
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

Custom column names

```
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

Source code in `src/fenic/api/dataframe/dataframe.py`

```
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
```

### filter

```
filter(condition: Column) -> DataFrame
```

Filters rows using the given condition.

Parameters:

- **`condition`**
  (`Column`)
  –

  A Column expression that evaluates to a boolean

Returns:

- **`DataFrame`** ( `DataFrame`
  ) –

  Filtered DataFrame

Filter with numeric comparison

```
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

Filter with semantic predicate

```
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

Filter with multiple conditions

```
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

Source code in `src/fenic/api/dataframe/dataframe.py`

```
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
```

### group_by

```
group_by(*cols: ColumnOrName) -> GroupedData
```

Groups the DataFrame using the specified columns.

Parameters:

- **`*cols`**
  (`ColumnOrName`, default:
  `()`
  )
  –

  Columns to group by. Can be column names as strings or Column expressions.

Returns:

- **`GroupedData`** ( `GroupedData`
  ) –

  Object for performing aggregations on the grouped data.

Group by single column

```
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

Group by multiple columns

```
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

Group by expression

```
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

Source code in `src/fenic/api/dataframe/dataframe.py`

```
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
```

### join

```
join(other: DataFrame, on: Union[str, List[str]], *, how: JoinType = 'inner') -> DataFrame
```

```
join(other: DataFrame, *, left_on: Union[ColumnOrName, List[ColumnOrName]], right_on: Union[ColumnOrName, List[ColumnOrName]], how: JoinType = 'inner') -> DataFrame
```

```
join(other: DataFrame, on: Optional[Union[str, List[str]]] = None, *, left_on: Optional[Union[ColumnOrName, List[ColumnOrName]]] = None, right_on: Optional[Union[ColumnOrName, List[ColumnOrName]]] = None, how: JoinType = 'inner') -> DataFrame
```

Joins this DataFrame with another DataFrame.

The Dataframes must have no duplicate column names between them. This API only supports equi-joins.
For non-equi-joins, use session.sql().

Parameters:

- **`other`**
  (`DataFrame`)
  –

  DataFrame to join with.
- **`on`**
  (`Optional[Union[str, List[str]]]`, default:
  `None`
  )
  –

  Join condition(s). Can be:
  - A column name (str)
  - A list of column names (List[str])
  - A Column expression (e.g., col('a'))
  - A list of Column expressions
  - `None` for cross joins
- **`left_on`**
  (`Optional[Union[ColumnOrName, List[ColumnOrName]]]`, default:
  `None`
  )
  –

  Column(s) from the left DataFrame to join on. Can be:
  - A column name (str)
  - A Column expression (e.g., col('a'), col('a') + 1)
  - A list of column names or expressions
- **`right_on`**
  (`Optional[Union[ColumnOrName, List[ColumnOrName]]]`, default:
  `None`
  )
  –

  Column(s) from the right DataFrame to join on. Can be:
  - A column name (str)
  - A Column expression (e.g., col('b'), upper(col('b')))
  - A list of column names or expressions
- **`how`**
  (`JoinType`, default:
  `'inner'`
  )
  –

  Type of join to perform.

Returns:

- `DataFrame`
  –

  Joined DataFrame.

Raises:

- `ValidationError`
  –

  If cross join is used with an ON clause.
- `ValidationError`
  –

  If join condition is invalid.
- `ValidationError`
  –

  If both 'on' and 'left_on'/'right_on' parameters are provided.
- `ValidationError`
  –

  If only one of 'left_on' or 'right_on' is provided.
- `ValidationError`
  –

  If 'left_on' and 'right_on' have different lengths

Inner join on column name

```
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

Join with expression

```
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

Cross join

```
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

Source code in `src/fenic/api/dataframe/dataframe.py`

```
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
```

### limit

```
limit(n: int) -> DataFrame
```

Limits the number of rows to the specified number.

Parameters:

- **`n`**
  (`int`)
  –

  Maximum number of rows to return.

Returns:

- **`DataFrame`** ( `DataFrame`
  ) –

  DataFrame with at most n rows.

Raises:

- `TypeError`
  –

  If n is not an integer.

Limit rows

```
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

Limit with other operations

```
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

Source code in `src/fenic/api/dataframe/dataframe.py`

```
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
```

### lineage

```
lineage() -> Lineage
```

Create a Lineage object to trace data through transformations.

The Lineage interface allows you to trace how specific rows are transformed
through your DataFrame operations, both forwards and backwards through the
computation graph.

Returns:

- **`Lineage`** ( `Lineage`
  ) –

  Interface for querying data lineage

Example

```
# Create lineage query
lineage = df.lineage()

# Trace specific rows backwards through transformations
source_rows = lineage.backward(["result_uuid1", "result_uuid2"])

# Or trace forwards to see outputs
result_rows = lineage.forward(["source_uuid1"])
```

See Also

LineageQuery: Full documentation of lineage querying capabilities

Source code in `src/fenic/api/dataframe/dataframe.py`

```
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
```

### order_by

```
order_by(cols: Union[ColumnOrName, List[ColumnOrName], None] = None, ascending: Optional[Union[bool, List[bool]]] = None) -> DataFrame
```

Sort the DataFrame by the specified columns. Alias for sort().

Returns:

- **`DataFrame`** ( `DataFrame`
  ) –

  sorted Dataframe.

See Also

sort(): Full documentation of sorting behavior and parameters.

Source code in `src/fenic/api/dataframe/dataframe.py`

```
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
```

### persist

```
persist() -> DataFrame
```

Mark this DataFrame to be persisted after first computation.

The persisted DataFrame will be cached after its first computation,
avoiding recomputation in subsequent operations. This is useful for:
- DataFrames that are created once and reused multiple times in your workflow
- DataFrames that are computationally expensive (large joins, aggregations, etc.)

Returns:

- **`DataFrame`** ( `DataFrame`
  ) –

  Same DataFrame, but marked for persistence

Example

```
# Cache intermediate results for reuse
filtered_df = (df
    .filter(col("age") > 25)
    .persist()  # Cache these results
)

# Both operations will use cached results
result1 = filtered_df.group_by("department").count()
result2 = filtered_df.select("name", "salary")
```

Source code in `src/fenic/api/dataframe/dataframe.py`

```
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
```

### posexplode

```
posexplode(column: ColumnOrName) -> DataFrame
```

Create a new row for each element in an array column, with the element's position in the array and its value.

This is a PySpark-compatible alias for explode_with_index.
Creates two columns: 'pos' (0-based position) and 'col' (the array element value).
These columns replace the original column in the output DataFrame.

Parameters:

- **`column`**
  (`ColumnOrName`)
  –

  Name of array column to explode (as string) or Column expression.

Returns:

- **`DataFrame`** ( `DataFrame`
  ) –

  New DataFrame with 'pos' and 'col' columns, plus all other original columns.

PySpark-style posexplode

```
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

Source code in `src/fenic/api/dataframe/dataframe.py`

```
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
```

### posexplode_outer

```
posexplode_outer(column: ColumnOrName) -> DataFrame
```

Create a new row for each element in an array column with position and value, preserving null/empty arrays.

This is a PySpark-compatible alias for explode_with_index with keep_null_and_empty=True.
Creates two columns: 'pos' (0-based position) and 'col' (the array element value).
Rows with null or empty arrays produce (null, null).

Parameters:

- **`column`**
  (`ColumnOrName`)
  –

  Name of array column to explode (as string) or Column expression.

Returns:

- **`DataFrame`** ( `DataFrame`
  ) –

  New DataFrame with 'pos' and 'col' columns, plus all other original columns.
- `DataFrame`
  –

  Rows with null or empty arrays are preserved with (null, null).

PySpark-style posexplode_outer

```
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

Source code in `src/fenic/api/dataframe/dataframe.py`

```
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
```

### select

```
select(*cols: ColumnOrName) -> DataFrame
```

Projects a set of Column expressions or column names.

Parameters:

- **`*cols`**
  (`ColumnOrName`, default:
  `()`
  )
  –

  Column expressions to select. Can be:
  - String column names (e.g., "id", "name")
  - Column objects (e.g., col("id"), col("age") + 1)

Returns:

- **`DataFrame`** ( `DataFrame`
  ) –

  A new DataFrame with selected columns

Select by column names

```
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

Select with expressions

```
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

Mix strings and expressions

```
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

Source code in `src/fenic/api/dataframe/dataframe.py`

```
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
```

### show

```
show(n: int = 10, explain_analyze: bool = False) -> None
```

Display the DataFrame content in a tabular form.

This is an action that triggers computation of the DataFrame.
The output is printed to stdout in a formatted table.

Parameters:

- **`n`**
  (`int`, default:
  `10`
  )
  –

  Number of rows to display
- **`explain_analyze`**
  (`bool`, default:
  `False`
  )
  –

  Whether to print the explain analyze plan

Source code in `src/fenic/api/dataframe/dataframe.py`

```
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
```

### sort

```
sort(cols: Union[ColumnOrName, List[ColumnOrName], None] = None, ascending: Optional[Union[bool, List[bool]]] = None) -> DataFrame
```

Sort the DataFrame by the specified columns.

Parameters:

- **`cols`**
  (`Union[ColumnOrName, List[ColumnOrName], None]`, default:
  `None`
  )
  –

  Columns to sort by. This can be:
  - A single column name (str)
  - A Column expression (e.g., `col("name")`)
  - A list of column names or Column expressions
  - Column expressions may include sorting directives such as `asc("col")`, `desc("col")`,
  `asc_nulls_last("col")`, etc.
  - If no columns are provided, the operation is a no-op.
- **`ascending`**
  (`Optional[Union[bool, List[bool]]]`, default:
  `None`
  )
  –

  A boolean or list of booleans indicating sort order.
  - If `True`, sorts in ascending order; if `False`, descending.
  - If a list is provided, its length must match the number of columns.
  - Cannot be used if any of the columns use `asc()`/`desc()` expressions.
  - If not specified and no sort expressions are used, columns will be sorted in ascending order by default.

Returns:

- **`DataFrame`** ( `DataFrame`
  ) –

  A new DataFrame sorted by the specified columns.

Raises:

- `ValueError`
  –

  - If `ascending` is provided and its length does not match `cols`
  - If both `ascending` and column expressions like `asc()`/`desc()` are used
- `TypeError`
  –

  - If `cols` is not a column name, Column, or list of column names/Columns
  - If `ascending` is not a boolean or list of booleans

Sort in ascending order

```
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

Sort in descending order

```
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

Sort with boolean ascending parameter

```
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

Multiple columns with different sort orders

```
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

Multiple columns with list of ascending strategies

```
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

Source code in `src/fenic/api/dataframe/dataframe.py`

```
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
```

### to_arrow

```
to_arrow() -> pa.Table
```

Execute the DataFrame computation and return an Apache Arrow Table.

This is an action that triggers computation of the DataFrame query plan.
All transformations and operations are executed, and the results are
materialized into an Apache Arrow Table with columnar memory layout
optimized for analytics and zero-copy data exchange.

Returns:

- `Table`
  –

  pa.Table: An Apache Arrow Table containing the computed results

Source code in `src/fenic/api/dataframe/dataframe.py`

```
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
```

### to_pandas

```
to_pandas() -> pd.DataFrame
```

Execute the DataFrame computation and return a Pandas DataFrame.

This is an action that triggers computation of the DataFrame query plan.
All transformations and operations are executed, and the results are
materialized into a Pandas DataFrame.

Returns:

- `DataFrame`
  –

  pd.DataFrame: A Pandas DataFrame containing the computed results with

Source code in `src/fenic/api/dataframe/dataframe.py`

```
def to_pandas(self) -> pd.DataFrame:
    """Execute the DataFrame computation and return a Pandas DataFrame.

    This is an action that triggers computation of the DataFrame query plan.
    All transformations and operations are executed, and the results are
    materialized into a Pandas DataFrame.

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the computed results with
    """
    return self.collect("pandas").data
```

### to_polars

```
to_polars() -> pl.DataFrame
```

Execute the DataFrame computation and return the result as a Polars DataFrame.

This is an action that triggers computation of the DataFrame query plan.
All transformations and operations are executed, and the results are
materialized into a Polars DataFrame.

Returns:

- `DataFrame`
  –

  pl.DataFrame: A Polars DataFrame with materialized results. A no-op
  local plan may return a frame that aliases its in-memory Polars
  input rather than copying it.

Source code in `src/fenic/api/dataframe/dataframe.py`

```
def to_polars(self) -> pl.DataFrame:
    """Execute the DataFrame computation and return the result as a Polars DataFrame.

    This is an action that triggers computation of the DataFrame query plan.
    All transformations and operations are executed, and the results are
    materialized into a Polars DataFrame.

    Returns:
        pl.DataFrame: A Polars DataFrame with materialized results. A no-op
            local plan may return a frame that aliases its in-memory Polars
            input rather than copying it.
    """
    return self.collect("polars").data
```

### to_pydict

```
to_pydict() -> Dict[str, List[Any]]
```

Execute the DataFrame computation and return a dictionary of column arrays.

This is an action that triggers computation of the DataFrame query plan.
All transformations and operations are executed, and the results are
materialized into a Python dictionary where each column becomes a list of values.

Returns:

- `Dict[str, List[Any]]`
  –

  Dict[str, List[Any]]: A dictionary containing the computed results with:
  - Keys: Column names as strings
  - Values: Lists containing all values for each column

Source code in `src/fenic/api/dataframe/dataframe.py`

```
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
```

### to_pylist

```
to_pylist() -> List[Dict[str, Any]]
```

Execute the DataFrame computation and return a list of row dictionaries.

This is an action that triggers computation of the DataFrame query plan.
All transformations and operations are executed, and the results are
materialized into a Python list where each element is a dictionary
representing one row with column names as keys.

Returns:

- `List[Dict[str, Any]]`
  –

  List[Dict[str, Any]]: A list containing the computed results with:
  - Each element: A dictionary representing one row
  - Dictionary keys: Column names as strings
  - Dictionary values: Cell values in Python native types
  - List length equals number of rows in the result

Source code in `src/fenic/api/dataframe/dataframe.py`

```
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
```

### union

```
union(other: DataFrame) -> DataFrame
```

Return a new DataFrame containing the union of rows in this and another DataFrame.

This is equivalent to UNION ALL in SQL. To remove duplicates, use drop_duplicates() after union().

Parameters:

- **`other`**
  (`DataFrame`)
  –

  Another DataFrame with the same schema.

Returns:

- **`DataFrame`** ( `DataFrame`
  ) –

  A new DataFrame containing rows from both DataFrames.

Raises:

- `ValueError`
  –

  If the DataFrames have different schemas.
- `TypeError`
  –

  If other is not a DataFrame.

Union two DataFrames

```
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

Union with duplicates

```
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

Source code in `src/fenic/api/dataframe/dataframe.py`

```
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
```

### unnest

```
unnest(*col_names: str) -> DataFrame
```

Unnest the specified struct columns into separate columns.

This operation flattens nested struct data by expanding each field of a struct
into its own top-level column.

For each specified column containing a struct:
1. Each field in the struct becomes a separate column.
2. New columns are named after the corresponding struct fields.
3. The new columns are inserted into the DataFrame in place of the original struct column.
4. The overall column order is preserved.

Parameters:

- **`*col_names`**
  (`str`, default:
  `()`
  )
  –

  One or more struct columns to unnest. Each can be a string (column name)
  or a Column expression.

Returns:

- **`DataFrame`** ( `DataFrame`
  ) –

  A new DataFrame with the specified struct columns expanded.

Raises:

- `TypeError`
  –

  If any argument is not a string or Column.
- `ValueError`
  –

  If a specified column does not contain struct data.

Unnest struct column

```
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

Unnest multiple struct columns

```
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

Source code in `src/fenic/api/dataframe/dataframe.py`

```
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
```

### where

```
where(condition: Column) -> DataFrame
```

Filters rows using the given condition (alias for filter()).

Parameters:

- **`condition`**
  (`Column`)
  –

  A Column expression that evaluates to a boolean

Returns:

- **`DataFrame`** ( `DataFrame`
  ) –

  Filtered DataFrame

See Also

filter(): Full documentation of filtering behavior

Source code in `src/fenic/api/dataframe/dataframe.py`

```
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
```

### with_column

```
with_column(col_name: str, col: Union[Any, Column, Series, Series]) -> DataFrame
```

Add a new column or replace an existing column.

Parameters:

- **`col_name`**
  (`str`)
  –

  Name of the new column
- **`col`**
  (`Union[Any, Column, Series, Series]`)
  –

  Column expression, Series, or value to assign to the column:

  - Column: A Column expression (e.g., `col("age") + 1`)
  - `pl.Series` or `pd.Series`: A Polars or pandas Series with data
    - **Note: Series length MUST match the DataFrame height**
  - Any other value: Treated as a literal value (broadcast to all rows)

Returns:

- **`DataFrame`** ( `DataFrame`
  ) –

  New DataFrame with added/replaced column

Raises:

- `ExecutionError`
  –

  - If a Series length does not match the DataFrame height
- `ValidationError`
  –

  - If the Series contains all null values and no dtype is specified
  - If the Series has length 0

Notes:
- The name of the created column will be the name defined in col_name, even if input is a Series with a different name.

Add literal column

```
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

Add computed column

```
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

Replace existing column

```
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

Add column with complex expression

```
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

Add column from Polars Series

```
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

Add column from pandas Series

```
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

Source code in `src/fenic/api/dataframe/dataframe.py`

```
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
```

### with_column_renamed

```
with_column_renamed(col_name: str, new_col_name: str) -> DataFrame
```

Rename a column. No-op if the column does not exist.

Parameters:

- **`col_name`**
  (`str`)
  –

  Name of the column to rename.
- **`new_col_name`**
  (`str`)
  –

  New name for the column.

Returns:

- **`DataFrame`** ( `DataFrame`
  ) –

  New DataFrame with the column renamed.

Rename a column

```
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

Rename multiple columns

```
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

Source code in `src/fenic/api/dataframe/dataframe.py`

```
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
```

### with_columns

```
with_columns(cols_map: Dict[str, Union[Any, Column, Series, Series]]) -> DataFrame
```

Add multiple new columns or replace existing columns.

Parameters:

- **`cols_map`**
  (`Dict[str, Union[Any, Column, Series, Series]]`)
  –

  A dictionary where keys are column names and values are:

  - Column: Column expressions (e.g., col("age") + 1)
  - pl.Series or pd.Series: Series with data
    - **Note: Series length MUST match the DataFrame height**
  - Any other value: Treated as literal values (broadcast to all rows)

Returns:

- **`DataFrame`** ( `DataFrame`
  ) –

  New DataFrame with added/replaced columns

Raises:

- `ValueError`
  –

  - If two columns being created in the same `with_columns` call depend on each other
- `ExecutionError`
  –

  - If any Series length does not match the DataFrame height
- `ValidationError`
  –

  - If any Series contains all null values and no dtype is specified
  - If any Series has length 0

Notes:
- All columns are created at once, so new columns cannot depend on each other.
- The name of the created column will be the name defined in cols_map, even if input is a Series with a different name.

Add multiple columns

```
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

Replace and add columns

```
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

Complex expressions

```
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

Add columns from Series

```
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

Mix Series with Column expressions

```
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

Error when adding columns that depend on each other

```
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

Source code in `src/fenic/api/dataframe/dataframe.py`

```
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
```

## GroupedData

```
GroupedData(df: DataFrame, by: Optional[List[ColumnOrName]] = None)
```

Bases: `BaseGroupedData`

```
              flowchart TD
              fenic.api.dataframe.GroupedData[GroupedData]
              fenic.api.dataframe._base_grouped_data.BaseGroupedData[BaseGroupedData]

                              fenic.api.dataframe._base_grouped_data.BaseGroupedData --> fenic.api.dataframe.GroupedData

              click fenic.api.dataframe.GroupedData href "" "fenic.api.dataframe.GroupedData"
              click fenic.api.dataframe._base_grouped_data.BaseGroupedData href "" "fenic.api.dataframe._base_grouped_data.BaseGroupedData"
```

Methods for aggregations on a grouped DataFrame.

Initialize grouped data.

Parameters:

- **`df`**
  (`DataFrame`)
  –

  The DataFrame to group.
- **`by`**
  (`Optional[List[ColumnOrName]]`, default:
  `None`
  )
  –

  Optional list of columns to group by.

Methods:

- **`agg`**
  –

  Compute aggregations on grouped data and return the result as a DataFrame.

Source code in `src/fenic/api/dataframe/grouped_data.py`

```
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
```

### agg

```
agg(*exprs: Union[Column, Dict[str, str]]) -> DataFrame
```

Compute aggregations on grouped data and return the result as a DataFrame.

This method applies aggregate functions to the grouped data.

Parameters:

- **`*exprs`**
  (`Union[Column, Dict[str, str]]`, default:
  `()`
  )
  –

  Aggregation expressions. Can be:

  - Column expressions with aggregate functions (e.g., `count("*")`, `sum("amount")`)
  - A dictionary mapping column names to aggregate function names (e.g., `{"amount": "sum", "age": "avg"}`)

Returns:

- **`DataFrame`** ( `DataFrame`
  ) –

  A new DataFrame with one row per group and columns for group keys and aggregated values

Raises:

- `ValueError`
  –

  If arguments are not Column expressions or a dictionary
- `ValueError`
  –

  If dictionary values are not valid aggregate function names

Count employees by department

```
# Group by department and count employees
df.group_by("department").agg(count("*").alias("employee_count"))
```

Multiple aggregations

```
# Multiple aggregations
df.group_by("department").agg(
    count("*").alias("employee_count"),
    avg("salary").alias("avg_salary"),
    max("age").alias("max_age")
)
```

Dictionary style aggregations

```
# Dictionary style for simple aggregations
df.group_by("department", "location").agg({"salary": "avg", "age": "max"})
```

Source code in `src/fenic/api/dataframe/grouped_data.py`

```
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
```

## SemanticExtensions

```
SemanticExtensions(df: DataFrame)
```

A namespace for semantic dataframe operators.

Initialize semantic extensions.

Parameters:

- **`df`**
  (`DataFrame`)
  –

  The DataFrame to extend with semantic operations.

Methods:

- **`join`**
  –

  Performs a semantic join between two DataFrames using a natural language predicate.
- **`sim_join`**
  –

  Performs a semantic similarity join between two DataFrames using embedding expressions.
- **`with_cluster_labels`**
  –

  Cluster rows using K-means and add cluster metadata columns.

Source code in `src/fenic/api/dataframe/semantic_extensions.py`

```
def __init__(self, df: DataFrame):
    """Initialize semantic extensions.

    Args:
        df: The DataFrame to extend with semantic operations.
    """
    self._df = df
```

### join

```
join(other: DataFrame, predicate: str, left_on: Column, right_on: Column, strict: bool = True, examples: Optional[JoinExampleCollection] = None, model_alias: Optional[Union[str, ModelAlias]] = None, request_timeout: Optional[float] = None) -> DataFrame
```

Performs a semantic join between two DataFrames using a natural language predicate.

This method evaluates a boolean predicate for each potential row pair between the two DataFrames,
including only those pairs where the predicate evaluates to True.

The join process:
1. For each row in the left DataFrame, evaluates the predicate in the jinja template against each row in the right DataFrame
2. Includes row pairs where the predicate returns True
3. Excludes row pairs where the predicate returns False
4. Returns a new DataFrame containing all columns from both DataFrames for the matched pairs

The jinja template must use exactly two column placeholders:
- One from the left DataFrame: `{{ left_on }}`
- One from the right DataFrame: `{{ right_on }}`

Parameters:

- **`other`**
  (`DataFrame`)
  –

  The DataFrame to join with.
- **`predicate`**
  (`str`)
  –

  A Jinja2 template containing the natural language predicate.
  Must include placeholders for exactly one column from each DataFrame.
  The template is evaluated as a boolean - True includes the pair, False excludes it.
- **`left_on`**
  (`Column`)
  –

  The column from the left DataFrame (self) to use in the join predicate.
- **`right_on`**
  (`Column`)
  –

  The column from the right DataFrame (other) to use in the join predicate.
- **`strict`**
  (`bool`, default:
  `True`
  )
  –

  If True, when either the left_on or right_on column has a None value for a row pair,
  that pair is automatically excluded from the join (predicate is not evaluated).
  If False, None values are rendered according to Jinja2's null rendering behavior.
  Default is True.
- **`examples`**
  (`Optional[JoinExampleCollection]`, default:
  `None`
  )
  –

  Optional JoinExampleCollection containing labeled examples to guide the join.
  Each example should have:
  - left: Sample value from the left column
  - right: Sample value from the right column
  - output: Boolean indicating whether this pair should be joined (True) or not (False)
- **`model_alias`**
  (`Optional[Union[str, ModelAlias]]`, default:
  `None`
  )
  –

  Optional alias for the language model to use. If None, uses the default model.
- **`request_timeout`**
  (`Optional[float]`, default:
  `None`
  )
  –

  Optional timeout in seconds for a single LLM request. If None, uses the default timeout (120 seconds).

Returns:

- **`DataFrame`** ( `DataFrame`
  ) –

  A new DataFrame containing matched row pairs with all columns from both DataFrames.

Basic semantic join

```
# Match job listings with candidate resumes based on title/skills
# Only includes pairs where the predicate evaluates to True
df_jobs.semantic.join(df_resumes,
    predicate=dedent('''                    Job Description: {{left_on}}
        Candidate Background: {{right_on}}
        The candidate is qualified for the job.'''),
    left_on=col("job_description"),
    right_on=col("work_experience"),
    examples=examples
)
```

Semantic join with examples

```
# Improve join quality with examples
examples = JoinExampleCollection()
examples.create_example(JoinExample(
    left="5 years experience building backend services in Python using asyncio, FastAPI, and PostgreSQL",
    right="Senior Software Engineer - Backend",
    output=True))  # This pair WILL be included in similar cases
examples.create_example(JoinExample(
    left="5 years experience with growth strategy, private equity due diligence, and M&A",
    right="Product Manager - Hardware",
    output=False))  # This pair will NOT be included in similar cases
df_jobs.semantic.join(
    other=df_resumes,
    predicate=dedent('''                    Job Description: {{left_on}}
        Candidate Background: {{right_on}}
        The candidate is qualified for the job.'''),
    left_on=col("job_description"),
    right_on=col("work_experience"),
    examples=examples
)
```

Source code in `src/fenic/api/dataframe/semantic_extensions.py`

```
def join(
    self,
    other: DataFrame,
    predicate: str,
    left_on: Column,
    right_on: Column,
    strict: bool = True,
    examples: Optional[JoinExampleCollection] = None,
    model_alias: Optional[Union[str, ModelAlias]] = None,
    request_timeout: Optional[float] = None,
) -> DataFrame:
    """Performs a semantic join between two DataFrames using a natural language predicate.

    This method evaluates a boolean predicate for each potential row pair between the two DataFrames,
    including only those pairs where the predicate evaluates to True.

    The join process:
    1. For each row in the left DataFrame, evaluates the predicate in the jinja template against each row in the right DataFrame
    2. Includes row pairs where the predicate returns True
    3. Excludes row pairs where the predicate returns False
    4. Returns a new DataFrame containing all columns from both DataFrames for the matched pairs

    The jinja template must use exactly two column placeholders:
    - One from the left DataFrame: `{{ left_on }}`
    - One from the right DataFrame: `{{ right_on }}`

    Args:
        other: The DataFrame to join with.
        predicate: A Jinja2 template containing the natural language predicate.
            Must include placeholders for exactly one column from each DataFrame.
            The template is evaluated as a boolean - True includes the pair, False excludes it.
        left_on: The column from the left DataFrame (self) to use in the join predicate.
        right_on: The column from the right DataFrame (other) to use in the join predicate.
        strict: If True, when either the left_on or right_on column has a None value for a row pair,
                that pair is automatically excluded from the join (predicate is not evaluated).
                If False, None values are rendered according to Jinja2's null rendering behavior.
                Default is True.
        examples: Optional JoinExampleCollection containing labeled examples to guide the join.
            Each example should have:
            - left: Sample value from the left column
            - right: Sample value from the right column
            - output: Boolean indicating whether this pair should be joined (True) or not (False)
        model_alias: Optional alias for the language model to use. If None, uses the default model.
        request_timeout: Optional timeout in seconds for a single LLM request. If None, uses the default timeout (120 seconds).

    Returns:
        DataFrame: A new DataFrame containing matched row pairs with all columns from both DataFrames.

    Example: Basic semantic join
        ```python
        # Match job listings with candidate resumes based on title/skills
        # Only includes pairs where the predicate evaluates to True
        df_jobs.semantic.join(df_resumes,
            predicate=dedent('''\
                Job Description: {{left_on}}
                Candidate Background: {{right_on}}
                The candidate is qualified for the job.'''),
            left_on=col("job_description"),
            right_on=col("work_experience"),
            examples=examples
        )
        ```

    Example: Semantic join with examples
        ```python
        # Improve join quality with examples
        examples = JoinExampleCollection()
        examples.create_example(JoinExample(
            left="5 years experience building backend services in Python using asyncio, FastAPI, and PostgreSQL",
            right="Senior Software Engineer - Backend",
            output=True))  # This pair WILL be included in similar cases
        examples.create_example(JoinExample(
            left="5 years experience with growth strategy, private equity due diligence, and M&A",
            right="Product Manager - Hardware",
            output=False))  # This pair will NOT be included in similar cases
        df_jobs.semantic.join(
            other=df_resumes,
            predicate=dedent('''\
                Job Description: {{left_on}}
                Candidate Background: {{right_on}}
                The candidate is qualified for the job.'''),
            left_on=col("job_description"),
            right_on=col("work_experience"),
            examples=examples
        )
        ```
    """
    from fenic.api.dataframe.dataframe import DataFrame

    if not isinstance(other, DataFrame):
        raise ValidationError(f"other argument must be a DataFrame, got {type(other)}")

    if not isinstance(predicate, str):
        raise ValidationError(
            f"The `predicate` argument to `semantic.join` must be a string, got {type(predicate)}"
        )
    if not isinstance(left_on, Column):
        raise ValidationError(f"`left_on` argument must be a Column, got {type(left_on)} instead.")
    if not isinstance(right_on, Column):
        raise ValidationError(f"`right_on` argument must be a Column, got {type(right_on)} instead.")
    if examples is not None and not isinstance(examples, JoinExampleCollection):
        raise ValidationError(f"`examples` argument must be a JoinExampleCollection, got {type(examples)} instead.")
    if model_alias is not None and not isinstance(model_alias, (str, ModelAlias)):
        raise ValidationError(f"`model_alias` argument must be a string or ModelAlias, got {type(model_alias)} instead.")

    # Validate request_timeout
    request_timeout = validate_timeout(request_timeout)

    resolved_model_alias = _resolve_model_alias(model_alias)
    DataFrame._ensure_same_session(self._df._session_state, [other._session_state])

    return self._df._from_logical_plan(
        SemanticJoin.from_session_state(
            left=self._df._logical_plan,
            right=other._logical_plan,
            left_on=left_on._logical_expr,
            right_on=right_on._logical_expr,
            jinja_template=predicate,
            strict=strict,
            model_alias=resolved_model_alias,
            examples=examples,
            request_timeout=request_timeout,
            session_state=self._df._session_state,
        ),
        self._df._session_state,
    )
```

### sim_join

```
sim_join(other: DataFrame, left_on: ColumnOrName, right_on: ColumnOrName, k: int = 1, similarity_metric: SemanticSimilarityMetric = 'cosine', similarity_score_column: Optional[str] = None, request_timeout: Optional[float] = None) -> DataFrame
```

Performs a semantic similarity join between two DataFrames using embedding expressions.

For each row in the left DataFrame, returns the top `k` most semantically similar rows
from the right DataFrame based on the specified similarity metric.

Note

Local execution requires the `sim-join` extra: `pip install "fenic[sim-join]"`.

Parameters:

- **`other`**
  (`DataFrame`)
  –

  The right-hand DataFrame to join with.
- **`left_on`**
  (`ColumnOrName`)
  –

  Expression or column representing embeddings in the left DataFrame.
  If this is a named column, that column is treated as a user column and is
  included in the output. If this is an expression, it is treated as a
  temporary join key and is not included as an output column.
- **`right_on`**
  (`ColumnOrName`)
  –

  Expression or column representing embeddings in the right DataFrame.
  Named columns are included in the output; expression-derived join keys
  are temporary and are not included in the output.
- **`k`**
  (`int`, default:
  `1`
  )
  –

  Number of most similar matches to return per row.
- **`similarity_metric`**
  (`SemanticSimilarityMetric`, default:
  `'cosine'`
  )
  –

  Similarity metric to use: "l2", "cosine", or "dot".
- **`similarity_score_column`**
  (`Optional[str]`, default:
  `None`
  )
  –

  If set, adds a column with this name containing similarity scores.
  If None, the scores are omitted.
- **`request_timeout`**
  (`Optional[float]`, default:
  `None`
  )
  –

  Optional timeout in seconds for a single LLM request. If None, uses the default timeout (120 seconds).

Returns:

- `DataFrame`
  –

  A DataFrame containing one row for each of the top-k matches per row in the left DataFrame.
- `DataFrame`
  –

  The result includes all columns from both input DataFrames and, when `similarity_score_column`
- `DataFrame`
  –

  is provided, a similarity score column. Join key columns that already exist in the input
- `DataFrame`
  –

  DataFrames are preserved as normal user columns. Join keys derived from expressions are
- `DataFrame`
  –

  temporary execution columns and are not included in the result schema.

Raises:

- `ValidationError`
  –

  If `k` is not positive or if the columns are invalid.
- `ValidationError`
  –

  If `similarity_metric` is not one of "l2", "cosine", "dot"

Match queries to FAQ entries

```
# Match customer queries to FAQ entries
df_queries.semantic.sim_join(
    df_faqs,
    left_on=embeddings(col("query_text")),
    right_on=embeddings(col("faq_question")),
    k=1
)
```

Link headlines to articles

```
# Link news headlines to full articles
df_headlines.semantic.sim_join(
    df_articles,
    left_on=embeddings(col("headline")),
    right_on=embeddings(col("content")),
    k=3,
    return_similarity_scores=True
)
```

Find similar job postings

```
# Find similar job postings across two sources
df_linkedin.semantic.sim_join(
    df_indeed,
    left_on=embeddings(col("job_title")),
    right_on=embeddings(col("job_description")),
    k=2
)
```

Source code in `src/fenic/api/dataframe/semantic_extensions.py`

```
def sim_join(
    self,
    other: DataFrame,
    left_on: ColumnOrName,
    right_on: ColumnOrName,
    k: int = 1,
    similarity_metric: SemanticSimilarityMetric = "cosine",
    similarity_score_column: Optional[str] = None,
    request_timeout: Optional[float] = None,
) -> DataFrame:
    """Performs a semantic similarity join between two DataFrames using embedding expressions.

    For each row in the left DataFrame, returns the top `k` most semantically similar rows
    from the right DataFrame based on the specified similarity metric.

    Note:
        Local execution requires the `sim-join` extra: `pip install "fenic[sim-join]"`.

    Args:
        other: The right-hand DataFrame to join with.
        left_on: Expression or column representing embeddings in the left DataFrame.
            If this is a named column, that column is treated as a user column and is
            included in the output. If this is an expression, it is treated as a
            temporary join key and is not included as an output column.
        right_on: Expression or column representing embeddings in the right DataFrame.
            Named columns are included in the output; expression-derived join keys
            are temporary and are not included in the output.
        k: Number of most similar matches to return per row.
        similarity_metric: Similarity metric to use: "l2", "cosine", or "dot".
        similarity_score_column: If set, adds a column with this name containing similarity scores.
            If None, the scores are omitted.
        request_timeout: Optional timeout in seconds for a single LLM request. If None, uses the default timeout (120 seconds).

    Returns:
        A DataFrame containing one row for each of the top-k matches per row in the left DataFrame.
        The result includes all columns from both input DataFrames and, when `similarity_score_column`
        is provided, a similarity score column. Join key columns that already exist in the input
        DataFrames are preserved as normal user columns. Join keys derived from expressions are
        temporary execution columns and are not included in the result schema.

    Raises:
        ValidationError: If `k` is not positive or if the columns are invalid.
        ValidationError: If `similarity_metric` is not one of "l2", "cosine", "dot"

    Example: Match queries to FAQ entries
        ```python
        # Match customer queries to FAQ entries
        df_queries.semantic.sim_join(
            df_faqs,
            left_on=embeddings(col("query_text")),
            right_on=embeddings(col("faq_question")),
            k=1
        )
        ```

    Example: Link headlines to articles
        ```python
        # Link news headlines to full articles
        df_headlines.semantic.sim_join(
            df_articles,
            left_on=embeddings(col("headline")),
            right_on=embeddings(col("content")),
            k=3,
            return_similarity_scores=True
        )
        ```

    Example: Find similar job postings
        ```python
        # Find similar job postings across two sources
        df_linkedin.semantic.sim_join(
            df_indeed,
            left_on=embeddings(col("job_title")),
            right_on=embeddings(col("job_description")),
            k=2
        )
        ```
    """
    from fenic.api.dataframe.dataframe import DataFrame

    if not isinstance(right_on, ColumnOrName):
        raise ValidationError(
            f"The `right_on` argument must be a `Column` or a string representing a column name, "
            f"but got `{type(right_on).__name__}` instead."
        )
    if not isinstance(other, DataFrame):
        raise ValidationError(
                        f"The `other` argument to `sim_join()` must be a DataFrame`, but got `{type(other).__name__}`."
                    )
    if not (isinstance(k, int) and k > 0):
        raise ValidationError(
            f"The parameter `k` must be a positive integer, but received `{k}`."
        )
    args = get_args(SemanticSimilarityMetric)
    if similarity_metric not in args:
        raise ValidationError(
            f"The `similarity_metric` argument must be one of {args}, but got `{similarity_metric}`."
        )

    def _validate_column(column: ColumnOrName, name: str):
        if column is None:
            raise ValidationError(f"The `{name}` argument must not be None.")
        if not isinstance(column, ColumnOrName):
            raise ValidationError(
                f"The `{name}` argument must be a `Column` or a string representing a column name, "
                f"but got `{type(column).__name__}` instead."
            )

    _validate_column(left_on, "left_on")
    _validate_column(right_on, "right_on")

    # Validate request_timeout
    request_timeout = validate_timeout(request_timeout)
    DataFrame._ensure_same_session(self._df._session_state, [other._session_state])
    return self._df._from_logical_plan(
        SemanticSimilarityJoin.from_session_state(
            self._df._logical_plan,
            other._logical_plan,
            Column._from_col_or_name(left_on)._logical_expr,
            Column._from_col_or_name(right_on)._logical_expr,
            k,
            similarity_metric=similarity_metric,
            similarity_score_column=similarity_score_column,
            session_state=self._df._session_state,
            request_timeout=request_timeout,
        ),
        self._df._session_state,
    )
```

### with_cluster_labels

```
with_cluster_labels(by: ColumnOrName, num_clusters: int, max_iter: int = 300, num_init: int = 1, label_column: str = 'cluster_label', centroid_column: Optional[str] = None, request_timeout: Optional[float] = None) -> DataFrame
```

Cluster rows using K-means and add cluster metadata columns.

This method clusters rows based on the given embedding column or expression using K-means.
It adds a new column with cluster assignments, and optionally includes the centroid embedding
for each assigned cluster.

Note

Local execution requires the `cluster` extra: `pip install "fenic[cluster]"`.

Parameters:

- **`by`**
  (`ColumnOrName`)
  –

  Column or expression producing embeddings to cluster (e.g., `embed(col("text"))`).
- **`num_clusters`**
  (`int`)
  –

  Number of clusters to compute (must be > 0).
- **`max_iter`**
  (`int`, default:
  `300`
  )
  –

  Maximum iterations for a single run of the k-means algorithm. The algorithm stops when it either converges or reaches this limit.
- **`num_init`**
  (`int`, default:
  `1`
  )
  –

  Number of independent runs of k-means with different centroid seeds. The best result is selected.
- **`label_column`**
  (`str`, default:
  `'cluster_label'`
  )
  –

  Name of the output column for cluster IDs. Default is "cluster_label".
- **`centroid_column`**
  (`Optional[str]`, default:
  `None`
  )
  –

  If provided, adds a column with this name containing the centroid embedding
  for each row's assigned cluster.
- **`request_timeout`**
  (`Optional[float]`, default:
  `None`
  )
  –

  Optional timeout in seconds for a single LLM request. If None, uses the default timeout (120 seconds).

Returns:

- `DataFrame`
  –

  A DataFrame with all original columns plus:
- `DataFrame`
  –

  - `<label_column>`: integer cluster assignment (0 to num_clusters - 1)
- `DataFrame`
  –

  - `<centroid_column>`: cluster centroid embedding, if specified

Basic clustering

```
# Cluster customer feedback and add cluster metadata
clustered_df = df.semantic.with_cluster_labels("feedback_embeddings", num_clusters=5)

# Then use regular operations to analyze clusters
clustered_df.group_by("cluster_label").agg(count("*"), avg("rating"))
```

Filter outliers using centroids

```
# Cluster and filter out rows far from their centroid
clustered_df = df.semantic.with_cluster_labels("embeddings", num_clusters=3, num_init=10, centroid_column="cluster_centroid")
clean_df = clustered_df.filter(
    embedding.compute_similarity("embeddings", "cluster_centroid", metric="cosine") > 0.7
)
```

Source code in `src/fenic/api/dataframe/semantic_extensions.py`

```
def with_cluster_labels(
    self,
    by: ColumnOrName,
    num_clusters: int,
    max_iter: int = 300,
    num_init: int = 1,
    label_column: str = "cluster_label",
    centroid_column: Optional[str] = None,
    request_timeout: Optional[float] = None,
) -> DataFrame:
    """Cluster rows using K-means and add cluster metadata columns.

    This method clusters rows based on the given embedding column or expression using K-means.
    It adds a new column with cluster assignments, and optionally includes the centroid embedding
    for each assigned cluster.

    Note:
        Local execution requires the `cluster` extra: `pip install "fenic[cluster]"`.

    Args:
        by: Column or expression producing embeddings to cluster (e.g., `embed(col("text"))`).
        num_clusters: Number of clusters to compute (must be > 0).
        max_iter: Maximum iterations for a single run of the k-means algorithm. The algorithm stops when it either converges or reaches this limit.
        num_init: Number of independent runs of k-means with different centroid seeds. The best result is selected.
        label_column: Name of the output column for cluster IDs. Default is "cluster_label".
        centroid_column: If provided, adds a column with this name containing the centroid embedding
                        for each row's assigned cluster.
        request_timeout: Optional timeout in seconds for a single LLM request. If None, uses the default timeout (120 seconds).

    Returns:
        A DataFrame with all original columns plus:
        - `<label_column>`: integer cluster assignment (0 to num_clusters - 1)
        - `<centroid_column>`: cluster centroid embedding, if specified

    Example: Basic clustering
        ```python
        # Cluster customer feedback and add cluster metadata
        clustered_df = df.semantic.with_cluster_labels("feedback_embeddings", num_clusters=5)

        # Then use regular operations to analyze clusters
        clustered_df.group_by("cluster_label").agg(count("*"), avg("rating"))
        ```

    Example: Filter outliers using centroids
        ```python
        # Cluster and filter out rows far from their centroid
        clustered_df = df.semantic.with_cluster_labels("embeddings", num_clusters=3, num_init=10, centroid_column="cluster_centroid")
        clean_df = clustered_df.filter(
            embedding.compute_similarity("embeddings", "cluster_centroid", metric="cosine") > 0.7
        )
        ```
    """
    # Validate request_timeout
    request_timeout = validate_timeout(request_timeout)

    # Validate num_clusters
    if not isinstance(num_clusters, int) or num_clusters <= 0:
        raise ValidationError("`num_clusters` must be a positive integer.")

    # Validate max_iter
    if not isinstance(max_iter, int) or max_iter <= 0:
        raise ValidationError("`max_iter` must be a positive integer.")

    # Validate num_init
    if not isinstance(num_init, int) or num_init <= 0:
        raise ValidationError("`num_init` must be a positive integer.")

    # Validate clustering target
    if not isinstance(by, ColumnOrName):
        raise ValidationError(
            f"Invalid cluster by: expected a column name (str) or Column object, got {type(by).__name__}."
        )

    # Validate label_column
    if not isinstance(label_column, str) or not label_column:
        raise ValidationError("`label_column` must be a non-empty string.")

    # Validate centroid_column if provided
    if centroid_column is not None:
        if not isinstance(centroid_column, str) or not centroid_column:
            raise ValidationError("`centroid_column` must be a non-empty string if provided.")

    # Check that the expression isn't a literal
    by_expr = Column._from_col_or_name(by)._logical_expr
    if isinstance(by_expr, LiteralExpr):
        raise ValidationError(
            f"Invalid cluster by: Cannot cluster by a literal value: {by_expr}."
        )

    return self._df._from_logical_plan(
        SemanticCluster.from_session_state(
            self._df._logical_plan,
            by_expr,
            num_clusters=num_clusters,
            max_iter=max_iter,
            num_init=num_init,
            label_column=label_column,
            centroid_column=centroid_column,
            session_state=self._df._session_state,
            request_timeout=request_timeout,
        ),
        self._df._session_state,
    )
```
