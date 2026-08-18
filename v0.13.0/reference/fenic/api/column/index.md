# fenic.api.column

Canonical HTML: https://docs.fenic.ai/latest/reference/fenic/api/column/

Column API for Fenic DataFrames - represents column expressions and operations.

Classes:

- **`Column`**
  –

  A column expression in a DataFrame.

## Column

A column expression in a DataFrame.

This class represents a column expression that can be used in DataFrame operations.
It provides methods for accessing, transforming, and combining column data.

Create a column reference

```
# Reference a column by name using col() function
col("column_name")
```

Use column in operations

```
# Perform arithmetic operations
df.select(col("price") * col("quantity"))
```

Chain column operations

```
# Chain multiple operations
df.select(col("name").upper().contains("John"))
```

Methods:

- **`alias`**
  –

  Create an alias for this column.
- **`asc`**
  –

  Mark this column for ascending sort order.
- **`asc_nulls_first`**
  –

  Alias for asc().
- **`asc_nulls_last`**
  –

  Mark this column for ascending sort order with nulls last.
- **`cast`**
  –

  Cast the column to a new data type.
- **`contains`**
  –

  Check if the column contains a substring.
- **`contains_any`**
  –

  Check if the column contains any of the specified substrings.
- **`desc`**
  –

  Mark this column for descending sort order.
- **`desc_nulls_first`**
  –

  Alias for desc().
- **`desc_nulls_last`**
  –

  Mark this column for descending sort order with nulls last.
- **`ends_with`**
  –

  Check if the column ends with a substring.
- **`get_item`**
  –

  Access an item in a struct or array column.
- **`ilike`**
  –

  Check if the column matches a SQL LIKE pattern (case-insensitive).
- **`is_in`**
  –

  Check if the column is in a list of values or a column expression.
- **`is_not_null`**
  –

  Check if the column contains non-NULL values.
- **`is_null`**
  –

  Check if the column contains NULL values.
- **`like`**
  –

  Check if the column matches a SQL LIKE pattern.
- **`otherwise`**
  –

  Returns a value when no prior conditions are True.
- **`rlike`**
  –

  Check if the column matches a regular expression pattern.
- **`starts_with`**
  –

  Check if the column starts with a substring.
- **`when`**
  –

  Evaluates a condition for each row and returns a value when true.

### alias

```
alias(name: str) -> Column
```

Create an alias for this column.

This method assigns a new name to the column expression, which is useful
for renaming columns or providing names for complex expressions.

Parameters:

- **`name`**
  (`str`)
  –

  The alias name to assign

Returns:

- **`Column`** ( `Column`
  ) –

  Column with the specified alias

Rename a column

```
# Rename a column to a new name
df.select(col("original_name").alias("new_name"))
```

Name a complex expression

```
# Give a name to a calculated column
df.select((col("price") * col("quantity")).alias("total_value"))
```

Source code in `src/fenic/api/column.py`

```
def alias(self, name: str) -> Column:
    """Create an alias for this column.

    This method assigns a new name to the column expression, which is useful
    for renaming columns or providing names for complex expressions.

    Args:
        name (str): The alias name to assign

    Returns:
        Column: Column with the specified alias

    Example: Rename a column
        ```python
        # Rename a column to a new name
        df.select(col("original_name").alias("new_name"))
        ```

    Example: Name a complex expression
        ```python
        # Give a name to a calculated column
        df.select((col("price") * col("quantity")).alias("total_value"))
        ```
    """
    return Column._from_logical_expr(AliasExpr(self._logical_expr, name))
```

### asc

```
asc() -> Column
```

Mark this column for ascending sort order.

Returns:

- **`Column`** ( `Column`
  ) –

  A sort expression with ascending order and nulls first.

Sort by age in ascending order

```
# Sort a dataframe by age in ascending order
df.sort(col("age").asc()).show()
```

Source code in `src/fenic/api/column.py`

```
def asc(self) -> Column:
    """Mark this column for ascending sort order.

    Returns:
        Column: A sort expression with ascending order and nulls first.

    Example: Sort by age in ascending order
        ```python
        # Sort a dataframe by age in ascending order
        df.sort(col("age").asc()).show()
        ```
    """
    return Column._from_logical_expr(SortExpr(self._logical_expr, ascending=True))
```

### asc_nulls_first

```
asc_nulls_first() -> Column
```

Alias for asc().

Returns:

- **`Column`** ( `Column`
  ) –

  A Column expression that provides a column and sort order to the sort function

Source code in `src/fenic/api/column.py`

```
def asc_nulls_first(self) -> Column:
    """Alias for asc().

    Returns:
        Column: A Column expression that provides a column and sort order to the sort function
    """
    return self.asc()
```

### asc_nulls_last

```
asc_nulls_last() -> Column
```

Mark this column for ascending sort order with nulls last.

Returns:

- **`Column`** ( `Column`
  ) –

  A sort expression with ascending order and nulls last.

Sort by age in ascending order with nulls last

```
# Sort a dataframe by age in ascending order, with nulls appearing last
df.sort(col("age").asc_nulls_last()).show()
```

Source code in `src/fenic/api/column.py`

```
def asc_nulls_last(self) -> Column:
    """Mark this column for ascending sort order with nulls last.

    Returns:
        Column: A sort expression with ascending order and nulls last.

    Example: Sort by age in ascending order with nulls last
        ```python
        # Sort a dataframe by age in ascending order, with nulls appearing last
        df.sort(col("age").asc_nulls_last()).show()
        ```
    """
    return Column._from_logical_expr(
        SortExpr(self._logical_expr, ascending=True, nulls_last=True)
    )
```

### cast

```
cast(data_type: DataType) -> Column
```

Cast the column to a new data type.

This method creates an expression that casts the column to a specified data type.
The casting behavior depends on the source and target types:

Primitive type casting:

- Numeric types (IntegerType, FloatType, DoubleType) can be cast between each other
- Numeric types can be cast to/from StringType
- DateType and TimestampType can be cast between each other
- DateType and TimestampType can be cast to/from numeric types and StringType
- BooleanType can be cast to/from numeric types and StringType
- StringType cannot be directly cast to BooleanType (will raise TypeError)
- BooleanType cannot be cast to DateType or TimestampType (will raise TypeError)
- DateType and TimestampType cannot be cast to BooleanType (will raise TypeError)

Complex type casting:

- ArrayType can only be cast to another ArrayType (with castable element types)
- StructType can only be cast to another StructType (with matching/castable fields)
- Primitive types cannot be cast to/from complex types

Parameters:

- **`data_type`**
  (`DataType`)
  –

  The target DataType to cast the column to

Returns:

- **`Column`** ( `Column`
  ) –

  A Column representing the casted expression

Cast integer to string

```
# Convert an integer column to string type
df.select(col("int_col").cast(StringType))
```

Cast array of integers to array of strings

```
# Convert an array of integers to an array of strings
df.select(col("int_array").cast(ArrayType(element_type=StringType)))
```

Cast struct fields to different types

```
# Convert struct fields to different types
new_type = StructType([
    StructField("id", StringType),
    StructField("value", FloatType)
])
df.select(col("data_struct").cast(new_type))
```

Raises:

- `TypeError`
  –

  If the requested cast operation is not supported

Source code in `src/fenic/api/column.py`

```
def cast(self, data_type: DataType) -> Column:
    """Cast the column to a new data type.

    This method creates an expression that casts the column to a specified data type.
    The casting behavior depends on the source and target types:

    Primitive type casting:

    - Numeric types (IntegerType, FloatType, DoubleType) can be cast between each other
    - Numeric types can be cast to/from StringType
    - DateType and TimestampType can be cast between each other
    - DateType and TimestampType can be cast to/from numeric types and StringType
    - BooleanType can be cast to/from numeric types and StringType
    - StringType cannot be directly cast to BooleanType (will raise TypeError)
    - BooleanType cannot be cast to DateType or TimestampType (will raise TypeError)
    - DateType and TimestampType cannot be cast to BooleanType (will raise TypeError)

    Complex type casting:

    - ArrayType can only be cast to another ArrayType (with castable element types)
    - StructType can only be cast to another StructType (with matching/castable fields)
    - Primitive types cannot be cast to/from complex types

    Args:
        data_type (DataType): The target DataType to cast the column to

    Returns:
        Column: A Column representing the casted expression

    Example: Cast integer to string
        ```python
        # Convert an integer column to string type
        df.select(col("int_col").cast(StringType))
        ```

    Example: Cast array of integers to array of strings
        ```python
        # Convert an array of integers to an array of strings
        df.select(col("int_array").cast(ArrayType(element_type=StringType)))
        ```

    Example: Cast struct fields to different types
        ```python
        # Convert struct fields to different types
        new_type = StructType([
            StructField("id", StringType),
            StructField("value", FloatType)
        ])
        df.select(col("data_struct").cast(new_type))
        ```

    Raises:
        TypeError: If the requested cast operation is not supported
    """
    return Column._from_logical_expr(CastExpr(self._logical_expr, data_type))
```

### contains

```
contains(other: Union[str, Column]) -> Column
```

Check if the column contains a substring.

This method creates a boolean expression that checks if each value in the column
contains the specified substring. The substring can be either a literal string
or a column expression.

Parameters:

- **`other`**
  (`Union[str, Column]`)
  –

  The substring to search for (can be a string or column expression)

Returns:

- **`Column`** ( `Column`
  ) –

  A boolean column indicating whether each value contains the substring

Find rows where name contains "john"

```
# Filter rows where the name column contains "john"
df.filter(col("name").contains("john"))
```

Find rows where text contains a dynamic pattern

```
# Filter rows where text contains a value from another column
df.filter(col("text").contains(col("pattern")))
```

Source code in `src/fenic/api/column.py`

```
def contains(self, other: Union[str, Column]) -> Column:
    """Check if the column contains a substring.

    This method creates a boolean expression that checks if each value in the column
    contains the specified substring. The substring can be either a literal string
    or a column expression.

    Args:
        other (Union[str, Column]): The substring to search for (can be a string or column expression)

    Returns:
        Column: A boolean column indicating whether each value contains the substring

    Example: Find rows where name contains "john"
        ```python
        # Filter rows where the name column contains "john"
        df.filter(col("name").contains("john"))
        ```

    Example: Find rows where text contains a dynamic pattern
        ```python
        # Filter rows where text contains a value from another column
        df.filter(col("text").contains(col("pattern")))
        ```
    """
    if isinstance(other, str):
        other_expr = LiteralExpr(other, StringType)
    else:
        other_expr = other._logical_expr
    return Column._from_logical_expr(ContainsExpr(self._logical_expr, other_expr))
```

### contains_any

```
contains_any(others: List[str], case_insensitive: bool = True) -> Column
```

Check if the column contains any of the specified substrings.

This method creates a boolean expression that checks if each value in the column
contains any of the specified substrings. The matching can be case-sensitive or
case-insensitive.

Parameters:

- **`others`**
  (`List[str]`)
  –

  List of substrings to search for
- **`case_insensitive`**
  (`bool`, default:
  `True`
  )
  –

  Whether to perform case-insensitive matching (default: True)

Returns:

- **`Column`** ( `Column`
  ) –

  A boolean column indicating whether each value contains any substring

Find rows where name contains "john" or "jane" (case-insensitive)

```
# Filter rows where name contains either "john" or "jane"
df.filter(col("name").contains_any(["john", "jane"]))
```

Case-sensitive matching

```
# Filter rows with case-sensitive matching
df.filter(col("name").contains_any(["John", "Jane"], case_insensitive=False))
```

Source code in `src/fenic/api/column.py`

```
def contains_any(self, others: List[str], case_insensitive: bool = True) -> Column:
    """Check if the column contains any of the specified substrings.

    This method creates a boolean expression that checks if each value in the column
    contains any of the specified substrings. The matching can be case-sensitive or
    case-insensitive.

    Args:
        others (List[str]): List of substrings to search for
        case_insensitive (bool): Whether to perform case-insensitive matching (default: True)

    Returns:
        Column: A boolean column indicating whether each value contains any substring

    Example: Find rows where name contains "john" or "jane" (case-insensitive)
        ```python
        # Filter rows where name contains either "john" or "jane"
        df.filter(col("name").contains_any(["john", "jane"]))
        ```

    Example: Case-sensitive matching
        ```python
        # Filter rows with case-sensitive matching
        df.filter(col("name").contains_any(["John", "Jane"], case_insensitive=False))
        ```
    """
    return Column._from_logical_expr(
        ContainsAnyExpr(self._logical_expr, others, case_insensitive)
    )
```

### desc

```
desc() -> Column
```

Mark this column for descending sort order.

Returns:

- **`Column`** ( `Column`
  ) –

  A sort expression with descending order.

Sort by age in descending order

```
# Sort a dataframe by age in descending order
df.sort(col("age").desc()).show()
```

Source code in `src/fenic/api/column.py`

```
def desc(self) -> Column:
    """Mark this column for descending sort order.

    Returns:
        Column: A sort expression with descending order.

    Example: Sort by age in descending order
        ```python
        # Sort a dataframe by age in descending order
        df.sort(col("age").desc()).show()
        ```
    """
    return Column._from_logical_expr(
        SortExpr(self._logical_expr, ascending=False)
    )
```

### desc_nulls_first

```
desc_nulls_first() -> Column
```

Alias for desc().

Returns:

- **`Column`** ( `Column`
  ) –

  A sort expression with descending order and nulls first.

Sort by age in descending order with nulls first

```
df.sort(col("age").desc_nulls_first()).show()
```

Source code in `src/fenic/api/column.py`

```
def desc_nulls_first(self) -> Column:
    """Alias for desc().

    Returns:
        Column: A sort expression with descending order and nulls first.

    Example: Sort by age in descending order with nulls first
        ```python
        df.sort(col("age").desc_nulls_first()).show()
        ```
    """
    return self.desc()
```

### desc_nulls_last

```
desc_nulls_last() -> Column
```

Mark this column for descending sort order with nulls last.

Returns:

- **`Column`** ( `Column`
  ) –

  A sort expression with descending order and nulls last.

Sort by age in descending order with nulls last

```
# Sort a dataframe by age in descending order, with nulls appearing last
df.sort(col("age").desc_nulls_last()).show()
```

Source code in `src/fenic/api/column.py`

```
def desc_nulls_last(self) -> Column:
    """Mark this column for descending sort order with nulls last.

    Returns:
        Column: A sort expression with descending order and nulls last.

    Example: Sort by age in descending order with nulls last
        ```python
        # Sort a dataframe by age in descending order, with nulls appearing last
        df.sort(col("age").desc_nulls_last()).show()
        ```
    """
    return Column._from_logical_expr(
        SortExpr(self._logical_expr, ascending=False, nulls_last=True)
    )
```

### ends_with

```
ends_with(other: Union[str, Column]) -> Column
```

Check if the column ends with a substring.

This method creates a boolean expression that checks if each value in the column
ends with the specified substring. The substring can be either a literal string
or a column expression.

Parameters:

- **`other`**
  (`Union[str, Column]`)
  –

  The substring to check for at the end (can be a string or column expression)

Returns:

- **`Column`** ( `Column`
  ) –

  A boolean column indicating whether each value ends with the substring

Find rows where email ends with "@gmail.com"

```
df.filter(col("email").ends_with("@gmail.com"))
```

Find rows where text ends with a dynamic pattern

```
df.filter(col("text").ends_with(col("suffix")))
```

Raises:

- `ValueError`
  –

  If the substring ends with a regular expression anchor ($)

Source code in `src/fenic/api/column.py`

```
def ends_with(self, other: Union[str, Column]) -> Column:
    """Check if the column ends with a substring.

    This method creates a boolean expression that checks if each value in the column
    ends with the specified substring. The substring can be either a literal string
    or a column expression.

    Args:
        other (Union[str, Column]): The substring to check for at the end (can be a string or column expression)

    Returns:
        Column: A boolean column indicating whether each value ends with the substring

    Example: Find rows where email ends with "@gmail.com"
        ```python
        df.filter(col("email").ends_with("@gmail.com"))
        ```

    Example: Find rows where text ends with a dynamic pattern
        ```python
        df.filter(col("text").ends_with(col("suffix")))
        ```

    Raises:
        ValueError: If the substring ends with a regular expression anchor ($)
    """
    if isinstance(other, str):
        if other.endswith("$"):
            raise ValidationError("substr should not end with a regular expression anchor")
        other_expr = LiteralExpr(other, StringType)
    else:
        other_expr = other._logical_expr
    return Column._from_logical_expr(EndsWithExpr(self._logical_expr, other_expr))
```

### get_item

```
get_item(key: Union[str, int, Column]) -> Column
```

Access an item in a struct or array column.

This method allows accessing elements in complex data types:

- For array columns, the key should be an integer index or a column expression that evaluates to an integer
- For struct columns, the key should be a literal field name

Parameters:

- **`key`**
  (`Union[str, int]`)
  –

  The index (for arrays) or field name (for structs) to access

Returns:

- **`Column`** ( `Column`
  ) –

  A Column representing the accessed item

Access an array element

```
# Get the first element from an array column
df.select(col("array_column").get_item(0))
```

Access a struct field

```
# Get a field from a struct column
df.select(col("struct_column").get_item("field_name"))
```

Source code in `src/fenic/api/column.py`

```
def get_item(self, key: Union[str, int, Column]) -> Column:
    """Access an item in a struct or array column.

    This method allows accessing elements in complex data types:

    - For array columns, the key should be an integer index or a column expression that evaluates to an integer
    - For struct columns, the key should be a literal field name

    Args:
        key (Union[str, int]): The index (for arrays) or field name (for structs) to access

    Returns:
        Column: A Column representing the accessed item

    Example: Access an array element
        ```python
        # Get the first element from an array column
        df.select(col("array_column").get_item(0))
        ```

    Example: Access a struct field
        ```python
        # Get a field from a struct column
        df.select(col("struct_column").get_item("field_name"))
        ```
    """
    if isinstance(key, Column):
        return Column._from_logical_expr(IndexExpr(self._logical_expr, key._logical_expr))
    elif isinstance(key, str):
        return Column._from_logical_expr(IndexExpr(self._logical_expr, LiteralExpr(key, StringType)))
    else:
        return Column._from_logical_expr(IndexExpr(self._logical_expr, LiteralExpr(key, IntegerType)))
```

### ilike

```
ilike(other: Union[str, Column]) -> Column
```

Check if the column matches a SQL LIKE pattern (case-insensitive).

This method creates a boolean expression that checks if each value in the column
matches the specified SQL LIKE pattern, ignoring case.
The pattern can be a string or a a column expression that resolves to a string.

SQL LIKE pattern syntax:

- % matches any sequence of characters
- _ matches any single character

Parameters:

- **`other`**
  (`str`)
  –

  The SQL LIKE pattern to match against

Returns:

- **`Column`** ( `Column`
  ) –

  A boolean column indicating whether each value matches the pattern

Find rows where name starts with "j" and ends with "n" (case-insensitive)

```
# Filter rows where name matches the pattern "j%n" (case-insensitive)
df.filter(col("name").ilike("j%n"))
```

Find rows where code matches pattern (case-insensitive)

```
# Filter rows where code matches the pattern "a_b%" (case-insensitive)
df.filter(col("code").ilike("a_b%"))
```

Source code in `src/fenic/api/column.py`

```
def ilike(self, other: Union[str, Column]) -> Column:
    r"""Check if the column matches a SQL LIKE pattern (case-insensitive).

    This method creates a boolean expression that checks if each value in the column
    matches the specified SQL LIKE pattern, ignoring case.
    The pattern can be a string or a a column expression that resolves to a string.

    SQL LIKE pattern syntax:

    - % matches any sequence of characters
    - _ matches any single character

    Args:
        other (str): The SQL LIKE pattern to match against

    Returns:
        Column: A boolean column indicating whether each value matches the pattern

    Example: Find rows where name starts with "j" and ends with "n" (case-insensitive)
        ```python
        # Filter rows where name matches the pattern "j%n" (case-insensitive)
        df.filter(col("name").ilike("j%n"))
        ```

    Example: Find rows where code matches pattern (case-insensitive)
        ```python
        # Filter rows where code matches the pattern "a_b%" (case-insensitive)
        df.filter(col("code").ilike("a_b%"))
        ```
    """
    if isinstance(other, str):
        other_expr = LiteralExpr(other, StringType)
    else:
        other_expr = other._logical_expr
    return Column._from_logical_expr(ILikeExpr(self._logical_expr, other_expr))
```

### is_in

```
is_in(other: Union[List[Any], ColumnOrName]) -> Column
```

Check if the column is in a list of values or a column expression.

Parameters:

- **`other`**
  (`Union[List[Any], ColumnOrName]`)
  –

  A list of values or a Column expression

Returns:

- **`Column`** ( `Column`
  ) –

  A Column expression representing whether each element of Column is in the list

Check if name is in a list of values

```
# Filter rows where name is in a list of values
df.filter(col("name").is_in(["Alice", "Bob"]))
```

Check if value is in another column

```
# Filter rows where name is in another column
df.filter(col("name").is_in(col("other_column")))
```

Source code in `src/fenic/api/column.py`

```
def is_in(self, other: Union[List[Any], ColumnOrName]) -> Column:
    """Check if the column is in a list of values or a column expression.

    Args:
        other (Union[List[Any], ColumnOrName]): A list of values or a Column expression

    Returns:
        Column: A Column expression representing whether each element of Column is in the list

    Example: Check if name is in a list of values
        ```python
        # Filter rows where name is in a list of values
        df.filter(col("name").is_in(["Alice", "Bob"]))
        ```

    Example: Check if value is in another column
        ```python
        # Filter rows where name is in another column
        df.filter(col("name").is_in(col("other_column")))
        ```
    """
    if isinstance(other, list):
        try:
            type_ = infer_dtype_from_pyobj(other)
            return Column._from_logical_expr(InExpr(self._logical_expr, LiteralExpr(other, type_)))
        except TypeInferenceError as e:
            raise ValidationError(f"Cannot apply IN on {other}. List argument to IN must be be a valid Python List literal.") from e
    else:
        return Column._from_logical_expr(InExpr(self._logical_expr, other._logical_expr))
```

### is_not_null

```
is_not_null() -> Column
```

Check if the column contains non-NULL values.

This method creates an expression that evaluates to TRUE when the column value is not NULL.

Returns:

- **`Column`** ( `Column`
  ) –

  A Column representing a boolean expression that is TRUE when this column is not NULL

Filter rows where a column is not NULL

```
df.filter(col("some_column").is_not_null())
```

Use in a complex condition

```
df.filter(col("col1").is_not_null() & (col("col2") <= 50))
```

Source code in `src/fenic/api/column.py`

```
def is_not_null(self) -> Column:
    """Check if the column contains non-NULL values.

    This method creates an expression that evaluates to TRUE when the column value is not NULL.

    Returns:
        Column: A Column representing a boolean expression that is TRUE when this column is not NULL

    Example: Filter rows where a column is not NULL
        ```python
        df.filter(col("some_column").is_not_null())
        ```

    Example: Use in a complex condition
        ```python
        df.filter(col("col1").is_not_null() & (col("col2") <= 50))
        ```
    """
    return Column._from_logical_expr(IsNullExpr(self._logical_expr, False))
```

### is_null

```
is_null() -> Column
```

Check if the column contains NULL values.

This method creates an expression that evaluates to TRUE when the column value is NULL.

Returns:

- **`Column`** ( `Column`
  ) –

  A Column representing a boolean expression that is TRUE when this column is NULL

Filter rows where a column is NULL

```
# Filter rows where some_column is NULL
df.filter(col("some_column").is_null())
```

Use in a complex condition

```
# Filter rows where col1 is NULL or col2 is greater than 100
df.filter(col("col1").is_null() | (col("col2") > 100))
```

Source code in `src/fenic/api/column.py`

```
def is_null(self) -> Column:
    """Check if the column contains NULL values.

    This method creates an expression that evaluates to TRUE when the column value is NULL.

    Returns:
        Column: A Column representing a boolean expression that is TRUE when this column is NULL

    Example: Filter rows where a column is NULL
        ```python
        # Filter rows where some_column is NULL
        df.filter(col("some_column").is_null())
        ```

    Example: Use in a complex condition
        ```python
        # Filter rows where col1 is NULL or col2 is greater than 100
        df.filter(col("col1").is_null() | (col("col2") > 100))
        ```
    """
    return Column._from_logical_expr(IsNullExpr(self._logical_expr, True))
```

### like

```
like(other: Union[str, Column]) -> Column
```

Check if the column matches a SQL LIKE pattern.

This method creates a boolean expression that checks if each value in the column
matches the specified SQL LIKE pattern.
The pattern can be a string or a a column expression that resolves to a string.

SQL LIKE pattern syntax:

- % matches any sequence of characters
- _ matches any single character

Parameters:

- **`other`**
  (`str`)
  –

  The SQL LIKE pattern to match against

Returns:

- **`Column`** ( `Column`
  ) –

  A boolean column indicating whether each value matches the pattern

Find rows where name starts with "J" and ends with "n"

```
# Filter rows where name matches the pattern "J%n"
df.filter(col("name").like("J%n"))
```

Find rows where code matches specific pattern

```
# Filter rows where code matches the pattern "A_B%"
df.filter(col("code").like("A_B%"))
```

Source code in `src/fenic/api/column.py`

```
def like(self, other: Union[str, Column]) -> Column:
    r"""Check if the column matches a SQL LIKE pattern.

    This method creates a boolean expression that checks if each value in the column
    matches the specified SQL LIKE pattern.
    The pattern can be a string or a a column expression that resolves to a string.

    SQL LIKE pattern syntax:

    - % matches any sequence of characters
    - _ matches any single character

    Args:
        other (str): The SQL LIKE pattern to match against

    Returns:
        Column: A boolean column indicating whether each value matches the pattern

    Example: Find rows where name starts with "J" and ends with "n"
        ```python
        # Filter rows where name matches the pattern "J%n"
        df.filter(col("name").like("J%n"))
        ```

    Example: Find rows where code matches specific pattern
        ```python
        # Filter rows where code matches the pattern "A_B%"
        df.filter(col("code").like("A_B%"))
        ```
    """
    if isinstance(other, str):
        other_expr = LiteralExpr(other, StringType)
    else:
        other_expr = other._logical_expr
    return Column._from_logical_expr(LikeExpr(self._logical_expr, other_expr))
```

### otherwise

```
otherwise(value: Column) -> Column
```

Returns a value when no prior conditions are True.

This is the final part of a when-chain, like the 'else' in an if-elif-else.
All branches must return the same type.

Parameters:

- **`value`**
  (`Column`)
  –

  Value to return when no prior conditions are True

Returns:

- **`Column`** ( `Column`
  ) –

  The complete conditional expression chain

Raises:

- `ValidationError`
  –

  If called on a non-when expression

Example

```
# Create age-based categories
df = session.createDataFrame({"age": [8, 25, 67]})

result = df.select(
    fc.when(col("age") < 18, fc.lit("minor"))
    .when(col("age") < 65, fc.lit("adult"))
    .otherwise(fc.lit("senior"))
    .alias("category")
)

result.show()
# +--------+
# |category|
# +--------+
# |   minor|
# |   adult|
# |  senior|
# +--------+
```

Note

- If otherwise() is not called, unmatched rows return null
- Can only be called once per when-chain
- Typically the last method in the chain

Source code in `src/fenic/api/column.py`

```
def otherwise(self, value: Column) -> Column:
    """Returns a value when no prior conditions are True.

    This is the final part of a when-chain, like the 'else' in an if-elif-else.
    All branches must return the same type.

    Args:
        value: Value to return when no prior conditions are True

    Returns:
        Column: The complete conditional expression chain

    Raises:
        ValidationError: If called on a non-when expression

    Example:
        ```python
        # Create age-based categories
        df = session.createDataFrame({"age": [8, 25, 67]})

        result = df.select(
            fc.when(col("age") < 18, fc.lit("minor"))
            .when(col("age") < 65, fc.lit("adult"))
            .otherwise(fc.lit("senior"))
            .alias("category")
        )

        result.show()
        # +--------+
        # |category|
        # +--------+
        # |   minor|
        # |   adult|
        # |  senior|
        # +--------+
        ```

    Note:
        - If otherwise() is not called, unmatched rows return null
        - Can only be called once per when-chain
        - Typically the last method in the chain
    """
    return Column._from_logical_expr(OtherwiseExpr(self._logical_expr, value._logical_expr))
```

### rlike

```
rlike(other: Union[str, Column]) -> Column
```

Check if the column matches a regular expression pattern.

This method creates a boolean expression that checks if each value in the column
matches the specified regular expression pattern.

Parameters:

- **`other`**
  (`Union[str, Column]`)
  –

  The regular expression pattern to match against.
  Can be a string or a a column expression that resolves to a string.

Returns:

- **`Column`** ( `Column`
  ) –

  A boolean column indicating whether each value matches the pattern

Find rows where phone number matches pattern

```
# Filter rows where phone number matches a specific pattern
df.filter(col("phone").rlike(r"^\d{3}-\d{3}-\d{4}$"))
```

Find rows where text contains word boundaries

```
# Filter rows where text contains a word with boundaries
df.filter(col("text").rlike(r"\bhello\b"))
```

Source code in `src/fenic/api/column.py`

```
def rlike(self, other: Union[str, Column]) -> Column:
    r"""Check if the column matches a regular expression pattern.

    This method creates a boolean expression that checks if each value in the column
    matches the specified regular expression pattern.

    Args:
        other (Union[str, Column]): The regular expression pattern to match against.
              Can be a string or a a column expression that resolves to a string.

    Returns:
        Column: A boolean column indicating whether each value matches the pattern

    Example: Find rows where phone number matches pattern
        ```python
        # Filter rows where phone number matches a specific pattern
        df.filter(col("phone").rlike(r"^\d{3}-\d{3}-\d{4}$"))
        ```

    Example: Find rows where text contains word boundaries
        ```python
        # Filter rows where text contains a word with boundaries
        df.filter(col("text").rlike(r"\bhello\b"))
        ```
    """
    if isinstance(other, str):
        other_expr = LiteralExpr(other, StringType)
    else:
        other_expr = other._logical_expr
    return Column._from_logical_expr(RLikeExpr(self._logical_expr, other_expr))
```

### starts_with

```
starts_with(other: Union[str, Column]) -> Column
```

Check if the column starts with a substring.

This method creates a boolean expression that checks if each value in the column
starts with the specified substring. The substring can be either a literal string
or a column expression.

Parameters:

- **`other`**
  (`Union[str, Column]`)
  –

  The substring to check for at the start (can be a string or column expression)

Returns:

- **`Column`** ( `Column`
  ) –

  A boolean column indicating whether each value starts with the substring

Find rows where name starts with "Mr"

```
# Filter rows where name starts with "Mr"
df.filter(col("name").starts_with("Mr"))
```

Find rows where text starts with a dynamic pattern

```
# Filter rows where text starts with a value from another column
df.filter(col("text").starts_with(col("prefix")))
```

Raises:

- `ValueError`
  –

  If the substring starts with a regular expression anchor (^)

Source code in `src/fenic/api/column.py`

```
def starts_with(self, other: Union[str, Column]) -> Column:
    """Check if the column starts with a substring.

    This method creates a boolean expression that checks if each value in the column
    starts with the specified substring. The substring can be either a literal string
    or a column expression.

    Args:
        other (Union[str, Column]): The substring to check for at the start (can be a string or column expression)

    Returns:
        Column: A boolean column indicating whether each value starts with the substring

    Example: Find rows where name starts with "Mr"
        ```python
        # Filter rows where name starts with "Mr"
        df.filter(col("name").starts_with("Mr"))
        ```

    Example: Find rows where text starts with a dynamic pattern
        ```python
        # Filter rows where text starts with a value from another column
        df.filter(col("text").starts_with(col("prefix")))
        ```

    Raises:
        ValueError: If the substring starts with a regular expression anchor (^)
    """
    if isinstance(other, str):
        if other.startswith("^"):
            raise ValidationError("substr should not start with a regular expression anchor")
        other_expr = LiteralExpr(other, StringType)
    else:
        other_expr = other._logical_expr
    return Column._from_logical_expr(StartsWithExpr(self._logical_expr, other_expr))
```

### when

```
when(condition: Column, value: Column) -> Column
```

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

  A new when expression with this condition added to the chain

Raises:

- `ValidationError`
  –

  If called on a non-when expression (e.g., regular columns)

Example

```
# Build a multi-condition expression:
result = (
    fc.when(col("age") < 18, fc.lit("minor"))
    .when(col("age") < 65, fc.lit("adult"))  # Add another condition
    .when(col("age") >= 65, fc.lit("senior"))
    .otherwise(fc.lit("unknown"))
)

# This evaluates like:
# if age < 18: return "minor"
# elif age < 65: return "adult"
# elif age >= 65: return "senior"
# else: return "unknown"

# ERROR - cannot call .when() on non-when expressions:
# col("age").when(...)  # ValidationError!
```

Note: Conditions are evaluated in order. The first True condition wins.

Source code in `src/fenic/api/column.py`

```
def when(self, condition: Column, value: Column) -> Column:
    """Evaluates a condition for each row and returns a value when true.

    Can be chained with more .when() calls or finished with .otherwise().
    All branches must return the same type.

    Args:
        condition: Boolean expression to test
        value: Value to return when condition is True

    Returns:
        Column: A new when expression with this condition added to the chain

    Raises:
        ValidationError: If called on a non-when expression (e.g., regular columns)

    Example:
        ```python
        # Build a multi-condition expression:
        result = (
            fc.when(col("age") < 18, fc.lit("minor"))
            .when(col("age") < 65, fc.lit("adult"))  # Add another condition
            .when(col("age") >= 65, fc.lit("senior"))
            .otherwise(fc.lit("unknown"))
        )

        # This evaluates like:
        # if age < 18: return "minor"
        # elif age < 65: return "adult"
        # elif age >= 65: return "senior"
        # else: return "unknown"

        # ERROR - cannot call .when() on non-when expressions:
        # col("age").when(...)  # ValidationError!
        ```

    Note: Conditions are evaluated in order. The first True condition wins.
    """
    return Column._from_logical_expr(WhenExpr(self._logical_expr, condition._logical_expr, value._logical_expr))
```
