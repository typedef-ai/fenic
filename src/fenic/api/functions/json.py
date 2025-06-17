"""JSON functions."""
from pydantic import ConfigDict, validate_call

from fenic.api.column import Column, ColumnOrName
from fenic.core._logical_plan.expressions import (
    JqExpr,
    JsonContainsExpr,
    JsonTypeExpr,
)


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def jq(column: ColumnOrName, query: str) -> Column:
    """Applies a JQ query to a column containing JSON-formatted strings.

    Args:
        column (ColumnOrName): Input column of type `JsonType`.
        query (str): A [JQ](https://jqlang.org/) expression used to extract or transform values.

    Returns:
        Column: A column containing the result of applying the JQ query to each row's JSON input.

    Notes:
        - The input column *must* be of type `JsonType`. Use `cast(JsonType)` if needed to ensure correct typing.
        - This function supports extracting nested fields, transforming arrays/objects, and other standard JQ operations.

    Example: Extract nested field
        ```python
        # Extract the "user.name" field from a JSON column
        df.select(json.jq(col("json_col"), ".user.name"))
        ```

    Example: Cast to JsonType before querying
        ```python
        df.select(json.jq(col("raw_json").cast(JsonType), ".event.type"))
        ```

    Example: Work with arrays
        ```python
        # Work with arrays using JQ functions
        df.select(json.jq(col("json_array"), "map(.id)"))
        ```
    """
    return Column._from_logical_expr(
        JqExpr(Column._from_col_or_name(column)._logical_expr, query)
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def get_type(column: ColumnOrName) -> Column:
    """Get the JSON type of each value.

    Args:
        column (ColumnOrName): Input column of type `JsonType`.

    Returns:
        Column: A column of strings indicating the JSON type
                ("string", "number", "boolean", "array", "object", "null").

    Example: Get JSON types
        ```python
        df.select(json.get_type(col("json_data")))
        ```

    Example: Filter by type
        ```python
        # Filter by type
        df.filter(json.get_type(col("data")) == "array")
        ```
    """
    return Column._from_logical_expr(
        JsonTypeExpr(Column._from_col_or_name(column)._logical_expr)
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def contains(column: ColumnOrName, value: str) -> Column:
    """Check if a JSON value contains the specified value using recursive deep search.

    Args:
        column (ColumnOrName): Input column of type `JsonType`.
        value (str): Valid JSON string to search for.

    Returns:
        Column: A column of booleans indicating whether the JSON contains the value.

    Matching Rules:
        - **Objects**: Uses partial matching - `{"role": "admin"}` matches `{"role": "admin", "level": 5}`
        - **Arrays**: Uses exact matching - `[1, 2]` only matches exactly `[1, 2]`, not `[1, 2, 3]`
        - **Primitives**: Uses exact matching - `42` matches `42` but not `"42"`
        - **Search is recursive**: Searches at all nesting levels throughout the JSON structure
        - **Type-aware**: Distinguishes between `42` (number) and `"42"` (string)

    Example: Find objects with partial structure match
        ```python
        # Find objects with partial structure match (at any nesting level)
        df.select(json.contains(col("json_data"), '{"name": "Alice"}'))
        # Matches: {"name": "Alice", "age": 30} and {"user": {"name": "Alice"}}
        ```

    Example: Find exact array match
        ```python
        # Find exact array match (at any nesting level)
        df.select(json.contains(col("json_data"), '["read", "write"]'))
        # Matches: {"permissions": ["read", "write"]} but not ["read", "write", "admin"]
        ```

    Example: Find exact primitive values
        ```python
        # Find exact primitive values (at any nesting level)
        df.select(json.contains(col("json_data"), '"admin"'))
        # Matches: {"role": "admin"} and ["admin", "user"] but not {"role": "administrator"}
        ```

    Example: Type distinction matters
        ```python
        # Type distinction matters
        df.select(json.contains(col("json_data"), '42'))      # number 42
        df.select(json.contains(col("json_data"), '"42"'))    # string "42"
        ```

    Raises:
        ValidationError: If `value` is not valid JSON.
    """
    return Column._from_logical_expr(
        JsonContainsExpr(Column._from_col_or_name(column)._logical_expr, value)
    )
