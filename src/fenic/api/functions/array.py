"""Array functions for Fenic DataFrames.

This module provides array manipulation functions following PySpark conventions.
Functions are available via fc.arr.* namespace (e.g., fc.arr.size()).
"""

from typing import Union

from pydantic import ConfigDict, validate_call

from fenic.api.column import Column, ColumnOrName
from fenic.api.functions.core import lit
from fenic.core._logical_plan.expressions import (
    ArrayCompactExpr,
    ArrayContainsExpr,
    ArrayDistinctExpr,
    ArrayExceptExpr,
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
    ElementAtExpr,
)


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def size(column: ColumnOrName) -> Column:
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
        df.select(fc.arr.size("tags"))

        # Use with column reference
        df.select(fc.arr.size(fc.col("tags")))
        ```
    """
    return Column._from_logical_expr(
        ArrayLengthExpr(Column._from_col_or_name(column)._logical_expr)
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def distinct(column: ColumnOrName) -> Column:
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
        df.select(fc.arr.distinct("array_col").alias("distinct_array"))
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
def contains(
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
        df.select(fc.arr.contains("tags", "python"))

        # Check using a value from another column
        df.select(fc.arr.contains("tags", fc.col("search_term")))
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
def max(column: ColumnOrName) -> Column:
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

        result = df.select(fc.arr.max("numbers").alias("max_value"))
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

        result = df.select(fc.array.max("words").alias("max_word"))
        # Output: ["zebra", "dog"]
        ```
    """
    return Column._from_logical_expr(
        ArrayMaxExpr(Column._from_col_or_name(column)._logical_expr)
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def min(column: ColumnOrName) -> Column:
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

        result = df.select(fc.arr.min("numbers").alias("min_value"))
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

        result = df.select(fc.array.min("words").alias("min_word"))
        # Output: ["apple", "bat"]
        ```
    """
    return Column._from_logical_expr(
        ArrayMinExpr(Column._from_col_or_name(column)._logical_expr)
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def sort(column: ColumnOrName) -> Column:
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

        result = df.select(fc.array.sort("numbers").alias("sorted"))
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

        result = df.select(fc.array.sort("words").alias("sorted"))
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
            fc.array.reverse("numbers").alias("reversed_nums"),
            fc.array.reverse("words").alias("reversed_words")
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
def remove(column: ColumnOrName, element: Union[str, int, float, bool, Column]) -> Column:
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
            fc.array.remove("tags", "a").alias("no_a"),
            fc.array.remove("numbers", 5).alias("no_five")
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

        result = df.select(fc.array.remove("values", fc.col("to_remove")))
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
def union(col1: ColumnOrName, col2: ColumnOrName) -> Column:
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

        result = df.select(fc.array.union("tags1", "tags2").alias("all_tags"))
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

        result = df.select(fc.array.union("nums1", "nums2"))
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
def intersect(col1: ColumnOrName, col2: ColumnOrName) -> Column:
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

        result = df.select(fc.array.intersect("arr1", "arr2").alias("common"))
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

        result = df.select(fc.array.intersect("arr1", "arr2"))
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
def except_(col1: ColumnOrName, col2: ColumnOrName) -> Column:
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

        result = df.select(fc.array.except_("all_tags", "deprecated").alias("active"))
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

        result = df.select(fc.array.except_("arr1", "arr2"))
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
def compact(column: ColumnOrName) -> Column:
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

        result = df.select(fc.array.compact("values").alias("compact"))
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

        result = df.select(fc.array.compact("sparse"))
        # Output: [[1], []]
        ```
    """
    return Column._from_logical_expr(
        ArrayCompactExpr(Column._from_col_or_name(column)._logical_expr)
    )


@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def repeat(col: ColumnOrName, count: Union[int, ColumnOrName]) -> Column:
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
            fc.array.repeat(fc.lit("x"), 3).alias("repeated"),
            fc.array.repeat(fc.lit(0), 5).alias("zeros")
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

        result = df.select(fc.array.repeat(fc.col("value"), fc.col("count")))
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
            fc.array.slice("numbers", 1, 3).alias("first_three"),
            fc.array.slice("numbers", 2, 2).alias("middle_two")
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
        result = df.select(fc.array.slice("arr", -3, 3))
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
            fc.array.slice("values", fc.col("start_idx"), fc.col("num_elements"))
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
            fc.array.element_at("numbers", 1).alias("first"),
            fc.array.element_at("numbers", 2).alias("second")
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
            fc.array.element_at("arr", -1).alias("last"),
            fc.array.element_at("arr", -2).alias("second_last")
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

        result = df.select(fc.array.element_at("values", fc.col("position")))
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
def overlap(col1: ColumnOrName, col2: ColumnOrName) -> Column:
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

        result = df.select(fc.array.overlap("arr1", "arr2").alias("has_overlap"))
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
        result = df.filter(fc.array.overlap("user_tags", "required"))
        # Output: Rows with indices 0 and 2 (have "python" tag)
        ```

    Example: Numeric arrays
        ```python
        df = fc.Session.local().create_dataframe({
            "nums1": [[1, 2, 3], [4, 5, 6]],
            "nums2": [[3, 4, 5], [7, 8, 9]]
        })

        result = df.select(fc.array.overlap("nums1", "nums2"))
        # Output: [true, false]
        ```
    """
    return Column._from_logical_expr(
        ArraysOverlapExpr(
            Column._from_col_or_name(col1)._logical_expr,
            Column._from_col_or_name(col2)._logical_expr
        )
    )
