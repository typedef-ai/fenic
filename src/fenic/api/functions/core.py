"""Core functions for Fenic DataFrames."""

from typing import Any

from pydantic import ConfigDict, validate_call

from fenic.api.column import Column
from fenic.core._logical_plan.expressions import LiteralExpr
from fenic.core._utils.type_inference import (
    TypeInferenceError,
    infer_dtype_from_pyobj,
)
from fenic.core.error import ValidationError
from fenic.core.types.datatypes import ArrayType, DataType, StructType


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

def none(data_type: DataType) -> Column:
    """Creates a Column expression representing a null value of the given type.

    No matter the data type, the value that will be populated in the column will be `None`.
    This is useful for creating columns with null values of a specific type.

    Args:
        data_type: The data type of the null value

    Returns:
        A Column expression representing the null value

    Raises:
        ValidationError: If the data type is not a valid data type

    Example: Creating a column with a null value of a primitive type
        ```python
        # The newly created `b` column will have a value of `None` for all rows
        df.select(fc.col("a"), fc.none(fc.IntegerType).alias("b"))
        ```

    Example: Creating a column with a null value of an array/struct type
        ```python
        # The newly created `b` and `c` columns will have a value of `None` for all rows
        df.select(
            fc.col("a"),
            fc.none(fc.ArrayType(fc.IntegerType)).alias("b"),
            fc.none(fc.StructType([fc.StructField("b", fc.IntegerType)])).alias("c"),
        )
        ```

    """
    return Column._from_logical_expr(LiteralExpr(None, data_type))

def empty(data_type: DataType) -> Column:
    """Creates a Column expression representing an empty value of the given type.

    - If the data type is an array or struct, the empty value will be an empty array or struct.
    - Otherwise, the empty value will be `None`, as if you had called `none()` on the data type.
      This is useful for creating columns with empty values of a specific type.

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
    return none(data_type)

def lit(value: Any) -> Column:
    """Creates a Column expression representing a literal value.

    Args:
        value: The literal value to create a column for


    Returns:
        A Column expression representing the literal value
    Raises:
        ValidationError: If the type of the value cannot be inferred and no dtype is provided
    """
    if not value:
        if value is None:
            raise ValidationError("Cannot create a literal with value `None`. Use `none()` instead.")
        else:
            raise ValidationError(f"Cannot create a literal with empty value `{value}` Use `empty()` instead.")
    try:
        inferred_type = infer_dtype_from_pyobj(value)
    except TypeInferenceError as e:
        raise ValidationError(f"`lit` failed to infer type for value `{value}`") from e
    literal_expr = LiteralExpr(value, inferred_type)
    return Column._from_logical_expr(literal_expr)