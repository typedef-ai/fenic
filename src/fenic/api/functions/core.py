"""Core functions for Fenic DataFrames."""

from typing import Any, Optional

from pydantic import ConfigDict, validate_call

from fenic.api.column import Column
from fenic.core._logical_plan.expressions import LiteralExpr
from fenic.core._utils.type_inference import (
    TypeInferenceError,
    infer_dtype_from_pyobj,
)
from fenic.core.error import ValidationError
from fenic.core.types.datatypes import DataType


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


def lit(value: Any, dtype: Optional[DataType] = None) -> Column:
    """Creates a Column expression representing a literal value.

    Args:
        value: The literal value to create a column for
        dtype: The data type of the literal value (only required if the type cannot be inferred, like `None` or an empty list/dict)

    Returns:
        A Column expression representing the literal value

    Example: Create a literal with a struct
        ```python
        # Create a literal with a struct
        df.select(
            lit({"a": 1, "b": 2}).alias("struct")
        )
        ```

    Example: Create a literal with a list
        ```python
        # Create a literal with a list
        df.select(
            lit([1, 2, 3]).alias("list")
        )
        ```

    Example: Create a literal with a None
        ```python
        # Create a literal with a None
        df.select(
            lit(None, dtype=IntegerType).alias("none")
        )
        ```

    Example: Create a literal with an empty list
        ```python
        # Create a literal with an empty list
        df.select(
            lit([], dtype=ArrayType(IntegerType)).alias("empty_list")
        )
        ```
    Raises:
        ValidationError: If the type of the value cannot be inferred and no dtype is provided
    """
    # Handle special cases with helpful error messages
    if value == {} and dtype is None:
        raise ValidationError("`lit` failed to infer type for value `{}`. "
                              "If you are trying to create a literal with an empty struct, "
                              "you must specify the dtype explicitly. "
                              "For example, `lit({}, dtype=StructType([StructField(name='c', data_type=IntegerType)]))`.")

    # Attempt type inference
    try:
        inferred_type = infer_dtype_from_pyobj(value)
    except TypeInferenceError as e:
        if dtype is None:
            raise ValidationError(f"`lit` failed to infer type for value `{value}`. "
                                  f"If you are trying to create a literal with `None` or an empty list, "
                                  f"you must specify the dtype explicitly.") from e
        inferred_type = dtype
    else:
        # Handle dtype override/validation
        if dtype is not None:
            if value == {}:  # Empty dict: allow dtype override
                inferred_type = dtype
            elif inferred_type != dtype:  # Type mismatch: error
                raise ValidationError(f"User provided dtype {dtype} does not match inferred type {inferred_type} for value `{value}` "
                                      "If value is not None or an empty list, you need not specify a dtype.")

    return Column._from_logical_expr(LiteralExpr(value, inferred_type))
