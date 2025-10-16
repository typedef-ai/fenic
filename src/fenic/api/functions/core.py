"""Core functions for Fenic DataFrames."""
from typing import Any

from pydantic import ConfigDict, validate_call

from fenic.api.column import Column
from fenic.core._logical_plan.expressions import LiteralExpr
from fenic.core._logical_plan.expressions.basic import UnresolvedLiteralExpr
from fenic.core._utils.type_inference import (
    TypeInferenceError,
    infer_dtype_from_pyobj,
)
from fenic.core.error import ValidationError
from fenic.core.types.datatypes import ArrayType, DataType, StructType, _LogicalType


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