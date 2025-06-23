"""Utilities for converting between different schema representations."""
import typing
from typing import Union, get_origin

import polars as pl
from pydantic import BaseModel

from fenic.core.types.datatypes import (
    ArrayType,
    BooleanType,
    DataType,
    DoubleType,
    EmbeddingType,
    FloatType,
    IntegerType,
    JsonType,
    StringType,
    StructField,
    StructType,
    _JsonType,
    _PrimitiveType,
)
from fenic.core.types.schema import (
    ColumnField,
    Schema,
)


def convert_polars_schema_to_custom_schema(
    polars_schema: pl.Schema,
) -> Schema:
    """Convert a Polars schema to an Omnitype Schema.

    Args:
        polars_schema: The Polars schema to convert

    Returns:
        The corresponding Omnitype Schema with equivalent column fields

    Example:
        >>> custom_schema = convert_polars_schema_to_custom_schema(df.schema)
    """
    return Schema(
        column_fields=[
            ColumnField(
                name=col_name,
                data_type=_convert_polars_dtype_to_custom_dtype(polars_dtype),
            )
            for col_name, polars_dtype in polars_schema.items()
        ]
    )


def convert_custom_schema_to_polars_schema(
    custom_schema: Schema,
) -> pl.Schema:
    """Convert an Omnitype Schema to a Polars schema.

    Args:
        custom_schema: The Omnitype Schema to convert

    Returns:
        The corresponding Polars schema with equivalent fields

    Example:
        >>> polars_schema = convert_custom_schema_to_polars_schema(custom_schema)
    """
    return pl.Schema(
        {
            field.name: convert_custom_dtype_to_polars(field.data_type)
            for field in custom_schema.column_fields
        }
    )


def convert_pydantic_type_to_custom_struct_type(
    model: type[BaseModel],
) -> StructType:
    """Convert a Pydantic model to a custom StructType.

    Args:
        model: The Pydantic model to convert (either an instance or a class)

    Returns:
        The corresponding custom StructType

    Raises:
        ValueError: If the model is not a Pydantic model

    Example:
        >>> struct_type = convert_pydantic_to_custom_struct_type(model)
    """
    if not (isinstance(model, type) and issubclass(model, BaseModel)):
        raise ValueError(
            f"Expected a pydantic model class, got type:{type(model).__name__}"
        )

    fields = []
    for field_name, field_info in model.model_fields.items():
        if field_info.description is None:
            raise ValueError(
                f"Field {field_name} has no description.  Please specify a description for each field."
            )
        if get_origin(field_info.annotation) is list:
            element_type = field_info.annotation.__args__[0]
            if isinstance(element_type, type) and issubclass(element_type, BaseModel):
                fields.append(
                    StructField(
                        field_name,
                        data_type=ArrayType(
                            element_type=convert_pydantic_type_to_custom_struct_type(
                                element_type
                            )
                        ),
                    )
                )
            else:
                fields.append(
                    StructField(
                        field_name,
                        data_type=ArrayType(
                            element_type=_convert_pytype_to_custom_dtype(element_type)
                        ),
                    )
                )
        elif isinstance(field_info.annotation, type) and issubclass(
            field_info.annotation, BaseModel
        ):
            fields.append(
                StructField(
                    field_name,
                    convert_pydantic_type_to_custom_struct_type(field_info.annotation),
                )
            )
        else:
            fields.append(
                StructField(
                    field_name,
                    data_type=_convert_pytype_to_custom_dtype(field_info.annotation),
                )
            )

    return StructType(fields)


def convert_custom_dtype_to_polars(
    custom_dtype: Union[
        _PrimitiveType,
        ArrayType,
        StructType,
        _JsonType,
    ],
) -> pl.DataType:
    """Convert custom data type to the Polars data type.

    Args:
        custom_dtype: Custom data type

    Returns:
        pl.DataType: Corresponding Polars data type

    Raises:
        ValueError: If the custom data type is not supported
    """
    if isinstance(custom_dtype, _PrimitiveType):
        if custom_dtype == IntegerType:
            return pl.Int64
        elif custom_dtype == FloatType:
            return pl.Float32
        elif custom_dtype == DoubleType:
            return pl.Float64
        elif custom_dtype == StringType:
            return pl.String
        elif custom_dtype == BooleanType:
            return pl.Boolean
    elif isinstance(custom_dtype, ArrayType):
        return pl.List(convert_custom_dtype_to_polars(custom_dtype.element_type))
    elif isinstance(custom_dtype, StructType):
        return pl.Struct(
            [
                pl.Field(field.name, convert_custom_dtype_to_polars(field.data_type))
                for field in custom_dtype.struct_fields
            ]
        )
    elif custom_dtype == JsonType:
        return pl.String
    elif isinstance(custom_dtype, EmbeddingType):
        return pl.Array(pl.Float32, custom_dtype.dimensions)
    else:
        raise ValueError(f"Unsupported custom data type: {custom_dtype}")


def _convert_pytype_to_custom_dtype(py_type: type) -> _PrimitiveType:
    """Convert a basic Python type to a PrimitiveType."""
    if py_type is str:
        return StringType
    elif py_type is int:
        return IntegerType
    elif py_type is float:
        return DoubleType
    elif py_type is bool:
        return BooleanType
    elif hasattr(py_type, '__origin__') and py_type.__origin__ is typing.Literal:
        return StringType
    else:
        raise ValueError(f"Unsupported Python type: {py_type.__name__}")


def _convert_polars_dtype_to_custom_dtype(
    polars_dtype: pl.DataType,
) -> DataType:
    """Convert Polars data type to the custom PrimitiveType enum or complex type.

    Args:
        polars_dtype: Polars data type

    Returns:
        Union[PrimitiveType, ArrayType, StructType]: Corresponding custom data type

    Raises:
        ValueError: If the Polars data type is not supported
    """
    if isinstance(polars_dtype, (pl.Int32, pl.Int64, pl.Int128, pl.Int16, pl.Int8)):
        return IntegerType
    elif isinstance(polars_dtype, (pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)):
        return IntegerType
    elif isinstance(polars_dtype, pl.Float32):
        return FloatType
    elif isinstance(polars_dtype, pl.Float64):
        return DoubleType
    elif isinstance(polars_dtype, pl.Utf8):
        return StringType
    elif isinstance(polars_dtype, pl.Boolean):
        return BooleanType
    elif isinstance(polars_dtype, pl.List):
        element_type = _convert_polars_dtype_to_custom_dtype(polars_dtype.inner)
        return ArrayType(element_type=element_type)
    elif isinstance(polars_dtype, pl.Array):
        element_type = _convert_polars_dtype_to_custom_dtype(polars_dtype.inner)
        return ArrayType(element_type=element_type)
    elif isinstance(polars_dtype, pl.Struct):
        # Convert each field in the StructType
        struct_fields = [
            StructField(
                name=field.name,
                data_type=_convert_polars_dtype_to_custom_dtype(field.dtype),
            )
            for field in polars_dtype.fields
        ]
        return StructType(struct_fields=struct_fields)
    elif isinstance(polars_dtype, (pl.Date, pl.Datetime)):
        # pl.Time is not properly supported by DuckDB, it is suggested that users
        # use timestamp instead.
        # https://duckdb.org/docs/sql/types/time
        return StringType
    else:
        raise ValueError(f"Unsupported Polars data type: {polars_dtype}")
