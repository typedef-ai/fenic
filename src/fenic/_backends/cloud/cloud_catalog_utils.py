from typing import Union

import pyarrow as pa

from fenic.core.types.datatypes import (
    ArrayType,
    BooleanType,
    DocumentPathType,
    DoubleType,
    EmbeddingType,
    FloatType,
    IntegerType,
    StringType,
    StructType,
    TranscriptType,
    _HtmlType,
    _JsonType,
    _MarkdownType,
    _PrimitiveType,
)
from fenic.core.types.schema import Schema


def convert_custom_dtype_to_pyarrow(
    custom_dtype: Union[
        _PrimitiveType,
        ArrayType,
        StructType,
        _JsonType,
        _MarkdownType,
        _HtmlType,
        TranscriptType,
        DocumentPathType,
    ]
) -> object:
    """Convert a custom data type to the corresponding PyArrow data type.

    Importing PyArrow inside the function avoids a hard dependency unless used.
    """

    if isinstance(custom_dtype, _PrimitiveType):
        if custom_dtype == IntegerType:
            return pa.int64()
        elif custom_dtype == FloatType:
            return pa.float32()
        elif custom_dtype == DoubleType:
            return pa.float64()
        elif custom_dtype == StringType:
            return pa.string()
        elif custom_dtype == BooleanType:
            return pa.bool_()
    elif isinstance(custom_dtype, ArrayType):
        return pa.list(convert_custom_dtype_to_pyarrow(custom_dtype.element_type))
    elif isinstance(custom_dtype, StructType):
        return pa.struct(
            [
                pa.field(field.name, convert_custom_dtype_to_pyarrow(field.data_type))
                for field in custom_dtype.struct_fields
            ]
        )
    elif isinstance(custom_dtype, EmbeddingType):
        return pa.list_(pa.float32(), custom_dtype.dimensions)
    elif isinstance(
        custom_dtype, (_JsonType, _MarkdownType, _HtmlType, TranscriptType, DocumentPathType)
    ):
        return pa.string()
    else:
        raise ValueError(f"Unsupported custom data type: {custom_dtype}")



def convert_custom_schema_to_pyarrow_schema(custom_schema: Schema) -> object:
    """Convert a fenic Schema to a PyArrow schema.

    Returns a ``pyarrow.Schema`` with fields mapped from the custom schema
    using ``convert_custom_dtype_to_pyarrow``.
    """
    return pa.schema(
        [
            pa.field(field.name, convert_custom_dtype_to_pyarrow(field.data_type))
            for field in custom_schema.column_fields
        ]
    )
