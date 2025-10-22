import datetime
from typing import Any

import polars as pl

from fenic.core.types.datatypes import (
    ArrayType,
    BooleanType,
    DataType,
    DateType,
    DocumentPathType,
    DoubleType,
    EmbeddingType,
    FloatType,
    HtmlType,
    IntegerType,
    JsonType,
    MarkdownType,
    StringType,
    StructField,
    StructType,
    TimestampType,
    TranscriptType,
)


class TypeInferenceError(ValueError):
    def __init__(self, message: str, path: str = ""):
        full_message = f"{message} at {path}" if path else message
        super().__init__(full_message)
        self.path = path

def infer_pytype_from_dtype(dtype: DataType) -> type:
    if dtype == BooleanType:
        return bool
    elif dtype == IntegerType:
        return int
    elif dtype == FloatType:
        return float
    elif dtype == StringType:
        return str
    elif dtype == DateType:
        return datetime.date
    elif dtype == TimestampType:
        return datetime.datetime(tzinfo=datetime.timezone.utc)
    elif dtype == JsonType or dtype == MarkdownType or dtype == HtmlType:
        return str
    elif isinstance(dtype, (TranscriptType, DocumentPathType)):
        return str
    elif isinstance(dtype, ArrayType):
        return list
    elif isinstance(dtype, StructType):
        return dict
    elif isinstance(dtype, EmbeddingType):
        return list[float]
    else:
        raise TypeInferenceError(f"Unsupported type {dtype}")

def infer_dtype_from_pyobj(value: Any, path="") -> DataType:
    if isinstance(value, bool):
        return BooleanType
    elif isinstance(value, int):
        return IntegerType
    elif isinstance(value, float):
        return FloatType
    elif isinstance(value, str):
        return StringType
    elif isinstance(value, datetime.datetime):
        return TimestampType
    elif isinstance(value, datetime.date):
        return DateType
    elif value is None:
        raise TypeInferenceError("Null value; please provide a concrete type", path)

    elif isinstance(value, list):
        if not value:
            raise TypeInferenceError("Empty list; cannot infer element type", path)

        element_types = []
        for i, el in enumerate(value):
            current_path = f"{path}[{i}]" if path else f"[{i}]"
            el_type = infer_dtype_from_pyobj(el, path=current_path)
            element_types.append(el_type)

        common_type = element_types[0]
        for et in element_types[1:]:
            common_type = _find_common_supertype(common_type, et, path=path)

        return ArrayType(common_type)

    elif isinstance(value, dict):
        fields = []
        for k in value.keys():
            current_path = f"{path}.{k}" if path else k
            dt = infer_dtype_from_pyobj(value[k], path=current_path)
            fields.append(StructField(name=k, data_type=dt))
        return StructType(fields)

    raise TypeInferenceError(f"Unsupported type {type(value).__name__}", path)


def _find_common_supertype(type1: DataType, type2: DataType, path="") -> DataType:
    if type1 == type2:
        return type1

    numeric_order = [IntegerType, FloatType, DoubleType]
    if type1 in numeric_order and type2 in numeric_order:
        idx1 = numeric_order.index(type1)
        idx2 = numeric_order.index(type2)
        return numeric_order[max(idx1, idx2)]

    if isinstance(type1, ArrayType) and isinstance(type2, ArrayType):
        element_super = _find_common_supertype(
            type1.element_type, type2.element_type, path=path
        )
        return ArrayType(element_super)

    if isinstance(type1, StructType) and isinstance(type2, StructType):
        all_field_names = {f.name for f in type1.struct_fields} | {
            f.name for f in type2.struct_fields
        }
        merged_fields = []
        for name in sorted(all_field_names):
            f1 = next((f for f in type1.struct_fields if f.name == name), None)
            f2 = next((f for f in type2.struct_fields if f.name == name), None)

            if f1 and f2:
                field_path = f"{path}.{name}" if path else name
                common_field_type = _find_common_supertype(
                    f1.data_type, f2.data_type, path=field_path
                )
            else:
                common_field_type = f1.data_type if f1 else f2.data_type

            merged_fields.append(StructField(name=name, data_type=common_field_type))

        return StructType(merged_fields)

    raise TypeInferenceError(f"Incompatible types: {type1} vs {type2}", path)


def infer_dtype_from_polars(pl_dtype: pl.DataType) -> DataType:
    """Convert a Polars data type to a Fenic DataType.

    Args:
        pl_dtype: A Polars data type

    Returns:
        The corresponding Fenic DataType

    Raises:
        TypeInferenceError: If the Polars dtype cannot be mapped to a Fenic type
    """
    if isinstance(pl_dtype, pl.Boolean):
        return BooleanType
    elif isinstance(pl_dtype, (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Int128, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)):
        return IntegerType
    elif isinstance(pl_dtype, pl.Float32):
        return FloatType
    elif isinstance(pl_dtype, (pl.Float64, pl.Decimal)):
        return DoubleType
    elif isinstance(pl_dtype, pl.Utf8):
        return StringType
    elif isinstance(pl_dtype, pl.Date):
        return DateType
    elif isinstance(pl_dtype, (pl.Datetime, pl.Time)):
        return TimestampType
    elif isinstance(pl_dtype, pl.Categorical):
        # Categorical types are represented as strings in Fenic
        return StringType
    elif isinstance(pl_dtype, (pl.List, pl.Array)):
        element_type = infer_dtype_from_polars(pl_dtype.inner)
        return ArrayType(element_type)
    elif isinstance(pl_dtype, pl.Struct):
        fields = []
        for field in pl_dtype.to_schema():
            field_dtype = pl_dtype.to_schema()[field]
            fields.append(StructField(name=field, data_type=infer_dtype_from_polars(field_dtype)))
        return StructType(fields)
    else:
        raise TypeInferenceError(f"Unsupported Polars dtype: {pl_dtype}")
