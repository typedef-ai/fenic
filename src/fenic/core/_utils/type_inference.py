from typing import Any

from fenic.core.types.datatypes import (
    ArrayType,
    BooleanType,
    DataType,
    DoubleType,
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)


class TypeInferenceError(ValueError):
    def __init__(self, message: str, path: str = ""):
        full_message = f"{message} at {path}" if path else message
        super().__init__(full_message)
        self.path = path

def infer_dtype_from_pyobj(value: Any, path="") -> DataType:
    if isinstance(value, bool):
        return BooleanType
    elif isinstance(value, int):
        return IntegerType
    elif isinstance(value, float):
        return FloatType
    elif isinstance(value, str):
        return StringType
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
        for k in sorted(value.keys()):
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
