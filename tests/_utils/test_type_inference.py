import re

import pytest

from fenic.core._utils.type_inference import (
    TypeInferenceError,
    infer_dtype_from_pyobj,
)
from fenic.core.types.datatypes import (
    ArrayType,
    BooleanType,
    FloatType,
    IntegerType,
    StringType,
    StructType,
)


def test_infer_primitives():
    assert infer_dtype_from_pyobj(True) == BooleanType
    assert infer_dtype_from_pyobj(42) == IntegerType
    assert infer_dtype_from_pyobj(3.14) == FloatType
    assert infer_dtype_from_pyobj("hello") == StringType


def test_infer_none_raises():
    with pytest.raises(TypeInferenceError, match="Null value; please provide a concrete type"):
        infer_dtype_from_pyobj(None)


def test_infer_empty_list_raises():
    with pytest.raises(TypeInferenceError, match="Empty list; cannot infer element type"):
        infer_dtype_from_pyobj([])


def test_infer_list_of_ints():
    dt = infer_dtype_from_pyobj([1, 2, 3])
    assert isinstance(dt, ArrayType)
    assert dt.element_type == IntegerType


def test_infer_list_of_mixed_numeric():
    dt = infer_dtype_from_pyobj([1, 2.0, 3.5])
    assert isinstance(dt, ArrayType)
    assert dt.element_type == FloatType


def test_infer_list_with_incompatible_types_raises():
    with pytest.raises(TypeInferenceError, match="Incompatible types: IntegerType vs StringType"):
        infer_dtype_from_pyobj([1, "str"])


def test_infer_dict_simple():
    dt = infer_dtype_from_pyobj({"a": 1, "b": True})
    assert isinstance(dt, StructType)
    assert len(dt.struct_fields) == 2
    # assert fields and their types
    field_map = {f.name: f.data_type for f in dt.struct_fields}
    assert field_map["a"] == IntegerType
    assert field_map["b"] == BooleanType


def test_infer_dict_nested():
    dt = infer_dtype_from_pyobj(
        {
            "a": 1,
            "b": {"x": 3.14, "y": False},
        }
    )
    assert isinstance(dt, StructType)
    assert len(dt.struct_fields) == 2
    field_map = {f.name: f.data_type for f in dt.struct_fields}
    assert field_map["a"] == IntegerType

    b_type = field_map["b"]
    assert isinstance(b_type, StructType)
    b_fields = {f.name: f.data_type for f in b_type.struct_fields}
    assert b_fields["x"] == FloatType
    assert b_fields["y"] == BooleanType


def test_infer_list_of_structs_with_different_keys():
    val = [
        {"a": 1, "b": 2.0},
        {"a": 3, "c": True},
    ]
    dt = infer_dtype_from_pyobj(val)
    assert isinstance(dt, ArrayType)
    elem_type = dt.element_type
    assert isinstance(elem_type, StructType)

    field_map = {f.name: f.data_type for f in elem_type.struct_fields}
    assert set(field_map.keys()) == {"a", "b", "c"}
    assert field_map["a"] == IntegerType
    # 'b' from first element is float, no 'b' in second element -> FloatType
    assert field_map["b"] == FloatType
    # 'c' from second element is bool, no 'c' in first element -> BooleanType
    assert field_map["c"] == BooleanType


def test_infer_nested_complex():
    val = {
        "users": [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25, "premium": True},
        ],
        "count": 2,
    }
    dt = infer_dtype_from_pyobj(val)
    assert isinstance(dt, StructType)
    assert len(dt.struct_fields) == 2

    field_map = {f.name: f.data_type for f in dt.struct_fields}
    count_type = field_map["count"]
    users_type = field_map["users"]

    assert count_type == IntegerType

    assert isinstance(users_type, ArrayType)
    user_struct = users_type.element_type
    assert isinstance(user_struct, StructType)

    user_fields = {f.name: f.data_type for f in user_struct.struct_fields}
    assert set(user_fields.keys()) == {"name", "age", "premium"}
    assert user_fields["name"] == StringType
    assert user_fields["age"] == IntegerType
    assert user_fields["premium"] == BooleanType


def test_infer_list_of_lists():
    val = [[1, 2], [3, 4]]
    dt = infer_dtype_from_pyobj(val)
    assert isinstance(dt, ArrayType)
    inner = dt.element_type
    assert isinstance(inner, ArrayType)
    assert inner.element_type == IntegerType


def test_infer_list_of_lists_mixed_numeric():
    val = [[1, 2.0], [3.0, 4]]
    dt = infer_dtype_from_pyobj(val)
    assert isinstance(dt, ArrayType)
    inner = dt.element_type
    assert isinstance(inner, ArrayType)
    assert inner.element_type == FloatType


def test_infer_list_with_none_raises():
    val = [1, None, 3]
    with pytest.raises(TypeInferenceError, match=re.escape("Null value; please provide a concrete type at [1]")):
        infer_dtype_from_pyobj(val)


def test_infer_list_with_none_raises_in_nested_list():
    val = [[1], [3, None]]
    with pytest.raises(TypeInferenceError, match=re.escape("Null value; please provide a concrete type at [1][1]")):
        infer_dtype_from_pyobj(val)
