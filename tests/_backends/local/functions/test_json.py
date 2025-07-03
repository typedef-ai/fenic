import polars as pl
import pytest

from fenic import (
    BooleanType,
    ColumnField,
    JsonType,
    StringType,
    col,
    json,
)
from fenic.core.error import ValidationError


def test_json_type(local_session):
    data = {
        "json_strings": [
            '{"name": "Alice", "age": 30, "active": true, "scores": [95, 87], "meta": {"role": "admin"}}',
            '"just a string"',
            '42',
            'true',
            '[1, 2, 3]',
            'null',
        ]
    }
    df = local_session.create_dataframe(data)
    df = df.select(col("json_strings").cast(JsonType).alias("json_col"))
    df = df.select(json.get_type(col("json_col")).alias("type"))

    assert df.schema.column_fields == [ColumnField(name="type", data_type=StringType)]
    result = df.to_polars()
    expected = pl.DataFrame({
        "type": ["object", "string", "number", "boolean", "array", "null"]
    })
    assert result.equals(expected)


def test_json_contains_primitives(local_session):
    data = {
        "json_strings": [
            '{"text": "hello", "number": 42, "flag": true, "empty": null}',
            '{"values": [42, "hello", false, null], "nested": {"deep": 3.14}}',
            '42',
            '"hello"',
            'true',
            'false',
            'null',
            '3.14',
        ]
    }
    df = local_session.create_dataframe(data)
    df = df.select(col("json_strings").cast(JsonType).alias("json_col"))

    # Test string matching
    df_contains = df.select(json.contains(col("json_col"), '"hello"').alias("result"))
    assert df_contains.schema.column_fields == [ColumnField(name="result", data_type=BooleanType)]
    result_str = df_contains.to_polars()
    expected_str = pl.DataFrame({"result": [True, True, False, True, False, False, False, False]})
    assert result_str.equals(expected_str)

    # Test integer matching
    result_int = df.select(json.contains(col("json_col"), '42').alias("result")).to_polars()
    expected_int = pl.DataFrame({"result": [True, True, True, False, False, False, False, False]})
    assert result_int.equals(expected_int)

    # Test boolean matching
    result_bool = df.select(json.contains(col("json_col"), 'true').alias("result")).to_polars()
    expected_bool = pl.DataFrame({"result": [True, False, False, False, True, False, False, False]})
    assert result_bool.equals(expected_bool)

    # Test null matching
    result_null = df.select(json.contains(col("json_col"), 'null').alias("result")).to_polars()
    expected_null = pl.DataFrame({"result": [True, True, False, False, False, False, True, False]})
    assert result_null.equals(expected_null)

    # Test float matching
    result_float = df.select(json.contains(col("json_col"), '3.14').alias("result")).to_polars()
    expected_float = pl.DataFrame({"result": [False, True, False, False, False, False, False, True]})
    assert result_float.equals(expected_float)


def test_json_contains_objects(local_session):
    data = {
        "json_strings": [
            '{"name": "Alice", "age": 30, "meta": {"role": "admin", "level": 5}}',
            '{"name": "Bob", "meta": {"role": "user"}}',
            '{"profile": {"meta": {"role": "admin"}}, "name": "David"}',
            '{"config": {"database": {"settings": {"role": "admin", "timeout": 30}}}}',
            '{}',
        ]
    }
    df = local_session.create_dataframe(data)
    df = df.select(col("json_strings").cast(JsonType).alias("json_col"))

    # Test partial object matching at different nesting levels
    result_admin = df.select(json.contains(col("json_col"), '{"role": "admin"}').alias("result")).to_polars()
    expected_admin = pl.DataFrame({"result": [True, False, True, True, False]})  # Alice, David, and deep config
    assert result_admin.equals(expected_admin)

    # Test nested object structure matching
    result_meta = df.select(json.contains(col("json_col"), '{"meta": {"role": "admin"}}').alias("result")).to_polars()
    expected_meta = pl.DataFrame({"result": [True, False, True, False, False]})  # Alice and David have meta.role=admin
    assert result_meta.equals(expected_meta)

    # Test empty object matching
    result_empty = df.select(json.contains(col("json_col"), '{}').alias("result")).to_polars()
    expected_empty = pl.DataFrame({"result": [True, True, True, True, True]})  # all objects contain empty object
    assert result_empty.equals(expected_empty)


def test_json_contains_arrays(local_session):
    data = {
        "json_strings": [
            '[1, 2, 3]',
            '{"tags": [1, 2, 3], "permissions": ["read", "write"]}',
            '{"nested": {"arrays": [[1, 2], [3, 4]]}}',
            '[{"type": "admin", "permissions": ["read", "write"]}, {"type": "user"}]',
            '[]',
            '{"data": []}',
        ]
    }
    df = local_session.create_dataframe(data)
    df = df.select(col("json_strings").cast(JsonType).alias("json_col"))

    # Test exact array matching
    result_array = df.select(json.contains(col("json_col"), '[1, 2, 3]').alias("result")).to_polars()
    expected_array = pl.DataFrame({"result": [True, True, False, False, False, False]})  # top level and nested
    assert result_array.equals(expected_array)

    # Test nested array matching
    result_nested = df.select(json.contains(col("json_col"), '[1, 2]').alias("result")).to_polars()
    expected_nested = pl.DataFrame({"result": [False, False, True, False, False, False]})  # only in nested arrays
    assert result_nested.equals(expected_nested)

    # Test array in object context
    result_perms = df.select(json.contains(col("json_col"), '["read", "write"]').alias("result")).to_polars()
    expected_perms = pl.DataFrame({"result": [False, True, False, True, False, False]})  # in permissions
    assert result_perms.equals(expected_perms)

    # Test empty array matching
    result_empty = df.select(json.contains(col("json_col"), '[]').alias("result")).to_polars()
    expected_empty = pl.DataFrame({"result": [False, False, False, False, True, True]})  # only exact empty arrays
    assert result_empty.equals(expected_empty)


def test_json_contains_validation(local_session):
    data = {"json_strings": ['{"test": "value"}']}
    df = local_session.create_dataframe(data)
    df = df.select(col("json_strings").cast(JsonType).alias("json_col"))

    # Test invalid JSON raises ValidationError
    with pytest.raises(ValidationError, match="json.contains\\(\\) requires a valid JSON string"):
        df.select(json.contains(col("json_col"), '{invalid json}'))

    with pytest.raises(ValidationError, match="json.contains\\(\\) requires a valid JSON string"):
        df.select(json.contains(col("json_col"), 'not json at all'))


def test_json_contains_mixed_types(local_session):
    data = {
        "json_strings": [
            '{"number": 42, "string": "42", "boolean": true}',
            '[42, "42", true, "true"]',
            '{"nested": {"values": [1, "1", false, "false"]}}',
        ]
    }
    df = local_session.create_dataframe(data)
    df = df.select(col("json_strings").cast(JsonType).alias("json_col"))

    # Test number vs string distinction
    result_num = df.select(json.contains(col("json_col"), '42').alias("result")).to_polars()
    expected_num = pl.DataFrame({"result": [True, True, False]})  # matches number 42, not string "42"
    assert result_num.equals(expected_num)

    result_str = df.select(json.contains(col("json_col"), '"42"').alias("result")).to_polars()
    expected_str = pl.DataFrame({"result": [True, True, False]})  # matches string "42", not number 42
    assert result_str.equals(expected_str)

    # Test boolean vs string distinction
    result_bool = df.select(json.contains(col("json_col"), 'true').alias("result")).to_polars()
    expected_bool = pl.DataFrame({"result": [True, True, False]})  # matches boolean true, not string "true"
    assert result_bool.equals(expected_bool)
