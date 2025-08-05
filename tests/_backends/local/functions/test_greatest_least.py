import re

import pytest

from fenic import ColumnField, col, greatest, least
from fenic.core.error import TypeMismatchError, ValidationError
from fenic.core.types.datatypes import BooleanType, DoubleType, IntegerType, StringType


@pytest.fixture
def greatest_least_source(local_session):
    """Create a test dataframe source with various data types."""
    return local_session.create_dataframe(
        {
            "int": [1, 3, None],
            "int2": [2, 4, None],
            "int3": [3, None, None],
            "list": [["any"], ["string"], ["list"]],
            "list2": [["other"], ["string"], ["list"]],
            "bool": [True, False, None],
            "bool2": [False, True, None],
            "float": [1.0, 2.0, None],
            "float2": [2.0, 1.0, None],
            "float3": [1.0, 1.0, None],
            "float4": [2.0, 2.0, None],
            "str": ["a", "b", None],
            "str2": ["c", "d", None],
            "struct": [{"a": 1, "b": 2}, {"a": 3, "b": 4}, None],
            "struct2": [{"a": 5, "b": 6}, {"a": 7, "b": 8}, None]
        }
    )

def test_greatest(greatest_least_source):
    df = greatest_least_source.select(greatest(col("int"), col("int2"), col("int3")).alias("greatest"))
    assert df.schema.column_fields == [
        ColumnField(name="greatest", data_type=IntegerType)
    ]
    result = df.to_polars()["greatest"]
    assert result[0] == 3
    assert result[1] == 4
    assert result[2] is None

    df = greatest_least_source.select(greatest(col("bool"), col("bool2")).alias("greatest"))
    assert df.schema.column_fields == [
        ColumnField(name="greatest", data_type=BooleanType)
    ]
    result = df.to_polars()["greatest"]
    assert result[0]
    assert result[1]
    assert result[2] is None

    df = greatest_least_source.select(greatest(col("float"), col("float2"), col("float3"), col("float4")).alias("greatest"))
    assert df.schema.column_fields == [
        ColumnField(name="greatest", data_type=DoubleType)
    ]
    result = df.to_polars()["greatest"]
    assert result[0] == 2.0
    assert result[1] == 2.0
    assert result[2] is None

    df = greatest_least_source.select(greatest(col("str"), col("str2")).alias("greatest"))
    assert df.schema.column_fields == [
        ColumnField(name="greatest", data_type=StringType)
    ]
    result = df.to_polars()["greatest"]
    assert result[0] == "c"
    assert result[1] == "d"
    assert result[2] is None

    with pytest.raises(TypeMismatchError):
        df = greatest_least_source.select(greatest(col("list"), col("list2")).alias("greatest"))

    with pytest.raises(TypeMismatchError):
        df = greatest_least_source.select(greatest(col("struct"), col("struct2")).alias("greatest"))

    with pytest.raises(ValidationError, match=re.escape("greatest() requires at least 2 columns, got 1")):
        df = greatest_least_source.select(greatest(col("int")).alias("greatest"))

    with pytest.raises(TypeMismatchError, match="greatest expects all arguments to have the same type"):
        df = greatest_least_source.select(greatest(col("int"), col("str")).alias("greatest"))

def test_least(greatest_least_source):
    df = greatest_least_source.select(least(col("int"), col("int2"), col("int3")).alias("least"))
    assert df.schema.column_fields == [
        ColumnField(name="least", data_type=IntegerType)
    ]
    result = df.to_polars()["least"]
    assert result[0] == 1
    assert result[1] == 3
    assert result[2] is None

    df = greatest_least_source.select(least(col("bool"), col("bool2")).alias("least"))
    assert df.schema.column_fields == [
        ColumnField(name="least", data_type=BooleanType)
    ]
    result = df.to_polars()["least"]
    assert not result[0]
    assert not result[1]
    assert result[2] is None

    df = greatest_least_source.select(least(col("float"), col("float2"), col("float3"), col("float4")).alias("least"))
    assert df.schema.column_fields == [
        ColumnField(name="least", data_type=DoubleType)
    ]
    result = df.to_polars()["least"]
    assert result[0] == 1.0
    assert result[1] == 1.0
    assert result[2] is None

    df = greatest_least_source.select(least(col("str"), col("str2")).alias("least"))
    assert df.schema.column_fields == [
        ColumnField(name="least", data_type=StringType)
    ]
    result = df.to_polars()["least"]
    assert result[0] == "a"
    assert result[1] == "b"
    assert result[2] is None

    with pytest.raises(TypeMismatchError):
        df = greatest_least_source.select(least(col("struct"), col("struct2")).alias("least"))

    with pytest.raises(TypeMismatchError):
        df = greatest_least_source.select(least(col("list"), col("list2")).alias("least"))

    with pytest.raises(ValidationError, match=re.escape("least() requires at least 2 columns, got 1")):
        df = greatest_least_source.select(least(col("int")).alias("least"))

    with pytest.raises(TypeMismatchError, match="least expects all arguments to have the same type"):
        df = greatest_least_source.select(least(col("int"), col("str")).alias("least"))
