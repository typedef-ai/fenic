import pytest

from fenic import (
    ArrayType,
    BooleanType,
    ColumnField,
    FloatType,
    IntegerType,
    Schema,
    StringType,
    StructField,
    StructType,
    col,
    lit,
)
from fenic.core.error import ValidationError


def test_lit_primitive(local_session):
    df = local_session.create_dataframe({"a": [1, 2, 3]})
    df = df.select(
        col("a"),
        lit(1).alias("b"),
        lit(True).alias("c"),
        lit(1.0).alias("d"),
        lit("foo").alias("e"),
    )
    expected_schema = Schema(
        [
            ColumnField(name="a", data_type=IntegerType),
            ColumnField(name="b", data_type=IntegerType),
            ColumnField(name="c", data_type=BooleanType),
            ColumnField(name="d", data_type=FloatType),
            ColumnField(name="e", data_type=StringType),
        ]
    )
    assert df.schema == expected_schema
    result = df.to_polars()
    assert result["a"][0] == 1
    assert result["b"][0] == 1
    assert result["c"][0]
    assert result["d"][0] == 1.0
    assert result["e"][0] == "foo"


def test_lit_array(local_session):
    df = local_session.create_dataframe({"a": [1, 2, 3]})
    df = df.select(col("a"), lit([1, 2, 3]).alias("b"))
    expected_schema = Schema(
        [
            ColumnField(name="a", data_type=IntegerType),
            ColumnField(name="b", data_type=ArrayType(IntegerType)),
        ]
    )
    assert df.schema == expected_schema
    result = df.to_polars()
    assert result["a"][0] == 1
    assert result["b"][0].to_list() == [1, 2, 3]


def test_lit_struct(local_session):
    df = local_session.create_dataframe({"a": [1, 2, 3]})
    df = df.select(col("a"), lit({"c": 1, "d": 2}).alias("b"))
    expected_schema = Schema(
        [
            ColumnField(name="a", data_type=IntegerType),
            ColumnField(
                name="b",
                data_type=StructType(
                    [
                        StructField(name="c", data_type=IntegerType),
                        StructField(name="d", data_type=IntegerType),
                    ]
                ),
            ),
        ]
    )
    assert df.schema == expected_schema
    result = df.to_polars()
    assert result["a"][0] == 1
    assert result["b"][0] == {"c": 1, "d": 2}


def test_lit_list_struct(local_session):
    df = local_session.create_dataframe({"a": [1, 2, 3]})
    df = df.select(
        col("a"), lit([{"c": 1, "d": 2}, {"c": 3.0, "d": 4, "e": True}]).alias("b")
    )
    expected_schema = Schema(
        [
            ColumnField(name="a", data_type=IntegerType),
            ColumnField(
                name="b",
                data_type=ArrayType(
                    StructType(
                        [
                            StructField(name="c", data_type=FloatType),
                            StructField(name="d", data_type=IntegerType),
                            StructField(name="e", data_type=BooleanType),
                        ]
                    )
                ),
            ),
        ]
    )
    assert df.schema == expected_schema
    result = df.to_polars()
    assert result["a"][0] == 1
    assert result["b"][0].to_list() == [
        {"c": 1.0, "d": 2, "e": None},
        {"c": 3.0, "d": 4, "e": True},
    ]


def test_lit_struct_with_list(local_session):
    df = local_session.create_dataframe({"a": [1, 2, 3]})
    df = df.select(col("a"), lit({"c": [1, 2, 3], "d": 2}).alias("b"))
    expected_schema = Schema(
        [
            ColumnField(name="a", data_type=IntegerType),
            ColumnField(
                name="b",
                data_type=StructType(
                    [
                        StructField(name="c", data_type=ArrayType(IntegerType)),
                        StructField(name="d", data_type=IntegerType),
                    ]
                ),
            ),
        ]
    )
    assert df.schema == expected_schema
    result = df.to_polars()
    assert result["a"][0] == 1
    assert result["b"][0] == {"c": [1, 2, 3], "d": 2}

def test_lit_none_with_dtype(local_session):
    df = local_session.create_dataframe({"a": [1, 2, 3]})
    df = df.select(col("a"), lit(None, dtype=IntegerType).alias("b"))
    expected_schema = Schema(
        [
            ColumnField(name="a", data_type=IntegerType),
            ColumnField(name="b", data_type=IntegerType),
        ]
    )
    assert df.schema == expected_schema
    result = df.to_polars()
    assert result["a"][0] == 1
    assert result["b"][0] is None

def test_lit_none_without_dtype(local_session):
    df = local_session.create_dataframe({"a": [1, 2, 3]})
    with pytest.raises(ValidationError, match="`lit` failed to infer type for value `None`"):
        df = df.select(col("a"), lit(None).alias("b"))

def test_lit_empty_list_with_dtype(local_session):
    df = local_session.create_dataframe({"a": [1, 2, 3]})
    df = df.select(col("a"), lit([], dtype=ArrayType(IntegerType)).alias("b"))
    assert df.schema == Schema([
        ColumnField(name="a", data_type=IntegerType),
        ColumnField(name="b", data_type=ArrayType(IntegerType)),
    ])
    result = df.to_polars()
    assert result["a"][0] == 1
    assert result["b"][0].to_list() == []

def test_lit_empty_struct_with_dtype(local_session):
    df = local_session.create_dataframe({"a": [1, 2, 3]})
    df = df.select(col("a"), lit({}, dtype=StructType([StructField(name="c", data_type=IntegerType)])).alias("b"))
    assert df.schema == Schema([
        ColumnField(name="a", data_type=IntegerType),
        ColumnField(name="b", data_type=StructType([StructField(name="c", data_type=IntegerType)])),
    ])
    result = df.to_polars()
    assert result["a"][0] == 1
    assert result["b"][0] == {"c": None}

def test_lit_struct_not_matching_dtype(local_session):
    df = local_session.create_dataframe({"a": [1, 2, 3]})
    with pytest.raises(ValidationError, match=("User provided dtype StructType\\(struct_fields=\\[StructField\\(name=c, data_type=FloatType\\)\\]\\) "
                                               "does not match inferred type StructType\\(struct_fields=\\[StructField\\(name=c, data_type=IntegerType\\)\\]\\) "
                                               "for value `\\{'c': 1\\}` If value is not None or an empty list, you need not specify a dtype.")):
        df = df.select(col("a"), lit({"c": 1}, dtype=StructType([StructField(name="c", data_type=FloatType)])).alias("b"))

def test_lit_empty_struct_without_dtype(local_session):
    df = local_session.create_dataframe({"a": [1, 2, 3]})
    with pytest.raises(ValidationError, match="`lit` failed to infer type for value `{}`. If you are trying to create a "
                                              "literal with an empty struct, you must specify the dtype explicitly. "
                                              "For example, `lit\\(\\{\\}, dtype=StructType\\(\\[StructField\\(name='c', data_type=IntegerType\\)\\]\\)\\)`."):
        df = df.select(col("a"), lit({}).alias("b"))


def test_lit_empty_list_without_dtype(local_session):
    df = local_session.create_dataframe({"a": [1, 2, 3]})
    with pytest.raises(ValidationError, match="`lit` failed to infer type for value"):
        df = df.select(col("a"), lit([]).alias("b"))

def test_lit_dtype_inferred_type_mismatch(local_session):
    df = local_session.create_dataframe({"a": [1, 2, 3]})
    with pytest.raises(ValidationError, match="User provided dtype IntegerType does not match inferred type FloatType for value `1.0`"):
        df = df.select(col("a"), lit(1.0, dtype=IntegerType).alias("b"))

def test_lit_dtype_inferred_type_mismatch_with_list(local_session):
    df = local_session.create_dataframe({"a": [1, 2, 3]})
    with pytest.raises(ValidationError, match="User provided dtype ArrayType\\(element_type=IntegerType\\) does not match "
                                              "inferred type ArrayType\\(element_type=FloatType\\) for value `\\[1.0\\]`"):
        df = df.select(col("a"), lit([1.0], dtype=ArrayType(IntegerType)).alias("b"))
