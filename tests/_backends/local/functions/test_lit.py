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
from fenic.api.functions.core import empty, null
from fenic.core.error import ValidationError
from fenic.core.types.datatypes import EmbeddingType, JsonType


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


def test_lit_none_should_raise(local_session):
    df = local_session.create_dataframe({"a": [1, 2, 3]})
    with pytest.raises(ValidationError, match="Cannot create a literal with value `None`. Use `null\\(...\\)` instead."):
        df = df.select(col("a"), lit(None).alias("b"))

def test_lit_empty_array_should_raise(local_session):
    df = local_session.create_dataframe({"a": [1, 2, 3]})
    with pytest.raises(ValidationError, match="Cannot create a literal with empty value `\\[\\]` Use `empty\\(ArrayType\\(...\\)\\)` instead."):
        df = df.select(col("a"), lit([]).alias("b"))

def test_lit_empty_struct_should_raise(local_session):
    df = local_session.create_dataframe({"a": [1, 2, 3]})
    with pytest.raises(ValidationError, match="Cannot create a literal with empty value `{}` Use `empty\\(StructType\\(...\\)\\)` instead."):
        df = df.select(col("a"), lit({}).alias("b"))

def test_empty_primitive_type(local_session):
    df = local_session.create_dataframe({"a": [1, 2, 3]})
    df = df.select(col("a"), empty(IntegerType).alias("b"))
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

def test_null_primitive_type(local_session):
    df = local_session.create_dataframe({"a": [1, 2, 3]})
    df = df.select(col("a"), null(IntegerType).alias("b"))
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

def test_empty_array_type(local_session):
    df = local_session.create_dataframe({"a": [1, 2, 3]})
    df = df.select(col("a"), empty(ArrayType(IntegerType)).alias("b"))
    assert df.schema == Schema([
        ColumnField(name="a", data_type=IntegerType),
        ColumnField(name="b", data_type=ArrayType(IntegerType)),
    ])
    result = df.to_polars()
    assert result["a"][0] == 1
    assert result["b"][0].to_list() == []

def test_null_array_type(local_session):
    df = local_session.create_dataframe({"a": [1, 2, 3]})
    df = df.select(col("a"), null(ArrayType(IntegerType)).alias("b"))
    assert df.schema == Schema([
        ColumnField(name="a", data_type=IntegerType),
        ColumnField(name="b", data_type=ArrayType(IntegerType)),
    ])
    result = df.to_polars()
    assert result["a"][0] == 1
    assert result["b"][0] is None

def test_null_struct_type(local_session):
    df = local_session.create_dataframe({"a": [1, 2, 3]})
    nested_struct_type = StructType([
            StructField(name="c", data_type=IntegerType),
            StructField(name="d", data_type=StructType([
                StructField(name="e", data_type=IntegerType),
                StructField(name="f", data_type=IntegerType),
            ])),
    ])
    df = df.select(col("a"), null(nested_struct_type).alias("b"))
    assert df.schema == Schema([
        ColumnField(name="a", data_type=IntegerType),
        ColumnField(name="b", data_type=nested_struct_type),
    ])
    result = df.to_polars()
    assert result["a"][0] == 1
    assert result["b"][0] is None
    # Ensuring that when the null struct is unnested, the fields are all None
    result = result.unnest("b")
    assert result["c"][0] is None
    assert result["d"][0] is None

def test_empty_struct_type(local_session):
    df = local_session.create_dataframe({"a": [1, 2, 3]})
    nested_struct_type = StructType([
            StructField(name="c", data_type=IntegerType),
            StructField(name="d", data_type=StructType([
                StructField(name="e", data_type=IntegerType),
                StructField(name="f", data_type=IntegerType),
            ])),
    ])
    df = df.select(col("a"), empty(nested_struct_type).alias("b"))
    assert df.schema == Schema([
        ColumnField(name="a", data_type=IntegerType),
        ColumnField(name="b", data_type=nested_struct_type),
    ])
    result = df.to_polars()
    assert result["a"][0] == 1
    assert result["b"][0] == {"c": None, "d": None}
    # Ensuring that when the empty struct is unnested, the fields are all None
    result = result.unnest("b")
    assert result["c"][0] is None
    assert result["d"][0] is None

def test_empty_embedding_type(local_session):
    df = local_session.create_dataframe({"a": [1, 2, 3]})
    df = df.select(col("a"), empty(EmbeddingType(dimensions=10, embedding_model="test")).alias("b"))
    assert df.schema == Schema([
        ColumnField(name="a", data_type=IntegerType),
        ColumnField(name="b", data_type=EmbeddingType(dimensions=10, embedding_model="test")),
    ])
    result = df.to_polars()
    assert result["a"][0] == 1
    assert result["b"][0] is None


def test_null_embedding_type(local_session):
    df = local_session.create_dataframe({"a": [1, 2, 3]})
    df = df.select(col("a"), null(EmbeddingType(dimensions=10, embedding_model="test")).alias("b"))
    assert df.schema == Schema([
        ColumnField(name="a", data_type=IntegerType),
        ColumnField(name="b", data_type=EmbeddingType(dimensions=10, embedding_model="test")),
    ])
    result = df.to_polars()
    assert result["a"][0] == 1
    assert result["b"][0] is None

def test_empty_string_backed_type(local_session):
    df = local_session.create_dataframe({"a": [1, 2, 3]})
    df = df.select(col("a"), empty(JsonType).alias("b"))
    assert df.schema == Schema([
        ColumnField(name="a", data_type=IntegerType),
        ColumnField(name="b", data_type=JsonType),
    ])
    result = df.to_polars()
    assert result["a"][0] == 1
    assert result["b"][0] is None

def test_null_string_backed_type(local_session):
    df = local_session.create_dataframe({"a": [1, 2, 3]})
    df = df.select(col("a"), null(JsonType).alias("b"))
    assert df.schema == Schema([
        ColumnField(name="a", data_type=IntegerType),
        ColumnField(name="b", data_type=JsonType),
    ])
    result = df.to_polars()
    assert result["a"][0] == 1
    assert result["b"][0] is None
