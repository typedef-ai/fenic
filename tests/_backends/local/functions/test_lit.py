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
