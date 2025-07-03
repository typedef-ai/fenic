from fenic import (
    ColumnField,
    IntegerType,
    Schema,
    StructField,
    StructType,
    col,
    struct,
)


def test_struct_aliasing(local_session):
    df = local_session.create_dataframe({"a": [1, 2, 3]})
    df = df.select(
        col("a"),
        struct(
            (col("a") + 1).alias("a_plus_1"), (col("a") + 2).alias("a_plus_2")
        ).alias("b"),
    )
    expected_schema = Schema(
        [
            ColumnField("a", IntegerType),
            ColumnField(
                "b",
                StructType(
                    [
                        StructField("a_plus_1", IntegerType),
                        StructField("a_plus_2", IntegerType),
                    ]
                ),
            ),
        ]
    )
    assert df.schema == expected_schema
    result = df.to_polars()
    assert result["b"].to_list() == [
        {"a_plus_1": 2, "a_plus_2": 3},
        {"a_plus_1": 3, "a_plus_2": 4},
        {"a_plus_1": 4, "a_plus_2": 5},
    ]

    df = df.with_column(
        "b", struct(col("b").a_plus_1.alias("x"), col("b").a_plus_2.alias("y"))
    )

    expected_schema = Schema(
        [
            ColumnField("a", IntegerType),
            ColumnField(
                "b",
                StructType(
                    [
                        StructField("x", IntegerType),
                        StructField("y", IntegerType),
                    ]
                ),
            ),
        ]
    )
    assert df.schema == expected_schema
    result = df.to_polars()
    assert result["b"].to_list() == [
        {"x": 2, "y": 3},
        {"x": 3, "y": 4},
        {"x": 4, "y": 5},
    ]
