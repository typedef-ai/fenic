from fenic import ArrayType, IntegerType, StructField, StructType, col, udf


def test_with_column_udf(sample_df):
    @udf(return_type=IntegerType)
    def add_one(x):
        return x + 1

    result = sample_df.select(add_one("age")).to_polars()
    assert "add_one(age)" in result.columns
    assert result["add_one(age)"][0] == 26


def test_with_column_udf_expr_input(sample_df):
    @udf(return_type=IntegerType)
    def add_one(x):
        return x + 1

    result = sample_df.with_column("age_plus_1", add_one(col("age") + 1)).to_polars()
    assert "age_plus_1" in result.columns
    assert result["age_plus_1"][0] == 27


def test_with_column_udf_multiple_args(local_session):
    @udf(return_type=IntegerType)
    def add_two_numbers(x, y):
        return x + y

    data1 = {"id1": [1, 2], "id2": [3, 4]}
    df1 = local_session.create_dataframe(data1)
    result = df1.with_column("added", add_two_numbers("id1", col("id2"))).to_polars()
    assert "added" in result.columns
    assert result["added"][0] == 4
    assert result["added"][1] == 6


def test_nested_udf(local_session):
    @udf(return_type=IntegerType)
    def add_two_numbers(x, y):
        return x + y

    @udf(return_type=IntegerType)
    def add_one(x):
        return x + 1

    data1 = {"id1": [1, 2], "id2": [3, 4]}
    df1 = local_session.create_dataframe(data1)
    result = df1.with_column(
        "nested_udf", add_one(add_two_numbers("id1", "id2")) + 1
    ).to_polars()
    assert "nested_udf" in result.columns
    assert result["nested_udf"][0] == 6
    assert result["nested_udf"][1] == 8


def test_udf_with_nested_types(local_session):
    @udf(return_type=IntegerType)
    def special_sum(x: dict[str, int], y: list[int]):
        return x["value1"] + x["value2"] + y[0]

    struct_data = {
        "struct_col": [{"value1": 10, "value2": 10}, {"value1": 20, "value2": 20}],
        "array_col": [[1, 2, 3], [4, 5, 6]],
    }
    struct_df = local_session.create_dataframe(struct_data)
    struct_result = struct_df.select(
        special_sum(col("struct_col"), col("array_col")).alias("result")
    ).to_polars()
    assert struct_result["result"].to_list() == [21, 44]

    struct_result_type = StructType(
        [
            StructField("value1", IntegerType),
            StructField("value2", IntegerType),
        ]
    )

    @udf(return_type=struct_result_type)
    def special_sum2(x: dict[str, int], y: list[int]):
        return {
            "value1": x["value1"] + x["value2"] + y[0],
            "value2": x["value1"] + x["value2"] + y[1],
        }

    struct_result = struct_df.select(
        special_sum2(col("struct_col"), col("array_col")).alias("result")
    ).to_polars()
    assert struct_result["result"].to_list() == [
        {"value1": 21, "value2": 22},
        {"value1": 44, "value2": 45},
    ]

    array_result_type = ArrayType(element_type=IntegerType)

    @udf(return_type=array_result_type)
    def special_sum3(x: dict[str, int], y: list[int]):
        return [x["value1"] + x["value2"]] + y

    struct_result = struct_df.select(
        special_sum3(col("struct_col"), col("array_col")).alias("result")
    ).to_polars()
    assert struct_result["result"].to_list() == [[20, 1, 2, 3], [40, 4, 5, 6]]
