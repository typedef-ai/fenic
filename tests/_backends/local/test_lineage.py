import polars as pl

from fenic import avg, col, count


def create_basic_test_df(local_session):
    """Creates a test DataFrame with sample data."""
    return local_session.create_dataframe(
        {
            "name": ["Phoebe", "Alice", "Bob", "Charlie", "David", "Eve"],
            "age": [25, 30, 35, 40, 45, 50],
            "has_pet": [True, True, True, True, True, False],
            "city": [
                "New York",
                "San Francisco",
                "San Francisco",
                "San Francisco",
                "New York",
                "San Francisco",
            ],
            "gender": ["Female", "Female", "Male", "Male", "Male", "Female"],
            "id": [1, 2, 3, 4, 5, 6],
            "hobbies": [
                ["reading", "yoga"],
                ["cooking", "traveling", "yoga", "reading"],
                ["cooking", "music", "traveling"],
                ["traveling", "reading", "yoga"],
                ["yoga", "traveling", "reading"],
                ["yoga", "traveling", "reading"],
            ],
        }
    )


def test_lineage_single_child_backward(local_session):
    source = create_basic_test_df(local_session)

    df = (
        source.filter(col("has_pet"))
        .explode("hobbies")
        .group_by("gender", "hobbies", "city")
        .agg(count("gender").alias("count"), avg("age").alias("avg_age"))
        .select(
            "gender",
            col("hobbies").alias("hobby"),
            "city",
            col("count"),
            (col("avg_age") * 7).alias("age_in_dog_years"),
        )
    )
    lineage = df.lineage()
    result = lineage.get_result_df()
    uuid_cols = (
        result.filter(
            (pl.col("gender") == "Male")
            & (pl.col("hobby") == "traveling")
            & (pl.col("city") == "San Francisco")
        )
        .select("_uuid")
        .to_series()
        .to_list()
    )
    backwards_result = lineage.backwards(uuid_cols)
    uuid_cols = backwards_result.select("_uuid").to_series().to_list()
    backwards_result = backwards_result.drop("_uuid").with_columns(
        pl.col("count").cast(pl.Int64)
    )
    assert backwards_result.equals(
        pl.DataFrame(
            {
                "gender": ["Male"],
                "hobbies": ["traveling"],
                "city": ["San Francisco"],
                "count": [2],
                "avg_age": [37.5],
            }
        )
    )
    backwards_result = lineage.backwards(uuid_cols)

    assert backwards_result.shape[0] == 2
    assert backwards_result.filter(pl.col("name") == "Charlie").shape[0] == 1
    uuid_cols = (
        backwards_result.filter(pl.col("name") == "Charlie")
        .select("_uuid")
        .to_series()
        .to_list()
    )

    backwards_result = lineage.backwards(uuid_cols).drop("_uuid")
    assert backwards_result.equals(source.filter(col("name") == "Charlie").to_polars())


def test_lineage_single_child_forward(local_session):
    source = create_basic_test_df(local_session)
    df = (
        source.filter(col("has_pet"))
        .explode("hobbies")
        .group_by("gender", "hobbies", "city")
        .agg(count("gender").alias("count"), avg("age").alias("avg_age"))
        .select(
            "gender",
            col("hobbies").alias("hobby"),
            "city",
            col("count"),
            (col("avg_age") * 7).alias("age_in_dog_years"),
        )
    )
    lineage_query = df.lineage()
    source_name = lineage_query.get_source_names()[0]
    lineage_query.start_from_source(source_name)
    source = lineage_query.get_source_df(source_name)
    uuid_cols = (
        source.filter(pl.col("name") == "Eve").select("_uuid").to_series().to_list()
    )
    forward = lineage_query.forwards(uuid_cols)
    # Eve has no pet, so no forward should be returned.
    assert forward.shape[0] == 0

    lineage_query.start_from_source(source_name)
    uuid_cols = (
        source.filter(pl.col("name") == "Charlie").select("_uuid").to_series().to_list()
    )
    forward = lineage_query.forwards(uuid_cols)
    assert forward.shape[0] == 1
    assert forward.filter(pl.col("name") == "Charlie").shape[0] == 1
    uuid_cols = forward.select("_uuid").to_series().to_list()

    forward = lineage_query.forwards(uuid_cols)
    assert forward.shape[0] == 3
    uuid_cols = forward.select("_uuid").to_series().to_list()
    forward = lineage_query.forwards(uuid_cols)

    uuid_cols = forward.select("_uuid").to_series().to_list()
    forward = lineage_query.forwards(uuid_cols)
    assert forward.shape[0] == 3


def test_lineage_single_child_backward_and_forward(local_session):
    source = create_basic_test_df(local_session)
    df = (
        source.filter(col("has_pet"))
        .explode("hobbies")
        .group_by("gender", "hobbies", "city")
        .agg(count("gender").alias("count"), avg("age").alias("avg_age"))
        .select(
            "gender",
            col("hobbies").alias("hobby"),
            "city",
            col("count"),
            (col("avg_age") * 7).alias("age_in_dog_years"),
        )
    )
    lineage = df.lineage()
    result = lineage.get_result_df()
    traversal = result
    for step in range(4):
        assert (
            not result.is_empty()
        ), f"Step {step}: Backward traversal returned empty result"

        uuid_cols = result.select(pl.col("_uuid")).to_series().to_list()
        traversal = lineage.backwards(uuid_cols)

    for step in range(4):
        assert (
            not result.is_empty()
        ), f"Step {step}: Forward traversal returned empty result"

        uuid_cols = result.select(pl.col("_uuid")).to_series().to_list()
        traversal = lineage.forwards(uuid_cols)

    assert traversal.equals(result)


def test_join_lineage(local_session):
    source1 = local_session.create_dataframe({"id": [1, 2, 3], "value1": ["a", "b", "c"]})
    source2 = local_session.create_dataframe({"id": [1, 4, 5], "value2": ["d", "e", "f"]})
    df = source1.join(source2, on="id", how="inner")
    lineage = df.lineage()
    result = lineage.get_result_df()

    uuid_cols = result.select("_uuid").to_series().to_list()
    backwards = lineage.backwards(uuid_cols, branch_side="left")
    assert backwards.drop("_uuid").equals(source1.filter(col("id") == 1).to_polars())

    uuid_cols = backwards.select("_uuid").to_series().to_list()
    forwards = lineage.forwards(uuid_cols)
    uuid_cols = forwards.select("_uuid").to_series().to_list()
    assert forwards.equals(result)

    backwards = lineage.backwards(uuid_cols, branch_side="right")
    source2_row = source2.filter(col("id") == 1).to_polars()
    assert backwards.drop("_uuid").equals(source2_row)


def test_union_lineage(local_session):
    source1 = local_session.create_dataframe({"id": [1, 2, 3], "value": ["a", "b", "c"]})
    source2 = local_session.create_dataframe({"id": [4, 5, 6], "value": ["d", "e", "f"]})
    df = source1.union(source2)
    lineage = df.lineage()
    result = lineage.get_result_df()

    uuid_cols = result.filter(pl.col("id") == 1).select("_uuid").to_series().to_list()
    backwards = lineage.backwards(uuid_cols, branch_side="left")
    assert backwards.drop("_uuid").equals(source1.filter(col("id") == 1).to_polars())

    uuid_cols = result.filter(pl.col("id") == 1).select("_uuid").to_series().to_list()
    forwards = lineage.forwards(uuid_cols)
    assert forwards.drop("_uuid").equals(source2.filter(col("id") == 1).to_polars())

    uuid_cols = result.filter(pl.col("id") == 1).select("_uuid").to_series().to_list()
    backwards = lineage.backwards(uuid_cols, branch_side="right")
    assert backwards.drop("_uuid").equals(source2.filter(col("id") == 1).to_polars())


def test_inmemory_source_lineage_with_transformation(local_session):
    """Test the lineage mechanism for an in-memory source when a transformation is applied.
    This test creates an in-memory DataFrame, applies a filter to create an operator above the source,
    then retrieves the lineage and traverses backward to verify that the original source row is returned.
    """
    # Create a simple in-memory DataFrame.
    data = {"id": [1, 2, 3], "value": ["a", "b", "c"]}
    pl_df = pl.DataFrame(data)
    df = local_session.create_dataframe(pl_df)

    # Apply a filter transformation to build a lineage chain.
    # This filter creates an intermediate operator whose child is the in-memory source.
    filtered_df = df.filter(col("id") == 2)

    # Obtain the lineage from the filtered DataFrame.
    lineage = filtered_df.lineage()
    result = lineage.get_result_df()

    # Verify that the lineage result includes the _uuid column.
    assert "_uuid" in result.columns, "Lineage result must include a '_uuid' column."

    # Extract the UUID(s) for the row where id == 2.
    uuid_cols = result.filter(pl.col("id") == 2).select("_uuid").to_series().to_list()

    # Traverse backwards from the filtered result to the original in-memory source row.
    backwards = lineage.backwards(uuid_cols)

    # Drop the _uuid column before comparing with the expected row.
    backwards_clean = backwards.drop("_uuid")
    expected = pl.DataFrame({"id": [2], "value": ["b"]})

    assert (
        backwards_clean.shape[0] == expected.shape[0]
    ), f"Expected row count {expected.shape[0]}, got {backwards_clean.shape[0]}"

    for column in expected.columns:
        actual_values = backwards_clean[column].to_list()
        expected_values = expected[column].to_list()
        assert (
            actual_values == expected_values
        ), f"Mismatch in column '{column}': expected {expected_values}, got {actual_values}"
