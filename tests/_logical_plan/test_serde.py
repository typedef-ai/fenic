import os

import polars as pl

from fenic import DataFrame, col, semantic
from fenic.core._interfaces.session_state import BaseSessionState
from fenic.core._logical_plan import LogicalPlan
from fenic.core._logical_plan.serde import LogicalPlanSerde


def _test_plan_serialization(
    plan: LogicalPlan, session: BaseSessionState
) -> LogicalPlan:
    """Helper method to test serialization/deserialization of a plan.

    TODO: Add special checking for subclass fields
    """
    # Serialize and deserialize
    serialized = LogicalPlanSerde.serialize(plan)
    deserialized = LogicalPlanSerde.deserialize(serialized)
    deserialized_with_session_state = (
        LogicalPlanSerde.build_logical_plan_with_session_state(deserialized, session)
    )
    deserialized_df = DataFrame._from_logical_plan(
        deserialized_with_session_state
    )
    deserialized_with_session_state._build_schema()
    plan._build_schema()

    # Test equivalence
    assert isinstance(deserialized_with_session_state, type(plan))
    assert plan._repr() == deserialized_with_session_state._repr()
    assert str(plan._build_schema()) == str(deserialized_with_session_state._build_schema())

    # Test children if any
    assert len(plan.children()) == len(deserialized_with_session_state.children())
    # for orig_child, deser_child in zip(plan.children(), deserialized.children()):
    #    assert orig_child._repr() == deser_child._repr()

    return deserialized_df


def test_basic_plan(local_session):
    # Create a simple DataFrame
    df = local_session.create_dataframe({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    plan = df._logical_plan
    _ = _test_plan_serialization(plan, local_session._session_state)


def test_transform_plans(local_session):
    # Create base DataFrame
    df = local_session.create_dataframe({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    expected = df.to_polars()

    # Test Projection
    projection = df.select("a", "b")
    deserialized_df = _test_plan_serialization(projection._logical_plan, local_session._session_state)
    result = deserialized_df.to_polars()
    expected = projection.to_polars()
    assert result.equals(expected)

    # Test Filter
    filter_plan = df.filter(df["a"] > 1)
    deserialized_df = _test_plan_serialization(filter_plan._logical_plan, local_session._session_state)
    result = deserialized_df.to_polars()
    expected = filter_plan.to_polars()
    assert result.equals(expected)

    # Test Sort
    sort_plan = df.sort("a", ascending=True)
    deserialized_df = _test_plan_serialization(sort_plan._logical_plan, local_session._session_state)
    result = deserialized_df.to_polars()
    expected = sort_plan.to_polars()
    assert result.equals(expected)

    # Test Limit
    limit_plan = df.limit(2)
    deserialized_df = _test_plan_serialization(limit_plan._logical_plan, local_session._session_state)
    result = deserialized_df.to_polars()
    expected = limit_plan.to_polars()
    assert result.equals(expected)

    # Test Union
    df2 = local_session.create_dataframe({"a": [4, 5, 6], "b": ["p", "q", "r"]})
    union_plan = df.union(df2)
    deserialized_df = _test_plan_serialization(union_plan._logical_plan, local_session._session_state)
    result = deserialized_df.to_polars()
    expected = union_plan.to_polars()
    assert result.equals(expected)

    # Test Unnest
    df = local_session.create_dataframe({"a": [{"b": 1, "c": 2}, {"b": 3, "c": 4}]})
    unnest_plan = df.unnest("a")
    deserialized_df = _test_plan_serialization(unnest_plan._logical_plan, local_session._session_state)
    result = deserialized_df.to_polars()
    expected = unnest_plan.to_polars()
    assert result.equals(expected)


def test_join_plans(local_session):
    # Create DataFrames for joining
    left = local_session.create_dataframe({"id": [1, 2, 3], "name": ["a", "b", "c"]})
    right = local_session.create_dataframe({"id": [1, 2, 3], "value": ["foo", "bar", "baz"]})

    # Test regular Join
    join_plan = left.join(right, "id").order_by("id")
    deserialized_df = _test_plan_serialization(join_plan._logical_plan, local_session._session_state)
    result = deserialized_df.to_polars()
    expected = join_plan.to_polars()
    assert result.equals(expected)

    right = right.select(col("id").alias("right_id"), col("value")).order_by("right_id")
    # Test SemanticJoin
    semantic_join = left.semantic.join(
        right, "match {name:left} to {value:right}"
    ).order_by("id")
    deserialized_df = _test_plan_serialization(semantic_join._logical_plan, local_session._session_state)
    result = deserialized_df.to_polars()
    expected = semantic_join.to_polars()
    assert result.schema == expected.schema

    # Test SemanticSimilarityJoin
    similarity_join = (
        left.with_column("name_embeddings", semantic.embed(col("name")))
        .semantic.sim_join(
            right.with_column("value_embeddings", semantic.embed(col("value"))),
            left_on=col("name_embeddings"),
            right_on=col("value_embeddings"),
            k=2,
            return_similarity_scores=True,
        )
        .order_by("id")
    )
    deserialized_df = _test_plan_serialization(similarity_join._logical_plan, local_session._session_state)
    result = deserialized_df.to_polars()
    expected = similarity_join.to_polars()
    assert result.schema == expected.schema


def test_aggregate_plans(local_session):
    # Create DataFrame for aggregation
    df = local_session.create_dataframe(
        {"group": ["a", "a", "b", "b"], "value": [1, 2, 3, 4]}
    )

    # Test regular Aggregate
    aggregate = df.group_by("group").agg({"value": "sum"}).order_by("group")
    deserialized_df = _test_plan_serialization(aggregate._logical_plan, local_session._session_state)
    result = deserialized_df.to_polars()
    expected = aggregate.to_polars()
    assert result.equals(expected)

    # Test SemanticAggregate
    semantic_aggregate = (
        df.with_column("group_embeddings", semantic.embed(col("group")))
        .semantic.group_by(col("group_embeddings"), 2)
        .agg({"value": "sum"})
    )
    deserialized_df = _test_plan_serialization(semantic_aggregate._logical_plan, local_session._session_state)
    result = deserialized_df.to_polars()
    expected = semantic_aggregate.to_polars()
    assert result.schema == expected.schema

def test_file_source_plans(local_session):
    test_data = """name,age,city
John,25,New York
Alice,30,San Francisco
Bob,35,Chicago
Carol,28,Boston
David,33,Seattle"""

    # Create a temporary CSV and Parquet file
    temp_csv_path = "temp_csv.csv"
    with open(temp_csv_path, "w") as f:
        f.write(test_data)

    try:
        temp_parquet_path = "temp_parquet.parquet"
        df_polars = pl.read_csv(temp_csv_path)
        df_polars.write_parquet(temp_parquet_path)

        # Simple plan with parquet file source
        df = local_session.read.parquet(temp_parquet_path)
        plan = df._logical_plan
        deserialized_df = _test_plan_serialization(plan, local_session._session_state)
        result = deserialized_df.to_polars()
        expected = df.to_polars()
        assert result.equals(expected)

        # Complex plan with csv file source
        df = local_session.read.csv(temp_csv_path)
        df2 = local_session.create_dataframe(
            {"a": [1, 2, 3, 4], "b": ["x", "y", "z", "w"], "age": [25, 26, 27, 28]}
        )
        df3 = df.join(other=df2, on="age").order_by("a").where(col("age") > 25)
        plan = df3._logical_plan
        deserialized_df = _test_plan_serialization(plan, local_session._session_state)
        result = deserialized_df.to_polars()
        expected = df3.to_polars()
        assert result.equals(expected)

    finally:
        if os.path.exists(temp_csv_path):
            os.remove(temp_csv_path)
        if os.path.exists(temp_parquet_path):
            os.remove(temp_parquet_path)



def test_table_source_plans(local_session):
    # Simple plan with table source
    df = local_session.create_dataframe({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    df.write.save_as_table("test_table", mode="overwrite")
    plan = local_session.table("test_table")._logical_plan
    deserialized_df = _test_plan_serialization(plan, local_session._session_state)
    result = deserialized_df.to_polars()
    expected = df.to_polars()
    assert result.equals(expected)

    # Complex plan with multiple table sources
    df = local_session.create_dataframe({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    df2 = local_session.create_dataframe({"a": [1, 2, 3], "c": ["a", "b", "c"]})
    df.write.save_as_table("test_table", mode="overwrite")
    df2.write.save_as_table("test_table2", mode="overwrite")
    df3 = local_session.table("test_table")
    df4 = local_session.table("test_table2")
    df5 = df3.join(other=df4, on="a").order_by("b").where(col("a") > 1)
    plan = df5._logical_plan
    deserialized_df = _test_plan_serialization(plan, local_session._session_state)
    result = deserialized_df.to_polars()
    expected = df5.to_polars()
    assert result.equals(expected)


    assert result.schema == expected.schema
    # grouping is not deterministic, so just test the schema matches
