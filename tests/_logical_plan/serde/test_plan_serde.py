import os
from enum import Enum
from typing import List, Optional

import polars as pl
import pytest
from pydantic import BaseModel, Field

from fenic import (
    ColumnField,
    DataFrame,
    IntegerType,
    Schema,
    col,
    count,
    lit,
    semantic,
    sum,
    text,
)
from fenic.core._interfaces.session_state import BaseSessionState
from fenic.core._logical_plan import LogicalPlan

# Import plan classes for the examples dictionary
from fenic.core._logical_plan.expressions.base import LogicalExpr
from fenic.core._logical_plan.plans import (
    SQL,
    Aggregate,
    DocSource,
    DropDuplicates,
    Explode,
    FileSink,
    FileSource,
    Filter,
    InMemorySource,
    Join,
    Limit,
    Projection,
    SemanticCluster,
    SemanticJoin,
    SemanticSimilarityJoin,
    Sort,
    TableSink,
    TableSource,
    Union,
    Unnest,
)
from fenic.core._serde.cloudpickle_serde import CloudPickleSerde
from fenic.core._serde.proto.errors import SerializationError
from fenic.core._serde.proto.plan_serde import serialize_logical_plan
from fenic.core._serde.proto.proto_serde import ProtoSerde
from fenic.core._serde.proto.serde_context import SerdeContext
from fenic.core._serde.serde_protocol import SupportsLogicalPlanSerde
from fenic.core.types import ClassDefinition
from fenic.core.types.datatypes import MarkdownType
from fenic.core.types.semantic_examples import MapExample, MapExampleCollection


def _create_plan_examples(session, temp_dir_with_test_files):
    """Create all plan examples upfront."""
    # Create base dataframes for testing
    df1 = session.create_dataframe({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    df2 = session.create_dataframe({"d": [1, 2, 3], "c": ["foo", "bar", "baz"]})
    # Create additional dataframes for specific tests
    df_with_array = session.create_dataframe({"a": [1, 2], "arr": [[1, 2], [3, 4]]})
    df_with_dupes = session.create_dataframe({"c1": [1, 1, 2], "c2": [1, 2, 2]})
    df_with_struct = session.create_dataframe({"a": [{"b": 1, "c": 2}, {"b": 3, "c": 4}]})

    # Create a test table for TableSource testing
    test_table_df = session.create_dataframe({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    test_table_df.write.save_as_table("test_table", mode="overwrite")
        # Get the absolute path of the current file
    current_file_path = os.path.abspath(__file__)

    # Get the directory containing the current file
    current_file_directory = os.path.dirname(current_file_path)

    return {
        # Basic source plans
        DocSource: [
            ("doc_source", DocSource.from_session_state(
                paths=[temp_dir_with_test_files],
                valid_file_extension=".md",
                recursive=True,
                session_state=session._session_state,
            )),
        ],
        InMemorySource: [
            ("basic_dataframe", df1._logical_plan),
        ],
        FileSource: [
            ("csv_source", FileSource.from_session_state(
                paths=[os.path.join(current_file_directory, "test_data", "test.csv")],
                file_format="csv",
                session_state=session._session_state,
            )),
        ],
        TableSource: [
            ("table_source", TableSource.from_session_state(
                table_name="test_table",
                session_state=session._session_state,
            )),
        ],
        # Transform plans
        Projection: [
            ("simple_select", df1.select("a")._logical_plan),
            ("complex_select", df1.select("a", col("b").alias("b_alias"))._logical_plan),
        ],
        Filter: [
            ("simple_filter", df1.filter(col("a") > 1)._logical_plan),
            ("complex_filter", df1.filter((col("a") > 1) & (col("b") == "y"))._logical_plan),
        ],
        Join: [
            ("inner_join", df1.join(df2, left_on="a", right_on="d")._logical_plan),
            ("left_join", df1.join(df2, left_on="a", right_on="d", how="left")._logical_plan),
        ],
        SemanticJoin: [
            ("semantic_join", df1.semantic.join(df2, "match {{left_on}} to {{right_on}}",
                                               left_on=col("b"), right_on=col("c"))._logical_plan),
        ],
        SemanticSimilarityJoin: [
            ("similarity_join", (df1.with_column("emb1", semantic.embed(col("b")))
                               .semantic.sim_join(
                                   df2.with_column("emb2", semantic.embed(col("c"))),
                                   left_on=col("emb1"), right_on=col("emb2"), k=2
                               ).cache()._logical_plan)),
        ],
        Aggregate: [
            ("simple_aggregate", df1.group_by("b").agg({"a": "sum"})._logical_plan),
            ("complex_aggregate", df1.group_by("b").agg(sum("a").alias("sum_a"), count("a").alias("count_a"))._logical_plan),
        ],
        SemanticCluster: [
            ("semantic_cluster", (df1.with_column("emb", semantic.embed(col("b")))
                                .semantic.with_cluster_labels(col("emb"), 2)._logical_plan)),
        ],
        Union: [
            ("simple_union", df1.union(df1.select("a", "b"))._logical_plan),
        ],
        Limit: [
            ("simple_limit", df1.limit(2)._logical_plan),
        ],
        Explode: [
            ("array_explode", df_with_array.explode("arr")._logical_plan),
        ],
        DropDuplicates: [
            ("drop_duplicates", df_with_dupes.drop_duplicates(["c1", "c2"])._logical_plan),
        ],
        Sort: [
            ("simple_sort", df1.sort("a")._logical_plan),
            ("complex_sort", df1.sort(["a", "b"])._logical_plan),
        ],
        Unnest: [
            ("struct_unnest", df_with_struct.unnest("a")._logical_plan),
        ],
        SQL: [
            ("sql_query", session.sql("SELECT a, b FROM {df1}", df1=df1)._logical_plan),
        ],

        # Sink plans
        FileSink: [
            ("csv_sink", FileSink.from_session_state(
                child=df1._logical_plan,
                sink_type="csv",
                path="/tmp/test.csv",
                mode="overwrite",
                session_state=session._session_state,
            )),
            ("parquet_sink", FileSink.from_session_state(
                child=df1._logical_plan,
                sink_type="parquet",
                path="/tmp/test.parquet",
                mode="overwrite",
                session_state=session._session_state,
            )),
        ],
        TableSink: [
            ("table_sink", TableSink.from_session_state(
                child=df1._logical_plan,
                table_name="test_table",
                mode="overwrite",
                session_state=session._session_state,
            )),
        ],
    }

def _test_df_serialization(df: DataFrame, session: BaseSessionState,
                           serde_implementation: SupportsLogicalPlanSerde) -> DataFrame:
    """Helper method to test serialization/deserialization of a DataFrame."""
    plan = df._logical_plan
    deserialized_df = _test_plan_serialization(plan, session, serde_implementation)
    return deserialized_df


def _test_plan_serialization(
    plan: LogicalPlan, session_state: BaseSessionState, serde_implementation: SupportsLogicalPlanSerde
) -> LogicalPlan:
    """Helper method to test serialization/deserialization of a plan.

    TODO: Add special checking for subclass fields
    """
    # Serialize and deserialize
    serialized = serde_implementation.serialize(plan)
    deserialized = serde_implementation.deserialize(serialized)
    deserialized_df = DataFrame._from_logical_plan(
        deserialized,
        session_state
    )
    deserialized._build_schema(session_state)
    plan._build_schema(session_state)

    # Test equivalence
    assert isinstance(deserialized, type(plan))
    assert plan == deserialized
    assert plan._build_schema(session_state) == deserialized._build_schema(session_state)

    return deserialized_df


class CategoryEnum(Enum):
    A = "a"
    B = "b"
    C = "c"


class BasicReviewModel(BaseModel):
    positive_feature: str = Field(
        ..., description="Positive feature described in the review"
    )


serde_implementations = [
    ProtoSerde,
    CloudPickleSerde,
]


@pytest.mark.parametrize("serde_implementation", serde_implementations)
def test_basic_plan(local_session, serde_implementation: SupportsLogicalPlanSerde):
    # Create a simple DataFrame
    df = local_session.create_dataframe({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    plan = df._logical_plan
    _ = _test_plan_serialization(plan, local_session._session_state, serde_implementation)


@pytest.mark.parametrize("serde_implementation", serde_implementations)
def test_transform_plans(local_session, serde_implementation: SupportsLogicalPlanSerde):
    # Create base DataFrame
    df = local_session.create_dataframe({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    expected = df.to_polars()

    # Test Projection
    projection = df.select("a", "b")
    deserialized_df = _test_plan_serialization(projection._logical_plan, local_session._session_state,
                                               serde_implementation)
    result = deserialized_df.to_polars()
    expected = projection.to_polars()
    assert result.equals(expected)

    # Test Filter
    filter_plan = df.filter(df["a"] > 1)
    deserialized_df = _test_plan_serialization(filter_plan._logical_plan, local_session._session_state,
                                               serde_implementation)
    result = deserialized_df.to_polars()
    expected = filter_plan.to_polars()
    assert result.equals(expected)

    # Test Sort
    sort_plan = df.sort("a", ascending=True)
    deserialized_df = _test_plan_serialization(sort_plan._logical_plan, local_session._session_state,
                                               serde_implementation)
    result = deserialized_df.to_polars()
    expected = sort_plan.to_polars()
    assert result.equals(expected)

    # Test Limit
    limit_plan = df.limit(2)
    deserialized_df = _test_plan_serialization(limit_plan._logical_plan, local_session._session_state,
                                               serde_implementation)
    result = deserialized_df.to_polars()
    expected = limit_plan.to_polars()
    assert result.equals(expected)

    # Test Union
    df2 = local_session.create_dataframe({"a": [4, 5, 6], "b": ["p", "q", "r"]})
    union_plan = df.union(df2)
    deserialized_df = _test_plan_serialization(union_plan._logical_plan, local_session._session_state,
                                               serde_implementation)
    result = deserialized_df.to_polars()
    expected = union_plan.to_polars()
    assert result.equals(expected)

    # Test Unnest
    df = local_session.create_dataframe({"a": [{"b": 1, "c": 2}, {"b": 3, "c": 4}]})
    unnest_plan = df.unnest("a")
    deserialized_df = _test_plan_serialization(unnest_plan._logical_plan, local_session._session_state,
                                               serde_implementation)
    result = deserialized_df.to_polars()
    expected = unnest_plan.to_polars()
    assert result.equals(expected)


@pytest.mark.parametrize("serde_implementation", serde_implementations)
def test_join_plans(local_session, serde_implementation: SupportsLogicalPlanSerde):
    # Create DataFrames for joining
    left = local_session.create_dataframe({"id": [1, 2, 3], "name": ["a", "b", "c"]})
    right = local_session.create_dataframe({"id": [1, 2, 3], "value": ["foo", "bar", "baz"]})

    # Test regular Join
    join_plan = left.join(right, "id").order_by("id")
    deserialized_df = _test_plan_serialization(join_plan._logical_plan, local_session._session_state,
                                               serde_implementation)
    result = deserialized_df.to_polars()
    expected = join_plan.to_polars()
    assert result.equals(expected)

    right = right.select(col("id").alias("right_id"), col("value")).order_by("right_id")
    # Test SemanticJoin
    semantic_join = left.semantic.join(
        right, "match {{left_on}} to {{right_on}}", left_on=col("name"), right_on=col("value")
    ).order_by("id")
    deserialized_df = _test_plan_serialization(semantic_join._logical_plan, local_session._session_state,
                                               serde_implementation)
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
            similarity_score_column="similarity_score",
        )
        .order_by("id")
    )
    deserialized_df = _test_plan_serialization(similarity_join._logical_plan, local_session._session_state,
                                               serde_implementation)
    result = deserialized_df.to_polars()
    expected = similarity_join.to_polars()
    assert result.schema == expected.schema


@pytest.mark.parametrize("serde_implementation", serde_implementations)
def test_aggregate_plans(local_session, serde_implementation: SupportsLogicalPlanSerde):
    # Create DataFrame for aggregation
    df = local_session.create_dataframe(
        {"group": ["a", "a", "b", "b"], "value": [1, 2, 3, 4]}
    )

    # Test regular Aggregate
    aggregate = df.group_by("group").agg({"value": "sum"}).order_by("group")
    deserialized_df = _test_plan_serialization(aggregate._logical_plan, local_session._session_state,
                                               serde_implementation)
    result = deserialized_df.to_polars()
    expected = aggregate.to_polars()
    assert result.equals(expected)

    # Test SemanticAggregate
    semantic_aggregate = (
        df.with_column("group_embeddings", semantic.embed(col("group")))
        .semantic.with_cluster_labels(col("group_embeddings"), 2)
        .group_by(col("cluster_label"))
        .agg({"value": "sum"})
    )
    deserialized_df = _test_plan_serialization(semantic_aggregate._logical_plan, local_session._session_state,
                                               serde_implementation)
    result = deserialized_df.to_polars()
    expected = semantic_aggregate.to_polars()
    assert result.schema == expected.schema


@pytest.mark.parametrize("serde_implementation", serde_implementations)
def test_file_source_plans(local_session, serde_implementation: SupportsLogicalPlanSerde, temp_dir_with_test_files):
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
        deserialized_df = _test_plan_serialization(plan, local_session._session_state, serde_implementation)
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
        deserialized_df = _test_plan_serialization(plan, local_session._session_state, serde_implementation)
        result = deserialized_df.to_polars()
        expected = df3.to_polars()
        assert result.equals(expected)
    finally:
        if os.path.exists(temp_csv_path):
            os.remove(temp_csv_path)
        if os.path.exists(temp_parquet_path):
            os.remove(temp_parquet_path)


@pytest.mark.parametrize("serde_implementation", serde_implementations)
def test_table_source_plans(local_session, serde_implementation: SupportsLogicalPlanSerde):
    # Simple plan with table source
    df = local_session.create_dataframe({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    df.write.save_as_table("test_table", mode="overwrite")
    plan = local_session.table("test_table")._logical_plan
    deserialized_df = _test_plan_serialization(plan, local_session._session_state, serde_implementation)
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
    deserialized_df = _test_plan_serialization(plan, local_session._session_state, serde_implementation)
    result = deserialized_df.to_polars()
    expected = df5.to_polars()
    assert result.equals(expected)

    assert result.schema == expected.schema
    # grouping is not deterministic, so just test the schema matches

@pytest.mark.parametrize("serde_implementation", serde_implementations)
def test_doc_source_plans(local_session, serde_implementation: SupportsLogicalPlanSerde, temp_dir_with_test_files):
    df_docs = local_session.read.docs(
        [temp_dir_with_test_files],
        data_type=MarkdownType,
        recursive=True
    )
    plan = df_docs._logical_plan
    _test_plan_serialization(plan, local_session._session_state, serde_implementation)

@pytest.mark.parametrize("serde_implementation", serde_implementations)
def test_semantic_cluster(local_session, serde_implementation: SupportsLogicalPlanSerde):
    # semantic cluster
    source = local_session.create_dataframe(
        {
            "blurb": [
                "Rust is a memory-safe systems programming language with zero-cost abstractions.",
                None,
            ],
        }
    )
    df = (
        source.with_column("embeddings", semantic.embed(col("blurb")))
        .semantic.with_cluster_labels(col("embeddings"), 2, centroid_column="cluster_centroid")
    )
    deserialized_df = _test_df_serialization(df, local_session._session_state, serde_implementation)
    assert deserialized_df


@pytest.mark.parametrize("serde_implementation", serde_implementations)
def test_semantic_classify(local_session, serde_implementation: SupportsLogicalPlanSerde):
    # semantic classify
    # test with a list of strings as categories
    df = local_session.create_dataframe({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    df = df.select(semantic.classify(col("b"), ["a", "b", "c"]))
    deserialized_df = _test_df_serialization(df, local_session._session_state, serde_implementation)
    assert deserialized_df

    # test with a list of class definitions as categories
    df = local_session.create_dataframe({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    df = df.select(semantic.classify(col("b"), [ClassDefinition(label="a", description="a"),
                                                ClassDefinition(label="b", description="b"),
                                                ClassDefinition(label="c", description="c")]))
    deserialized_df = _test_df_serialization(df, local_session._session_state, serde_implementation)
    assert deserialized_df


@pytest.mark.parametrize("serde_implementation", serde_implementations)
def test_semantic_sim_join(local_session, serde_implementation: SupportsLogicalPlanSerde):
    # semantic sim join
    left = local_session.create_dataframe(
        {
            "course_id": [1, 2, ],
            "course_name": [
                "History of The Atlantic World",
                "Riemann Geometry",
            ],
            "other_col_left": ["a", "b"],
        }
    )
    right = local_session.create_dataframe(
        {
            "skill_id": [1, 2],
            "skill": ["Math", "Computer Science"],
            "other_col_right": ["g", "h"],
        }
    )
    df = (
        left.with_column("course_embeddings", semantic.embed(col("course_name")))
        .semantic.sim_join(
            right.with_column("skill_embeddings", semantic.embed(col("skill"))),
            left_on="course_embeddings",
            right_on="skill_embeddings",
            k=1,
            similarity_metric="cosine",
        )
    )
    deserialized_df = _test_df_serialization(df, local_session._session_state, serde_implementation)
    assert deserialized_df


@pytest.mark.parametrize("serde_implementation", serde_implementations)
def test_semantic_analyze_sentiment(local_session, serde_implementation: SupportsLogicalPlanSerde):
    # semantic analyze sentiment
    comments_data = {
        "user_comments": [
            "best product ever",
        ]
    }
    comments_df = local_session.create_dataframe(comments_data)
    categorized_comments_df = comments_df.select(
        col("user_comments"),
        semantic.analyze_sentiment(text.concat(col("user_comments"), lit(" "))).alias(
            "sentiment"
        ),
    )
    deserialized_df = _test_df_serialization(categorized_comments_df, local_session._session_state,
                                             serde_implementation)
    assert deserialized_df


@pytest.mark.parametrize("serde_implementation", serde_implementations)
def test_semantic_extract(local_session, extract_data_df: DataFrame, serde_implementation: SupportsLogicalPlanSerde):
    # test extract with the base model
    df = extract_data_df.select(
        semantic.extract(col("review"), BasicReviewModel).alias("review_out")
    )
    deserialized_df = _test_df_serialization(df, local_session._session_state, serde_implementation)
    assert deserialized_df


@pytest.mark.parametrize("serde_implementation", serde_implementations)
def test_semantic_map(local_session, serde_implementation: SupportsLogicalPlanSerde):
    # semantic map
    source = local_session.create_dataframe({"name": ["Alice"], "city": ["New York"]})
    state_prompt = "What state does {{name}} live in given that they live in {{city}}?"
    df_select = source.select(
        semantic.map(state_prompt, name=col("name"), city=col("city")).alias("state"),
        col("name"),
        semantic.map("What is the typical weather in {{city}} in summer?", city=col("city")).alias("weather"),
    )
    deserialized_df = _test_df_serialization(df_select, local_session._session_state, serde_implementation)
    assert deserialized_df

@pytest.mark.parametrize("serde_implementation", serde_implementations)
def test_semantic_map_with_examples(local_session, serde_implementation: SupportsLogicalPlanSerde):
# semantic map with examples
    source = local_session.create_dataframe({"name": ["Alice"], "city": ["New York"]})
    state_prompt = "What state does {{name}} live in given that they live in {{city}}?"
    df_select = source.select(
        semantic.map(
            state_prompt,
            examples=MapExampleCollection(
                [
                    MapExample(input={"name": "Alice", "city": "New York City"}, output="New York"),
                    MapExample(input={"name": "Bob", "city": "Chicago"}, output="Illinois"),
                ]
            ),
            name=col("name"), city=col("city")
        ).alias("state"),
        col("name"),
        semantic.map(
            "What is the typical weather in {{city}} in summer?",
            examples=MapExampleCollection([MapExample(input={"city": "New York"}, output="hot"),
                                            MapExample(input={"city": "Chicago"}, output="cool")]),
            city=col("city")
        ).alias("weather"),
    )
    deserialized_df = _test_df_serialization(df_select, local_session._session_state, serde_implementation)
    assert deserialized_df


@pytest.mark.parametrize("serde_implementation", serde_implementations)
def test_semantic_predicate(local_session, serde_implementation: SupportsLogicalPlanSerde):
    # semantic predicate
    instruction = "Review: '{{blurb}}'. The review speaks positively about apache spark."
    source = local_session.create_dataframe(
        {
            "blurb": [
                "Apache Spark is the worst piece of software I've ever used. It's so slow and inefficient and I hate the JVM.",
            ]
        }
    )
    df = source.filter(
        semantic.predicate(instruction, blurb=col("blurb"))
    )
    deserialized_df = _test_df_serialization(df, local_session._session_state, serde_implementation)
    assert deserialized_df


@pytest.mark.parametrize("serde_implementation", serde_implementations)
def test_transform_explode_and_drop_duplicates(local_session, serde_implementation: SupportsLogicalPlanSerde):
    df = local_session.create_dataframe({"a": [1, 2], "arr": [[1, 2], [3, 4]]})
    exploded = df.explode("arr")
    deserialized_df = _test_plan_serialization(exploded._logical_plan, local_session._session_state, serde_implementation)
    assert deserialized_df.to_polars().equals(exploded.to_polars())

    df2 = local_session.create_dataframe({"c1": [1, 1, 2], "c2": [1, 2, 2]})
    dd = df2.drop_duplicates(["c1", "c2"])  # ordering may differ
    deserialized_df = _test_plan_serialization(dd._logical_plan, local_session._session_state, serde_implementation)
    # Compare ignoring row order
    result = deserialized_df.to_polars().sort(["c1", "c2"])
    expected = dd.to_polars().sort(["c1", "c2"])
    assert result.equals(expected)


@pytest.mark.parametrize("serde_implementation", serde_implementations)
def test_sql_plan(local_session, serde_implementation: SupportsLogicalPlanSerde):
    left = local_session.create_dataframe({"id": [1, 2, 3], "name1": ["a", "b", "c"]})
    right = local_session.create_dataframe({"id": [1, 2, 3], "name2": ["d", "e", "f"]})
    df = local_session.sql("SELECT {df1}.id, {df1}.name1, {df2}.name2 FROM {df1} JOIN {df2} USING (id)", df1=left, df2=right)
    deserialized_df = _test_df_serialization(df, local_session._session_state, serde_implementation)
    assert deserialized_df.to_polars().equals(df.to_polars())


@pytest.mark.parametrize("serde_implementation", serde_implementations)
def test_sink_plans(local_session, serde_implementation: SupportsLogicalPlanSerde):
    df = local_session.create_dataframe({"a": [1, 2, 3]})

    file_sink_plan = FileSink.from_session_state(
        child=df._logical_plan,
        sink_type="csv",
        path="/tmp/dummy.csv",
        mode="overwrite",
        session_state=local_session._session_state,
    )
    _ = _test_plan_serialization(file_sink_plan, local_session._session_state, serde_implementation)

    table_sink_plan = TableSink.from_session_state(
        child=df._logical_plan,
        table_name="tmp_table",
        mode="overwrite",
        session_state=local_session._session_state,
    )
    _ = _test_plan_serialization(table_sink_plan, local_session._session_state, serde_implementation)

def test_serialize_unregistered_plan_type():
    """Test that serializing an unregistered expression type raises an error."""

    # Create a mock expression that's not registered
    class MockLogicalPlan(LogicalPlan):
        def __init__(self, schema: Schema):
            super().__init__(schema=schema)

        def __str__(self):
            return "mock_expr"

        def _eq_specific(self, other: LogicalPlan) -> bool:
            return True

        def _build_schema(self, session_state: BaseSessionState) -> Schema:
            raise NotImplementedError()

        def with_children(self, children: List[LogicalPlan], session_state: Optional[BaseSessionState] = None) -> LogicalPlan:
            raise NotImplementedError()

        def exprs(self) -> List[LogicalExpr]:
            return []

        def _repr(self) -> str:
            return "mock_expr"

        def to_column_field(self, plan):
            return None

        def children(self):
            return []

    mock_expr = MockLogicalPlan(Schema([ColumnField("a", IntegerType)]))
    context = SerdeContext()
    with pytest.raises(SerializationError, match="Serialization not implemented for Logical Plan"):
        serialize_logical_plan(mock_expr, context)



@pytest.mark.parametrize("serde_implementation", serde_implementations)
def test_all_plan_types_with_examples(local_session, serde_implementation, temp_dir_with_test_files):
    """Test all plan types with comprehensive examples using parameterized tests."""

    test_cases = []
    examples = _create_plan_examples(local_session, temp_dir_with_test_files)
    for plan_class, examples_list in examples.items():
        for example_name, plan in examples_list:
            test_cases.append((plan_class, example_name, plan))

    # Run all test cases
    for plan_class, example_name, plan in test_cases:
        try:
            # Test serialization/deserialization
            _test_plan_serialization(plan, local_session._session_state, serde_implementation)

        except Exception as e:
            pytest.fail(f"Test failed for {plan_class.__name__}.{example_name}: {e}")




def test_plan_type_coverage(local_session, temp_dir_with_test_files):
    """Test that all concrete LogicalPlan subclasses are covered in the test file."""
    import importlib
    import inspect

    from fenic.core._logical_plan import LogicalPlan

    # Find all concrete LogicalPlan subclasses
    concrete_subclasses = set()
    try:
        module = importlib.import_module("fenic.core._logical_plan.plans")
        for _name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and
                issubclass(obj, LogicalPlan) and
                obj != LogicalPlan and
                not inspect.isabstract(obj)):
                concrete_subclasses.add(obj.__name__)
    except ImportError:
        pass

    # Get all tested plan classes from the examples
    examples = _create_plan_examples(local_session, temp_dir_with_test_files)
    tested_classes = set(cls.__name__ for cls in examples.keys())

    # Find missing classes
    missing = concrete_subclasses - tested_classes

    if missing:
        pytest.fail(
            f"Missing {len(missing)} concrete LogicalPlan subclasses from tests: {sorted(missing)}. "
            f"Add them to the _create_plan_examples function in this test file."
        )

    # Optional: Check for extra classes (not LogicalPlan subclasses)
    extra = tested_classes - concrete_subclasses
    if extra:
        print(f"Warning: {len(extra)} tested classes are not concrete LogicalPlan subclasses: {sorted(extra)}")

    # Verify coverage
    coverage = len(concrete_subclasses - missing) / len(concrete_subclasses) * 100
    assert coverage == 100.0, f"Plan coverage is {coverage:.1f}%, expected 100%"

    # Count total examples
    total_examples = __builtins__['sum'](len(examples) for examples in examples.values())
    print(f"Total plan examples: {total_examples}")
