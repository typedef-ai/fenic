import polars as pl

from fenic.core._logical_plan.expressions.aggregate import SumExpr
from fenic.core._logical_plan.expressions.base import Operator
from fenic.core._logical_plan.expressions.basic import (
    AliasExpr,
    ColumnExpr,
    LiteralExpr,
    SortExpr,
)
from fenic.core._logical_plan.expressions.comparison import EqualityComparisonExpr
from fenic.core._logical_plan.plans.aggregate import Aggregate
from fenic.core._logical_plan.plans.join import Join
from fenic.core._logical_plan.plans.source import InMemorySource
from fenic.core._logical_plan.plans.transform import (
    DropDuplicates,
    Explode,
    Filter,
    Limit,
    Projection,
    SemanticCluster,
    Sort,
    Union,
    Unnest,
)
from fenic.core.types.datatypes import (
    EmbeddingType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)
from fenic.core.types.schema import ColumnField, Schema


def make_source_with_xy():
    df = pl.DataFrame({"x": [1, 2], "y": ["a", "b"]})
    schema = Schema([ColumnField("x", IntegerType), ColumnField("y", StringType)])
    return InMemorySource.from_schema(df, schema), schema


def test_projection_exprs():
    src, schema = make_source_with_xy()
    col_x = ColumnExpr("x")
    col_y = ColumnExpr("y")
    node = Projection.from_schema(src, [col_x, col_y], Schema([ColumnField("x", IntegerType), ColumnField("y", StringType)]))
    assert node.exprs() == [col_x, col_y]


def test_filter_exprs():
    src, schema = make_source_with_xy()
    pred = EqualityComparisonExpr(ColumnExpr("x"), LiteralExpr(1, IntegerType), Operator.EQ)
    node = Filter.from_schema(src, pred, schema)
    assert node.exprs() == [pred]


def test_join_exprs():
    src, schema = make_source_with_xy()
    left_on = [ColumnExpr("x")]
    right_on = [ColumnExpr("x")]
    node = Join.from_schema(src, src, left_on, right_on, how="inner", schema=schema)
    assert node.exprs() == left_on + right_on


def test_aggregate_exprs():
    src, schema = make_source_with_xy()
    groups = [ColumnExpr("x")]
    aggs = [AliasExpr(SumExpr(ColumnExpr("x")), "sum_x")]
    out_schema = Schema([ColumnField("x", IntegerType), ColumnField("sum_x", IntegerType)])
    node = Aggregate.from_schema(src, groups, aggs, out_schema)
    assert node.exprs() == groups + aggs


def test_sort_exprs():
    src, schema = make_source_with_xy()
    sort_exprs = [SortExpr(ColumnExpr("x"))]
    node = Sort.from_schema(src, sort_exprs, schema)
    assert node.exprs() == sort_exprs


def test_explode_exprs():
    src, _ = make_source_with_xy()
    node = Explode.from_schema(src, ColumnExpr("y"), Schema([ColumnField("y", StringType)]))
    assert node.exprs() == [ColumnExpr("y")]


def test_drop_duplicates_exprs():
    src, schema = make_source_with_xy()
    cols = [ColumnExpr("x")]
    node = DropDuplicates.from_schema(src, cols, schema)
    assert node.exprs() == cols


def test_unnest_exprs():
    df = pl.DataFrame({"s": [{"a": 1}, {"a": 2}]})
    schema = Schema([ColumnField("s", StructType([StructField("a", IntegerType)]))])
    src = InMemorySource.from_schema(df, schema)
    node = Unnest.from_schema(src, [ColumnExpr("s")], Schema([ColumnField("a", IntegerType)]))
    assert node.exprs() == [ColumnExpr("s")]


def test_limit_union_sql_exprs_empty():
    src, schema = make_source_with_xy()
    assert Limit.from_schema(src, 5, schema).exprs() == []
    assert Union.from_schema([src, src], schema).exprs() == []


def test_semantic_cluster_exprs():
    # Use EmbeddingType for by_expr
    src, schema = make_source_with_xy()
    emb_schema = Schema([ColumnField("e", EmbeddingType(dimensions=3, embedding_model="test"))])
    emb_df = pl.DataFrame({"e": [[0.0, 0.0, 0.0]]})
    emb_src = InMemorySource.from_schema(emb_df, emb_schema)
    node = SemanticCluster.from_schema(emb_src, ColumnExpr("e"), num_clusters=2, max_iter=10, num_init=1, label_column="label", centroid_column=None, schema=Schema(emb_schema.column_fields + [ColumnField("label", IntegerType)]))
    assert node.exprs() == [ColumnExpr("e")]
