"""Tests for LogicalPlan equality implementation.

This test suite focuses on:
1. One comprehensive test for recursive __eq__ logic
2. Minimal tests for each plan's unique equality attributes
"""

import polars as pl

from fenic.core._logical_plan.expressions.aggregate import (
    AvgExpr,
    SumExpr,
)
from fenic.core._logical_plan.expressions.base import Operator
from fenic.core._logical_plan.expressions.basic import (
    AliasExpr,
    ColumnExpr,
    LiteralExpr,
    SortExpr,
)
from fenic.core._logical_plan.expressions.comparison import (
    EqualityComparisonExpr,
)
from fenic.core._logical_plan.expressions.semantic import (
    EmbeddingsExpr,
)
from fenic.core._logical_plan.plans.aggregate import Aggregate
from fenic.core._logical_plan.plans.join import (
    Join,
    SemanticJoin,
    SemanticSimilarityJoin,
)
from fenic.core._logical_plan.plans.sink import FileSink, TableSink
from fenic.core._logical_plan.plans.source import (
    FileSource,
    InMemorySource,
    TableSource,
)
from fenic.core._logical_plan.plans.transform import (
    SQL,
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
from fenic.core.types.datatypes import FloatType, IntegerType, StringType
from fenic.core.types.schema import ColumnField, Schema


class TestRecursiveEquality:
    """Test the base __eq__ recursive logic once."""

    def test_recursive_equality_comprehensive(self):
        """Test that __eq__ correctly handles recursion, type checking, and edge cases."""
        # Create test data
        test_df = pl.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})

        # Create schema
        schema = Schema([ColumnField("x", IntegerType), ColumnField("y", StringType)])

        # Create base sources
        source1 = InMemorySource.from_schema(test_df, schema)
        source2 = InMemorySource.from_schema(test_df, schema)

        # Create expressions
        col_x = ColumnExpr("x")
        col_y = ColumnExpr("y")

        # Create nested plan structure: projection(filter(source))
        filter_pred1 = EqualityComparisonExpr(
            col_x, LiteralExpr(1, IntegerType), Operator.EQ
        )
        filter_pred2 = EqualityComparisonExpr(
            col_x, LiteralExpr(1, IntegerType), Operator.EQ
        )
        filter_pred3 = EqualityComparisonExpr(
            col_x, LiteralExpr(2, IntegerType), Operator.EQ
        )

        filtered1 = Filter.from_schema(source1, filter_pred1, schema)
        filtered2 = Filter.from_schema(source2, filter_pred2, schema)
        filtered3 = Filter.from_schema(source1, filter_pred3, schema)

        proj_exprs = [col_x, col_y]
        projected1 = Projection.from_schema(filtered1, proj_exprs, schema)
        projected2 = Projection.from_schema(filtered2, proj_exprs, schema)
        projected3 = Projection.from_schema(filtered3, proj_exprs, schema)

        # Identical nested structures should be equal
        assert projected1 == projected2

        # Different deep expressions should not be equal
        assert projected1 != projected3

        # Test type checking
        assert projected1 != col_x  # Different types
        assert projected1 != "string"  # Non-LogicalPlan
        assert projected1 is not None  # None

        # Test self equality
        assert projected1 == projected1

    def test_children_count_mismatch(self):
        """Test that plans with different child counts are not equal."""
        test_df = pl.DataFrame({"x": [1, 2, 3]})
        schema = Schema([ColumnField("x", IntegerType)])

        # Single child plan
        source = InMemorySource.from_schema(test_df, schema)
        projection = Projection.from_schema(source, [ColumnExpr("x")], schema)

        # Multiple child plan
        union = Union.from_schema([source, source], schema)

        assert projection != union

    def test_schema_mismatch(self):
        """Test that plans with different schemas are not equal."""
        test_df = pl.DataFrame({"x": [1, 2, 3]})

        schema1 = Schema([ColumnField("x", IntegerType)])
        schema2 = Schema([ColumnField("x", StringType)])  # Different type

        source1 = InMemorySource.from_schema(test_df, schema1)
        source2 = InMemorySource.from_schema(test_df, schema2)

        assert source1 != source2


class TestSourcePlans:
    """Test equality for source plans."""

    def test_inmemory_source(self):
        """Test InMemorySource compares DataFrame content."""
        df1 = pl.DataFrame({"x": [1, 2, 3]})
        df2 = pl.DataFrame({"x": [1, 2, 3]})
        df3 = pl.DataFrame({"x": [4, 5, 6]})

        schema = Schema([ColumnField("x", IntegerType)])

        source1 = InMemorySource.from_schema(df1, schema)
        source2 = InMemorySource.from_schema(df2, schema)
        source3 = InMemorySource.from_schema(df3, schema)

        # Same data should be equal
        assert source1 == source2

        # Different data should not be equal
        assert source1 != source3

    def test_file_source(self):
        """Test FileSource compares paths and format."""
        schema = Schema([ColumnField("x", IntegerType)])
        
        # Same paths and format
        file1 = FileSource.from_schema(["data.csv"], "csv", None, schema)
        file2 = FileSource.from_schema(["data.csv"], "csv", None, schema)
        assert file1 == file2
        
        # Different paths
        file3 = FileSource.from_schema(["other.csv"], "csv", None, schema)
        assert file1 != file3
        
        # Different format
        file4 = FileSource.from_schema(["data.csv"], "parquet", None, schema)
        assert file1 != file4
        
        # Different number of paths
        file5 = FileSource.from_schema(["data.csv", "data2.csv"], "csv", None, schema)
        assert file1 != file5

    def test_table_source(self):
        """Test TableSource compares table name."""
        schema = Schema([ColumnField("x", IntegerType)])

        # Same table name
        table1 = TableSource.from_schema("users", schema)
        table2 = TableSource.from_schema("users", schema)
        assert table1 == table2

        # Different table name
        table3 = TableSource.from_schema("products", schema)
        assert table1 != table3


class TestTransformPlans:
    """Test equality for transform plans."""

    def test_projection(self):
        """Test Projection compares expressions (set equality)."""
        test_df = pl.DataFrame({"x": [1, 2], "y": ["a", "b"]})
        schema = Schema([ColumnField("x", IntegerType), ColumnField("y", StringType)])
        source = InMemorySource.from_schema(test_df, schema)

        col_x = ColumnExpr("x")
        col_y = ColumnExpr("y")

        # Same expressions (same order)
        proj1 = Projection.from_schema(source, [col_x, col_y], schema)
        proj2 = Projection.from_schema(source, [col_x, col_y], schema)
        assert proj1 == proj2

        # Same expressions (different order) - should NOT be equal because order matters for schema
        proj3 = Projection.from_schema(source, [col_y, col_x], Schema([ColumnField("y", StringType), ColumnField("x", IntegerType)]))
        assert proj1 != proj3

        # Different expressions
        proj4 = Projection.from_schema(
            source, [col_x], Schema([ColumnField("x", IntegerType)])
        )
        assert proj1 != proj4

    def test_filter(self):
        """Test Filter compares predicate."""
        test_df = pl.DataFrame({"x": [1, 2, 3]})
        schema = Schema([ColumnField("x", IntegerType)])
        source = InMemorySource.from_schema(test_df, schema)

        col_x = ColumnExpr("x")

        # Same predicate
        pred1 = EqualityComparisonExpr(col_x, LiteralExpr(1, IntegerType), Operator.EQ)
        pred2 = EqualityComparisonExpr(col_x, LiteralExpr(1, IntegerType), Operator.EQ)
        filter1 = Filter.from_schema(source, pred1, schema)
        filter2 = Filter.from_schema(source, pred2, schema)
        assert filter1 == filter2

        # Different predicate
        pred3 = EqualityComparisonExpr(col_x, LiteralExpr(2, IntegerType), Operator.EQ)
        filter3 = Filter.from_schema(source, pred3, schema)
        assert filter1 != filter3

    def test_union(self):
        """Test Union has no specific attributes (only children matter)."""
        test_df = pl.DataFrame({"x": [1, 2, 3]})
        schema = Schema([ColumnField("x", IntegerType)])

        source1 = InMemorySource.from_schema(test_df, schema)
        source2 = InMemorySource.from_schema(test_df, schema)
        source3 = InMemorySource.from_schema(
            test_df.with_columns(pl.col("x") + 1), schema
        )

                # Same children
        union1 = Union.from_schema([source1, source2], schema)
        union2 = Union.from_schema([source1, source2], schema)
        assert union1 == union2
        
        # Different children
        union3 = Union.from_schema([source1, source3], schema)
        assert union1 != union3
        
        # Different order of children (using different sources so order matters)
        union4 = Union.from_schema([source1, source3], schema)
        union5 = Union.from_schema([source3, source1], schema)
        assert union4 != union5  # Order matters for children

    def test_limit(self):
        """Test Limit compares n value."""
        test_df = pl.DataFrame({"x": [1, 2, 3]})
        schema = Schema([ColumnField("x", IntegerType)])
        source = InMemorySource.from_schema(test_df, schema)

        # Same limit
        limit1 = Limit.from_schema(source, 10, schema)
        limit2 = Limit.from_schema(source, 10, schema)
        assert limit1 == limit2

        # Different limit
        limit3 = Limit.from_schema(source, 5, schema)
        assert limit1 != limit3

    def test_explode(self):
        """Test Explode compares expression."""
        test_df = pl.DataFrame({"arr": [[1, 2], [3, 4]]})
        schema = Schema([ColumnField("arr", IntegerType)])  # Simplified for test
        source = InMemorySource.from_schema(test_df, schema)

        col_arr = ColumnExpr("arr")
        col_other = ColumnExpr("other")

        # Same expression
        explode1 = Explode.from_schema(source, col_arr, schema)
        explode2 = Explode.from_schema(source, col_arr, schema)
        assert explode1 == explode2

        # Different expression
        explode3 = Explode.from_schema(source, col_other, schema)
        assert explode1 != explode3

    def test_drop_duplicates(self):
        """Test DropDuplicates compares subset columns."""
        test_df = pl.DataFrame({"x": [1, 2], "y": ["a", "b"]})
        schema = Schema([ColumnField("x", IntegerType), ColumnField("y", StringType)])
        source = InMemorySource.from_schema(test_df, schema)

        col_x = ColumnExpr("x")
        col_y = ColumnExpr("y")

        # Same subset
        dedup1 = DropDuplicates.from_schema(source, [col_x], schema)
        dedup2 = DropDuplicates.from_schema(source, [col_x], schema)
        assert dedup1 == dedup2

        # Different subset
        dedup3 = DropDuplicates.from_schema(source, [col_y], schema)
        assert dedup1 != dedup3

        # Different subset size
        dedup4 = DropDuplicates.from_schema(source, [col_x, col_y], schema)
        assert dedup1 != dedup4

    def test_sort(self):
        """Test Sort compares sort expressions (set equality)."""
        test_df = pl.DataFrame({"x": [1, 2], "y": ["a", "b"]})
        schema = Schema([ColumnField("x", IntegerType), ColumnField("y", StringType)])
        source = InMemorySource.from_schema(test_df, schema)

        col_x = ColumnExpr("x")
        col_y = ColumnExpr("y")
        sort_x_asc = SortExpr(col_x, ascending=True)
        sort_x_desc = SortExpr(col_x, ascending=False)
        sort_y_asc = SortExpr(col_y, ascending=True)

        # Same sort expressions
        sort1 = Sort.from_schema(source, [sort_x_asc, sort_y_asc], schema)
        sort2 = Sort.from_schema(source, [sort_x_asc, sort_y_asc], schema)
        assert sort1 == sort2

        # Same expressions, different order - should NOT be equal because order matters for sorting
        sort3 = Sort.from_schema(source, [sort_y_asc, sort_x_asc], schema)
        assert sort1 != sort3

        # Different sort expressions
        sort4 = Sort.from_schema(source, [sort_x_desc], schema)
        assert sort1 != sort4

    def test_unnest(self):
        """Test Unnest compares expressions (set equality)."""
        # Mock schema for struct type
        schema = Schema([ColumnField("x", IntegerType)])
        source = InMemorySource.from_schema(pl.DataFrame({"x": [1]}), schema)

        col_x = ColumnExpr("x")
        col_y = ColumnExpr("y")

        # Same expressions
        unnest1 = Unnest.from_schema(source, [col_x], schema)
        unnest2 = Unnest.from_schema(source, [col_x], schema)
        assert unnest1 == unnest2

        # Different expressions
        unnest3 = Unnest.from_schema(source, [col_y], schema)
        assert unnest1 != unnest3

    def test_sql(self):
        """Test SQL compares template names and SQL query."""
        schema = Schema([ColumnField("x", IntegerType)])
        source = InMemorySource.from_schema(pl.DataFrame({"x": [1]}), schema)

        # Same template names and query
        sql1 = SQL.from_schema([source], ["table1"], "SELECT * FROM ${table1}", schema)
        sql2 = SQL.from_schema([source], ["table1"], "SELECT * FROM ${table1}", schema)
        assert sql1 == sql2
        
        # Different template names
        sql3 = SQL.from_schema([source], ["table2"], "SELECT * FROM ${table2}", schema)
        assert sql1 != sql3
        
        # Different SQL query
        sql4 = SQL.from_schema([source], ["table1"], "SELECT x FROM ${table1}", schema)
        assert sql1 != sql4

    def test_semantic_cluster(self):
        """Test SemanticCluster compares clustering parameters."""
        schema = Schema(
            [
                ColumnField("x", IntegerType),
                ColumnField("embedding", FloatType),  # Simplified
            ]
        )
        source = InMemorySource.from_schema(
            pl.DataFrame({"x": [1], "embedding": [0.1]}), schema
        )

        by_expr = EmbeddingsExpr(ColumnExpr("x"))

        # Same parameters
        cluster1 = SemanticCluster.from_schema(
            source,
            by_expr,
            num_clusters=5,
            max_iter=100,
            num_init=10,
            label_column="cluster",
            centroid_column="centroid",
            schema=schema,
        )
        cluster2 = SemanticCluster.from_schema(
            source,
            by_expr,
            num_clusters=5,
            max_iter=100,
            num_init=10,
            label_column="cluster",
            centroid_column="centroid",
            schema=schema,
        )
        assert cluster1 == cluster2

        # Different num_clusters
        cluster3 = SemanticCluster.from_schema(
            source,
            by_expr,
            num_clusters=3,
            max_iter=100,
            num_init=10,
            label_column="cluster",
            centroid_column="centroid",
            schema=schema,
        )
        assert cluster1 != cluster3

        # Different max_iter
        cluster4 = SemanticCluster.from_schema(
            source,
            by_expr,
            num_clusters=5,
            max_iter=50,
            num_init=10,
            label_column="cluster",
            centroid_column="centroid",
            schema=schema,
        )
        assert cluster1 != cluster4

        # Different label_column
        cluster5 = SemanticCluster.from_schema(
            source,
            by_expr,
            num_clusters=5,
            max_iter=100,
            num_init=10,
            label_column="group",
            centroid_column="centroid",
            schema=schema,
        )
        assert cluster1 != cluster5


class TestJoinPlans:
    """Test equality for join plans."""

    def test_join(self):
        """Test Join compares join conditions and type."""
        schema_x = Schema([ColumnField("x", IntegerType)])
        schema_y = Schema([ColumnField("y", IntegerType)])
        left = InMemorySource.from_schema(pl.DataFrame({"x": [1]}), schema_x)
        right_x = InMemorySource.from_schema(pl.DataFrame({"x": [1]}), schema_x)
        right_y = InMemorySource.from_schema(pl.DataFrame({"y": [1]}), schema_y)

        col_x = ColumnExpr("x")
        col_y = ColumnExpr("y")

        # Same join conditions and type
        join1 = Join.from_schema(left, right_x, [col_x], [col_x], "inner", schema_x)
        join2 = Join.from_schema(left, right_x, [col_x], [col_x], "inner", schema_x)
        assert join1 == join2

        # Different join conditions (different right column)
        join3 = Join.from_schema(left, right_y, [col_x], [col_y], "inner", schema_x)
        assert join1 != join3

        # Different join type
        join4 = Join.from_schema(left, right_x, [col_x], [col_x], "left", schema_x)
        assert join1 != join4

    def test_semantic_join(self):
        """Test SemanticJoin compares embedding expressions and parameters."""
        schema = Schema([ColumnField("x", IntegerType)])
        left = InMemorySource.from_schema(pl.DataFrame({"x": [1]}), schema)
        right = InMemorySource.from_schema(pl.DataFrame({"x": [1]}), schema)

        left_embed = EmbeddingsExpr(ColumnExpr("x"))
        right_embed = EmbeddingsExpr(ColumnExpr("x"))

        template = "Compare {{left_on}} with {{right_on}}"

        # Same parameters
        sem_join1 = SemanticJoin.from_schema(
            left,
            right,
            left_embed,
            right_embed,
            template,
            True,
            0.5,
            schema=schema,
        )
        sem_join2 = SemanticJoin.from_schema(
            left,
            right,
            left_embed,
            right_embed,
            template,
            True,
            0.5,
            schema=schema,
        )
        assert sem_join1 == sem_join2

        # Different template
        sem_join3 = SemanticJoin.from_schema(
            left,
            right,
            left_embed,
            right_embed,
            "Different template {{left_on}} {{right_on}}",
            True,
            0.5,
            schema=schema,
        )
        assert sem_join1 != sem_join3

        # Different strict
        sem_join4 = SemanticJoin.from_schema(
            left,
            right,
            left_embed,
            right_embed,
            template,
            False,
            0.5,
            schema=schema,
        )
        assert sem_join1 != sem_join4

        # Different temperature
        sem_join5 = SemanticJoin.from_schema(
            left,
            right,
            left_embed,
            right_embed,
            template,
            True,
            0.8,
            schema=schema,
        )
        assert sem_join1 != sem_join5

    def test_semantic_similarity_join(self):
        """Test SemanticSimilarityJoin compares expressions, k, and similarity metric."""
        schema = Schema([ColumnField("x", IntegerType)])
        left = InMemorySource.from_schema(pl.DataFrame({"x": [1]}), schema)
        right = InMemorySource.from_schema(pl.DataFrame({"x": [1]}), schema)

        left_embed = EmbeddingsExpr(ColumnExpr("x"))
        right_embed = EmbeddingsExpr(ColumnExpr("x"))

        # Same parameters
        sim_join1 = SemanticSimilarityJoin.from_schema(
            left,
            right,
            left_embed,
            right_embed,
            k=5,
            similarity_metric="cosine",
            schema=schema,
        )
        sim_join2 = SemanticSimilarityJoin.from_schema(
            left,
            right,
            left_embed,
            right_embed,
            k=5,
            similarity_metric="cosine",
            schema=schema,
        )
        assert sim_join1 == sim_join2

        # Different k
        sim_join3 = SemanticSimilarityJoin.from_schema(
            left,
            right,
            left_embed,
            right_embed,
            k=10,
            similarity_metric="cosine",
            schema=schema,
        )
        assert sim_join1 != sim_join3

        # Different similarity metric
        sim_join4 = SemanticSimilarityJoin.from_schema(
            left,
            right,
            left_embed,
            right_embed,
            k=5,
            similarity_metric="l2",
            schema=schema,
        )
        assert sim_join1 != sim_join4


class TestSinkPlans:
    """Test equality for sink plans."""

    def test_file_sink(self):
        """Test FileSink compares path and format."""
        schema = Schema([ColumnField("x", IntegerType)])
        source = InMemorySource.from_schema(pl.DataFrame({"x": [1]}), schema)

        # Same path and format
        sink1 = FileSink.from_schema(source, "csv", "output.csv", "overwrite", schema)
        sink2 = FileSink.from_schema(source, "csv", "output.csv", "overwrite", schema)
        assert sink1 == sink2

        # Different path
        sink3 = FileSink.from_schema(source, "csv", "other.csv", "overwrite", schema)
        assert sink1 != sink3

        # Different format
        sink4 = FileSink.from_schema(source, "parquet", "output.csv", "overwrite", schema)
        assert sink1 != sink4

    def test_table_sink(self):
        """Test TableSink compares table name and mode."""
        schema = Schema([ColumnField("x", IntegerType)])
        source = InMemorySource.from_schema(pl.DataFrame({"x": [1]}), schema)

        # Same table name and mode
        sink1 = TableSink.from_schema(source, "users", "overwrite", schema)
        sink2 = TableSink.from_schema(source, "users", "overwrite", schema)
        assert sink1 == sink2

        # Different table name
        sink3 = TableSink.from_schema(source, "products", "overwrite", schema)
        assert sink1 != sink3

        # Different mode
        sink4 = TableSink.from_schema(source, "users", "append", schema)
        assert sink1 != sink4


class TestAggregateePlans:
    """Test equality for aggregate plans."""

    def test_aggregate(self):
        """Test Aggregate compares group and aggregate expressions (set equality)."""
        schema = Schema(
            [
                ColumnField("x", IntegerType),
                ColumnField("y", StringType),
                ColumnField("sum_x", IntegerType),
            ]
        )
        source = InMemorySource.from_schema(
            pl.DataFrame({"x": [1, 2], "y": ["a", "b"]}),
            Schema([ColumnField("x", IntegerType), ColumnField("y", StringType)]),
        )

        col_x = ColumnExpr("x")
        col_y = ColumnExpr("y")
        group_exprs = [col_y]
        agg_exprs = [AliasExpr(SumExpr(col_x), "sum_x")]

        # Same group and agg expressions
        agg1 = Aggregate.from_schema(source, group_exprs, agg_exprs, schema)
        agg2 = Aggregate.from_schema(source, group_exprs, agg_exprs, schema)
        assert agg1 == agg2

        # Different group expressions
        different_group = [col_x]
        agg3 = Aggregate.from_schema(source, different_group, agg_exprs, schema)
        assert agg1 != agg3

        # Different agg expressions
        different_agg = [AliasExpr(AvgExpr(col_x), "avg_x")]
        agg4 = Aggregate.from_schema(source, group_exprs, different_agg, schema)
        assert agg1 != agg4


class TestCrossTypeInequality:
    """Test that different plan types are never equal."""

    def test_source_types_never_equal(self):
        """Test different source plan types are never equal."""
        test_df = pl.DataFrame({"x": [1, 2, 3]})
        schema = Schema([ColumnField("x", IntegerType)])

        inmemory = InMemorySource.from_schema(test_df, schema)
        file_source = FileSource.from_schema(["data.csv"], "csv", None, schema)
        table_source = TableSource.from_schema("users", schema)

        sources = [inmemory, file_source, table_source]
        for i, source1 in enumerate(sources):
            for j, source2 in enumerate(sources):
                if i != j:
                    assert source1 != source2

    def test_transform_types_never_equal(self):
        """Test different transform plan types are never equal."""
        test_df = pl.DataFrame({"x": [1, 2, 3]})
        schema = Schema([ColumnField("x", IntegerType)])
        source = InMemorySource.from_schema(test_df, schema)

        col_x = ColumnExpr("x")

        projection = Projection.from_schema(source, [col_x], schema)
        filter_plan = Filter.from_schema(
            source,
            EqualityComparisonExpr(col_x, LiteralExpr(1, IntegerType), Operator.EQ),
            schema,
        )
        limit = Limit.from_schema(source, 10, schema)
        sort = Sort.from_schema(source, [SortExpr(col_x, ascending=True)], schema)

        transforms = [projection, filter_plan, limit, sort]
        for i, transform1 in enumerate(transforms):
            for j, transform2 in enumerate(transforms):
                if i != j:
                    assert transform1 != transform2

    def test_all_plan_categories_never_equal(self):
        """Test different plan categories are never equal."""
        test_df = pl.DataFrame({"x": [1, 2, 3]})
        schema = Schema([ColumnField("x", IntegerType)])

        # One from each category
        source = InMemorySource.from_schema(test_df, schema)
        transform = Projection.from_schema(source, [ColumnExpr("x")], schema)
        sink = FileSink.from_schema(source, "csv", "output.csv", "overwrite", schema)
        join = Join.from_schema(
            source,
            source,
            [ColumnExpr("x")],
            [ColumnExpr("x")],
            "inner",
            schema,
        )
        aggregate = Aggregate.from_schema(
            source, [], [AliasExpr(SumExpr(ColumnExpr("x")), "sum_x")], schema
        )

        plans = [source, transform, sink, join, aggregate]
        for i, plan1 in enumerate(plans):
            for j, plan2 in enumerate(plans):
                if i != j:
                    assert plan1 != plan2
