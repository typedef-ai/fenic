"""Unit tests for physical plan optimizer."""

import polars as pl

from fenic import ColumnField, Schema, StringType
from fenic._backends.local.physical_plan import (
    DuckDBTableSinkExec,
    DuckDBTableSourceExec,
    FileSinkExec,
    FileSourceExec,
    FilterExec,
    SQLExec,
)
from fenic._backends.local.physical_plan.optimizer import (
    MergeDuckDBNodesRule,
    PhysicalPlanOptimizer,
)
from fenic._backends.local.physical_plan.transform import MergedDuckDBExec
from fenic.api.session.session import Session
from fenic.core._logical_plan.plans import CacheInfo


class TestMergeDuckDBNodesRule:
    """Test the DuckDB node merging optimization rule."""

    def test_simple_chain_merging(self, local_session: Session):
        """Test merging of a simple linear chain of DuckDB operations."""
        session_state = local_session._session_state
        # Build plan: FileSource -> SQL -> SQL
        file_source = FileSourceExec(
            paths=["data.parquet"],
            file_format="parquet",
            session_state=session_state
        )
        sql1 = SQLExec(
            children=[file_source],
            query="SELECT * FROM file_source WHERE x > 10",
            cache_info=None,
            session_state=session_state,
            arrow_view_names=["view1"]
        )
        sql2 = SQLExec(
            children=[sql1],
            query="SELECT * FROM sql1 GROUP BY y",
            cache_info=None,
            session_state=session_state,
            arrow_view_names=["view2"]
        )
        duckdb_sink = DuckDBTableSinkExec(
            child=sql2,
            table_name="sink_table",
            mode="overwrite",
            cache_info=None,
            session_state=session_state,
            schema=Schema(
                column_fields=[
                    ColumnField(name="x", data_type=StringType),
                    ColumnField(name="y", data_type=StringType),
                ]
            )
        )

        # Apply optimization
        optimizer = PhysicalPlanOptimizer(session_state, [MergeDuckDBNodesRule()])
        result = optimizer.optimize(duckdb_sink)

        # Should be optimized
        assert result.optimized is True
        merged_node = result.plan
        assert isinstance(merged_node, MergedDuckDBExec)


        assert merged_node.merge_root is duckdb_sink
        assert merged_node.merge_root.children[0].query == "SELECT * FROM sql1 GROUP BY y"

        assert merged_node.merge_root.children[0].children[0].query == "SELECT * FROM file_source WHERE x > 10"

        assert merged_node.merge_root.children[0].children[0].children[0].paths == ["data.parquet"]
        assert merged_node.merge_root.children[0].children[0].children[0].file_format == "parquet"

        assert len(merged_node.children) == 0  # No external inputs

    def test_merge_file_sink(self, local_session: Session):
        """Test merging of a file sink node."""
        session_state = local_session._session_state
        # Build plan: FileSource -> FileSink
        file_source = FileSourceExec(
            paths=["data1.parquet"],
            file_format="parquet",
            session_state=session_state
        )
        file_sink = FileSinkExec(
            child=file_source,
            path="data2.parquet",
            file_type="parquet",
            mode="overwrite",
            cache_info=None,
            session_state=session_state
        )

        # Apply optimization
        optimizer = PhysicalPlanOptimizer(session_state, [MergeDuckDBNodesRule()])
        result = optimizer.optimize(file_sink)

        assert result.optimized is True
        assert isinstance(result.plan, MergedDuckDBExec)
        assert result.plan.merge_root is file_sink
        assert result.plan.merge_root.children[0] is file_source
        assert len(result.plan.children) == 0

    def test_mixed_duckdb_non_duckdb_chain(self, local_session: Session):
        """Test that mixing DuckDB and non-DuckDB operations prevents merging."""
        session_state = local_session._session_state
        # Build plan: FileSource -> Filter -> SQL
        file_source = FileSourceExec(
            paths=["data.csv"],
            file_format="csv",
            session_state=session_state
        )
        filter_exec = FilterExec(
            child=file_source,
            predicate=pl.col("x") > 10,
            cache_info=None,
            session_state=session_state
        )
        sql = SQLExec(
            children=[filter_exec],
            query="SELECT * FROM filter_exec",
            cache_info=None,
            session_state=session_state,
            arrow_view_names=["view1"]
        )

        # Apply optimization
        optimizer = PhysicalPlanOptimizer(session_state, [MergeDuckDBNodesRule()])
        result = optimizer.optimize(sql)
        assert result.optimized is False

        # Should not be optimized because FilterExec breaks the DuckDB chain
        # The plan should remain as-is with optimized children
        assert result.plan is sql
        assert result.plan.children[0] is filter_exec
        assert result.plan.children[0].children[0] is file_source

    def test_single_node_no_merging(self, local_session: Session):
        """Test that single DuckDB nodes are not merged."""
        session_state = local_session._session_state
        # Single SQL node
        file_source = FileSourceExec(
            paths=["data.parquet"],
            file_format="parquet",
            session_state=session_state
        )

        # Apply optimization
        optimizer = PhysicalPlanOptimizer(session_state, [MergeDuckDBNodesRule()])
        result = optimizer.optimize(file_source)

        # Should not be optimized (single node doesn't benefit from merging)
        assert result.optimized is False
        assert result.plan is file_source

    def test_cache_boundary_prevents_merging(self, local_session: Session):
        """Test that cache boundaries prevent merging."""
        session_state = local_session._session_state

        # Build plan with cache in middle
        source = DuckDBTableSourceExec(
            table_name="source_table",
            session_state=session_state
        )
        sql1 = SQLExec(
            children=[source],
            query="SELECT * FROM source_table",
            cache_info=CacheInfo(duckdb_table_name="cached_table"),  # Cache here
            session_state=session_state,
            arrow_view_names=["view1"]
        )
        sql2 = SQLExec(
            children=[sql1],
            query="SELECT * FROM sql1",
            cache_info=None,
            session_state=session_state,
            arrow_view_names=["view2"]
        )
        sql3 = SQLExec(
            children=[sql2],
            query="SELECT * FROM sql2",
            cache_info=None,
            session_state=session_state,
            arrow_view_names=["view3"]
        )

        # Apply optimization
        optimizer = PhysicalPlanOptimizer(session_state, [MergeDuckDBNodesRule()])
        result = optimizer.optimize(sql3)

        assert result.optimized is True
        assert isinstance(result.plan, MergedDuckDBExec)
        assert result.plan.merge_root is sql3
        assert result.plan.merge_root.children[0] is sql2
        assert result.plan.merge_root.children[0].children[0] is sql1
        assert result.plan.merge_root.children[0].children[0].children[0] is source

        assert len(result.plan.children) == 1
        assert isinstance(result.plan.children[0], SQLExec)
        assert result.plan.children[0] is sql1
        assert len(result.plan.children[0].children) == 1
        assert result.plan.children[0].children[0] is source

    def test_cache_root_prevents_merging(self, local_session: Session):
        """Test that cache boundaries prevent merging."""
        session_state = local_session._session_state

        # Build plan with cache in middle
        source = FileSourceExec(
            paths=["data.parquet"],
            file_format="parquet",
            session_state=session_state
        )
        sql1 = SQLExec(
            children=[source],
            query="SELECT * FROM source_table",
            cache_info=None,
            session_state=session_state,
            arrow_view_names=["view1"]
        )
        sql2 = SQLExec(
            children=[sql1],
            query="SELECT * FROM sql1",
            cache_info=None,
            session_state=session_state,
            arrow_view_names=["view2"]
        )
        sql3 = SQLExec(
            children=[sql2],
            query="SELECT * FROM sql2",
            cache_info=CacheInfo(duckdb_table_name="cached_table"),  # Cache here
            session_state=session_state,
            arrow_view_names=["view3"]
        )

        # Apply optimization
        optimizer = PhysicalPlanOptimizer(session_state, [MergeDuckDBNodesRule()])
        result = optimizer.optimize(sql3)

        assert result.optimized is True
        assert isinstance(result.plan, SQLExec)
        assert isinstance(result.plan.children[0], MergedDuckDBExec)
        assert len(result.plan.children[0].children) == 0

        merged_node = result.plan.children[0]
        assert merged_node.merge_root is sql2
        assert merged_node.merge_root.children[0] is sql1
        assert merged_node.merge_root.children[0].children[0] is source


    def test_multi_child_sql(self, local_session: Session):
        """Test merging with JOIN pattern (multiple children)."""
        session_state = local_session._session_state
        # Build plan: two DuckDB sources -> SQL JOIN
        left_source = FileSourceExec(
            paths=["left.parquet"],
            file_format="parquet",
            session_state=session_state
        )
        right_source = DuckDBTableSourceExec(
            table_name="right_table",
            session_state=session_state
        )
        join_sql = SQLExec(
            children=[left_source, right_source],
            query="SELECT * FROM left_source JOIN right_source ON left_source.id = right_source.id",
            cache_info=None,
            session_state=session_state,
            arrow_view_names=["left", "right"]
        )

        # Apply optimization
        optimizer = PhysicalPlanOptimizer(session_state, [MergeDuckDBNodesRule()])
        result = optimizer.optimize(join_sql)

        # Should be optimized
        assert result.optimized is True
        assert isinstance(result.plan, MergedDuckDBExec)

        # Check subtree
        merged_node = result.plan
        assert merged_node.merge_root is join_sql
        assert merged_node.merge_root.children[0] is left_source
        assert merged_node.merge_root.children[1] is right_source
        assert len(merged_node.children) == 0  # All inputs are DuckDB
