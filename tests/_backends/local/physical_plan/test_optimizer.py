"""Unit tests for physical plan optimizer."""

import polars as pl

from fenic._backends.local.physical_plan import (
    DuckDBTableSourceExec,
    FileSourceExec,
    FilterExec,
    ProjectionExec,
    SQLExec,
)
from fenic._backends.local.physical_plan.optimizer import (
    MergeDuckDBNodesRule,
    PhysicalPlanOptimizer,
)
from fenic._backends.local.physical_plan.transform import MergedDuckDBExec
from fenic.api.session.session import Session


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
            templated_query="SELECT * FROM {0} WHERE x > 10",
            cache_info=None,
            session_state=session_state,
            arrow_view_names=["view1"]
        )
        sql2 = SQLExec(
            children=[sql1],
            templated_query="SELECT * FROM {0} GROUP BY y",
            cache_info=None,
            session_state=session_state,
            arrow_view_names=["view2"]
        )

        # Apply optimization
        optimizer = PhysicalPlanOptimizer(session_state, [MergeDuckDBNodesRule()])
        result = optimizer.optimize(sql2)

        # Should be optimized
        assert result.optimized is True
        assert isinstance(result.plan, MergedDuckDBExec)

        # Check that the subtree contains all original nodes
        merged_node = result.plan
        assert merged_node.subtree.root == sql2
        assert len(merged_node.subtree.external_inputs) == 0  # No external inputs

    def test_mixed_duckdb_non_duckdb(self, local_session: Session):
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
            templated_query="SELECT * FROM {0}",
            cache_info=None,
            session_state=session_state,
            arrow_view_names=["view1"]
        )

        # Apply optimization
        optimizer = PhysicalPlanOptimizer(session_state, [MergeDuckDBNodesRule()])
        result = optimizer.optimize(sql)

        # Should not be optimized because FilterExec breaks the DuckDB chain
        # The plan should remain as-is with optimized children
        assert isinstance(result.plan, SQLExec)
        assert isinstance(result.plan.children[0], FilterExec)
        assert isinstance(result.plan.children[0].children[0], FileSourceExec)

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
        assert isinstance(result.plan, FileSourceExec)

    def test_cache_boundary_prevents_merging(self, local_session: Session):
        """Test that cache boundaries prevent merging."""
        session_state = local_session._session_state
        from fenic.core._logical_plan.plans import CacheInfo

        # Build plan with cache in middle
        source = DuckDBTableSourceExec(
            table_name="source_table",
            session_state=session_state
        )
        sql1 = SQLExec(
            children=[source],
            templated_query="SELECT * FROM {0}",
            cache_info=CacheInfo(cache_key="cached_table"),  # Cache here
            session_state=session_state,
            arrow_view_names=["view1"]
        )
        sql2 = SQLExec(
            children=[sql1],
            templated_query="SELECT * FROM {0}",
            cache_info=None,
            session_state=session_state,
            arrow_view_names=["view2"]
        )

        # Apply optimization
        optimizer = PhysicalPlanOptimizer(session_state, [MergeDuckDBNodesRule()])
        result = optimizer.optimize(sql2)

        # Cache should prevent merging - the cached node becomes an external input
        assert result.optimized is False
        assert isinstance(result.plan, SQLExec)

    def test_join_pattern_merging(self, local_session: Session):
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
            templated_query="SELECT * FROM {0} JOIN {1} ON {0}.id = {1}.id",
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
        assert merged_node.subtree.root == join_sql
        assert len(merged_node.subtree.external_inputs) == 0  # All inputs are DuckDB

    def test_external_inputs_detection(self, local_session: Session):
        """Test correct detection of external inputs in mixed plans."""
        session_state = local_session._session_state
        # Build plan: DuckDB source + non-DuckDB operation -> SQL
        duckdb_source = FileSourceExec(
            paths=["data.parquet"],
            file_format="parquet",
            session_state=session_state
        )

        # Non-DuckDB chain
        projection = ProjectionExec(
            child=duckdb_source,
            projections=[pl.col("x"), pl.col("y")],
            cache_info=None,
            session_state=session_state
        )
        filter_exec = FilterExec(
            child=projection,
            predicate=pl.col("x") > 5,
            cache_info=None,
            session_state=session_state
        )

        # DuckDB operation
        sql = SQLExec(
            children=[filter_exec],
            templated_query="SELECT x, COUNT(*) FROM {0} GROUP BY x",
            cache_info=None,
            session_state=session_state,
            arrow_view_names=["filtered_data"]
        )

        # Apply optimization
        optimizer = PhysicalPlanOptimizer(session_state, [MergeDuckDBNodesRule()])
        result = optimizer.optimize(sql)

        # The SQL node should not be merged because it has non-DuckDB input
        # But the children should be optimized recursively
        assert isinstance(result.plan, SQLExec)
        assert isinstance(result.plan.children[0], FilterExec)


class TestPhysicalPlanOptimizer:
    """Test the physical plan optimizer framework."""

    def test_optimizer_initialization(self, local_session: Session):
        """Test optimizer initialization with session state."""
        session_state = local_session._session_state
        rules = [MergeDuckDBNodesRule()]
        optimizer = PhysicalPlanOptimizer(session_state, rules)

        assert optimizer.session_state == session_state
        assert optimizer.rules == rules

    def test_empty_rules_list(self, local_session: Session):
        """Test optimizer with no rules."""
        session_state = local_session._session_state
        # Simple plan
        source = FileSourceExec(
            paths=["data.parquet"],
            file_format="parquet",
            session_state=session_state
        )

        # No rules
        optimizer = PhysicalPlanOptimizer(session_state, [])
        result = optimizer.optimize(source)

        # Should not be optimized
        assert result.optimized is False
        assert result.plan == source

    def test_multiple_rules_application(self, local_session: Session):
        """Test that multiple rules are applied in sequence."""
        session_state = local_session._session_state
        # For now we only have one rule, but this tests the framework
        rules = [MergeDuckDBNodesRule()]
        optimizer = PhysicalPlanOptimizer(session_state, rules)

        # Build a chain that can be optimized
        file_source = FileSourceExec(
            paths=["data.parquet"],
            file_format="parquet",
            session_state=session_state
        )
        sql1 = SQLExec(
            children=[file_source],
            templated_query="SELECT * FROM {0}",
            cache_info=None,
            session_state=session_state,
            arrow_view_names=["view1"]
        )
        sql2 = SQLExec(
            children=[sql1],
            templated_query="SELECT * FROM {0}",
            cache_info=None,
            session_state=session_state,
            arrow_view_names=["view2"]
        )

        result = optimizer.optimize(sql2)

        # Should be optimized by the DuckDB rule
        assert result.optimized is True
        assert isinstance(result.plan, MergedDuckDBExec)
