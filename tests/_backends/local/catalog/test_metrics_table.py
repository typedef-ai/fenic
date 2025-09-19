import os

import pytest

from fenic import (
    DataFrame,
    Session,
    SessionConfig,
    col,
)
from fenic.core.error import CatalogError


def test_metrics_table_created_automatically(local_session: Session):
    """Test that metrics table is automatically created and appears in catalog."""
    # Metrics table should be created automatically when session starts
    assert local_session.catalog.does_table_exist("fenic_system.query_metrics")

    # Should be listed in tables in the fenic_system database
    current_db = local_session.catalog.get_current_database()
    local_session.catalog.set_current_database("fenic_system")
    tables = local_session.catalog.list_tables()
    assert "query_metrics" in tables
    local_session.catalog.set_current_database(current_db)


def test_metrics_table_schema(local_session: Session):
    """Test that metrics table has correct schema."""
    table_metadata = local_session.catalog.describe_table("fenic_system.query_metrics")

    # Check that all required columns exist
    column_names = [field.name for field in table_metadata.schema.column_fields]
    expected_columns = [
        "execution_id", "session_id", "execution_time_ms",
        "num_output_rows", "start_ts", "end_ts", "total_lm_cost",
        "total_lm_uncached_input_tokens", "total_lm_cached_input_tokens",
        "total_lm_output_tokens", "total_lm_requests", "total_rm_cost",
        "total_rm_input_tokens", "total_rm_requests"
    ]

    for expected_column in expected_columns:
        assert expected_column in column_names, f"Missing column: {expected_column}"


def test_metrics_table_read_only_protection(local_session: Session):
    """Test that metrics table cannot be modified by users."""
    with pytest.raises(CatalogError, match="Cannot drop table.*from read-only system database"):
        local_session.catalog.drop_table("fenic_system.query_metrics")


def test_metrics_collected_on_dataframe_operations(local_session: Session, sample_df: DataFrame):
    """Test that metrics are collected when DataFrames are executed."""
    # Get initial row count
    initial_metrics = local_session.table("fenic_system.query_metrics")
    initial_count = initial_metrics.count()

    # Execute some operations
    sample_df.show()
    sample_df.count()
    sample_df.collect()
    sample_df.select("name").filter(col("name") == "Alice").show()
    sample_df.write.save_as_table("test_table")
    sample_df.write.save_as_table("test_table", mode="ignore")
    try:
        sample_df.write.parquet("test_table.parquet")
        sample_df.write.parquet("test_table.parquet", mode="ignore")
    finally:
        os.remove("test_table.parquet")

    # Check that metrics were added
    final_metrics = local_session.table("fenic_system.query_metrics")
    final_count = final_metrics.count() # This adds a 9th entry

    assert final_count == initial_count + 9, f"Expected 9 new metrics, got {final_count - initial_count}"


def test_metrics_session_id(local_session: Session, sample_df: DataFrame):
    """Test that we can read metrics table as a DataFrame."""
    # Execute an operation to generate metrics
    sample_df.show()

    # Read metrics table as DataFrame
    metrics_df = local_session.table("fenic_system.query_metrics")
    result = metrics_df.collect()

    assert len(result.data) > 0
    # Check that session_id matches current session
    querymetrics_session_id = result.metrics.session_id
    table_session_id = result.data["session_id"][0]

    assert local_session._session_state.session_id == querymetrics_session_id
    assert local_session._session_state.session_id == table_session_id


def test_metrics_table_contains_execution_data(local_session: Session, sample_df: DataFrame):
    """Test that metrics table contains expected execution data."""
    # Get initial state
    initial_metrics = local_session.table("fenic_system.query_metrics")
    initial_metrics.count()

    # Execute operation
    result = sample_df.collect()
    expected_rows = len(result.data)
    execution_id = result.metrics.execution_id

    # Get final state
    metrics_data = local_session.table("fenic_system.query_metrics").collect().data

    # Should have new metric row
    assert len(metrics_data) > 0

    # Check the latest metric entry
    latest_metric = metrics_data[-1]
    assert latest_metric["session_id"][0] == local_session._session_state.session_id
    assert latest_metric["num_output_rows"][0] == expected_rows
    assert latest_metric["execution_time_ms"][0] >= 0
    assert latest_metric["execution_id"][0] == execution_id
    assert latest_metric["start_ts"][0] is not None
    assert latest_metric["end_ts"][0] is not None


def test_multiple_sessions_different_metrics(tmp_path, local_session_config: SessionConfig):
    """Test that different sessions have separate metrics tracking."""
    # Create first session and run queries
    session1 = Session.get_or_create(local_session_config)
    df1 = session1.create_dataframe({"a": [1, 2, 3]})
    df1.show()  # 1 query
    df1.count()  # 1 query

    session1_id = session1._session_state.session_id

    # Create second session with different config
    session_config2 = SessionConfig(
        app_name="metrics_test_2",
        semantic=local_session_config.semantic,
        db_path=tmp_path,
    )
    session2 = Session.get_or_create(session_config2)
    df2 = session2.create_dataframe({"a": [1, 2, 3]})
    df2.show()  # 1 query

    session2_id = session2._session_state.session_id

    try:
        # Check metrics for session1 - need to account for count() calls adding metrics
        metrics1_data = session1.table("fenic_system.query_metrics").filter(
            col("session_id") == session1_id).collect().data
        count1 = len(metrics1_data)
        session1.table("fenic_system.query_metrics").show()

        # Check metrics for session2 - need to account for collect() call adding metrics
        metrics2_data = session2.table("fenic_system.query_metrics").filter(
            col("session_id") == session2_id).collect().data
        count2 = len(metrics2_data)
        session2.table("fenic_system.query_metrics").show()

        # Session1 should have 2 metrics, for show and count
        assert count1 == 2, f"Session1 should have 2 metrics, got {count1}"

        # Session2 should have 1 metric, for show
        assert count2 == 1, f"Session2 should have 1 metric, got {count2}"

        assert session1_id != session2_id

    finally:
        # Clean up sessions
        session1.stop()
        session2.stop()

