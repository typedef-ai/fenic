
from urllib.parse import urlparse

import boto3
import pytest
from botocore.session import get_session
from pydantic import ValidationError

from fenic._backends.local.utils.io_utils import does_path_exist
from fenic.api.session import Session
from fenic.core._logical_plan.plans import FileSink, TableSink
from fenic.core.error import ConfigurationError, PlanError
from fenic.core.error import ValidationError as FenicValidationError


def test_csv_writer_overwrite_mode(local_session, temp_dir):
    """Test CSV writer with overwrite mode."""
    # Create output path
    output_path = f"{temp_dir.path}/test_overwrite.csv"

    # Write a file first
    df1 = local_session.create_dataframe({"a": [1, 2, 3]})
    df1.write.csv(output_path, mode="overwrite")
    assert does_path_exist(output_path, local_session._session_state.s3_session)

    # Write again with overwrite mode
    df2 = local_session.create_dataframe({"a": [4, 5, 6, 7]})
    df2.write.csv(output_path, mode="overwrite")

    # Verify file was overwritten by reading it
    result_df = local_session.read.csv(output_path)
    count = result_df.count()
    assert count == 4  # Should have the count from the second dataframe


def test_csv_writer_error_mode(local_session, temp_dir):
    """Test CSV writer with error mode."""
    # Create output path
    output_path = f"{temp_dir.path}/test_error.csv"

    # Write a file first
    df1 = local_session.create_dataframe({"a": [1, 2, 3]})
    df1.write.csv(output_path, mode="overwrite")
    assert does_path_exist(output_path, local_session._session_state.s3_session)

    # Attempt to write again with error mode - should raise an error
    df2 = local_session.create_dataframe({"a": [4, 5, 6]})
    with pytest.raises(PlanError, match=".*already exists and mode is 'error'.*"):
        df2.write.csv(output_path, mode="error")


def test_csv_writer_ignore_mode(local_session, temp_dir):
    """Test CSV writer with ignore mode."""
    # Create output path
    output_path = f"{temp_dir.path}/test_ignore.csv"

    # Write a file first
    df1 = local_session.create_dataframe({"a": [1, 2, 3]})
    df1.write.csv(output_path, mode="overwrite")
    assert does_path_exist(output_path, local_session._session_state.s3_session)

    # Write again with ignore mode
    df2 = local_session.create_dataframe({"a": [4, 5, 6, 7]})
    df2.write.csv(output_path, mode="ignore")

    # Verify file was not overwritten by reading it
    result_df = local_session.read.csv(output_path)
    count = result_df.count()
    assert count == 3  # Should still have the count from the first dataframe


def test_csv_writer_invalid_mode(local_session, temp_dir):
    """Test CSV writer with an invalid mode."""
    output_path = f"{temp_dir.path}/test_invalid.csv"
    df = local_session.create_dataframe({"a": [1, 2, 3]})

    # Test with an invalid mode
    with pytest.raises(ValidationError):
        df.write.csv(output_path, mode="append")


def test_parquet_writer_overwrite_mode(local_session, temp_dir):
    """Test Parquet writer with overwrite mode."""
    # Create output path
    output_path = f"{temp_dir.path}/test_overwrite.parquet"

    # Write a file first
    df1 = local_session.create_dataframe({"a": [1, 2, 3]})
    df1.write.parquet(output_path, mode="overwrite")
    assert does_path_exist(output_path, local_session._session_state.s3_session)

    # Write again with overwrite mode
    df2 = local_session.create_dataframe({"a": [4, 5, 6, 7]})
    df2.write.parquet(output_path, mode="overwrite")

    # Verify file was overwritten by reading it
    result_df = local_session.read.parquet(output_path)
    count = result_df.count()
    assert count == 4  # Should have the count from the second dataframe


def test_parquet_writer_error_mode(local_session, temp_dir):
    """Test Parquet writer with error mode."""
    # Create output path
    output_path = f"{temp_dir.path}/test_error.parquet"

    # Write a file first
    df1 = local_session.create_dataframe({"a": [1, 2, 3]})
    df1.write.parquet(output_path, mode="overwrite")
    assert does_path_exist(output_path, local_session._session_state.s3_session)

    # Attempt to write again with error mode - should raise an error
    df2 = local_session.create_dataframe({"a": [4, 5, 6]})
    with pytest.raises(PlanError, match=".*already exists and mode is 'error'.*"):
        df2.write.parquet(output_path, mode="error")


def test_parquet_writer_ignore_mode(local_session, temp_dir):
    """Test Parquet writer with ignore mode."""
    # Create output path
    output_path = f"{temp_dir.path}/test_ignore.parquet"

    # Write a file first
    df1 = local_session.create_dataframe({"a": [1, 2, 3]})
    df1.write.parquet(output_path, mode="overwrite")
    assert does_path_exist(output_path, local_session._session_state.s3_session)

    # Write again with ignore mode
    df2 = local_session.create_dataframe({"a": [4, 5, 6, 7]})
    df2.write.parquet(output_path, mode="ignore")

    # Verify file was not overwritten by reading it
    result_df = local_session.read.parquet(output_path)
    count = result_df.count()
    assert count == 3  # Should still have the count from the first dataframe


def test_parquet_writer_invalid_mode(local_session, temp_dir):
    """Test Parquet writer with an invalid mode."""
    output_path = f"{temp_dir.path}/test_invalid.parquet"
    df = local_session.create_dataframe({"a": [1, 2, 3]})

    # Test with an invalid mode
    with pytest.raises(ValidationError):
        df.write.parquet(output_path, mode="append")


def test_table_writer_overwrite_mode(local_session):
    """Test table writer with overwrite mode."""
    table_name = "test_overwrite_table"

    # Write a table first
    df1 = local_session.create_dataframe({"a": [1, 2, 3]})
    df1.write.save_as_table(table_name, mode="overwrite")

    # Verify the table was created
    count = local_session.table(table_name).count()
    assert count == 3

    # Write again with overwrite mode
    df2 = local_session.create_dataframe({"a": [4, 5, 6, 7]})
    df2.write.save_as_table(table_name, mode="overwrite")

    # Verify table was overwritten
    count = local_session.table(table_name).count()
    assert count == 4  # Should have the count from the second dataframe


def test_table_writer_append_mode_same_schema(local_session):
    """Test table writer with append mode."""
    table_name = "test_append_table"

    # Write a table first
    df1 = local_session.create_dataframe({"a": [1, 2, 3]})
    df1.write.save_as_table(table_name, mode="overwrite")

    # Verify the table was created
    count = local_session.table(table_name).count()
    assert count == 3

    # Write again with append mode
    df2 = local_session.create_dataframe({"a": [4, 5, 6, 7]})
    df2.write.save_as_table(table_name, mode="append")

    # Verify data was appended
    count = local_session.table(table_name).count()
    assert count == 7  # Should have the combined count from both dataframes


def test_table_writer_append_mode_different_schema(local_session):
    """Test table writer with append mode."""
    table_name = "test_append_table"

    # Write a table first
    df1 = local_session.create_dataframe({"a": [1, 2, 3]})
    df1.write.save_as_table(table_name, mode="overwrite")

    # Verify the table was created
    count = local_session.table(table_name).count()
    assert count == 3

    # Write again with append mode
    df2 = local_session.create_dataframe({"a": [4, 5, 6, 7], "b": [8, 9, 10, 11]})
    with pytest.raises(
        PlanError,
        match="Cannot append to table 'test_append_table' - schema mismatch detected",
    ):
        df2.write.save_as_table(table_name, mode="append")


def test_table_writer_error_mode(local_session):
    """Test table writer with error mode."""
    table_name = "test_error_table"

    # Write a table first
    df1 = local_session.create_dataframe({"a": [1, 2, 3]})
    df1.write.save_as_table(table_name, mode="overwrite")

    # Verify the table was created
    count = local_session.table(table_name).count()
    assert count == 3

    # Attempt to write again with error mode - should raise an error
    df2 = local_session.create_dataframe({"a": [4, 5, 6]})
    with pytest.raises(
        PlanError,
        match=f"Cannot save to table '{table_name}' - it already exists and mode is 'error'",
    ):
        df2.write.save_as_table(table_name, mode="error")


def test_table_writer_ignore_mode(local_session):
    """Test table writer with ignore mode."""
    table_name = "test_ignore_table"

    # Write a table first
    df1 = local_session.create_dataframe({"a": [1, 2, 3]})
    df1.write.save_as_table(table_name, mode="overwrite")

    # Verify the table was created
    count = local_session.table(table_name).count()
    assert count == 3

    # Write again with ignore mode
    df2 = local_session.create_dataframe({"a": [4, 5, 6, 7]})
    df2.write.save_as_table(table_name, mode="ignore")

    # Verify table was not modified
    count = local_session.table(table_name).count()
    assert count == 3  # Should still have the count from the first dataframe


def test_logical_plan_sink_nodes(local_session):
    """Test creation of logical plan sink nodes directly."""
    df = local_session.create_dataframe({"a": [1, 2, 3]})

    # Create a FileSink node
    file_sink = FileSink.from_session_state(
        child=df._logical_plan,
        sink_type="csv",
        path="test.csv",
        mode="overwrite",
        session_state=df._session_state,
    )

    # Create a TableSink node
    table_sink = TableSink.from_session_state(
        child=df._logical_plan,
        table_name="test_table",
        mode="overwrite",
        session_state=df._session_state
    )
    file_sink_columns = file_sink.schema().column_names()
    table_sink_columns = table_sink.schema().column_names()

    # Verify the nodes have the correct schema
    for col in df.columns:
        assert col in file_sink_columns
        assert col in table_sink_columns

    # Verify string representation
    assert "FileSink" in file_sink._repr()
    assert "TableSink" in table_sink._repr()

    # Verify children
    assert file_sink.children() == [df._logical_plan]
    assert table_sink.children() == [df._logical_plan]


def test_write_with_no_aws_credentials(local_session_config, temp_dir):
    """Test that local write queries work and that write queries to s3 will fail without aws credentials."""
    session = Session.get_or_create(local_session_config)
    # remove credentials from session and patch the fenic session
    botocore_session = get_session()
    botocore_session.set_credentials(None, None)
    session._session_state.s3_session = boto3.Session(botocore_session=botocore_session)

    df1 = session.create_dataframe({"a": [1, 2, 3]})

    # Test that local file csv and parquet writes will succeed without credentials
    # skip if tests are running on s3
    if urlparse(temp_dir.path).scheme != "s3":
        output_path = f"{temp_dir.path}/test.csv"
        df1.write.csv(output_path, mode="overwrite")
        assert does_path_exist(output_path, session._session_state.s3_session)

        output_path = f"{temp_dir.path}/test.parquet"
        df1.write.parquet(output_path, mode="overwrite")
        assert does_path_exist(output_path, session._session_state.s3_session)

    # Test that s3 writes will fail without credentials
    with pytest.raises(ConfigurationError, match="Unable to locate AWS credentials."):
        df1.write.csv("s3://test-bucket/test-file.csv", mode="overwrite")
    with pytest.raises(ConfigurationError, match="Unable to locate AWS credentials."):
        df1.write.parquet("s3://test-bucket/test-file.parquet", mode="overwrite")

    session.stop()

def test_write_with_invalid_file_path(local_session, temp_dir):
    """Test that write queries to invalid file paths will fail."""
    df1 = local_session.create_dataframe({"a": [1, 2, 3]})

    output_path = f"{temp_dir.path}/bad_test_file"
    with pytest.raises(FenicValidationError, match="CSV writer requires a '.csv' file extension."):
        df1.write.csv(output_path, mode="overwrite")

    output_path = f"{temp_dir.path}/bad_test_file"
    with pytest.raises(FenicValidationError, match="Parquet writer requires a '.parquet' file extension."):
        df1.write.parquet(output_path, mode="overwrite")
