import os
from unittest.mock import Mock, patch

import pytest

from fenic._backends.local.utils.io_utils import (
    build_read_query,
    build_write_query,
)
from fenic.core.error import ConfigurationError, InternalError
from fenic.core.types.datatypes import FloatType, IntegerType, StringType
from fenic.core.types.schema import ColumnField, Schema


class TestBuildReadQuery:
    """Test cases for build_read_query function."""

    def setup_method(self):
        """Set up test fixtures."""
        # Mock S3 session with credentials
        self.mock_s3_session = Mock()
        self.mock_creds = Mock()
        self.mock_creds.get_frozen_credentials.return_value = Mock(
            access_key="test_key",
            secret_key="test_secret",
            token="test_token"
        )
        self.mock_s3_session.get_credentials.return_value = self.mock_creds
        self.mock_s3_session.region_name = "us-east-1"

        # Mock S3 session without credentials
        self.mock_s3_session_no_creds = Mock()
        self.mock_s3_session_no_creds.get_credentials.return_value = None

    def test_local_csv_simple(self):
        """Test building query for local CSV files."""
        result = build_read_query(
            paths=["data.csv", "data2.csv"],
            file_type="csv",
            s3_session=self.mock_s3_session
        )

        assert result.sql == "SELECT * FROM read_csv(['data.csv', 'data2.csv'])"
        assert result.s3_status is None
        assert result.hf_status is None

    def test_local_parquet_simple(self):
        """Test building query for local Parquet files."""
        result = build_read_query(
            paths=["data.parquet"],
            file_type="parquet",
            s3_session=self.mock_s3_session
        )

        assert result.sql == "SELECT * FROM read_parquet(['data.parquet'])"
        assert result.s3_status is None
        assert result.hf_status is None

    def test_csv_with_schema(self):
        """Test CSV query with explicit schema."""
        schema = Schema([
            ColumnField(name="id", data_type=IntegerType),
            ColumnField(name="name", data_type=StringType),
            ColumnField(name="score", data_type=FloatType)
        ])

        result = build_read_query(
            paths=["data.csv"],
            file_type="csv",
            s3_session=self.mock_s3_session,
            schema=schema
        )

        expected = "SELECT * FROM read_csv(['data.csv'], columns = {'id': 'BIGINT', 'name': 'VARCHAR', 'score': 'FLOAT'})"
        assert result.sql == expected

    def test_csv_merge_schemas(self):
        """Test CSV query with merge_schemas option."""
        result = build_read_query(
            paths=["data1.csv", "data2.csv"],
            file_type="csv",
            s3_session=self.mock_s3_session,
            merge_schemas=True
        )

        assert result.sql == "SELECT * FROM read_csv(['data1.csv', 'data2.csv'], union_by_name=true)"

    def test_parquet_merge_schemas(self):
        """Test Parquet query with merge_schemas option."""
        result = build_read_query(
            paths=["data1.parquet", "data2.parquet"],
            file_type="parquet",
            s3_session=self.mock_s3_session,
            merge_schemas=True
        )

        assert result.sql == "SELECT * FROM read_parquet(['data1.parquet', 'data2.parquet'], union_by_name=true)"

    def test_schema_inference(self):
        """Test query with schema inference enabled."""
        result = build_read_query(
            paths=["data.csv"],
            file_type="csv",
            s3_session=self.mock_s3_session,
            schema_inference=True
        )

        expected = "PRAGMA disable_optimizer; SELECT * FROM read_csv(['data.csv']) WHERE 1=0"
        assert result.sql == expected

    def test_s3_paths_with_credentials(self):
        """Test S3 paths with valid credentials."""
        result = build_read_query(
            paths=["s3://bucket/data.csv"],
            file_type="csv",
            s3_session=self.mock_s3_session
        )

        expected_parts = [
            "INSTALL httpfs",
            "LOAD httpfs",
            "SET s3_region='us-east-1'",
            "SET s3_access_key_id='test_key'",
            "SET s3_secret_access_key='test_secret'",
            "SET s3_session_token='test_token'",
            "SELECT * FROM read_csv(['s3://bucket/data.csv'])"
        ]
        assert result.sql == "; ".join(expected_parts)
        assert result.s3_status is True
        assert result.hf_status is None

    def test_s3_paths_without_credentials(self):
        """Test S3 paths without credentials."""
        result = build_read_query(
            paths=["s3://bucket/data.csv"],
            file_type="csv",
            s3_session=self.mock_s3_session_no_creds
        )

        expected_parts = [
            "INSTALL httpfs",
            "LOAD httpfs",
            "SELECT * FROM read_csv(['s3://bucket/data.csv'])"
        ]
        assert result.sql == "; ".join(expected_parts)
        assert result.s3_status is False
        assert result.hf_status is None

    @patch.dict(os.environ, {"HF_TOKEN": "test_hf_token"})
    def test_hf_paths_with_token(self):
        """Test HuggingFace paths with token."""
        result = build_read_query(
            paths=["hf://datasets/org/dataset/file.parquet"],
            file_type="parquet",
            s3_session=self.mock_s3_session
        )

        expected_parts = [
            "INSTALL httpfs",
            "LOAD httpfs",
            "CREATE SECRET hf_token (TYPE HUGGINGFACE, TOKEN 'test_hf_token')",
            "SELECT * FROM read_parquet(['hf://datasets/org/dataset/file.parquet'])"
        ]
        assert result.sql == "; ".join(expected_parts)
        assert result.s3_status is None
        assert result.hf_status is True

    @patch.dict(os.environ, {}, clear=True)
    def test_hf_paths_without_token(self):
        """Test HuggingFace paths without token."""
        # Ensure HF_TOKEN is not set
        if "HF_TOKEN" in os.environ:
            del os.environ["HF_TOKEN"]

        result = build_read_query(
            paths=["hf://datasets/org/dataset/file.parquet"],
            file_type="parquet",
            s3_session=self.mock_s3_session
        )

        expected_parts = [
            "INSTALL httpfs",
            "LOAD httpfs",
            "SELECT * FROM read_parquet(['hf://datasets/org/dataset/file.parquet'])"
        ]
        assert result.sql == "; ".join(expected_parts)
        assert result.s3_status is None
        assert result.hf_status is False

    def test_mixed_paths(self):
        """Test query with mixed path types."""
        with patch.dict(os.environ, {"HF_TOKEN": "test_token"}):
            result = build_read_query(
                paths=[
                    "local.csv",
                    "s3://bucket/data.csv",
                    "hf://datasets/org/data.csv"
                ],
                file_type="csv",
                s3_session=self.mock_s3_session
            )

            expected_parts = [
                "INSTALL httpfs",
                "LOAD httpfs",
                "SET s3_region='us-east-1'",
                "SET s3_access_key_id='test_key'",
                "SET s3_secret_access_key='test_secret'",
                "SET s3_session_token='test_token'",
                "CREATE SECRET hf_token (TYPE HUGGINGFACE, TOKEN 'test_token')",
                "SELECT * FROM read_csv(['local.csv', 's3://bucket/data.csv', 'hf://datasets/org/data.csv'])"
            ]
            assert result.sql == "; ".join(expected_parts)
            assert result.s3_status is True
            assert result.hf_status is True

    def test_schema_inference_with_s3(self):
        """Test schema inference combined with S3 paths."""
        result = build_read_query(
            paths=["s3://bucket/data.parquet"],
            file_type="parquet",
            s3_session=self.mock_s3_session,
            schema_inference=True
        )

        parts = result.sql.split("; ")
        assert parts[0] == "PRAGMA disable_optimizer"
        assert parts[-1] == "SELECT * FROM read_parquet(['s3://bucket/data.parquet']) WHERE 1=0"


class TestBuildWriteQuery:
    """Test cases for build_write_query function."""

    def setup_method(self):
        """Set up test fixtures."""
        # Mock S3 session with credentials
        self.mock_s3_session = Mock()
        self.mock_creds = Mock()
        self.mock_creds.get_frozen_credentials.return_value = Mock(
            access_key="test_key",
            secret_key="test_secret",
            token=None  # No session token for writes
        )
        self.mock_s3_session.get_credentials.return_value = self.mock_creds
        self.mock_s3_session.region_name = "us-west-2"

        # Mock S3 session without credentials
        self.mock_s3_session_no_creds = Mock()
        self.mock_s3_session_no_creds.get_credentials.return_value = None

    def test_local_csv_write(self):
        """Test writing to local CSV file."""
        result = build_write_query(
            source_table="my_table",
            path="output.csv",
            file_type="csv",
            s3_session=self.mock_s3_session
        )

        assert result.sql == "COPY my_table TO 'output.csv' (FORMAT CSV, HEADER TRUE)"
        assert result.is_s3 is False

    def test_local_parquet_write(self):
        """Test writing to local Parquet file."""
        result = build_write_query(
            source_table="my_view",
            path="/path/to/output.parquet",
            file_type="parquet",
            s3_session=self.mock_s3_session
        )

        assert result.sql == "COPY my_view TO '/path/to/output.parquet' (FORMAT PARQUET)"
        assert result.is_s3 is False

    def test_s3_csv_write_with_credentials(self):
        """Test writing CSV to S3 with valid credentials."""
        result = build_write_query(
            source_table="my_table",
            path="s3://bucket/output.csv",
            file_type="csv",
            s3_session=self.mock_s3_session
        )

        expected_parts = [
            "INSTALL httpfs",
            "LOAD httpfs",
            "SET s3_region='us-west-2'",
            "SET s3_access_key_id='test_key'",
            "SET s3_secret_access_key='test_secret'",
            "COPY my_table TO 's3://bucket/output.csv' (FORMAT CSV, HEADER TRUE)"
        ]
        assert result.sql == "; ".join(expected_parts)
        assert result.is_s3 is True

    def test_s3_parquet_write_with_credentials(self):
        """Test writing Parquet to S3 with valid credentials."""
        result = build_write_query(
            source_table="analytics_data",
            path="s3://bucket/folder/output.parquet",
            file_type="parquet",
            s3_session=self.mock_s3_session
        )

        expected_parts = [
            "INSTALL httpfs",
            "LOAD httpfs",
            "SET s3_region='us-west-2'",
            "SET s3_access_key_id='test_key'",
            "SET s3_secret_access_key='test_secret'",
            "COPY analytics_data TO 's3://bucket/folder/output.parquet' (FORMAT PARQUET)"
        ]
        assert result.sql == "; ".join(expected_parts)
        assert result.is_s3 is True

    def test_s3_write_with_session_token(self):
        """Test S3 write with session token."""
        # Update mock to include session token
        self.mock_creds.get_frozen_credentials.return_value = Mock(
            access_key="test_key",
            secret_key="test_secret",
            token="session_token_123"
        )

        result = build_write_query(
            source_table="my_table",
            path="s3://bucket/data.csv",
            file_type="csv",
            s3_session=self.mock_s3_session
        )

        expected_parts = [
            "INSTALL httpfs",
            "LOAD httpfs",
            "SET s3_region='us-west-2'",
            "SET s3_access_key_id='test_key'",
            "SET s3_secret_access_key='test_secret'",
            "SET s3_session_token='session_token_123'",
            "COPY my_table TO 's3://bucket/data.csv' (FORMAT CSV, HEADER TRUE)"
        ]
        assert result.sql == "; ".join(expected_parts)
        assert result.is_s3 is True

    def test_s3_write_without_credentials(self):
        """Test writing to S3 without credentials (should raise error)."""
        with pytest.raises(ConfigurationError, match="AWS credentials were not found"):
            build_write_query(
                source_table="my_table",
                path="s3://bucket/output.csv",
                file_type="csv",
                s3_session=self.mock_s3_session_no_creds
            )

    def test_invalid_file_type_write(self):
        """Test writing with invalid file type."""
        with pytest.raises(InternalError, match="Unsupported file type"):
            build_write_query(
                source_table="my_table",
                path="output.json",
                file_type="json",
                s3_session=self.mock_s3_session
            )
