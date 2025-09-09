import os
from io import StringIO
from pathlib import Path
from typing import Literal, Union
from urllib.parse import urlparse

import boto3
import polars as pl
import pytest
from botocore.session import get_session
from duckdb import HTTPException

from fenic import (
    ArrayType,
    BooleanType,
    ColumnField,
    DoubleType,
    FloatType,
    IntegerType,
    MarkdownType,
    Schema,
    StringType,
    col,
    markdown,
)
from fenic._backends.local.utils.io_utils import (
    _build_query_with_hf_creds,
    _build_query_with_httpfs_extensions,
    _build_query_with_s3_creds,
    write_file,
)
from fenic.api.session import Session
from fenic.core.error import (
    ConfigurationError,
    InternalError,
    PlanError,
    UnsupportedFileTypeError,
    ValidationError,
)

COLUMNS = {"name", "age", "city"}

def write_test_file(
    path: str,
    content: Union[pl.DataFrame, str],
    local_session: Session,
    file_type: Literal["csv", "parquet"],
):
    """Write test file content to either local or S3 path."""
    if isinstance(content, str):
        if file_type == "csv":
            df = pl.read_csv(StringIO(content))
        elif file_type == "parquet":
            raise InternalError("Writing parquet test files expects a polars DataFrame")
    else:
        df = content
    write_file(df=df, path=path, s3_session=local_session._session_state.s3_session, file_type=file_type)


# =============================================================================
# Basic Reading Tests
# =============================================================================


def test_csv_single_file(local_session, temp_dir):
    """Test reading a single CSV file.

    This tests:
    - Basic CSV file reading functionality
    - Schema inference
    - Data loading
    """
    # Create test data
    test_data = """name,age,city
John,25,New York
Alice,30,San Francisco
Bob,35,Chicago
Carol,28,Boston
David,33,Seattle"""

    test_file_path = f"{temp_dir.path}/test_csv_single_file.csv"
    write_test_file(test_file_path, test_data, local_session, "csv")

    # Test DataFrame creation and schema inference
    df = local_session.read.csv(test_file_path)

    # Verify schema through DataFrame API
    schema = df.schema
    actual_columns = {field.name for field in schema.column_fields}
    assert (
        actual_columns == COLUMNS
    ), f"Expected columns {COLUMNS}, got {actual_columns}"

    # Collect the full DataFrame and verify data
    collected = df.to_polars()
    assert collected.height == 5, f"Expected 5 rows, got {collected.height}"

    # Compare the collected data against expected values
    data = collected.to_dict(as_series=False)
    assert data["name"] == ["John", "Alice", "Bob", "Carol", "David"]
    assert data["age"] == [25, 30, 35, 28, 33]
    assert data["city"] == ["New York", "San Francisco", "Chicago", "Boston", "Seattle"]

    if urlparse(test_file_path).scheme == "s3":
        return
    # Test both Path and string path formats
    df = local_session.read.csv(Path(test_file_path))
    schema = df.schema
    actual_columns = {field.name for field in schema.column_fields}
    assert (
        actual_columns == COLUMNS
    ), f"Expected columns {COLUMNS}, got {actual_columns}"


def test_csv_multiple_files_same_schema(local_session, temp_dir):
    """Test reading multiple CSV files.

    This tests:
    - Reading with glob patterns
    - Reading with explicit file lists
    - Schema consistency across files
    """
    # Create test data
    test_data1 = """name,age,city
John,25,New York
Alice,30,San Francisco
Bob,35,Chicago
Carol,28,Boston
David,33,Seattle"""

    test_data2 = """name,age,city
Michael,40,Dallas
Sarah,22,Denver
Tom,29,Atlanta
Lisa,31,Miami
Ryan,27,Portland
Emma,36,Phoenix"""

    test_file_path1 = f"{temp_dir.path}/test_csv_multiple_1.csv"
    test_file_path2 = f"{temp_dir.path}/test_csv_multiple_2.csv"

    write_test_file(test_file_path1, test_data1, local_session, "csv")
    write_test_file(test_file_path2, test_data2, local_session, "csv")

    # Test with glob pattern
    csv_glob = f"{temp_dir.path}/test_csv_multiple_*.csv"
    df = local_session.read.csv(csv_glob)

    schema = df.schema
    actual_columns = {field.name for field in schema.column_fields}
    assert (
        actual_columns == COLUMNS
    ), f"Expected columns {COLUMNS}, got {actual_columns}"

    collected = df.to_polars()
    assert collected.height == 11, f"Expected 11 rows, got {collected.height}"

    # Test with explicit list of files
    df = local_session.read.csv([test_file_path1, test_file_path2])
    schema = df.schema
    actual_columns = {field.name for field in schema.column_fields}
    assert (
        actual_columns == COLUMNS
    ), f"Expected columns {COLUMNS}, got {actual_columns}"
    collected = df.to_polars()
    assert collected.height == 11, f"Expected 11 rows, got {collected.height}"



def test_parquet_single_file(local_session, temp_dir):
    """Test reading a single Parquet file.

    This tests:
    - Basic Parquet file reading functionality
    - Schema inference
    - Data loading
    """
    # Create test data as CSV first
    test_data = """name,age,city
John,25,New York
Alice,30,San Francisco
Bob,35,Chicago
Carol,28,Boston
David,33,Seattle"""

    test_file_path = f"{temp_dir.path}/test_parquet_single_file.parquet"

    # Convert to parquet using polars
    df_polars = pl.read_csv(StringIO(test_data))
    write_test_file(test_file_path, df_polars, local_session, "parquet")

    # Test DataFrame creation and schema inference
    df = local_session.read.parquet(test_file_path)

    # Verify schema through DataFrame API
    schema = df.schema
    actual_columns = {field.name for field in schema.column_fields}
    assert (
        actual_columns == COLUMNS
    ), f"Expected columns {COLUMNS}, got {actual_columns}"

    # Collect the full DataFrame and verify data
    collected = df.to_polars()
    assert collected.height == 5, f"Expected 5 rows, got {collected.height}"

    # Compare the collected data against expected values
    data = collected.to_dict(as_series=False)
    assert data["name"] == ["John", "Alice", "Bob", "Carol", "David"]
    assert data["age"] == [25, 30, 35, 28, 33]
    assert data["city"] == ["New York", "San Francisco", "Chicago", "Boston", "Seattle"]

    if urlparse(test_file_path).scheme == "s3":
        return
    # Test both Path and string path formats
    df = local_session.read.parquet(Path(test_file_path))
    schema = df.schema
    actual_columns = {field.name for field in schema.column_fields}
    assert (
        actual_columns == COLUMNS
    ), f"Expected columns {COLUMNS}, got {actual_columns}"


def test_parquet_multiple_files_same_schema(local_session, temp_dir):
    """Test reading multiple Parquet files.

    This tests:
    - Reading with glob patterns
    - Reading with explicit file lists
    - Schema consistency across files
    """
    # Create test data
    test_data1 = """name,age,city
John,25,New York
Alice,30,San Francisco
Bob,35,Chicago
Carol,28,Boston
David,33,Seattle"""

    test_data2 = """name,age,city
Michael,40,Dallas
Sarah,22,Denver
Tom,29,Atlanta
Lisa,31,Miami
Ryan,27,Portland
Emma,36,Phoenix"""

    test_file_path1 = f"{temp_dir.path}/test_parquet_multiple_1.parquet"
    test_file_path2 = f"{temp_dir.path}/test_parquet_multiple_2.parquet"

    # Convert to parquet using polars
    df_polars1 = pl.read_csv(StringIO(test_data1))
    df_polars2 = pl.read_csv(StringIO(test_data2))

    write_test_file(test_file_path1, df_polars1, local_session, "parquet")
    write_test_file(test_file_path2, df_polars2, local_session, "parquet")

    # Test with glob pattern
    parquet_glob = f"{temp_dir.path}/test_parquet_multiple_*.parquet"
    df = local_session.read.parquet(parquet_glob)

    # Verify schema
    schema = df.schema
    actual_columns = {field.name for field in schema.column_fields}
    assert (
        actual_columns == COLUMNS
    ), f"Expected columns {COLUMNS}, got {actual_columns}"

    # Verify row count
    collected = df.to_polars()
    assert collected.height == 11, f"Expected 11 rows, got {collected.height}"

    # Test with list of file paths
    parquet_files = [test_file_path1, test_file_path2]
    df = local_session.read.parquet(parquet_files)

    # Verify schema
    schema = df.schema
    actual_columns = {field.name for field in schema.column_fields}
    assert (
        actual_columns == COLUMNS
    ), f"Expected columns {COLUMNS}, got {actual_columns}"

    # Verify row count
    collected = df.to_polars()
    assert collected.height == 11, f"Expected 11 rows, got {collected.height}"


# =============================================================================
# Schema Customization Tests
# =============================================================================


def test_csv_with_explicit_schema(local_session, temp_dir):
    """Test the CSV reader with an explicit schema.

    This tests:
    - Providing a custom schema definition
    - Type conversion during loading
    - Schema validation
    - Error handling for incompatible schema types
    """
    # Create test data
    test_data = """name,age,score
Alice,25,95.5
Bob,30,88.3
Carol,28,91.7"""

    test_file_path = f"{temp_dir.path}/test_csv_with_explicit_schema.csv"
    write_test_file(test_file_path, test_data, local_session, "csv")

    # Define schema with specific types
    schema = Schema(
        [
            ColumnField(name="name", data_type=StringType),
            ColumnField(name="age", data_type=IntegerType),
            ColumnField(name="score", data_type=DoubleType),
        ]
    )

    # Read with schema
    df = local_session.read.csv(test_file_path, schema=schema)

    # Verify schema
    df_schema = df.schema
    assert df_schema.column_fields == [
        ColumnField("name", StringType),
        ColumnField("age", IntegerType),
        ColumnField("score", DoubleType),
    ]
    # Verify data
    result = df.to_polars()
    data = result.to_dict(as_series=False)

    assert data["name"] == ["Alice", "Bob", "Carol"]
    assert data["age"] == [25, 30, 28]
    assert data["score"] == [95.5, 88.3, 91.7]

    # Edge case: Incompatible schema types (numeric to boolean conversion not possible)
    with pytest.raises(PlanError, match="Schema mismatch"):
        schema = Schema(
            [
                ColumnField(name="name", data_type=StringType),
                ColumnField(name="age", data_type=IntegerType),
                ColumnField(
                    name="score", data_type=BooleanType
                ),  # Not castable to boolean
            ]
        )
        df = local_session.read.csv(test_file_path, schema=schema)

    # Test with unsupported data type in explicit schema
    with pytest.raises(ValidationError, match="CSV files only support primitive data types in schema definitions."):
        schema = Schema(
            [
                ColumnField(name="name", data_type=ArrayType(StringType)),
                ColumnField(name="age", data_type=IntegerType),
                ColumnField(
                    name="score", data_type=BooleanType
                ),
            ]
        )
        df = local_session.read.csv(test_file_path, schema=schema)



def test_csv_merge_schemas_true_union_columns(local_session, temp_dir):
    """Test CSV schema merging with merge_schemas=True.

    This tests:
    - Union of columns across files with different schemas
    - Type inference
    """
    # Create test files with different columns and overlapping columns with different types
    file1_data = """id1,id2,name,age,truthy
1,1,Alice,25,true
2,2,Bob,30,false"""

    file2_data = """id1,id2,city,country,truthy
3,cat,New York,USA,True
4,dog,London,UK,False"""

    file1_path = f"{temp_dir.path}/test_csv_merge_schemas_true_union_columns_1.csv"
    file2_path = f"{temp_dir.path}/test_csv_merge_schemas_true_union_columns_2.csv"

    write_test_file(file1_path, file1_data, local_session, "csv")
    write_test_file(file2_path, file2_data, local_session, "csv")

    # Read with merge_schemas=True
    df = local_session.read.csv([file1_path, file2_path], merge_schemas=True)

    # Verify schema contains all columns with correct types
    schema = df.schema
    assert schema.column_fields == [
        ColumnField("id1", IntegerType),
        # Type widening occurs: numeric + string -> string
        ColumnField("id2", StringType),
        ColumnField("name", StringType),
        ColumnField("age", IntegerType),
        ColumnField("truthy", BooleanType),
        ColumnField("city", StringType),
        ColumnField("country", StringType),
    ]

    # Verify merged data
    result = df.to_polars()
    data = result.to_dict(as_series=False)
    assert data["id1"] == [1, 2, 3, 4]
    assert data["name"] == ["Alice", "Bob", None, None]  # Nulls for missing values
    assert data["age"] == [25, 30, None, None]  # Nulls for missing values
    assert data["city"] == [
        None,
        None,
        "New York",
        "London",
    ]  # Nulls for missing values
    assert data["country"] == [None, None, "USA", "UK"]  # Nulls for missing values
    assert data["id2"] == ["1", "2", "cat", "dog"]  # Type widened to string
    assert result.height == 4, f"Expected 4 rows, got {result.height}"

    # Verify that merge_schemas=False fails when schemas don't match
    with pytest.raises(PlanError, match="Failed to infer schema"):
        df = local_session.read.csv([file1_path, file2_path], merge_schemas=False)


def test_csv_merge_schemas_false_first_file_determines_columns(local_session, temp_dir):
    """Test CSV schema behavior with merge_schemas=False (default).

    This tests:
    - Schema inference
    - Order dependency (first file determines schema)
    """
    file1_data = """id1,id2
1,2
3,4"""
    file2_data = """id1,id2,city
5,6,New York
7,8,London"""

    file1_path = (
        f"{temp_dir.path}/test_csv_merge_schemas_false_first_file_determines_columns_1.csv"
    )
    file2_path = (
        f"{temp_dir.path}/test_csv_merge_schemas_false_first_file_determines_columns_2.csv"
    )

    write_test_file(file1_path, file1_data, local_session, "csv")
    write_test_file(file2_path, file2_data, local_session, "csv")

    df = local_session.read.csv([file1_path, file2_path], merge_schemas=False)
    schema = df.schema
    assert schema.column_fields == [
        ColumnField("id1", IntegerType),
        ColumnField("id2", IntegerType),
    ]
    result = df.to_polars()
    data = result.to_dict(as_series=False)
    assert data["id1"] == [1, 3, 5, 7]
    assert data["id2"] == [2, 4, 6, 8]

    df = local_session.read.csv(
        [
            f"{temp_dir.path}/test_csv_merge_schemas_false_first_file_determines_columns_*.csv"
        ],
        merge_schemas=False,
    )
    schema = df.schema
    assert schema.column_fields == [
        ColumnField("id1", IntegerType),
        ColumnField("id2", IntegerType),
    ]

    file1_data = """id1,id2
1,2
3,4"""
    file2_data = """id1
5
7"""

    file1_path = (
        f"{temp_dir.path}/test_csv_merge_schemas_false_first_file_determines_columns_1.csv"
    )
    file2_path = (
        f"{temp_dir.path}/test_csv_merge_schemas_false_first_file_determines_columns_2.csv"
    )

    write_test_file(file1_path, file1_data, local_session, "csv")
    write_test_file(file2_path, file2_data, local_session, "csv")

    with pytest.raises(PlanError, match="Failed to infer schema"):
        df = local_session.read.csv([file1_path, file2_path], merge_schemas=False)


def test_csv_merge_schemas_false_type_inference(local_session, temp_dir):
    """Test CSV schema behavior with merge_schemas=False (default).

    This tests:
    - Schema inference
    - Type widening behavior
    - Order dependency (first file determines schema)
    """
    # Create test files with same column names but different types
    file1_data = """id, id2
1, 1
2, 1.0"""

    file2_data = """id, id2
cat, 1.1
dog, 1.2"""

    file1_path = f"{temp_dir.path}/test_csv_merge_schemas_false_type_inference_1.csv"
    file2_path = f"{temp_dir.path}/test_csv_merge_schemas_false_type_inference_2.csv"

    write_test_file(file1_path, file1_data, local_session, "csv")
    write_test_file(file2_path, file2_data, local_session, "csv")

    # Case 1: First file has integers, second has mixed types
    # This works by widening integers to strings
    df = local_session.read.csv([file1_path, file2_path], merge_schemas=False)
    schema = df.schema
    assert schema.column_fields == [
        ColumnField("id", StringType),
        ColumnField("id2", DoubleType),
    ]

    # Case 2: Reverse order does not affect schema inference for csv files
    df = local_session.read.csv([file2_path, file1_path])
    schema = df.schema
    assert schema.column_fields == [
        ColumnField("id", StringType),
        ColumnField("id2", DoubleType),
    ]

def test_parquet_merge_schemas_true_union_columns(local_session, temp_dir):
    """Test Parquet schema merging with merge_schemas=True.

    This tests:
    - Union of columns across Parquet files with different schemas
    """
    data1 = {"id": [1, 2], "id2": ["1", "2"]}
    data2 = {"id": [3, 4], "id3": ["cat", "dog"]}

    file1_path = f"{temp_dir.path}/test_parquet_merge_schemas_true_union_columns_1.parquet"
    file2_path = f"{temp_dir.path}/test_parquet_merge_schemas_true_union_columns_2.parquet"

    # Create and write parquet files
    write_test_file(file1_path, pl.DataFrame(data1), local_session, "parquet")
    write_test_file(file2_path, pl.DataFrame(data2), local_session, "parquet")

    # Read with merge_schemas=True
    df = local_session.read.parquet([file1_path, file2_path], merge_schemas=True)

    # Verify schema with correct types
    schema = df.schema
    assert schema.column_fields == [
        ColumnField("id", IntegerType),
        ColumnField("id2", StringType),
        ColumnField("id3", StringType),
    ]
    # assert None for missing values
    result = df.to_polars()
    data = result.to_dict(as_series=False)
    assert data["id"] == [1, 2, 3, 4]
    assert data["id2"] == ["1", "2", None, None]
    assert data["id3"] == [None, None, "cat", "dog"]
    with pytest.raises(PlanError, match="Failed to infer schema"):
        df = local_session.read.parquet([file1_path, file2_path], merge_schemas=False)


def test_parquet_merge_schemas_true_type_widening(local_session, temp_dir):
    """Test Parquet schema merging with merge_schemas=True.

    This tests:
    - Type consistency and widening
    - Handling of null values for missing columns
    """
    # Create test data and write as parquet files with overlapping columns
    data1 = {"id": [1, 2], "id2": ["1", "2"]}
    data2 = {"id": ["3", "4"], "id2": [3, 4]}

    file1_path = f"{temp_dir.path}/test_parquet_merge_schemas_true_type_widening_1.parquet"
    file2_path = f"{temp_dir.path}/test_parquet_merge_schemas_true_type_widening_2.parquet"

    # Create and write parquet files
    write_test_file(file1_path, pl.DataFrame(data1), local_session, "parquet")
    write_test_file(file2_path, pl.DataFrame(data2), local_session, "parquet")

    # Read with merge_schemas=True
    df = local_session.read.parquet([file1_path, file2_path], merge_schemas=True)

    # Verify schema with correct types
    schema = df.schema
    assert schema.column_fields == [
        ColumnField("id", StringType),
        ColumnField("id2", StringType),
    ]

    # Verify merged data
    result = df.to_polars()
    data = result.to_dict(as_series=False)
    assert data["id"] == ["1", "2", "3", "4"]
    assert data["id2"] == ["1", "2", "3", "4"]
    assert result.height == 4, f"Expected 4 rows, got {result.height}"


def test_parquet_merge_schemas_false_first_file_determines_schema(local_session, temp_dir):
    """Test Parquet schema behavior with merge_schemas=False (default).

    This tests:
    - First file determines schema and subsequent files are cast if possible to match the schema
    """
    data1 = {"id": [1, 2], "id2": ["1", "2"]}
    data2 = {"id": ["3", "4"], "id2": [3, 4], "id3": ["cat", "dog"]}

    file1_path = f"{temp_dir.path}/test_parquet_merge_schemas_false_first_file_determines_schema_1.parquet"
    file2_path = f"{temp_dir.path}/test_parquet_merge_schemas_false_first_file_determines_schema_2.parquet"

    write_test_file(file1_path, pl.DataFrame(data1), local_session, "parquet")
    write_test_file(file2_path, pl.DataFrame(data2), local_session, "parquet")

    df = local_session.read.parquet([file1_path, file2_path])
    schema = df.schema
    assert schema.column_fields == [
        ColumnField("id", IntegerType),
        ColumnField("id2", StringType),
    ]
    df = local_session.read.parquet(
        [
            f"{temp_dir.path}/test_parquet_merge_schemas_false_first_file_determines_schema_*.parquet"
        ]
    )
    schema = df.schema
    assert schema.column_fields == [
        ColumnField("id", IntegerType),
        ColumnField("id2", StringType),
    ]


    data1 = {"id": [1, 2], "id2": ["1", "2"]}
    data2 = {"id": ["cat", "dog"], "id2": [3, 4]}

    file1_path = f"{temp_dir.path}/test_parquet_merge_schemas_false_first_file_determines_schema_1.parquet"
    file2_path = f"{temp_dir.path}/test_parquet_merge_schemas_false_first_file_determines_schema_2.parquet"

    write_test_file(file1_path, pl.DataFrame(data1), local_session, "parquet")
    write_test_file(file2_path, pl.DataFrame(data2), local_session, "parquet")

    with pytest.raises(PlanError, match="Failed to infer schema"):
        df = local_session.read.parquet([file1_path, file2_path])


    data1 = {"id": [1, 2], "id2": ["1", "2"]}
    data2 = {"id2": [3, 4]}

    file1_path = f"{temp_dir.path}/parquet_merge_test1.parquet"
    file2_path = f"{temp_dir.path}/parquet_merge_test2.parquet"

    write_test_file(file1_path, pl.DataFrame(data1), local_session, "parquet")
    write_test_file(file2_path, pl.DataFrame(data2), local_session, "parquet")

    with pytest.raises(PlanError, match="Failed to infer schema"):
        df = local_session.read.parquet([file1_path, file2_path])


# =============================================================================
# Schema Validation and Error Handling Tests
# =============================================================================


def test_csv_incompatible_schema_error(local_session, temp_dir):
    """Test CSV reader error handling for schema incompatibilities.

    This tests:
    - Type overriding with explicit schema
    - Error handling for incomplete schemas
    """
    csv_data = """name,age,city
Alice,25,New York
Bob,30,Chicago"""

    test_file_path = f"{temp_dir.path}/incompatible.csv"
    write_test_file(test_file_path, csv_data, local_session, "csv")

    # Case 1: Override inferred types with explicit schema
    schema = Schema(
        [
            ColumnField(name="name", data_type=StringType),
            ColumnField(
                name="age", data_type=StringType
            ),  # Override age to be string instead of integer
            ColumnField(name="city", data_type=StringType),
        ]
    )

    df = local_session.read.csv(test_file_path, schema=schema)
    result = df.to_polars()

    data = result.to_dict(as_series=False)
    assert isinstance(
        data["age"][0], str
    ), "Age should be a string due to schema override"

    # Case 2: Error on incomplete schema (missing columns)
    with pytest.raises(PlanError, match="Schema mismatch"):
        missing_schema = Schema(
            [
                ColumnField(name="name", data_type=StringType),
                ColumnField(name="age", data_type=IntegerType),
                # Missing city column
            ]
        )
        df = local_session.read.csv(test_file_path, schema=missing_schema)


def test_csv_invalid_options_combination(local_session, temp_dir):
    """Test error handling for invalid option combinations.

    This tests:
    - Error when both schema and merge_schemas are provided
    """
    csv_data = """name,age,city
Alice,25,New York
Bob,30,Chicago"""

    test_file_path = f"{temp_dir.path}/incompatible.csv"
    write_test_file(test_file_path, csv_data, local_session, "csv")

    with pytest.raises(
        ValidationError, match="Cannot specify both 'schema' and 'merge_schemas=True' - these options conflict."
    ):
        schema = {"name": StringType, "age": IntegerType}
        local_session.read.csv(test_file_path, schema=schema, merge_schemas=True)


# =============================================================================
# AWS Credentials Tests
# =============================================================================

def test_read_query_setup_with_aws_credentials(local_session_config, monkeypatch):
    """Test that a local session can be created with AWS credentials.

    test that our read queries are setup with the credentials.
    test that we error out if we don't have credentials.
    """
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test_access_key_id")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test_secret_access_key")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "test_session_token")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")

    session = Session.get_or_create(local_session_config)

    # Get configured credentials from session
    aws_credentials = session._session_state.s3_session.get_credentials()
    frozen_credentials = aws_credentials.get_frozen_credentials()
    access_key = frozen_credentials.access_key
    secret_key = frozen_credentials.secret_key
    token = frozen_credentials.token
    region = session._session_state.s3_session.region_name
 
    # Test that read queries to s3 have the configured credentials
    paths = ["s3://test-bucket/test-file.csv"]
    query = session._session_state.execution._build_read_csv_query(
        paths,
        infer_schema =True,
    )
    query = _build_query_with_httpfs_extensions(query)
    query = _build_query_with_s3_creds(query, session._session_state.s3_session)
    assert "http"
    assert f"SET s3_access_key_id='{access_key}'" in query
    assert f"SET s3_secret_access_key='{secret_key}'" in query
    assert f"SET s3_session_token='{token}'" in query
    assert f"SET s3_region='{region}'" in query

    session.stop()

def test_read_queries_with_no_aws_credentials(local_session_config, temp_dir):
    """Test that local read queries work and that read queries to s3 will fail without aws credentials."""
    session = Session.get_or_create(local_session_config)
    # remove credentials from session and patch the fenic session
    botocore_session = get_session()
    botocore_session.set_credentials(None, None)
    session._session_state.s3_session = boto3.Session(botocore_session=botocore_session)

    # Test that local file csv and parquet queries will succeed without credentials
    # skip if tests are running on s3
    if urlparse(temp_dir.path).scheme != "s3":
        test_file_path = f"{temp_dir.path}/test.csv"
        write_test_file(test_file_path, "name,age\nAlice,25\nBob,30", session, "csv")
        df = session.read.csv(test_file_path)
        result = df.to_polars()
        assert result.height == 2, f"Expected 2 rows, got {result.height}"

        test_file_path = f"{temp_dir.path}/test.parquet"
        write_test_file(test_file_path, pl.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]}), session, "parquet")
        df = session.read.parquet(test_file_path)
        result = df.to_polars()
        assert result.height == 2, f"Expected 2 rows, got {result.height}"

    # Test that read queries to s3 will fail without credentials
    with pytest.raises(PlanError, match="Failed to infer schema from CSV files") as exc_info:
        session.read.csv("s3://test-bucket/test-file.csv")
    assert isinstance(exc_info.value.__cause__, ConfigurationError)
    assert str(exc_info.value.__cause__) == "Unable to locate AWS credentials."

    with pytest.raises(PlanError, match="Failed to infer schema from Parquet files") as exc_info:
        session.read.parquet("s3://test-bucket/test-file.parquet")
    assert isinstance(exc_info.value.__cause__, ConfigurationError)
    assert str(exc_info.value.__cause__) == "Unable to locate AWS credentials."

    session.stop()

# =============================================================================
# HuggingFace Credentials Tests
# =============================================================================

def test_read_queries_with_invalid_huggingface_credentials(local_session_config, monkeypatch):
    """Test that read queries to private huggingface datasets will fail without hf credentials."""
    session = Session.get_or_create(local_session_config)
    paths = ["hf://datasets/typedef-ai/fenic-test-datasets-private/last_names_1.csv"]

    # Test with no token
    if os.getenv("HF_TOKEN"):
        monkeypatch.delenv("HF_TOKEN")

    with pytest.raises(PlanError, match="Failed to infer schema from CSV files") as exc_info:
        session.read.csv(paths[0])
    assert isinstance(exc_info.value.__cause__, ConfigurationError)
    assert str(exc_info.value.__cause__) == "HuggingFace token not found. Set HF_TOKEN environment variable."

    # Test with invalid token
    monkeypatch.setenv("HF_TOKEN", "invalid_token")
    with pytest.raises(PlanError, match="Failed to infer schema from CSV files") as exc_info:
        session.read.csv(paths[0])
    assert isinstance(exc_info.value.__cause__, HTTPException)
    assert str(exc_info.value.__cause__) == "HTTP Error: HTTP GET error on 'https://huggingface.co/datasets/typedef-ai/fenic-test-datasets-private/resolve/main/last_names_1.csv' (HTTP 401)"


def test_read_query_setup_with_huggingface_credentials(local_session_config, monkeypatch):
    """Test that read queries to huggingface datasets will succeed with hf credentials."""
    monkeypatch.setenv("HF_TOKEN", "test_token")
    session = Session.get_or_create(local_session_config)

    # Test that read queries to s3 have the configured credentials
    paths = ["hf://datasets/typedef-ai/fenic-test-datasets-private/last_names_1.csv"]
    query = session._session_state.execution._build_read_csv_query(
        paths,
        infer_schema=True,
    )
    query = _build_query_with_httpfs_extensions(query)
    query = _build_query_with_hf_creds(query)
    assert "INSTALL httpfs; LOAD httpfs;" in query
    assert "CREATE SECRET hf_token (TYPE HUGGINGFACE, TOKEN 'test_token');" in query


def test_read_query_setup_with_huggingface_credentials_and_s3_credentials(local_session_config, monkeypatch):
    """Test that read queries to huggingface datasets will succeed with hf credentials."""
    monkeypatch.setenv("HF_TOKEN", "test_token")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test_access_key_id")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test_secret_access_key")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "test_session_token")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")

    session = Session.get_or_create(local_session_config)

    # Get configured credentials from session
    aws_credentials = session._session_state.s3_session.get_credentials()
    frozen_credentials = aws_credentials.get_frozen_credentials()
    access_key = frozen_credentials.access_key
    secret_key = frozen_credentials.secret_key
    token = frozen_credentials.token
    region = session._session_state.s3_session.region_name
    
    # Test that read queries to s3 have the configured credentials
    paths = ["hf://datasets/typedef-ai/fenic-test-datasets-private/last_names_1.csv"]
    query = session._session_state.execution._build_read_csv_query(
        paths,
        infer_schema=True,
    )
    query = _build_query_with_httpfs_extensions(query)
    query = _build_query_with_s3_creds(query, session._session_state.s3_session)
    query = _build_query_with_hf_creds(query)
    assert "INSTALL httpfs; LOAD httpfs;" in query
    assert "CREATE SECRET hf_token (TYPE HUGGINGFACE, TOKEN 'test_token');" in query
    assert f"SET s3_access_key_id='{access_key}'" in query
    assert f"SET s3_secret_access_key='{secret_key}'" in query
    assert f"SET s3_session_token='{token}'" in query
    assert f"SET s3_region='{region}'" in query

# =============================================================================
# HuggingFace Test Reads on Public Datasets
# =============================================================================

TEST_CSV_SCHEMA = Schema(column_fields=[ColumnField(name="last_name", data_type=StringType)])
TEST_PARQUET_SCHEMA = Schema(column_fields=[
    ColumnField(name="id", data_type=IntegerType),
    ColumnField(name="name", data_type=StringType),
    ColumnField(name="occupation", data_type=StringType),
    ColumnField(name="city", data_type=StringType),
    ColumnField(name="age", data_type=IntegerType),
    ColumnField(name="salary", data_type=IntegerType),
    ColumnField(name="years_experience", data_type=IntegerType),
    ColumnField(name="is_remote", data_type=BooleanType),
    ColumnField(name="salary_tier", data_type=StringType),
    ColumnField(name="experience_category", data_type=StringType),
    ColumnField(name="location_type", data_type=StringType),
])

def test_read_public_huggingface_datasets(request, local_session_config, temp_dir):
    """Test that read queries to public huggingface datasets will succeed without hf credentials."""
    if not request.config.getoption("--test-huggingface-reads"):
        pytest.skip("Skipping HuggingFace tests. To test these, run with --test-huggingface-reads flag")

    session = Session.get_or_create(local_session_config)

    # Test with single csv file
    csv_path = "hf://datasets/typedef-ai/fenic-test-datasets-public/last_names_1.csv"
    df = session.read.csv(csv_path)
    assert df.schema == TEST_CSV_SCHEMA, f"Expected {TEST_CSV_SCHEMA}, got {df.schema}"
    assert df.count() == 10


    # Test with single parquet file
    parquet_path = "hf://datasets/typedef-ai/fenic-test-datasets-public/names_and_occupations_3.parquet"
    df = session.read.parquet(parquet_path)
    assert df.schema == TEST_PARQUET_SCHEMA, f"Expected {TEST_PARQUET_SCHEMA}, got {df.schema}"
    assert df.count() == 10

    # Test with multiple parquet files
    parquet_paths = [
        "hf://datasets/typedef-ai/fenic-test-datasets-public/names_and_occupations_1.parquet",
        "hf://datasets/typedef-ai/fenic-test-datasets-public/names_and_occupations_2.parquet",
        "hf://datasets/typedef-ai/fenic-test-datasets-public/names_and_occupations_3.parquet",
    ]
    df = session.read.parquet(parquet_paths)
    assert df.count() == 30
    assert df.schema == TEST_PARQUET_SCHEMA

    # Test multiple parquet files with glob 
    parquet_path = "hf://datasets/typedef-ai/fenic-test-datasets-public/names_and_occupations_*.parquet"
    df = session.read.parquet(parquet_path)
    assert df.count() == 30
    assert df.schema == TEST_PARQUET_SCHEMA

    # Test with a parquet file at commit 
    parquet_path = "hf://datasets/typedef-ai/fenic-test-datasets-public@58138e6c12e70033f6e36307c7a149a1c13dfc38/names_and_occupations_2.parquet"
    df = session.read.parquet(parquet_path)
    assert df.count() == 10
    assert df.schema == TEST_PARQUET_SCHEMA


def test_read_public_huggingface_datasets_with_other_sources(request, local_session_config, temp_dir):
    """Test that read queries to public huggingface datasets will succeed with other schemes."""
    if not request.config.getoption("--test-huggingface-reads"):
        pytest.skip("Skipping HuggingFace tests. To test these, run with --test-huggingface-reads flag")

    session = Session.get_or_create(local_session_config)
    test_local_csv_data = """last_name
Jones
Smith
Francis
"""
    write_test_file(f"{temp_dir.path}/test.csv", test_local_csv_data, session, "csv")
    paths = ["hf://datasets/typedef-ai/fenic-test-datasets-public/last_names_1.csv", f"{temp_dir.path}/test.csv"]
    df = session.read.csv(paths)
    assert df.count() == 13
    assert df.schema == TEST_CSV_SCHEMA

# =============================================================================
# Ingestion Type Coercions Tests
# =============================================================================


def test_ingest_date_type(local_session, temp_dir):
    """Test automatic conversion of date columns to strings.

    This tests:
    - Date column representation as strings
    - Filtering on date string values
    - Consistency across read methods (parquet vs in-memory)
    """
    PARQUET_FILE_NAME = f"{temp_dir.path}/test.parquet"
    DATE_COLUMN_NAME = "some_date"
    CSV_FILE_NAME = f"{temp_dir.path}/test.csv"

    # Create a dataframe with a date column
    df = pl.DataFrame(
        {
            "month": [1, 2, 3],
            "day": [4, 5, 6],
        }
    )
    polars_df = df.with_columns(
        pl.date(2024, pl.col("month"), pl.col("day")).alias(DATE_COLUMN_NAME)
    )
    write_test_file(PARQUET_FILE_NAME, polars_df, local_session, "parquet")

    # Test 1: Reading from Parquet file
    fenic_df = local_session.read.parquet(PARQUET_FILE_NAME)
    fenic_df = fenic_df.filter(col(DATE_COLUMN_NAME) == "2024-01-04")
    result = fenic_df.to_polars()

    # Verify schema and type conversion
    expected_schema = pl.Schema(
        {"month": pl.Int64, "day": pl.Int64, DATE_COLUMN_NAME: pl.String}
    )
    assert result.schema == expected_schema, "Date should be converted to String"
    assert result[DATE_COLUMN_NAME].to_list() == ["2024-01-04"]

    # Test 2: Using in-memory dataframe
    fenic_df = local_session.create_dataframe(polars_df)
    fenic_df = fenic_df.filter(col(DATE_COLUMN_NAME) == "2024-01-04")
    result = fenic_df.to_polars()

    # Test 3: CSV file
    polars_df.write_csv(CSV_FILE_NAME)
    fenic_df = local_session.read.csv(CSV_FILE_NAME)
    fenic_df = fenic_df.filter(col(DATE_COLUMN_NAME) == "2024-01-04")
    result = fenic_df.to_polars()
    assert result[DATE_COLUMN_NAME].to_list() == ["2024-01-04"]

    assert result.schema == expected_schema, "Date should be converted to String"
    assert result[DATE_COLUMN_NAME].to_list() == ["2024-01-04"]


def test_ingest_datetime_type(local_session, temp_dir):
    """Test automatic conversion of datetime columns to strings.

    This tests:
    - Datetime column representation as strings
    - Filtering on datetime string values
    - Consistency across read methods (parquet vs in-memory)
    """
    PARQUET_FILE_NAME = f"{temp_dir.path}/test.parquet"
    DATETIME_COLUMN_NAME = "some_datetime"
    CSV_FILE_NAME = f"{temp_dir.path}/test.csv"

    # Create a dataframe with a datetime column
    df = pl.DataFrame(
        {
            "month": [1, 2, 3],
            "day": [4, 5, 6],
            "hour": [7, 8, 9],
            "minute": [10, 11, 12],
            "second": [13, 14, 15],
        }
    )
    polars_df = df.with_columns(
        pl.datetime(
            2024,
            pl.col("month"),
            pl.col("day"),
            pl.col("hour"),
            pl.col("minute"),
            pl.col("second"),
        ).alias(DATETIME_COLUMN_NAME)
    )
    write_test_file(PARQUET_FILE_NAME, polars_df, local_session, "parquet")

    # Test 1: Reading from Parquet file
    fenic_df = local_session.read.parquet(PARQUET_FILE_NAME)
    fenic_df = fenic_df.filter(
        col(DATETIME_COLUMN_NAME) == "2024-01-04 07:10:13.000000"
    )
    result = fenic_df.to_polars()

    expected_schema = pl.Schema(
        {
            "month": pl.Int64,
            "day": pl.Int64,
            "hour": pl.Int64,
            "minute": pl.Int64,
            "second": pl.Int64,
            DATETIME_COLUMN_NAME: pl.String,
        }
    )
    assert (
        result.schema == expected_schema
    ), "Datetime should be converted to String"
    assert result[DATETIME_COLUMN_NAME].to_list() == ["2024-01-04 07:10:13.000000"]

    # Test 2: Using in-memory dataframe
    fenic_df = local_session.create_dataframe(polars_df)
    fenic_df = fenic_df.filter(
        col(DATETIME_COLUMN_NAME) == "2024-01-04 07:10:13.000000"
    )
    result = fenic_df.to_polars()

    assert (
        result.schema == expected_schema
    ), "Datetime should be converted to String"
    assert result[DATETIME_COLUMN_NAME].to_list() == ["2024-01-04 07:10:13.000000"]

    # Test 3: CSV file
    write_test_file(CSV_FILE_NAME, polars_df, local_session, "csv")
    fenic_df = local_session.read.csv(CSV_FILE_NAME)
    fenic_df = fenic_df.filter(
        col(DATETIME_COLUMN_NAME) == "2024-01-04 07:10:13.000000"
    )
    result = fenic_df.to_polars()
    assert result[DATETIME_COLUMN_NAME].to_list() == ["2024-01-04 07:10:13.000000"]

def test_ingest_array_type(local_session, temp_dir):
    """Test automatic conversion of array columns to lists."""
    PARQUET_FILE_NAME = f"{temp_dir.path}/test.parquet"
    ARRAY_COLUMN_NAME = "array_column"

    # Create a dataframe with an array column
    df = pl.DataFrame(
        {
            "list_column": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        }
    )
    polars_df = df.with_columns(
        pl.col("list_column").cast(pl.Array(pl.Float32, 3)).alias(ARRAY_COLUMN_NAME)
    )
    write_test_file(PARQUET_FILE_NAME, polars_df, local_session, "parquet")

    # Test 1: Reading from Parquet file
    fenic_df = local_session.read.parquet(PARQUET_FILE_NAME)
    assert fenic_df.schema.column_fields == [
        ColumnField(name="list_column", data_type=ArrayType(IntegerType)),
        ColumnField(name="array_column", data_type=ArrayType(FloatType)),
    ]
    result = fenic_df.to_polars()
    assert result.schema == pl.Schema(
        {
            "list_column": pl.List(pl.Int64),
            "array_column": pl.List(pl.Float32),
        }
    )

    fenic_df = local_session.create_dataframe(polars_df)
    result = fenic_df.to_polars()
    assert result.schema == pl.Schema(
        {
            "list_column": pl.List(pl.Int64),
            "array_column": pl.List(pl.Float32),
        }
    )


def test_nested_views(local_session, temp_dir):
    """Test that views can be unioned and that the unioned view can be queried."""
    test_data = """name,age,city
John,25,New York
David,33,Seattle"""

    test_file_path = f"{temp_dir.path}/test_csv_single_file.csv"
    write_test_file(test_file_path, test_data, local_session, "csv")

    df_csv = local_session.read.csv(test_file_path)
    df_csv.write.save_as_view("view")
    df_csv_view = local_session.view("view")
    df_csv_view.write.save_as_view("view_nested")
    df_view_nested = local_session.view("view_nested")

    table_name = "test_overwrite_table"
    df = local_session.create_dataframe([{"name": "Alice", "age": 30, "city": "SF"},
                                          {"name": "Carol", "age": 34, "city": "Boston"}])
    df.write.save_as_table(table_name, mode="overwrite")
    df_table = local_session.table(table_name)
    df_table.write.save_as_view("df_view")

    df_view = local_session.view("df_view")

    df_union = df_view_nested.union(df_view)
    df_union.write.save_as_view("df_view_union")

    df_view_union = local_session.view("df_view_union")
    assert df_view_union.columns == ["name", "age", "city"]

    result_name = df_view_union.select(col("name")).collect("polars").data
    values_name = result_name["name"].to_list()
    assert values_name == ["John", "David", "Alice", "Carol"]

def test_view_schema_validation(local_session, temp_dir):
    """Test that views are validated against the current state of the source."""
    test_data = """name,age,city
John,25,New York
David,33,Seattle"""

    test_file_path = f"{temp_dir.path}/test_csv_single_file.csv"
    write_test_file(test_file_path, test_data, local_session, "csv")

    df_csv = local_session.read.csv(test_file_path)
    df_csv.write.save_as_view("df_csv_view")
    df_csv_view = local_session.view("df_csv_view")
    assert df_csv_view.schema == df_csv.schema

    # Make a quick change in the file.
    test_data_2 = """name,age
John,25
David,33"""
    write_test_file(test_file_path, test_data_2, local_session, "csv")
    with pytest.raises(PlanError):
        local_session.view("df_csv_view")


def test_read_docs(local_session, temp_dir_with_test_files):
    """Test that reading from a folder works."""
    df = local_session.read.docs(
        get_globbed_path(temp_dir_with_test_files, "**/*.md"),
        data_type=MarkdownType,
        recursive=True)
    df.collect()
    assert df.schema == Schema(
        [
            ColumnField(name="file_path", data_type=StringType),
            ColumnField(name="error", data_type=StringType),
            ColumnField(name="content", data_type=MarkdownType),
        ]
    )
    # generate the toc
    df = df.select(
        col("file_path"),
        markdown.generate_toc(col("content")).alias("toc")
    )
    dict = df.to_pydict()
    assert len(dict["file_path"]) == 5
    assert "2 Background" in dict["toc"][0]


def test_read_docs_invalid_path(local_session):
    """Test that reading from an invalid path fails."""
    with pytest.raises(ValidationError):
        local_session.read.docs(
            "/invalid/path",
            data_type=MarkdownType,
            recursive=True)


def test_read_docs_invalid_type(local_session, temp_dir_with_test_files):
    """Test that reading from an invalid path fails."""
    with pytest.raises(UnsupportedFileTypeError):
        local_session.read.docs(
            get_globbed_path(temp_dir_with_test_files, "**/*.md"),
            data_type=StringType,
            recursive=True)

def test_read_docs_no_wildcard_only_valid_files(local_session, temp_dir_just_one_file):
    """Test that reading from a path with (and no wild card) only valid files works."""
    df = local_session.read.docs(
        [temp_dir_just_one_file],
        data_type=MarkdownType)
    df.collect()
    dict = df.to_pydict()
    assert len(dict["file_path"]) == 1

def test_read_docs_no_files_valid_paths(local_session, temp_dir_with_test_files):
    """Test that if no files are found, we'll get a dataframe with the path and an error message."""
    df = local_session.read.docs(
        get_globbed_path(temp_dir_with_test_files, "**/*.unknown_extension"),
        data_type=MarkdownType,
        recursive=True)
    df.collect()
    results = df.to_pydict()
    assert len(results["file_path"]) == 0

def test_read_docs_no_wildcard_path_is_file(local_session, temp_dir_just_one_file):
    """Test that reading from a path to a file works."""
    df = local_session.read.docs(
        [str(Path.joinpath(Path(temp_dir_just_one_file), "file1.md"))],
        data_type=MarkdownType)
    df.collect()
    dict = df.to_pydict()
    assert len(dict["file_path"]) == 1
    assert "file1.md" in dict["file_path"][0]

def get_globbed_path(path: str, file_extension: str) -> list[str]:
    return [str(Path.joinpath(Path(path), file_extension))]
