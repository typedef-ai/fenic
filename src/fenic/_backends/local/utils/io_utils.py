import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple
from urllib.parse import urlparse

import duckdb
import polars as pl
from boto3.session import Session as BotoSession
from botocore.credentials import ReadOnlyCredentials

from fenic.core.error import (
    ConfigurationError,
    FileLoaderError,
    InternalError,
    ValidationError,
)
from fenic.core.types.datatypes import (
    BooleanType,
    DateType,
    DoubleType,
    FloatType,
    IntegerType,
    StringType,
    TimestampType,
)
from fenic.core.types.schema import Schema

logger = logging.getLogger(__name__)

@dataclass
class ReadQueryResult:
    sql: str
    s3_status: Optional[bool] = None  # None = no S3 paths, True = has creds, False = no creds
    hf_status: Optional[bool] = None  # None = no HF paths, True = has creds, False = no creds


@dataclass
class WriteQueryResult:
    sql: str
    is_s3: bool = False  # Whether writing to S3 (credentials are always required for writes)

class PathScheme(Enum):
    S3 = "s3"
    HF = "hf"
    LOCALFS = "localfs"

def does_path_exist(path: str, s3_session: BotoSession) -> bool:
    """Check if a s3, hf, or local path exists."""
    scheme = urlparse(path).scheme
    if scheme == "s3":
        _, _ = _fetch_and_validate_s3_credentials(s3_session)
        s3 = s3_session.client("s3")
        parsed = urlparse(path)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        try:
            s3.head_object(Bucket=bucket, Key=key)
            return True
        except s3.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise
    elif scheme == "file" or scheme == "":
        return os.path.exists(path)
    else:
        raise ValidationError(f"Unsupported file type: {scheme} for path: {path}.  Please use s3 scheme ('s3://'), hf scheme ('hf://'), or local scheme ('file://' or no prefix).")

def query_files(
    paths: List[str],
    file_type: Literal["csv", "parquet"],
    s3_session: BotoSession,
    **options
) -> pl.DataFrame:
    """Execute a file read query with proper error handling."""
    duckdb_conn = _configure_duckdb_conn(duckdb.connect())
    result = build_read_query(paths, file_type, s3_session, **options)

    try:
        return duckdb_conn.execute(result.sql).pl()
    except duckdb.HTTPException as e:
        logger.debug("DuckDB read query failed for paths=%s: %s", paths, e, exc_info=True)
        message = _format_http_read_error(e, result)
        raise FileLoaderError(message) from e
    except duckdb.Error as e:
        logger.debug("DuckDB read query failed for paths=%s: %s", paths, e, exc_info=True)
        raise FileLoaderError(e) from e

def write_file(
    df: pl.DataFrame,
    path: str,
    s3_session: BotoSession,
    file_type: Literal["csv", "parquet"],
):
    """Write file to local, s3, or hf path using duckdb."""
    duckdb_conn = _configure_duckdb_conn(duckdb.connect())
    arrow_table = df.to_arrow()
    duckdb_conn.register("df_view", arrow_table)
    result = build_write_query(source_table="df_view", path=path, file_type=file_type, s3_session=s3_session)
    try:
        duckdb_conn.execute(result.sql)
    except duckdb.Error as e:
        logger.debug("DuckDB write query failed for path=%s: %s", path, e, exc_info=True)
        if result.is_s3:
            message = f"Failed to write to S3. Ensure AWS credentials permit write access and the bucket/path exists. {e}"
        else:
            message = "Failed to write file. Verify the destination path exists and is writable."
        raise FileLoaderError(message) from e


def configure_duckdb_conn_for_path(db_path: Path) -> duckdb.DuckDBPyConnection:
    """Create a duckdb connection for a given path, applying common configuration."""
    db_conn = duckdb.connect(str(db_path))
    return _configure_duckdb_conn(db_conn)


def get_path_scheme(path: str) -> PathScheme:
    """Gets the path scheme."""
    scheme = urlparse(path).scheme
    if scheme == "s3":
        return PathScheme.S3
    elif scheme == "hf":
        return PathScheme.HF
    else:
        return PathScheme.LOCALFS


def build_read_query(
    paths: List[str],
    file_type: Literal["csv", "parquet"],
    s3_session: BotoSession,
    schema: Optional[Schema] = None,
    merge_schemas: bool = False,
    schema_inference: bool = False
) -> ReadQueryResult:
    """Build a complete SQL query string for reading files."""
    # Check what types of paths we have
    has_s3_paths = any(urlparse(path).scheme == "s3" for path in paths)
    has_hf_paths = any(urlparse(path).scheme == "hf" for path in paths)

    components = []
    s3_status = None
    hf_status = None

    # Add schema inference pragma if needed
    if schema_inference:
        components.append("PRAGMA disable_optimizer")

    # Add httpfs and credentials
    if has_s3_paths or has_hf_paths:
        components.extend(["INSTALL httpfs", "LOAD httpfs"])

        if has_s3_paths:
            try:
                s3_config = _build_s3_config_components(s3_session)
                components.extend(s3_config)
                s3_status = True
            except ConfigurationError:
                logger.warning(
                    "Unable to locate AWS credentials for fetching data from S3 -- will still attempt to do so, "
                    "but the query will fail unless the data is in a public bucket."
                )
                s3_status = False

        if has_hf_paths:
            hf_token = _fetch_hf_token()
            if hf_token:
                components.append(f"CREATE SECRET hf_token (TYPE HUGGINGFACE, TOKEN '{hf_token}')")
                hf_status = True
            else:
                logger.warning(
                    "HuggingFace token not found. Will attempt to read dataset, this will fail if the dataset is private or gated. "
                    "Set HF_TOKEN environment variable to authenticate to HuggingFace."
                )
                hf_status = False

    # Build the main query
    main_query = _build_file_scan_query(paths, file_type, schema, merge_schemas, schema_inference)
    components.append(main_query)

    return ReadQueryResult(
        sql="; ".join(components),
        s3_status=s3_status,
        hf_status=hf_status
    )


def build_write_query(
    source_table: str,
    path: str,
    file_type: Literal["csv", "parquet"],
    s3_session: BotoSession
) -> WriteQueryResult:
    """Build a complete SQL query string for writing files."""
    components = []
    is_s3 = urlparse(path).scheme == "s3"

    # Add httpfs and credentials if needed
    if is_s3:
        components.extend(["INSTALL httpfs", "LOAD httpfs"])

        s3_config = _build_s3_config_components(s3_session)
        components.extend(s3_config)

    # Build the COPY query
    copy_query = _build_copy_query(source_table, path, file_type)
    components.append(copy_query)

    return WriteQueryResult(
        sql="; ".join(components),
        is_s3=is_s3
    )

def _build_s3_config_components(s3_session: BotoSession) -> List[str]:
    """Build S3 configuration SQL statements. Raises ConfigurationError if credentials not found."""
    frozen_creds, region = _fetch_and_validate_s3_credentials(s3_session)
    config = [
        f"SET s3_region='{region}'",
        f"SET s3_access_key_id='{frozen_creds.access_key}'",
        f"SET s3_secret_access_key='{frozen_creds.secret_key}'"
    ]
    if frozen_creds.token:
        config.append(f"SET s3_session_token='{frozen_creds.token}'")
    return config


def _build_file_scan_query(
    paths: List[str],
    file_type: Literal["csv", "parquet"],
    schema: Optional[Schema],
    merge_schemas: bool,
    schema_inference: bool
) -> str:
    """Build the main file scanning query (read_csv or read_parquet)."""
    paths_str = "', '".join(paths)

    if file_type == "csv":
        query = _build_csv_scan_query(paths_str, schema, merge_schemas)
    elif file_type == "parquet":
        query = _build_parquet_scan_query(paths_str, merge_schemas)

    # Add WHERE clause for schema inference
    if schema_inference:
        query = f"{query} WHERE 1=0"

    return query


def _build_csv_scan_query(paths_str: str, schema: Optional[Schema], merge_schemas: bool) -> str:
    """Build a DuckDB read_csv query."""
    # trunk-ignore-begin(bandit/B608)
    if schema:
        duckdb_schema = _convert_schema_to_duckdb_dict(schema)
        duckdb_schema_string = json.dumps(duckdb_schema).replace('"', "'")
        return f"SELECT * FROM read_csv(['{paths_str}'], columns = {duckdb_schema_string})"
    elif merge_schemas:
        return f"SELECT * FROM read_csv(['{paths_str}'], union_by_name=true)"
    else:
        return f"SELECT * FROM read_csv(['{paths_str}'])"
    # trunk-ignore-end(bandit/B608)


def _build_parquet_scan_query(paths_str: str, merge_schemas: bool) -> str:
    """Build a DuckDB read_parquet query."""
    # trunk-ignore-begin(bandit/B608)
    if merge_schemas:
        return f"SELECT * FROM read_parquet(['{paths_str}'], union_by_name=true)"
    else:
        return f"SELECT * FROM read_parquet(['{paths_str}'])"
    # trunk-ignore-end(bandit/B608)


def _build_copy_query(source_table: str, path: str, file_type: Literal["csv", "parquet"]) -> str:
    """Build a DuckDB COPY query for writing files."""
    if file_type == "csv":
        return f"COPY {source_table} TO '{path}' (FORMAT CSV, HEADER TRUE)"
    elif file_type == "parquet":
        return f"COPY {source_table} TO '{path}' (FORMAT PARQUET)"
    else:
        raise InternalError(f"Unsupported file type: {file_type}.  Please use '.csv' or '.parquet'.")


def _convert_schema_to_duckdb_dict(schema: Schema) -> Dict[str, str]:
    """Convert a Fenic Schema to DuckDB column types dictionary."""
    type_mapping = {
        StringType: "VARCHAR",
        IntegerType: "BIGINT",
        FloatType: "FLOAT",
        DoubleType: "DOUBLE",
        BooleanType: "BOOLEAN",
        DateType: "DATE",
        TimestampType: "TIMESTAMPTZ",
    }

    duckdb_schema = {}
    for col_field in schema.column_fields:
        duckdb_type = type_mapping.get(col_field.data_type)
        if not duckdb_type:
            raise InternalError(
                f"Invalid column type for csv Schema: ColumnField(name='{col_field.name}', data_type={type(col_field.data_type).__name__}). "
                f"Expected one of: IntegerType, FloatType, DoubleType, BooleanType, StringType, DateType, or TimestampType as data_type"
                f"Example: Schema([ColumnField(name='id', data_type=IntegerType), ColumnField(name='name', data_type=StringType)])"
            )
        duckdb_schema[col_field.name] = duckdb_type

    return duckdb_schema


def _format_http_read_error(e: duckdb.HTTPException, result: ReadQueryResult) -> str:
    """Format HTTP errors into user-friendly messages for read operations."""
    if result.s3_status is not None:  # We have S3 paths
        return _format_s3_read_error(e, result.s3_status)
    elif result.hf_status is not None:  # We have HF paths
        return _format_hf_read_error(e, result.hf_status)
    else:
        return "Failed to read from HTTPFS. Verify the path(s) exist and are readable."


def _format_s3_read_error(e: duckdb.HTTPException, has_creds: bool) -> str:
    """Format S3-specific read errors."""
    if e.status_code == 404:
        return "Failed to read from S3. The object does not exist."
    elif e.status_code in (401, 403):
        if has_creds:
            return f"Failed to read from S3. The provided credentials do not have the required permissions. (Status code: {e.status_code})"
        else:
            return f"Failed to read from S3, the object is not publicly readable and no AWS credentials were provided. Configure AWS credentials (env/aws_config) or ensure the object is publicly readable. (Status code: {e.status_code})"
    else:
        return f"Failed to read from S3. {e}"


def _format_hf_read_error(e: duckdb.HTTPException, has_creds: bool) -> str:
    """Format HuggingFace-specific read errors."""
    if e.status_code == 404:
        return "Failed to read from Hugging Face. The object does not exist."
    elif e.status_code in (401, 403):
        if has_creds:
            return f"Failed to read from Hugging Face -- the provided credentials do not have the required permissions. (Status code: {e.status_code})"
        else:
            return f"Failed to read from Hugging Face -- credentials were not found and the dataset is private or gated. Set HF_TOKEN environment variable. (Status code: {e.status_code})"
    else:
        return f"Failed to read from Hugging Face. {e}"


def _fetch_and_validate_s3_credentials(s3_session: BotoSession) -> Tuple[ReadOnlyCredentials, str]:
    """Check if s3 credentials are present and valid."""
    credentials = s3_session.get_credentials()
    if not credentials:
        raise ConfigurationError("AWS credentials were not found. Configure AWS credentials (env/aws_config) to read or write to S3.")
    frozen_creds = credentials.get_frozen_credentials()
    region = s3_session.region_name
    if not frozen_creds.access_key or not frozen_creds.secret_key:
        raise ConfigurationError("AWS credentials were not found. Configure AWS credentials (env/aws_config) to read or write to S3.")
    if not region:
        raise ConfigurationError("No AWS region specified. Configure AWS region (env/aws_config) to read or write to S3.")
    return frozen_creds, region


def _fetch_hf_token() -> Optional[str]:
    """Get HuggingFace token from environment variable."""
    return os.environ.get("HF_TOKEN")

def _configure_duckdb_conn(db_conn: duckdb.DuckDBPyConnection) -> duckdb.DuckDBPyConnection:
    """Common configuration for duckdb connections."""
    db_conn.execute("set arrow_large_buffer_size=true;")
    return db_conn
