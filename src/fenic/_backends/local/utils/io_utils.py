import os
from pathlib import Path
from typing import List, Literal, Tuple
from urllib.parse import urlparse

import boto3
import duckdb
import polars as pl
from botocore.credentials import ReadOnlyCredentials

from fenic.core.error import ConfigurationError, ValidationError


def does_path_exist(path: str, s3_session:boto3.session.Session) -> bool:
    """Check if a s3 or local path exists."""
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
        raise ValidationError(f"Unsupported file type: {scheme} for path: {path}.  Please use s3 scheme ('s3://') or local scheme ('file://' or no prefix).")


def query_files(query: str, paths: List[str], s3_session: boto3.session.Session) -> pl.DataFrame:
    """Execute a DuckDB query with s3 credentials."""
    duckdb_conn = _configure_duckdb_conn(duckdb.connect())
    if any(urlparse(path).scheme == "s3" for path in paths):
        query = _build_query_with_s3_creds(query, s3_session)
    arrow_result = duckdb_conn.execute(query).arrow()
    return pl.from_arrow(arrow_result)


def write_file(
    df: pl.DataFrame,
    path: str,
    s3_session: boto3.session.Session,
    file_type: Literal["csv", "parquet"],
):
    """Write file to local or s3 path using duckdb."""
    duckdb_conn = _configure_duckdb_conn(duckdb.connect())
    arrow_table = df.to_arrow()
    duckdb_conn.register("df_view", arrow_table)
    if file_type == "csv":
        query = f"COPY df_view TO '{path}' (FORMAT CSV, HEADER TRUE)"
    elif file_type == "parquet":
        query = f"COPY df_view TO '{path}' (FORMAT PARQUET)"
    else:
        raise ValidationError(f"Unsupported file type: {file_type}.  Please use '.csv' or '.parquet'.")
    if urlparse(path).scheme == "s3":
        # grab the s3 credentials and add them to the query
        query = _build_query_with_s3_creds(query, s3_session)
    duckdb_conn.execute(query)


def _fetch_and_validate_s3_credentials(s3_session:boto3.session.Session) -> Tuple[ReadOnlyCredentials, str]:
    """Check if s3 credentials are present and valid."""
    credentials = s3_session.get_credentials()
    frozen_creds = credentials.get_frozen_credentials()
    region = s3_session.region_name
    if not frozen_creds.access_key or not frozen_creds.secret_key:
        raise ConfigurationError("Unable to locate AWS credentials.")
    if not region:
        raise ConfigurationError("No AWS region specified.")
    return frozen_creds, region


def _build_query_with_s3_creds(query: str, s3_session: boto3.session.Session) -> str:
    """Helper method to add AWS credentials to a DuckDB query."""
    frozen_creds, region = _fetch_and_validate_s3_credentials(s3_session)
    s3_setup_query = "INSTALL httpfs; LOAD httpfs;"
    s3_setup_query += f"SET s3_region='{region}'; "
    s3_setup_query += f"SET s3_access_key_id='{frozen_creds.access_key}'; "
    s3_setup_query += f"SET s3_secret_access_key='{frozen_creds.secret_key}'; "
    if frozen_creds.token:
        s3_setup_query += f"SET s3_session_token='{frozen_creds.token}'; "
    query = f"{s3_setup_query} {query}"
    return query


def configure_duckdb_conn_for_path(db_path: Path) -> duckdb.DuckDBPyConnection:
    """Create a duckdb connection for a given path, applying common configuration."""
    db_conn = duckdb.connect(str(db_path))
    return _configure_duckdb_conn(db_conn)


def _configure_duckdb_conn(db_conn: duckdb.DuckDBPyConnection) -> duckdb.DuckDBPyConnection:
    """Common configuration for duckdb connections."""
    db_conn.execute("set arrow_large_buffer_size=true;")
    return db_conn
