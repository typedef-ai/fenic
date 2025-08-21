import logging
import os
from enum import Enum
from pathlib import Path
from typing import List, Literal, Optional, Tuple
from urllib.parse import urlparse

import duckdb
import polars as pl
from boto3.session import Session as BotoSession
from botocore.credentials import ReadOnlyCredentials

from fenic.core.error import ConfigurationError, ValidationError

logger = logging.getLogger(__name__)

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
    elif scheme == "hf":
        # For HF paths, we assume they exist if we have a token
        hf_token = _fetch_hf_token()
        if not hf_token:
            logger.warning("HuggingFace token not found. Set HF_TOKEN environment variable.")
        return True  # We'll let DuckDB handle the actual path validation
    elif scheme == "file" or scheme == "":
        return os.path.exists(path)
    else:
        raise ValidationError(f"Unsupported file type: {scheme} for path: {path}.  Please use s3 scheme ('s3://'), hf scheme ('hf://'), or local scheme ('file://' or no prefix).")


def query_files(query: str, paths: List[str], s3_session: BotoSession) -> pl.DataFrame:
    """Execute a DuckDB query with s3 and/or hf credentials."""
    duckdb_conn = _configure_duckdb_conn(duckdb.connect())

    has_s3_paths = any(urlparse(path).scheme == "s3" for path in paths)
    has_hf_paths = any(urlparse(path).scheme == "hf" for path in paths)

    if has_s3_paths or has_hf_paths:
        query = _build_query_with_httpfs_extensions(query)
        if has_s3_paths:
            query = _build_query_with_s3_creds(query, s3_session)
        if has_hf_paths:
            query = _build_query_with_hf_creds(query)

    arrow_result = duckdb_conn.execute(query).arrow()
    return pl.from_arrow(arrow_result)


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
    if file_type == "csv":
        query = f"COPY df_view TO '{path}' (FORMAT CSV, HEADER TRUE)"
    elif file_type == "parquet":
        query = f"COPY df_view TO '{path}' (FORMAT PARQUET)"
    else:
        raise ValidationError(f"Unsupported file type: {file_type}.  Please use '.csv' or '.parquet'.")

    scheme = urlparse(path).scheme
    if scheme == "s3":
        query = _build_query_with_httpfs_extensions(query)
        query = _build_query_with_s3_creds(query, s3_session)

    duckdb_conn.execute(query)


def _fetch_and_validate_s3_credentials(s3_session: BotoSession) -> Tuple[ReadOnlyCredentials, str]:
    """Check if s3 credentials are present and valid."""
    credentials = s3_session.get_credentials()
    if not credentials:
        raise ConfigurationError("Unable to locate AWS credentials.")
    frozen_creds = credentials.get_frozen_credentials()
    region = s3_session.region_name
    if not frozen_creds.access_key or not frozen_creds.secret_key:
        raise ConfigurationError("Unable to locate AWS credentials.")
    if not region:
        raise ConfigurationError("No AWS region specified.")
    return frozen_creds, region


def _fetch_hf_token() -> Optional[str]:
    """Get HuggingFace token from environment variable."""
    hf_token = os.environ.get("HF_TOKEN")
    return hf_token

def _build_query_with_httpfs_extensions(query: str) -> str:
    """Helper method to add httpfs extensions to a DuckDB query."""
    return f"INSTALL httpfs; LOAD httpfs; {query}"

def _build_query_with_s3_creds(query: str, s3_session: BotoSession) -> str:
    """Helper method to add AWS credentials to a DuckDB query."""
    frozen_creds, region = _fetch_and_validate_s3_credentials(s3_session)
    s3_setup_query = f"SET s3_region='{region}'; "
    s3_setup_query += f"SET s3_access_key_id='{frozen_creds.access_key}'; "
    s3_setup_query += f"SET s3_secret_access_key='{frozen_creds.secret_key}'; "
    if frozen_creds.token:
        s3_setup_query += f"SET s3_session_token='{frozen_creds.token}'; "
    query = f"{s3_setup_query} {query}"
    return query


def _build_query_with_hf_creds(query: str) -> str:
    """Helper method to add HuggingFace credentials to a DuckDB query."""
    hf_token = _fetch_hf_token()
    if not hf_token:
        raise ConfigurationError("HuggingFace token not found. Set HF_TOKEN environment variable.")

    return f"CREATE SECRET hf_token (TYPE HUGGINGFACE, TOKEN '{hf_token}'); {query}"

def configure_duckdb_conn_for_path(db_path: Path) -> duckdb.DuckDBPyConnection:
    """Create a duckdb connection for a given path, applying common configuration."""
    db_conn = duckdb.connect(str(db_path))
    return _configure_duckdb_conn(db_conn)


def _configure_duckdb_conn(db_conn: duckdb.DuckDBPyConnection) -> duckdb.DuckDBPyConnection:
    """Common configuration for duckdb connections."""
    db_conn.execute("set arrow_large_buffer_size=true;")
    return db_conn

def get_path_scheme(path: str) -> PathScheme:
    """Gets the path scheme."""
    scheme = urlparse(path).scheme
    if scheme == "s3":
        return PathScheme.S3
    elif scheme == "hf":
        return PathScheme.HF
    else:
        return PathScheme.LOCALFS
