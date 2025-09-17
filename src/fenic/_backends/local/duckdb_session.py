"""Unified DuckDB client managing both main and intermediate databases."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import duckdb
import polars as pl

from fenic._backends.local.utils.io_utils import configure_duckdb_conn_for_path
from fenic.core._utils.misc import generate_unique_arrow_view_name

logger = logging.getLogger(__name__)

class IntermediateTableOps:
    """Operations for intermediate table storage."""

    @staticmethod
    def write_df(conn: duckdb.DuckDBPyConnection, df: pl.DataFrame, table_name: str) -> None:
        """Write a Polars DataFrame into the intermediate DB."""
        view_name = generate_unique_arrow_view_name()
        conn.register(view_name, df)
        # trunk-ignore-begin(bandit/B608)
        try:
            conn.execute(
                f"CREATE OR REPLACE TABLE __intermediate__.main.{table_name} AS SELECT * FROM {view_name}"
            )
        finally:
            conn.execute(f"DROP VIEW IF EXISTS {view_name}")
        # trunk-ignore-end(bandit/B608)

    @staticmethod
    def read_df(conn: duckdb.DuckDBPyConnection, table_name: str) -> pl.DataFrame:
        """Read a Polars DataFrame from the intermediate DB."""
        # trunk-ignore-begin(bandit/B608)
        result = conn.execute(f"SELECT * FROM __intermediate__.main.{table_name}")
        return pl.from_arrow(result.arrow())
        # trunk-ignore-end(bandit/B608)

    @staticmethod
    def exists(conn: duckdb.DuckDBPyConnection, table_name: str) -> bool:
        """Check if a DataFrame exists in the intermediate DB."""
        res = conn.execute(
            """
            SELECT COUNT(*) > 0
            FROM duckdb_tables()
            WHERE database_name = '__intermediate__'
            AND schema_name = 'main'
            AND table_name = ?
            """,
            (table_name,),
        ).fetchone()
        return res[0] if res else False


class DuckDBSession:
    """Session for DuckDB operations."""

    def __init__(self, db_path: Path, app_name: str):
        self.db_path = db_path
        self.app_name = app_name
        self.intermediate_path = db_path.parent / f"__{app_name}_tmp_dfs.duckdb"

    def create_connection(self) -> duckdb.DuckDBPyConnection:
        """Create a new connection for query execution."""
        conn = configure_duckdb_conn_for_path(self.db_path)
        conn.execute(f"ATTACH IF NOT EXISTS '{self.intermediate_path}' AS __intermediate__")
        return conn

    def close(self) -> None:
        """Close catalog connection and clean up."""

        # Clean up intermediate database
        if os.path.exists(self.intermediate_path):
            os.remove(self.intermediate_path)
