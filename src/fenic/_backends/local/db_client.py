"""Unified DuckDB client managing both main and intermediate databases."""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import polars as pl
import os
import duckdb

import fenic._backends.local.utils.io_utils
from fenic.core._utils.misc import generate_unique_arrow_view_name
from fenic.core.error import InternalError

class DBClient:
    """Unified DuckDB client managing both main and intermediate databases."""

    _connection: Optional[duckdb.DuckDBPyConnection] = None

    def __init__(self, db_path: Path, app_name: str):
        """Initialize DBClient with database paths.

        Args:
            db_path: Path to the main database file
            app_name: Application name for intermediate database naming
        """
        self._connection = fenic._backends.local.utils.io_utils.configure_duckdb_conn_for_path(
            db_path
        )

        # Attach intermediate database
        self._intermediate_path = db_path.parent / f"__{app_name}_tmp_dfs.duckdb"
        self._connection.execute(f"ATTACH '{self._intermediate_path}' AS __intermediate__")

    def cursor(self) -> duckdb.DuckDBPyConnection:
        """Get cursor from the unified connection.

        Returns:
            Cursor from the unified connection

        Raises:
            RuntimeError: If connection hasn't been established
        """
        if self._connection is None:
            raise InternalError("DBClient connection is closed.")
        return self._connection.cursor()

    def cleanup(self) -> None:
        """Close the unified connection."""
        if self._connection is None:
            self._connection.close()
            self.intermediate.cleanup()

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        """Get the unified connection."""
        if self._connection is None:
            raise InternalError("DBClient connection is closed.")
        return self._connection

    @property
    def intermediate(self) -> IntermediateDBClient:
        return IntermediateDBClient(self)

class IntermediateDBClient:
    """Client for the intermediate database."""

    def __init__(self, db_client: DBClient):
        self.db_client = db_client

    def is_df_cached(self, cache_name: str) -> bool:
        """Check if a Polars dataframe is stored in a DuckDB table in the 'main' schema."""
        # trunk-ignore-begin(bandit/B608)
        result = self.db_client.cursor().execute(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{cache_name}'"
        )
        return len(result.fetchall()) > 0
        # trunk-ignore-end(bandit/B608)

    def write_df(self, df: pl.DataFrame, table_name: str):
        """Write a Polars dataframe to a DuckDB table in the current DuckDB schema."""
        # trunk-ignore-begin(bandit/B608)
        view_name = generate_unique_arrow_view_name()
        cursor = self.db_client.cursor()
        cursor.register(view_name, df)
        cursor.execute(f"CREATE TABLE {table_name} AS SELECT * FROM __intermediate__.{view_name}")
        cursor.execute(f"DROP VIEW IF EXISTS {view_name}")
        # trunk-ignore-end(bandit/B608)

    def read_df(self, table_name: str) -> pl.DataFrame:
        """Read a Polars dataframe from a DuckDB table in the current DuckDB schema."""
        # trunk-ignore-begin(bandit/B608)
        result = self.db_client.cursor().execute(f"SELECT * FROM __intermediate__.{table_name}")
        arrow_table = result.arrow()
        return pl.from_arrow(arrow_table)
        # trunk-ignore-end(bandit/B608)

    def get_read_df_query(self, table_name: str) -> str:
        """Get a SQL query to read a Polars dataframe from a DuckDB table in the current DuckDB schema."""
        return f"SELECT * FROM __intermediate__.{table_name}"

    def cleanup(self) -> None:
        """Clean up the intermediate database."""
        if os.path.exists(self.db_client._intermediate_path):
            os.remove(self.db_client._intermediate_path)
