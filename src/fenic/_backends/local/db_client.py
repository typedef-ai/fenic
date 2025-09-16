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

class FenicDuckDBClient:
    """Client for interacting with main and intermediate DuckDB databases."""

    def __init__(self, db_path: Path, app_name: str):
        self.intermediate_path = db_path.parent / f"__{app_name}_tmp_dfs.duckdb"
        self.connection = configure_duckdb_conn_for_path(db_path)
        self.connection.execute(f"ATTACH '{self.intermediate_path}' AS __intermediate__")

    def write_intermediate_df(self, df: pl.DataFrame, table_name: str) -> None:
        """Write a Polars DataFrame into the intermediate DB."""
        cursor = self.cursor()
        view_name = generate_unique_arrow_view_name()
        cursor.register(view_name, df)
        # trunk-ignore-begin(bandit/B608)
        try:
            cursor.execute(
                f"CREATE OR REPLACE TABLE __intermediate__.main.{table_name} AS SELECT * FROM {view_name}"
            )
        finally:
            cursor.execute(f"DROP VIEW IF EXISTS {view_name}")
        print(self.read_intermediate_df(table_name))
        # trunk-ignore-end(bandit/B608)

    def read_intermediate_df(self, table_name: str) -> pl.DataFrame:
        """Read a Polars DataFrame from the intermediate DB."""
        # trunk-ignore-begin(bandit/B608)
        result = self.cursor().execute(f"SELECT * FROM __intermediate__.main.{table_name}")
        return pl.from_arrow(result.arrow())
        # trunk-ignore-end(bandit/B608)

    def does_intermediate_df_exist(self, table_name: str) -> bool:
        """Check if a Polars DataFrame exists in the intermediate DB."""
        res = self.cursor().execute(
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

    def cursor(self) -> duckdb.DuckDBPyConnection:
        """Get a cursor for the DuckDB connection."""
        return self.connection.cursor()

    def close(self) -> None:
        """Close the DuckDB connection."""
        if os.path.exists(self.intermediate_path):
            os.remove(self.intermediate_path)
        self.connection.close()
