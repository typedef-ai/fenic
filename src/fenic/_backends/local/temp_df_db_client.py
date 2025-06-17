import os
from pathlib import Path

import polars as pl

import fenic._backends.local.utils.io_utils
from fenic.core._utils.misc import generate_unique_arrow_view_name


class TempDFDBClient:
    """A client for reading and writing Polars dataframes to an ephemeral DuckDB database.
    Used for cached dataframes and intermediate dataframes used in joins and lineage graphs.
    """

    def __init__(self, app_name: str):
        self.duckdb_path = Path(f"__{app_name}_tmp_dfs.duckdb")
        if os.path.exists(self.duckdb_path):
            os.remove(self.duckdb_path)
        self.db_conn = fenic._backends.local.utils.io_utils.configure_duckdb_conn_for_path(Path(self.duckdb_path))

    def cleanup(self):
        """Clean up the ephemeral DuckDB database."""
        self.db_conn.close()
        if os.path.exists(self.duckdb_path):
            os.remove(self.duckdb_path)

    def read_df(self, table_name: str) -> pl.DataFrame:
        """Read a Polars dataframe from a DuckDB table in the current DuckDB schema."""
        # trunk-ignore-begin(bandit/B608)
        result = self.db_conn.execute(f"SELECT * FROM {table_name}")
        arrow_table = result.arrow()
        return pl.from_arrow(arrow_table)
        # trunk-ignore-end(bandit/B608)

    def write_df(self, df: pl.DataFrame, table_name: str):
        """Write a Polars dataframe to a DuckDB table in the current DuckDB schema."""
        # trunk-ignore-begin(bandit/B608)
        view_name = generate_unique_arrow_view_name()
        self.db_conn.register(view_name, df)
        self.db_conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM {view_name}")
        self.db_conn.execute(f"DROP VIEW IF EXISTS {view_name}")
        # trunk-ignore-end(bandit/B608)

    def is_df_cached(self, table_name: str) -> bool:
        """Check if a Polars dataframe is stored in a DuckDB table in the 'main' schema."""
        # trunk-ignore-begin(bandit/B608)
        result = self.db_conn.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
        )
        return len(result.fetchall()) > 0
        # trunk-ignore-end(bandit/B608)
