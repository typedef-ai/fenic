from pathlib import Path

import polars as pl

import fenic._backends.local.utils.io_utils
from fenic._backends.local.physical_plan.utils import apply_ingestion_coercions
from fenic.core._utils.misc import generate_unique_view_name


class TempDFDBClient:
    """A client for reading and writing Polars dataframes to an ephemeral DuckDB database.
    Used for cached dataframes and intermediate dataframes used in lineage graphs.
    """

    def __init__(self, duckdb_path: Path):
        self.duckdb_path = duckdb_path
        self._delete_intermediate_files()
        self.db_conn = fenic._backends.local.utils.io_utils.configure_duckdb_conn_for_path(self.duckdb_path)

    def cleanup(self):
        """Clean up the ephemeral DuckDB database."""
        self.db_conn.close()
        self._delete_intermediate_files()

    def read_df(self, table_name: str) -> pl.DataFrame:
        """Read a Polars dataframe from a DuckDB table in the current DuckDB schema."""
        # trunk-ignore-begin(bandit/B608)
        return apply_ingestion_coercions(self.db_conn.cursor().execute(f"SELECT * FROM {table_name}").pl(), coerce_array=False)
        # trunk-ignore-end(bandit/B608)

    def write_df(self, df: pl.DataFrame, table_name: str):
        """Write a Polars dataframe to a DuckDB table in the current DuckDB schema."""
        # trunk-ignore-begin(bandit/B608)
        view_name = generate_unique_view_name()
        cursor = self.db_conn.cursor()
        cursor.register(view_name, df)
        cursor.execute(f"CREATE TABLE {table_name} AS SELECT * FROM {view_name}")
        cursor.execute(f"DROP VIEW IF EXISTS {view_name}")
        # trunk-ignore-end(bandit/B608)

    def is_df_cached(self, table_name: str) -> bool:
        """Check if a Polars dataframe is stored in a DuckDB table in the 'main' schema."""
        # trunk-ignore-begin(bandit/B608)
        result = self.db_conn.cursor().execute(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
        )
        return len(result.fetchall()) > 0
        # trunk-ignore-end(bandit/B608)

    def _delete_intermediate_files(self):
        """Delete the intermediate files for the DuckDB database."""
        self.duckdb_path.unlink(missing_ok=True)
        self.duckdb_path.with_suffix(".wal").unlink(missing_ok=True)
