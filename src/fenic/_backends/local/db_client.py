"""Unified DuckDB client managing both main and intermediate databases."""

from pathlib import Path
from typing import Optional

import duckdb

import fenic._backends.local.utils.io_utils


class DBClient:
    """Unified DuckDB client managing both main and intermediate databases."""

    def __init__(self, main_db_path: Path, app_name: str):
        """Initialize DBClient with database paths.
        
        Args:
            main_db_path: Path to the main database file
            app_name: Application name for intermediate database naming
        """
        self.main_db_path = main_db_path
        self.app_name = app_name
        self._connection: Optional[duckdb.DuckDBPyConnection] = None

    def connect(self) -> None:
        """Create connection and attach intermediate database."""
        # Create main database connection using existing configuration
        self._connection = fenic._backends.local.utils.io_utils.configure_duckdb_conn_for_path(
            self.main_db_path
        )
        
        # Attach intermediate database
        intermediate_path = self.main_db_path.parent / f"__{self.app_name}_tmp_dfs.duckdb"
        self._connection.execute(f"ATTACH '{intermediate_path}' AS __intermediate__")

    def cursor(self) -> duckdb.DuckDBPyConnection:
        """Get cursor from the unified connection.
        
        Returns:
            Cursor from the unified connection
            
        Raises:
            RuntimeError: If connection hasn't been established
        """
        if self._connection is None:
            raise RuntimeError("DBClient connection not established. Call connect() first.")
        return self._connection.cursor()

    def close(self) -> None:
        """Close the unified connection."""
        if self._connection:
            self._connection.close()
            self._connection = None

    @property
    def is_connected(self) -> bool:
        """Check if the connection is established."""
        return self._connection is not None


def is_df_cached(db_client: DBClient, table_name: str) -> bool:
    """Check if a DataFrame is cached in the intermediate database.
    
    Args:
        db_client: DBClient instance
        table_name: Name of the table to check
        
    Returns:
        True if table exists in intermediate database, False otherwise
    """
    cursor = db_client.cursor()
    result = cursor.execute(
        "SELECT COUNT(*) FROM __intermediate__.information_schema.tables WHERE table_name = ?",
        [table_name]
    ).fetchone()[0]
    return result > 0