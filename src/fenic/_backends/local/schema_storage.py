"""Schema storage utilities for persisting and retrieving schema metadata.

This module handles the serialization, deserialization, and storage of
schema metadata, particularly for logical types that can't be directly
represented in the physical storage system.
"""

import logging
from typing import Optional

import duckdb

from fenic._backends.schema_serde import deserialize_schema, serialize_schema
from fenic._backends.utils.catalog_utils import normalize_object_name
from fenic.core.types import Schema

# Constants for system schema and table names
SYSTEM_SCHEMA_NAME = "__fenic_system"
SCHEMA_METADATA_TABLE = "table_schemas"

logger = logging.getLogger(__name__)


class SchemaStorage:
    """Handles storage and retrieval of schema metadata in the system tables. This is particularly important for logical types that can't be directly represented in the physical storage system."""

    def __init__(self, connection: duckdb.DuckDBPyConnection):
        """Initialize the schema storage with a DuckDB connection.

        Args:
            connection: An initialized DuckDB connection
        """
        self.db_conn = connection

    def initialize_system_schema(self) -> None:
        """Initialize the system schema and metadata table for storing table schemas including logical type information.

        Raises:
            RuntimeError: If the system schema or metadata table cannot be created.
        """
        try:
            # Create system schema if it doesn't exist
            self.db_conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{SYSTEM_SCHEMA_NAME}";')

            # Create the schema metadata table if it doesn't exist
            self.db_conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS "{SYSTEM_SCHEMA_NAME}"."{SCHEMA_METADATA_TABLE}" (
                    database_name TEXT NOT NULL,
                    table_name TEXT NOT NULL,
                    schema_blob TEXT NOT NULL,
                    PRIMARY KEY (database_name, table_name)
                );
            """
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize system schema and {SCHEMA_METADATA_TABLE} table: {e}"
            ) from e

        logger.debug(f"Initialized system schema and {SCHEMA_METADATA_TABLE} table")

    def save_schema(self, database_name: str, table_name: str, schema: Schema) -> None:
        """Save a table's schema metadata to the system table. This is used for storing logical type information that can't be directly represented in the physical storage.

        Args:
            database_name: The name of the database/schema
            table_name: The name of the table
            schema: The schema to store

        Raises:
            RuntimeError: If the schema cannot be saved
        """
        schema_blob = serialize_schema(schema)
        database_name = normalize_object_name(database_name)
        table_name = normalize_object_name(table_name)

        try:
            # Upsert the schema - replace if exists
            self.db_conn.execute(
                f"""
                INSERT OR REPLACE INTO "{SYSTEM_SCHEMA_NAME}"."{SCHEMA_METADATA_TABLE}" (
                    database_name, table_name, schema_blob
                ) VALUES (?, ?, ?)
            """,
                (database_name, table_name, schema_blob),
            )

            logger.debug(f"Saved schema metadata for {database_name}.{table_name}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to save schema metadata for {database_name}.{table_name}: {e}"
            ) from e

    def get_schema(self, database_name: str, table_name: str) -> Optional[Schema]:
        """Retrieve a table's schema metadata from the system table.

        Args:
            database_name: The name of the database/schema
            table_name: The name of the table

        Returns:
            The schema if found, None otherwise

        Raises:
            RuntimeError: If there's an error retrieving the schema
        """
        try:
            # trunk-ignore-begin(bandit/B608): No major risk of SQL injection here, because queries run on a client side DuckDB instance.
            result = self.db_conn.execute(
                f"""
                SELECT schema_blob
                FROM "{SYSTEM_SCHEMA_NAME}"."{SCHEMA_METADATA_TABLE}"
                WHERE database_name = ? AND table_name = ?
            """,
                (normalize_object_name(database_name), normalize_object_name(table_name)),
            ).fetchone()
            # trunk-ignore-end(bandit/B608)
            if result is None:
                logger.debug(
                    f"No schema metadata found for {database_name}.{table_name}"
                )
                return None

            schema_blob = result[0]
            return deserialize_schema(schema_blob)
        except Exception as e:
            raise RuntimeError(
                f"Failed to retrieve schema metadata for {database_name}.{table_name}: {e}"
            ) from e

    def delete_schema(self, database_name: str, table_name: str) -> bool:
        """Delete a table's schema metadata from the system table.

        Args:
            database_name: The name of the database/schema
            table_name: The name of the table

        Returns:
            True if the schema was deleted, False if it didn't exist

        Raises:
            RuntimeError: If there's an error deleting the schema
        """
        try:
            # trunk-ignore-begin(bandit/B608): No major risk of SQL injection here, because queries run on a client side DuckDB instance.
            result = self.db_conn.execute(
                f"""
                DELETE FROM "{SYSTEM_SCHEMA_NAME}"."{SCHEMA_METADATA_TABLE}"
                WHERE database_name = ? AND table_name = ?
            """,
                (normalize_object_name(database_name), normalize_object_name(table_name)),
            )
            # trunk-ignore-end(bandit/B608)
            rows_deleted = result.fetchone()[0]
            if rows_deleted == 0:
                logger.debug(
                    f"No schema metadata found to delete for {database_name}.{table_name}"
                )
                return False

            logger.debug(f"Deleted schema metadata for {database_name}.{table_name}")
            return True
        except Exception as e:
            raise RuntimeError(
                f"Failed to delete schema metadata for {database_name}.{table_name}: {e}"
            ) from e

    def delete_database_schemas(self, database_name: str) -> int:
        """Delete all schema metadata for a database.

        Args:
            database_name: The name of the database/schema

        Returns:
            The number of schema metadata entries deleted

        Raises:
            RuntimeError: If there's an error deleting the schemas
        """
        try:
            # trunk-ignore-begin(bandit/B608): No major risk of SQL injection here, because queries run on a client side DuckDB instance.
            result = self.db_conn.execute(
                f"""
                DELETE FROM "{SYSTEM_SCHEMA_NAME}"."{SCHEMA_METADATA_TABLE}"
                WHERE database_name = ?
            """,
                (normalize_object_name(database_name),),
            )
            # trunk-ignore-end(bandit/B608)
            rows_deleted = result.fetchone()[0]

            if rows_deleted > 0:
                logger.debug(
                    f"Deleted {rows_deleted} schema metadata entries for database {database_name}"
                )
            else:
                logger.debug(f"No schema metadata found for database {database_name}")

            return rows_deleted
        except Exception as e:
            raise RuntimeError(
                f"Failed to delete schema metadata for database {database_name}: {e}"
            ) from e
