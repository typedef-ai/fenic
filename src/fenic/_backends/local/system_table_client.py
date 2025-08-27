"""Schema storage utilities for persisting and retrieving schema metadata.

This module handles the serialization, deserialization, and storage of
schema metadata, particularly for logical types that can't be directly
represented in the physical storage system.
"""

import base64
import logging
from datetime import datetime
from typing import Dict, List, Optional

import duckdb

from fenic._backends.schema_serde import deserialize_schema, serialize_schema
from fenic._backends.utils.catalog_utils import normalize_object_name
from fenic.core._logical_plan.plans.base import LogicalPlan
from fenic.core._serde import LogicalPlanSerde
from fenic.core.error import CatalogError
from fenic.core.metrics import QueryMetrics
from fenic.core.types import ColumnField, Schema
from fenic.core.types.datatypes import (
    DoubleType,
    IntegerType,
    StringType,
)

# Constants for system schema and table names
SYSTEM_SCHEMA_NAME = "__fenic_system" 
SCHEMA_METADATA_TABLE = "table_schemas"
VIEWS_METADATA_TABLE = "table_views"

# Constants for read-only system schema and tables
READ_ONLY_SYSTEM_SCHEMA_NAME = "fenic_system"
METRICS_TABLE_NAME = "metrics"

logger = logging.getLogger(__name__)


class SystemTableClient:
    """Handles storage and retrieval of schema metadata in the system tables. This is particularly important for logical types that can't be directly represented in the physical storage system."""

    def __init__(self, connection: duckdb.DuckDBPyConnection):
        """Initialize the schema storage with a DuckDB connection.

        Args:
            connection: An initialized DuckDB connection

        Raises:
            CatalogError: If the initialization of tables for schema or view metadata fails
        """
        self.connection = connection
        self._initialize_system_schema()
        self._initialize_views_metadata()
        self._initialize_read_only_system_schema_and_tables()

    def save_schema(self, cursor: duckdb.DuckDBPyConnection, database_name: str, table_name: str, schema: Schema) -> None:
        """Save a table's schema metadata to the system table. This is used for storing logical type information that can't be directly represented in the physical storage.

        Args:
            database_name: The name of the database/schema
            table_name: The name of the table
            schema: The schema to store

        Raises:
            CatalogError: If the schema cannot be saved
        """
        schema_blob = serialize_schema(schema)
        database_name = normalize_object_name(database_name)
        table_name = normalize_object_name(table_name)

        try:
            # Upsert the schema - replace if exists
            cursor.execute(
                f"""
                INSERT OR REPLACE INTO "{SYSTEM_SCHEMA_NAME}"."{SCHEMA_METADATA_TABLE}" (
                    database_name, table_name, schema_blob
                ) VALUES (?, ?, ?)
            """,
                (database_name, table_name, schema_blob),
            )

            logger.debug(f"Saved schema metadata for {database_name}.{table_name}")
        except Exception as e:
            raise CatalogError(
                f"Failed to save schema metadata for {database_name}.{table_name}: {e}"
            ) from e

    def get_schema(self, cursor: duckdb.DuckDBPyConnection, database_name: str, table_name: str) -> Optional[Schema]:
        """Retrieve a table's schema metadata from the system table.

        Args:
            database_name: The name of the database/schema
            table_name: The name of the table

        Returns:
            The schema if found, None otherwise

        Raises:
            CatalogError: If there's an error retrieving the schema
        """
        try:
            # trunk-ignore-begin(bandit/B608): No major risk of SQL injection here, because queries run on a client side DuckDB instance.
            result = cursor.execute(
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
            raise CatalogError(
                f"Failed to retrieve schema metadata for {database_name}.{table_name}: {e}"
            ) from e

    def delete_schema(self, cursor: duckdb.DuckDBPyConnection, database_name: str, table_name: str) -> bool:
        """Delete a table's schema metadata from the system table.

        Args:
            database_name: The name of the database/schema
            table_name: The name of the table

        Returns:
            True if the schema was deleted, False if it didn't exist

        Raises:
            CatalogError: If there's an error deleting the schema
        """
        try:
            # trunk-ignore-begin(bandit/B608): No major risk of SQL injection here, because queries run on a client side DuckDB instance.
            result = cursor.execute(
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
            raise CatalogError(
                f"Failed to delete schema metadata for {database_name}.{table_name}: {e}"
            ) from e

    def delete_database_schemas(self, cursor: duckdb.DuckDBPyConnection, database_name: str) -> int:
        """Delete all schema metadata for a database.

        Args:
            database_name: The name of the database/schema

        Returns:
            The number of schema metadata entries deleted

        Raises:
            CatalogError: If there's an error deleting the schemas
        """
        try:
            # trunk-ignore-begin(bandit/B608): No major risk of SQL injection here, because queries run on a client side DuckDB instance.
            result = cursor.execute(
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
            raise CatalogError(
                f"Failed to delete schema metadata for database {database_name}: {e}"
            ) from e

    def save_view(
        self,
        cursor: duckdb.DuckDBPyConnection,
        database_name: str,
        view_name: str,
        logical_plan: LogicalPlan
    ) -> None:
        database_name = database_name.casefold()
        view_name = view_name.casefold()
        logical_plan_str = base64.b64encode(LogicalPlanSerde.serialize(logical_plan)).decode('utf-8')
        try:
            cursor.execute(
                f"""
                INSERT OR REPLACE INTO "{SYSTEM_SCHEMA_NAME}"."{VIEWS_METADATA_TABLE}" (
                    database_name, view_name, view_blob, creation_time
                ) VALUES (?, ?, ?, ?)
            """,
                (database_name, view_name, logical_plan_str, datetime.now()),
            )

            logger.debug(f"Saved View for {database_name}.{view_name}")
        except Exception as e:
            logger.error(f"View error while saving: {e}")
            raise CatalogError(
                f"Failed to save view for {database_name}.{view_name}"
            ) from e

    def get_view(
        self, cursor: duckdb.DuckDBPyConnection, database_name: str, view_name: str
    ) -> Optional[LogicalPlan]:
        try:
            # trunk-ignore-begin(bandit/B608): No major risk of SQL injection here, because queries run on a client side DuckDB instance.
            result = cursor.execute(
                f"""
                SELECT view_blob
                FROM "{SYSTEM_SCHEMA_NAME}"."{VIEWS_METADATA_TABLE}"
                WHERE database_name = ? AND view_name = ?
            """,
                (database_name, view_name),
            ).fetchone()
            # trunk-ignore-end(bandit/B608)
            if result is None:
                logger.debug(f"No view found for {database_name}.{view_name}")
                return None

            view_blob = base64.b64decode(result[0])
            return LogicalPlanSerde.deserialize(view_blob)
        except Exception as e:
            logger.error(f"View error: {e}")
            raise CatalogError(
                f"Failed to retrieve view for {database_name}.{view_name}"
            ) from e

    def list_views(
        self, cursor: duckdb.DuckDBPyConnection, database_name: str
    ) -> Optional[List[object]]:
        try:
            # trunk-ignore-begin(bandit/B608): No major risk of SQL injection here, because queries run on a client side DuckDB instance.
            result = cursor.execute(
                f"""
                SELECT view_name
                FROM "{SYSTEM_SCHEMA_NAME}"."{VIEWS_METADATA_TABLE}"
                WHERE database_name = ?
            """,
                (database_name,),
            ).fetchall()
            # trunk-ignore-end(bandit/B608)
            if result is None:
                logger.debug(f"No view found in {database_name}")
                return None

            return result
        except Exception as e:
            raise CatalogError(
                f"Failed to retrieve all views for {database_name}"
            ) from e

    def delete_view(self, cursor: duckdb.DuckDBPyConnection, database_name: str, view_name: str) -> bool:
        try:
            # trunk-ignore-begin(bandit/B608): No major risk of SQL injection here, because queries run on a client side DuckDB instance.
            result = cursor.execute(
                f"""
                DELETE FROM "{SYSTEM_SCHEMA_NAME}"."{VIEWS_METADATA_TABLE}"
                WHERE database_name = ? AND view_name = ?
            """,
                (database_name, view_name),
            )
            # trunk-ignore-end(bandit/B608)
            rows_deleted = result.fetchone()[0]
            if rows_deleted == 0:
                logger.debug(
                    f"No views found to delete for {database_name}.{view_name}"
                )
                return False

            logger.debug(f"Deleted views for {database_name}.{view_name}")
            return True
        except Exception as e:
            raise CatalogError(
                f"Failed to delete views for {database_name}.{view_name}"
            ) from e

    def delete_database_views(self, cursor: duckdb.DuckDBPyConnection, database_name: str) -> int:
        try:
            # trunk-ignore-begin(bandit/B608): No major risk of SQL injection here, because queries run on a client side DuckDB instance.
            result = cursor.execute(
                f"""
                DELETE FROM "{SYSTEM_SCHEMA_NAME}"."{VIEWS_METADATA_TABLE}"
                WHERE database_name = ?
            """,
                (database_name,),
            )
            # trunk-ignore-end(bandit/B608)
            rows_deleted = result.fetchone()[0]

            if rows_deleted > 0:
                logger.debug(
                    f"Deleted {rows_deleted} views metadata entries for database {database_name}"
                )
            else:
                logger.debug(f"No views metadata found for database {database_name}")

            return rows_deleted
        except Exception as e:
            raise CatalogError(
                f"Failed to delete views metadata for database {database_name}"
            ) from e

    def _initialize_system_schema(self) -> None:
        """Initialize the system schema and metadata table for storing table schemas including logical type information.

        Raises:
            CatalogError: If the system schema or metadata table cannot be created.
        """
        cursor = self.connection.cursor()
        try:
            # Create system schema if it doesn't exist
            cursor.execute(f'CREATE SCHEMA IF NOT EXISTS "{SYSTEM_SCHEMA_NAME}";')

            # Create the schema metadata table if it doesn't exist
            cursor.execute(
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
            raise CatalogError(
                f"Failed to initialize system schema and {SCHEMA_METADATA_TABLE} table: {e}"
            ) from e

        logger.debug(f"Initialized system schema and {SCHEMA_METADATA_TABLE} table")

    def _initialize_read_only_system_schema_and_tables(self) -> None:
        """Initialize the read-only system schema and tables, including the metrics table.
        Raises:
            CatalogError: If the read-only system schema or tables cannot be created.
        """
        cursor = self.connection.cursor()
        try:
            cursor.execute(f'CREATE SCHEMA IF NOT EXISTS "{READ_ONLY_SYSTEM_SCHEMA_NAME}";')

            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS "{READ_ONLY_SYSTEM_SCHEMA_NAME}"."{METRICS_TABLE_NAME}" (
                    index INTEGER PRIMARY KEY,
                    execution_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    execution_time_ms DOUBLE NOT NULL,
                    num_output_rows INTEGER NOT NULL,
                    start_ts TIMESTAMP NOT NULL,
                    end_ts TIMESTAMP NOT NULL,
                    total_lm_cost DOUBLE NOT NULL DEFAULT 0.0,
                    total_lm_uncached_input_tokens INTEGER NOT NULL DEFAULT 0,
                    total_lm_cached_input_tokens INTEGER NOT NULL DEFAULT 0,
                    total_lm_output_tokens INTEGER NOT NULL DEFAULT 0,
                    total_lm_requests INTEGER NOT NULL DEFAULT 0,
                    total_rm_cost DOUBLE NOT NULL DEFAULT 0.0,
                    total_rm_input_tokens INTEGER NOT NULL DEFAULT 0,
                    total_rm_requests INTEGER NOT NULL DEFAULT 0
                );
            """
            )
            
            # Define the schema for the system tables
            metrics_schema = Schema(column_fields=[
                ColumnField(name="index", data_type=IntegerType),
                ColumnField(name="execution_id", data_type=StringType),
                ColumnField(name="session_id", data_type=StringType),
                ColumnField(name="execution_time_ms", data_type=DoubleType),
                ColumnField(name="num_output_rows", data_type=IntegerType),
                ColumnField(name="start_ts", data_type=StringType),  # Store as ISO timestamp string
                ColumnField(name="end_ts", data_type=StringType),    # Store as ISO timestamp string
                ColumnField(name="total_lm_cost", data_type=DoubleType),
                ColumnField(name="total_lm_uncached_input_tokens", data_type=IntegerType),
                ColumnField(name="total_lm_cached_input_tokens", data_type=IntegerType),
                ColumnField(name="total_lm_output_tokens", data_type=IntegerType),
                ColumnField(name="total_lm_requests", data_type=IntegerType),
                ColumnField(name="total_rm_cost", data_type=DoubleType),
                ColumnField(name="total_rm_input_tokens", data_type=IntegerType),
                ColumnField(name="total_rm_requests", data_type=IntegerType),
            ])
            
            # Save the schema to system tables
            self.save_schema(
                cursor=cursor,
                database_name=READ_ONLY_SYSTEM_SCHEMA_NAME,
                table_name=METRICS_TABLE_NAME,
                schema=metrics_schema
            )
            
        except Exception as e:
            raise CatalogError(
                f"Failed to initialize read-only system schema and tables: {e}"
            ) from e

    def _initialize_views_metadata(self) -> None:
        """Initialize the table for storing views metadata.
        Raises:
            CatalogError: If the views metadata table cannot be created.
        """
        cursor = self.connection.cursor()
        try:
            # Create system schema if it doesn't exist
            cursor.execute(f'CREATE SCHEMA IF NOT EXISTS "{SYSTEM_SCHEMA_NAME}";')

            # Create the schema metadata table if it doesn't exist
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS "{SYSTEM_SCHEMA_NAME}"."{VIEWS_METADATA_TABLE}" (
                    database_name TEXT NOT NULL,
                    view_name TEXT NOT NULL,
                    view_blob TEXT NOT NULL,
                    creation_time TIMESTAMP NOT NULL,
                    PRIMARY KEY (database_name, view_name)
                );
            """
            )
        except Exception as e:
            raise CatalogError(
                f"Failed to initialize views and {VIEWS_METADATA_TABLE} table"
            ) from e

        logger.debug(f"Initialized views and {VIEWS_METADATA_TABLE} table")

    def insert_metrics(self, metrics: QueryMetrics) -> None:
        """Append query execution metrics to the metrics table.

        Uses atomic SQL to determine the next index value to prevent race conditions
        in parallel sessions.

        Args:
            metrics: The QueryMetrics instance to store

        Raises:
            CatalogError: If the metrics cannot be saved
        """
        metrics_dict = metrics.to_dict()
        cursor = self.connection.cursor()
        try:
            # trunk-ignore-begin(bandit/B608): No major risk of SQL injection here, this client is not exposed to the user.
            cursor.execute(
                f"""
                INSERT INTO "{READ_ONLY_SYSTEM_SCHEMA_NAME}"."{METRICS_TABLE_NAME}" (
                    index, execution_id, session_id, execution_time_ms, num_output_rows,
                    start_ts, end_ts, total_lm_cost, total_lm_uncached_input_tokens, 
                    total_lm_cached_input_tokens, total_lm_output_tokens, total_lm_requests,
                    total_rm_cost, total_rm_input_tokens, total_rm_requests
                )
                SELECT 
                    COALESCE(MAX(index), 0) + 1,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                FROM "{READ_ONLY_SYSTEM_SCHEMA_NAME}"."{METRICS_TABLE_NAME}"
            """,
                (
                    metrics_dict["execution_id"],
                    metrics_dict["session_id"],
                    metrics_dict["execution_time_ms"],
                    metrics_dict["num_output_rows"],
                    metrics_dict["start_ts"],
                    metrics_dict["end_ts"],
                    metrics_dict["total_lm_cost"],
                    metrics_dict["total_lm_uncached_input_tokens"],
                    metrics_dict["total_lm_cached_input_tokens"],
                    metrics_dict["total_lm_output_tokens"],
                    metrics_dict["total_lm_requests"],
                    metrics_dict["total_rm_cost"],
                    metrics_dict["total_rm_input_tokens"],
                    metrics_dict["total_rm_requests"],
                ),
            )
            # trunk-ignore-end(bandit/B608)

            logger.debug(f"Appended metrics for execution {metrics.execution_id}")
        except Exception as e:
            raise CatalogError(
                f"Failed to append metrics for execution {metrics.execution_id}: {e}"
            ) from e

    def get_metrics_for_session(self, session_id: str) -> Dict[str, float]:
        """Get aggregated metrics and costs for a specific session.

        Args:
            session_id: The session ID to aggregate costs for

        Returns:
            Dictionary containing aggregated cost information

        Raises:
            CatalogError: If there's an error retrieving the costs
        """
        cursor = self.connection.cursor()
        try:
            # trunk-ignore-begin(bandit/B608): No major risk of SQL injection here, this client is not exposed to the user.
            result = cursor.execute(
                f"""
                SELECT 
                    SUM(total_lm_cost) as total_lm_cost,
                    SUM(total_rm_cost) as total_rm_cost,
                    COUNT(*) as query_count,
                    SUM(execution_time_ms) as total_execution_time_ms,
                    SUM(num_output_rows) as total_output_rows
                FROM "{READ_ONLY_SYSTEM_SCHEMA_NAME}"."{METRICS_TABLE_NAME}"
                WHERE session_id = ?
            """,
                (session_id,),
            ).fetchone()
            # trunk-ignore-end(bandit/B608)
            if result is None:
                return {
                    "total_lm_cost": 0.0,
                    "total_rm_cost": 0.0,
                    "query_count": 0,
                    "total_execution_time_ms": 0.0,
                    "total_output_rows": 0,
                }

            return {
                "total_lm_cost": result[0],
                "total_rm_cost": result[1],
                "query_count": result[2],
                "total_execution_time_ms": result[3],
                "total_output_rows": result[4],
            }
            
        except Exception as e:
            raise CatalogError(
                f"Failed to get session aggregate costs for {session_id}: {e}"
            ) from e
