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
from fenic.core._serde.proto.serde_context import SerdeContext
from fenic.core._serde.proto.types import ToolDefinitionProto
from fenic.core.error import CatalogError
from fenic.core.mcp.types import UserDefinedTool
from fenic.core.metrics import QueryMetrics
from fenic.core.types import ColumnField, DatasetMetadata, Schema
from fenic.core.types.datatypes import (
    DoubleType,
    IntegerType,
    StringType,
    TimestampType,
)

# Constants for system schema and table names
SYSTEM_SCHEMA_NAME = "__fenic_system"
SCHEMA_METADATA_TABLE = "table_schemas"
VIEWS_METADATA_TABLE = "table_views"
TOOLS_METADATA_TABLE = "mcp_tools"

# Constants for read-only system schema and tables
READ_ONLY_SYSTEM_SCHEMA_NAME = "fenic_system"
METRICS_TABLE_NAME = "query_metrics"

logger = logging.getLogger(__name__)


class SystemTableClient:
    """Handles storage and retrieval of schema metadata in the system tables. This is particularly important for logical types that can't be directly represented in the physical storage system."""

    def __init__(self, cursor: duckdb.DuckDBPyConnection):
        """Initialize the schema storage with a DuckDB connection.

        Args:
            connection: An initialized DuckDB connection

        Raises:
            CatalogError: If the initialization of tables for schema or view metadata fails
        """
        self._initialize_system_schema(cursor)
        self._initialize_views_metadata(cursor)
        self._initialize_read_only_system_schema_and_tables(cursor)
        self._initialize_tools_metadata(cursor)
        self.serde_context = SerdeContext()

    def save_table(
        self,
        cursor: duckdb.DuckDBPyConnection,
        database_name: str,
        table_name: str,
        schema: Schema,
        description: Optional[str] = None
    ) -> None:
        """Save a table's schema metadata to the system table. This is used for storing logical type information that can't be directly represented in the physical storage.

        Args:
            cursor: A thread-safe DuckDB cursor
            database_name: The name of the database/schema
            table_name: The name of the table
            schema: The schema to store
            description: Optional description of the table
        Raises:
            CatalogError: If the schema cannot be saved
        """
        schema_blob = serialize_schema(schema)
        database_name = normalize_object_name(database_name)
        table_name = normalize_object_name(table_name)

        try:
            if description is None:
                # Preserve existing description if not provided
                existing_desc = self.get_table_description(cursor, database_name, table_name)
                description = existing_desc
            # Upsert the schema - replace if exists
            cursor.execute(
                f"""
                INSERT OR REPLACE INTO "{SYSTEM_SCHEMA_NAME}"."{SCHEMA_METADATA_TABLE}" (
                    database_name, table_name, schema_blob, description
                ) VALUES (?, ?, ?, ?)
            """,
                (database_name, table_name, schema_blob, description),
            )

            logger.debug(f"Saved schema metadata for {database_name}.{table_name}")
        except Exception as e:
            raise CatalogError(
                f"Failed to save schema metadata for {database_name}.{table_name}: {e}"
            ) from e

    def get_table_metadata(self, cursor: duckdb.DuckDBPyConnection, database_name: str, table_name: str) -> Optional[DatasetMetadata]:
        """Retrieve a table's schema metadata from the system table.

        Args:
            cursor: A thread-safe DuckDB cursor
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
                SELECT schema_blob, description
                FROM "{SYSTEM_SCHEMA_NAME}"."{SCHEMA_METADATA_TABLE}"
                WHERE database_name = ? AND table_name = ?
            """,
                (database_name, table_name),
            ).fetchone()
            # trunk-ignore-end(bandit/B608)
            if result is None:
                logger.debug(
                    f"No schema metadata found for {database_name}.{table_name}"
                )
                return None

            schema_blob = result[0]
            description = result[1]
            return DatasetMetadata(schema=deserialize_schema(schema_blob), description=description)

        except Exception as e:
            raise CatalogError(
                f"Failed to retrieve schema metadata for {database_name}.{table_name}: {e}"
            ) from e

    def get_table_description(self, cursor: duckdb.DuckDBPyConnection, database_name: str, table_name: str) -> Optional[str]:
        """Retrieve a table's description, if present."""
        try:
            # trunk-ignore-begin(bandit/B608): Query built from constants; parameters are bound.
            result = cursor.execute(
                f"""
                SELECT description
                FROM "{SYSTEM_SCHEMA_NAME}"."{SCHEMA_METADATA_TABLE}"
                WHERE database_name = ? AND table_name = ?
            """,
                (database_name, table_name),
            ).fetchone()
            # trunk-ignore-end(bandit/B608)
            if result is None:
                return None
            return result[0]
        except Exception as e:
            raise CatalogError(
                f"Failed to retrieve table description for {database_name}.{table_name}: {e}"
            ) from e

    def set_table_description(self, cursor: duckdb.DuckDBPyConnection, database_name: str, table_name: str,
                              description: Optional[str]) -> None:
        """Set or clear a table's description in the system table."""
        try:
            # trunk-ignore-begin(bandit/B608): Query built from constants; parameters are bound.
            cursor.execute(
                f"""
                UPDATE "{SYSTEM_SCHEMA_NAME}"."{SCHEMA_METADATA_TABLE}"
                SET description = ?
                WHERE database_name = ? AND table_name = ?
            """,
                (description, database_name, table_name),
            )
            # trunk-ignore-end(bandit/B608)
        except Exception as e:
            raise CatalogError(
                f"Failed to set table description for {database_name}.{table_name}: {e}"
            ) from e

    def delete_schema(self, cursor: duckdb.DuckDBPyConnection, database_name: str, table_name: str) -> bool:
        """Delete a table's schema metadata from the system table.

        Args:
            cursor: A thread-safe DuckDB cursor
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
            cursor: A thread-safe DuckDB cursor
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
        logical_plan: LogicalPlan,
        description: Optional[str] = None,
    ) -> None:
        database_name = database_name.casefold()
        view_name = view_name.casefold()
        logical_plan_str = base64.b64encode(LogicalPlanSerde.serialize(logical_plan)).decode('utf-8')
        try:
            if description is None:
                # Preserve existing description if not provided
                existing_desc = self.get_view_description(cursor, database_name, view_name)
                description = existing_desc
            cursor.execute(
                f"""
                INSERT OR REPLACE INTO "{SYSTEM_SCHEMA_NAME}"."{VIEWS_METADATA_TABLE}" (
                    database_name, view_name, view_blob, creation_time, description
                ) VALUES (?, ?, ?, ?, ?)
            """,
                (database_name, view_name, logical_plan_str, datetime.now(), description),
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

    def get_view_metadata(
        self,
        cursor: duckdb.DuckDBPyConnection,
        database_name: str,
        view_name: str
    ) -> Optional[DatasetMetadata]:
        """Retrieve a view's description, if present."""
        try:
            # trunk-ignore-begin(bandit/B608): Query built from constants; parameters are bound.
            result = cursor.execute(
                f"""
                SELECT description, view_blob
                FROM "{SYSTEM_SCHEMA_NAME}"."{VIEWS_METADATA_TABLE}"
                WHERE database_name = ? AND view_name = ?
            """,
                (database_name, view_name),
            ).fetchone()
            # trunk-ignore-end(bandit/B608)
            if result is None:
                return None
            view_blob = base64.b64decode(result[1])
            schema = LogicalPlanSerde.deserialize(view_blob).schema()
            return DatasetMetadata(schema=schema, description=result[0])
        except Exception as e:
            raise CatalogError(
                f"Failed to retrieve view metadata for {database_name}.{view_name}"
            ) from e

    def get_view_description(
        self,
        cursor: duckdb.DuckDBPyConnection,
        database_name: str,
        view_name: str
    ) -> Optional[str]:
        """Retrieve a view's description, if present."""
        try:
            # trunk-ignore-begin(bandit/B608): Query built from constants; parameters are bound.
            result = cursor.execute(
                f"""
                SELECT description
                FROM "{SYSTEM_SCHEMA_NAME}"."{VIEWS_METADATA_TABLE}"
                WHERE database_name = ? AND view_name = ?
            """,
                (database_name, view_name),
            ).fetchone()
            # trunk-ignore-end(bandit/B608)
            if result is None:
                return None
            return result[0]
        except Exception as e:
            raise CatalogError(
                f"Failed to retrieve view description for {database_name}.{view_name}"
            ) from e

    def set_view_description(self, cursor: duckdb.DuckDBPyConnection, database_name: str, view_name: str,
                             description: str) -> None:
        """Set a view's description in the system table."""
        try:
            # trunk-ignore-begin(bandit/B608): Query built from constants; parameters are bound.
            cursor.execute(
                f"""
                UPDATE "{SYSTEM_SCHEMA_NAME}"."{VIEWS_METADATA_TABLE}"
                SET description = ?
                WHERE database_name = ? AND view_name = ?
            """,
                (description, database_name, view_name),
            )
            # trunk-ignore-end(bandit/B608)
        except Exception as e:
            raise CatalogError(
                f"Failed to set view description for {database_name}.{view_name}"
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

    def save_tool(self, cursor: duckdb.DuckDBPyConnection, tool: UserDefinedTool) -> None:
        """Save a tool's metadata to the system table.
        Raises:
            CatalogError: If the tool metadata cannot be saved.
        """
        try:
            tool_proto = self.serde_context.serialize_tool_definition(tool)
            tool_blob = base64.b64encode(tool_proto.SerializeToString())
            cursor.execute(
                f"""
                INSERT OR REPLACE INTO "{SYSTEM_SCHEMA_NAME}"."{TOOLS_METADATA_TABLE}" (
                    tool_name, tool_blob
                ) VALUES (?, ?)
            """,
                (tool.name, tool_blob),
            )
        except Exception as e:
            raise CatalogError(
                f"Failed to save tool metadata for {tool.name}"
            ) from e

    def describe_tool(self, cursor: duckdb.DuckDBPyConnection, tool_name: str) -> Optional[UserDefinedTool]:
        """Get a tool's metadata from the system table.
        Raises:
            CatalogError: If the tool metadata cannot be retrieved.
        """
        try:
            result = cursor.execute(
                f"""
                SELECT tool_blob
                FROM "{SYSTEM_SCHEMA_NAME}"."{TOOLS_METADATA_TABLE}"
                WHERE tool_name = ?
            """, (tool_name,),# nosec: B608: No major risk of SQL injection here, because queries run on a client side DuckDB instance.
            ).fetchone()
            if result is None:
                logger.debug(f"No tool found for {tool_name}")
                return None
            return self._deserialize_and_resolve_tool(result)
        except Exception as e:
            raise CatalogError(
                f"Failed to retrieve tool metadata for {tool_name}"
            ) from e

    def list_tools(self, cursor: duckdb.DuckDBPyConnection) -> List[UserDefinedTool]:
        """List all tools in the system table.
        Raises:
            CatalogError: If the tools metadata cannot be retrieved.
        """
        try:
            result = cursor.execute(
                f"""
                SELECT tool_blob
                FROM "{SYSTEM_SCHEMA_NAME}"."{TOOLS_METADATA_TABLE}"
            """, # nosec: B608: No risk of injection, only uses fixed constants.
            ).fetchall()
            return [self._deserialize_and_resolve_tool(row) for row in result]
        except Exception as e:
            raise CatalogError(
                "Failed to list tools"
            ) from e

    def delete_tool(self, cursor: duckdb.DuckDBPyConnection, tool_name: str) -> bool:
        """Delete a tool's metadata from the system table.
        Raises:
            CatalogError: If the tool metadata cannot be deleted.
        """
        try:
            result = cursor.execute(
                f"""
                DELETE FROM "{SYSTEM_SCHEMA_NAME}"."{TOOLS_METADATA_TABLE}"
                WHERE tool_name = ?
            """, (tool_name,) # nosec: B608: No major risk of SQL injection here, because queries run on a client side DuckDB instance.
            ).fetchone()
            if result is None:
                logger.debug(f"No tool found for {tool_name}")
                return False
            return True
        except Exception as e:
            raise CatalogError(
                f"Failed to delete tool metadata for {tool_name}"
            ) from e

    def delete_all_tools(self, cursor: duckdb.DuckDBPyConnection) -> bool:
        """Delete all tools from the system table.
        Raises:
            CatalogError: If the tools metadata cannot be deleted.
        """
        try:
            cursor.execute(
                f"""
                DELETE FROM "{SYSTEM_SCHEMA_NAME}"."{TOOLS_METADATA_TABLE}"
            """, # nosec: B608: No risk of injection, only uses fixed constants.
            )
            return True
        except Exception as e:
            raise CatalogError(
                "Failed to delete all tools"
            ) from e

    def insert_query_metrics(self, cursor: duckdb.DuckDBPyConnection, metrics: QueryMetrics) -> None:
        """Append query execution metrics to the metrics table.

        Args:
            cursor: The thread-safe DuckDB cursor to use to store the metrics.
            metrics: The QueryMetrics instance to store

        Raises:
            CatalogError: If the metrics cannot be saved
        """
        metrics_dict = metrics.to_dict()
        try:
            # trunk-ignore-begin(bandit/B608): No major risk of SQL injection here, this client is not exposed to the user.
            cursor.execute(
                f"""
                INSERT INTO "{READ_ONLY_SYSTEM_SCHEMA_NAME}"."{METRICS_TABLE_NAME}" (
                    execution_id, session_id, execution_time_ms, num_output_rows,
                    start_ts, end_ts, total_lm_cost, total_lm_uncached_input_tokens,
                    total_lm_cached_input_tokens, total_lm_output_tokens, total_lm_requests,
                    total_rm_cost, total_rm_input_tokens, total_rm_requests
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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

    def get_metrics_for_session(self, cursor: duckdb.DuckDBPyConnection, session_id: str) -> Dict[str, float]:
        """Get aggregated metrics and costs for a specific session.

        Args:
            cursor: The thread-safe DuckDB cursor to use to get the metrics for the session.
            session_id: The session ID to aggregate costs for

        Returns:
            Dictionary containing aggregated cost information

        Raises:
            CatalogError: If there's an error retrieving the costs
        """
        try:
            # trunk-ignore-begin(bandit/B608): No major risk of SQL injection here, this client is not exposed to the user.
            result = cursor.execute(
                f"""
                SELECT
                    SUM(total_lm_cost) as total_lm_cost,
                    SUM(total_lm_uncached_input_tokens) as total_lm_uncached_input_tokens,
                    SUM(total_lm_cached_input_tokens) as total_lm_cached_input_tokens,
                    SUM(total_lm_output_tokens) as total_lm_output_tokens,
                    SUM(total_lm_requests) as total_lm_requests,
                    SUM(total_rm_input_tokens) as total_rm_input_tokens,
                    SUM(total_rm_requests) as total_rm_requests,
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
                    "total_lm_uncached_input_tokens": 0,
                    "total_lm_cached_input_tokens": 0,
                    "total_lm_output_tokens": 0,
                    "total_lm_requests": 0,
                    "total_rm_input_tokens": 0,
                    "total_rm_requests": 0,
                    "total_rm_cost": 0.0,
                    "query_count": 0,
                    "total_execution_time_ms": 0.0,
                    "total_output_rows": 0,
                }

            return {
                "total_lm_cost": result[0],
                "total_lm_uncached_input_tokens": result[1],
                "total_lm_cached_input_tokens": result[2],
                "total_lm_output_tokens": result[3],
                "total_lm_requests": result[4],
                "total_rm_input_tokens": result[5],
                "total_rm_requests": result[6],
                "total_rm_cost": result[7],
                "query_count": result[8],
                "total_execution_time_ms": result[9],
                "total_output_rows": result[10],
            }

        except Exception as e:
            raise CatalogError(
                f"Failed to get session aggregate costs for {session_id}: {e}"
            ) from e

    def _initialize_system_schema(self, cursor: duckdb.DuckDBPyConnection) -> None:
        """Initialize the system schema and metadata table for storing table schemas including logical type information.

        Args:
            cursor: The thread-safe DuckDB cursor to use to initialize the system schema and metadata table.

        Raises:
            CatalogError: If the system schema or metadata table cannot be created.
        """
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
                    description TEXT,
                    PRIMARY KEY (database_name, table_name)
                );
            """
            )
        except Exception as e:
            raise CatalogError(
                f"Failed to initialize system schema and {SCHEMA_METADATA_TABLE} table: {e}"
            ) from e

        logger.debug(f"Initialized system schema and {SCHEMA_METADATA_TABLE} table")

    def _initialize_read_only_system_schema_and_tables(self, cursor: duckdb.DuckDBPyConnection) -> None:
        """Initialize the read-only system schema and tables, including the metrics table.

        Args:
            cursor: The thread-safe DuckDB cursor to use to initialize the read-only system schema and tables.

        Raises:
            CatalogError: If the read-only system schema or tables cannot be created.
        """
        try:
            cursor.execute(f'CREATE SCHEMA IF NOT EXISTS "{READ_ONLY_SYSTEM_SCHEMA_NAME}";')

            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS "{READ_ONLY_SYSTEM_SCHEMA_NAME}"."{METRICS_TABLE_NAME}" (
                    execution_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    execution_time_ms DOUBLE NOT NULL,
                    num_output_rows INTEGER NOT NULL,
                    start_ts TIMESTAMPTZ NOT NULL,
                    end_ts TIMESTAMPTZ NOT NULL,
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
                ColumnField(name="execution_id", data_type=StringType),
                ColumnField(name="session_id", data_type=StringType),
                ColumnField(name="execution_time_ms", data_type=DoubleType),
                ColumnField(name="num_output_rows", data_type=IntegerType),
                ColumnField(name="start_ts", data_type=TimestampType),
                ColumnField(name="end_ts", data_type=TimestampType),
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
            self.save_table(
                cursor=cursor,
                database_name=READ_ONLY_SYSTEM_SCHEMA_NAME,
                table_name=METRICS_TABLE_NAME,
                schema=metrics_schema
            )

        except Exception as e:
            raise CatalogError(
                f"Failed to initialize read-only system schema and tables: {e}"
            ) from e

    def _initialize_views_metadata(self, cursor: duckdb.DuckDBPyConnection) -> None:
        """Initialize the table for storing views metadata.

        Args:
            cursor: The thread-safe DuckDB cursor to use to initialize the views metadata table.

        Raises:
            CatalogError: If the views metadata table cannot be created.
        """
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
                    description TEXT,
                    PRIMARY KEY (database_name, view_name)
                );
            """
            )
        except Exception as e:
            raise CatalogError(
                f"Failed to initialize views and {VIEWS_METADATA_TABLE} table"
            ) from e

        logger.debug(f"Initialized views and {VIEWS_METADATA_TABLE} table")

    def _initialize_tools_metadata(self, cursor: duckdb.DuckDBPyConnection) -> None:
        """Initialize the table for storing tools metadata.
        Raises:
            CatalogError: If the tools metadata table cannot be created.
        """
        try:
            # Create system schema if it doesn't exist
            cursor.execute(f'CREATE SCHEMA IF NOT EXISTS "{SYSTEM_SCHEMA_NAME}";')

            # Create the tools metadata table if it doesn't exist
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS "{SYSTEM_SCHEMA_NAME}"."{TOOLS_METADATA_TABLE}" (
                    tool_name TEXT NOT NULL,
                    tool_blob TEXT NOT NULL,
                    PRIMARY KEY (tool_name)
                );
            """
            )
        except Exception as e:
            raise CatalogError(
                f"Failed to initialize tools and {TOOLS_METADATA_TABLE} table"
            ) from e

    def _deserialize_and_resolve_tool(self, row: tuple) -> UserDefinedTool:
       decoded_tool = base64.b64decode(row[0])
       proto_tool = ToolDefinitionProto.FromString(decoded_tool)
       return self.serde_context.deserialize_tool_definition(proto_tool)
