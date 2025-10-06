import logging
import threading
from typing import Dict, List, Optional

import duckdb
import polars as pl

from fenic._backends.local.physical_plan.utils import apply_ingestion_coercions
from fenic._backends.local.system_table_client import (
    READ_ONLY_SYSTEM_SCHEMA_NAME,
    SYSTEM_SCHEMA_NAME,
    SystemTableClient,
)
from fenic._backends.utils.catalog_utils import (
    DBIdentifier,
    TableIdentifier,
    compare_object_names,
)
from fenic.core._interfaces.catalog import BaseCatalog
from fenic.core._logical_plan.plans.base import LogicalPlan
from fenic.core._utils.misc import generate_unique_view_name
from fenic.core._utils.schema import (
    convert_custom_schema_to_polars_schema,
)
from fenic.core.error import (
    CatalogError,
    DatabaseAlreadyExistsError,
    DatabaseNotFoundError,
    InternalError,
    TableAlreadyExistsError,
    TableNotFoundError,
    ToolAlreadyExistsError,
    ToolNotFoundError,
)
from fenic.core.mcp._tools import bind_tool
from fenic.core.mcp.types import ToolParam, UserDefinedTool
from fenic.core.metrics import QueryMetrics
from fenic.core.types import (
    DatasetMetadata,
    Schema,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

DB_IGNORE_LIST = [
    "main",
    "information_schema",
    "pg_catalog",
    SYSTEM_SCHEMA_NAME,
]
DEFAULT_CATALOG_NAME = "typedef_default"
DEFAULT_DATABASE_NAME = "typedef_default"


class DuckDBTransaction:
    """A context manager for DuckDB transactions."""

    def __init__(self, connection: duckdb.DuckDBPyConnection):
        self.connection = connection

    def __enter__(self):
        """Start a transaction when entering the context."""
        logger.debug("Beginning DuckDB transaction")
        self.connection.execute("BEGIN TRANSACTION")
        return self.connection

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Commit transaction if no exceptions, otherwise rollback and log error."""
        if exc_type is None:
            # No exception occurred, commit the transaction
            logger.debug("Committing DuckDB transaction")
            self.connection.execute("COMMIT")
        else:
            # Exception occurred, rollback the transaction and log error
            logger.error(
                f"Rolling back DuckDB transaction due to error: {exc_type.__name__}: {str(exc_val)}"
            )
            self.connection.execute("ROLLBACK")
            # Don't suppress the exception
            return False


class LocalCatalog(BaseCatalog):
    """A catalog for local execution mode implementing BaseCatalog.

    All table reads and writes go through this class for unified table name canonicalization.

    Thread Safety:
    - DuckDB handles concurrent table read/write access internally via MVCC and optimistic concurrency control
    - Locking Rules:
    * Write operations (create/drop/update) are locked to prevent race conditions in check-then-act patterns
    * Read operations do NOT require locks (DuckDB handles concurrent reads via MVCC)
    * Catalog metadata operations (e.g., current database access/modification) are protected by locks
    - Each thread must use its own cursor for concurrent operations to avoid segfaults
    See: https://duckdb.org/docs/stable/guides/python/multiple_threads.html#reader-and-writer-functions
    """

    def __init__(self, connection: duckdb.DuckDBPyConnection):
        self.db_conn: duckdb.DuckDBPyConnection = connection
        self.lock = threading.RLock()
        self.create_database(DEFAULT_DATABASE_NAME)
        self.current_database = DEFAULT_DATABASE_NAME
        self.system_tables = SystemTableClient(self.db_conn.cursor())

    def does_catalog_exist(self, catalog_name: str) -> bool:
        """Checks if a catalog with the specified name exists."""
        return compare_object_names(catalog_name, DEFAULT_CATALOG_NAME)

    def get_current_catalog(self) -> str:
        """Get the name of the current catalog."""
        return DEFAULT_CATALOG_NAME

    def set_current_catalog(self, catalog_name: str) -> None:
        """Set the current catalog."""
        if not compare_object_names(catalog_name, DEFAULT_CATALOG_NAME):
            raise CatalogError(
                f"Invalid catalog name '{catalog_name}'. Only the default catalog '{DEFAULT_CATALOG_NAME}' is supported in local execution mode."
            )
        # No actual action needed to set the catalog in this local setup

    def list_catalogs(self) -> List[str]:
        """Get a list of all catalogs."""
        return [DEFAULT_CATALOG_NAME]

    def create_catalog(self, catalog_name: str, ignore_if_exists: bool = True) -> bool:
        """Creates a new catalog."""
        raise CatalogError(
            "Catalog creation is not supported in local execution mode."
            f"Only one catalog: '{DEFAULT_CATALOG_NAME}' is supported in local execution mode.")

    def drop_catalog(
        self, catalog_name: str, ignore_if_not_exists: bool = True
    ) -> bool:
        """Drops a catalog."""
        raise CatalogError(
            "Catalog deletion is not supported in local execution mode."
            f"Only one catalog: '{DEFAULT_CATALOG_NAME}' is supported in local execution mode.")

    def does_database_exist(self, database_name: str) -> bool:
        """Checks if a database with the specified name exists."""
        with self.lock:
            db_identifier = DBIdentifier.from_string(database_name).enrich(self.get_current_catalog())
            _verify_db_catalog(db_identifier)
            return self._does_database_exist(self.db_conn.cursor(), db_identifier.db)

    def get_current_database(self) -> str:
        """Get the name of the current database in the current catalog."""
        with self.lock:
            try:
                return self.current_database
            except Exception as e:
                raise CatalogError("Failed to get current database") from e

    def create_database(
        self, database_name: str, ignore_if_exists: bool = True
    ) -> bool:
        """Create a new database."""
        with self.lock:
            db_identifier = DBIdentifier.from_string(database_name).enrich(self.get_current_catalog())
            _verify_db_catalog(db_identifier)
            cursor = self.db_conn.cursor()
            if self._does_database_exist(cursor, db_identifier.db):
                if ignore_if_exists:
                    return False
                raise DatabaseAlreadyExistsError(database_name)

            try:
                cursor.execute(f'CREATE SCHEMA IF NOT EXISTS "{db_identifier.db}";')
                return True
            except Exception as e:
                raise CatalogError(f"Failed to create database: {database_name}") from e

    def drop_database(
        self,
        database_name: str,
        cascade: bool = False,
        ignore_if_not_exists: bool = True,
    ) -> bool:
        """Drop a database."""
        with self.lock:
            db_identifier = DBIdentifier.from_string(database_name).enrich(self.get_current_catalog())
            _verify_db_catalog(db_identifier)
            if db_identifier.is_db_name_equal(READ_ONLY_SYSTEM_SCHEMA_NAME):
                raise CatalogError(
                    f"Cannot drop read-only system database '{READ_ONLY_SYSTEM_SCHEMA_NAME}'"
                )
            if db_identifier.is_db_name_equal(self.get_current_database()):
                raise CatalogError(
                    f"Cannot drop the current database '{database_name}'. Switch to another database first."
                )
            cursor = self.db_conn.cursor()
            if not self._does_database_exist(cursor, db_identifier.db):
                if ignore_if_not_exists:
                    return False
                raise DatabaseNotFoundError(database_name)
            try:
                with DuckDBTransaction(cursor):
                    if cascade:
                        cursor.execute(
                            f'DROP SCHEMA IF EXISTS "{db_identifier.db}" CASCADE;'
                        )
                        self.system_tables.delete_database_schemas(cursor, db_identifier.db)
                        self.system_tables.delete_database_views(cursor, db_identifier.db)
                    else:
                        if self.system_tables.list_views(cursor, db_identifier.db):
                            raise CatalogError(
                                f"Cannot drop database '{database_name}' because it contains views. Use CASCADE to drop the database and all its views."
                            )
                        cursor.execute(
                            f'DROP SCHEMA IF EXISTS "{db_identifier.db}";'
                        )
                return True
            except CatalogError:
                raise
            except Exception as e:
                raise CatalogError(f"Failed to drop database: {database_name}") from e

    def list_databases(self) -> List[str]:
        """Get a list of all databases in the current catalog."""
        try:
            cursor = self.db_conn.cursor()
            schemas = cursor.execute(
                "SELECT schema_name FROM duckdb_schemas();"
            ).fetchall()
            return [
                schema[0] for schema in schemas if schema[0] not in DB_IGNORE_LIST
            ]
        except Exception as e:
            raise CatalogError("Failed to list databases") from e

    def set_current_database(self, database_name: str) -> None:
        """Set the current database in the current catalog."""
        with self.lock:
            db_identifier = DBIdentifier.from_string(database_name).enrich(self.get_current_catalog())
            _verify_db_catalog(db_identifier)
            if not self._does_database_exist(self.db_conn.cursor(), db_identifier.db):
                raise DatabaseNotFoundError(database_name)
            self.current_database = db_identifier.db

    def does_table_exist(self, table_name: str) -> bool:
        """Checks if a table with the specified name exists."""
        table_identifier = TableIdentifier.from_string(table_name).enrich(
            self.get_current_catalog(),
            self.get_current_database())
        _verify_table_catalog(table_identifier)
        return self._does_table_exist(self.db_conn.cursor(), table_identifier)

    def does_view_exist(self, view_name: str) -> bool:
        """Checks if a view with the specified name exists in the current database."""
        view_identifier = TableIdentifier.from_string(view_name).enrich(
            self.get_current_catalog(),
            self.get_current_database())
        _verify_table_catalog(view_identifier)
        try:
            views = self.system_tables.get_view(self.db_conn.cursor(), view_identifier.db, view_identifier.table)
            return views is not None
        except Exception as e:
            raise CatalogError(
                f"Failed to check if view: `{view_identifier.db}.{view_identifier.table}` exists"
            ) from e

    def list_tables(self) -> List[str]:
        """Get a list of all tables in the current database."""
        current_db = self.get_current_database()
        cursor = self.db_conn.cursor()
        try:
            result = cursor.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = ?
                    AND table_type = 'BASE TABLE'
                """,
                (current_db,),
            )
            result_list = result.fetchall()

            if len(result_list) > 0:
                return [str(element[0]) for element in result_list]
            return []
        except Exception as e:
            raise CatalogError(
                f"Failed to list tables in database '{current_db}'"
            ) from e

    def list_views(self) -> List[str]:
        """Get a list of all views in the current database."""
        current_db = self.get_current_database()
        try:
            result_list = self.system_tables.list_views(self.db_conn.cursor(), current_db)

            if len(result_list) > 0:
                return [str(element[0]) for element in result_list]
            return []
        except Exception as e:
            raise CatalogError(
                f"Failed to list views in database '{self.get_current_database()}'"
            ) from e

    # Descriptions
    def set_table_description(self, table_name: str, description: Optional[str]) -> None:
        """Set the description for a table."""
        with self.lock:
            table_identifier = TableIdentifier.from_string(table_name).enrich(
                self.get_current_catalog(),
                self.get_current_database())
            _verify_table_catalog(table_identifier)
            if not self._does_table_exist(self.db_conn.cursor(), table_identifier):
                raise TableNotFoundError(table_identifier.table, table_identifier.db)
            cursor = self.db_conn.cursor()
            try:
                self.system_tables.set_table_description(cursor, table_identifier.db, table_identifier.table, description)
            except Exception as e:
                raise CatalogError(
                    f"Failed to set description for table: `{table_identifier.db}.{table_identifier.table}`"
                ) from e

    def get_table_description(self, table_name: str) -> Optional[str]:
        """Get description of the specified table."""
        table_identifier = TableIdentifier.from_string(table_name).enrich(
            self.get_current_catalog(),
            self.get_current_database())
        _verify_table_catalog(table_identifier)
        return self.system_tables.get_table_description(self.db_conn.cursor(), table_identifier.db, table_identifier.table)

    def describe_view(self, view_name: str) -> DatasetMetadata:
        """Get the schema and description of the specified view."""
        view_identifier = TableIdentifier.from_string(view_name).enrich(
            self.get_current_catalog(),
            self.get_current_database())
        _verify_table_catalog(view_identifier)
        return self.system_tables.get_view_metadata(self.db_conn.cursor(), view_identifier.db, view_identifier.table)

    def describe_table(self, table_name: str) -> DatasetMetadata:
        """Get the schema and description of the specified table."""
        table_identifier = TableIdentifier.from_string(table_name).enrich(
            self.get_current_catalog(),
            self.get_current_database())
        _verify_table_catalog(table_identifier)
        maybe_table_metadata = self.system_tables.get_table_metadata(
            self.db_conn.cursor(), table_identifier.db, table_identifier.table
        )
        if maybe_table_metadata is None:
            raise TableNotFoundError(table_identifier.table, table_identifier.db)
        return maybe_table_metadata

    def get_view_plan(self, view_name: str) -> LogicalPlan:
        """Get the LogicalPlan for the specified view."""
        view_identifier = TableIdentifier.from_string(view_name).enrich(
            self.get_current_catalog(),
            self.get_current_database())
        _verify_table_catalog(view_identifier)
        try:
            maybe_views = self.system_tables.get_view(
                self.db_conn.cursor(), view_identifier.db, view_identifier.table
            )
            if maybe_views is None:
                raise TableNotFoundError(view_identifier.table, view_identifier.db)
            return maybe_views
        except Exception as e:
            raise CatalogError(f"Failed to describe view: {view_name}") from e

    def drop_table(self, table_name: str, ignore_if_not_exists: bool = True) -> bool:
        """Drop a table."""
        with self.lock:
            table_identifier = TableIdentifier.from_string(table_name).enrich(
                self.get_current_catalog(),
                self.get_current_database())
            _verify_table_catalog(table_identifier)
            if table_identifier.is_db_name_equal(READ_ONLY_SYSTEM_SCHEMA_NAME):
                raise CatalogError(
                    f"Cannot drop table '{table_identifier}' from read-only system database"
                )
            cursor = self.db_conn.cursor()
            if not self._does_table_exist(cursor, table_identifier):
                if not ignore_if_not_exists:
                    raise TableNotFoundError(table_identifier.table, table_identifier.db)
                return False
            try:
                with DuckDBTransaction(cursor):
                    sql = f"DROP TABLE IF EXISTS {table_identifier.build_qualified_table_name()}"
                    cursor.execute(sql)
                    self.system_tables.delete_schema(cursor, table_identifier.db, table_identifier.table)
                return True
            except Exception as e:
                raise CatalogError(
                    f"Failed to drop table: `{table_identifier.db}.{table_identifier.table}`"
                ) from e

    def drop_view(self, view_name: str, ignore_if_not_exists: bool = True) -> bool:
        """Drop a view from the current database."""
        with self.lock:
            view_identifier = TableIdentifier.from_string(view_name).enrich(
                self.get_current_catalog(),
                self.get_current_database())
            _verify_table_catalog(view_identifier)
            if view_identifier.is_db_name_equal(READ_ONLY_SYSTEM_SCHEMA_NAME):
                raise CatalogError(
                    f"Cannot drop view '{view_identifier}' from read-only system database"
                )
            cursor = self.db_conn.cursor()
            if not self._does_view_exist(cursor, view_identifier):
                if ignore_if_not_exists:
                    return False
                raise TableNotFoundError(view_identifier.table, view_identifier.db)
            try:
                with DuckDBTransaction(cursor):
                    self.system_tables.delete_view(cursor, view_identifier.db, view_identifier.table)
                return True
            except Exception as e:
                raise CatalogError(
                    f"Failed to drop view: `{view_identifier.db}.{view_identifier.table}`"
                ) from e

    def create_table(
        self, table_name: str, schema: Schema, ignore_if_exists: bool = True, description: Optional[str] = None
    ) -> bool:
        """Create a new table."""
        temp_view_name = generate_unique_view_name()
        with self.lock:
            table_identifier = TableIdentifier.from_string(table_name).enrich(
                self.get_current_catalog(),
                self.get_current_database())
            _verify_table_catalog(table_identifier)
            if table_identifier.is_db_name_equal(READ_ONLY_SYSTEM_SCHEMA_NAME):
                raise CatalogError(
                    f"Cannot create table '{table_identifier}' in read-only system database"
                )
            cursor = self.db_conn.cursor()
            if self._does_table_exist(cursor, table_identifier):
                if ignore_if_exists:
                    return False
                raise TableAlreadyExistsError(table_identifier.table, table_identifier.db)
            polars_schema = convert_custom_schema_to_polars_schema(schema)
            try:
                with DuckDBTransaction(cursor):
                    cursor.register(
                        temp_view_name, pl.DataFrame(schema=polars_schema)
                    )
                    # trunk-ignore-begin(bandit/B608): No major risk of SQL injection here, because queries run on a client side DuckDB instance.
                    cursor.execute(
                        f"CREATE TABLE IF NOT EXISTS {table_identifier.build_qualified_table_name()} AS SELECT * FROM {temp_view_name} WHERE 1=0"
                    )
                    self.system_tables.save_table(
                        cursor, table_identifier.db, table_identifier.table, schema, description
                    )
                return True
            except Exception as e:
                raise CatalogError(
                    f"Failed to create table: `{table_identifier.db}.{table_identifier.table}`"
                ) from e
            finally:
                try:
                    cursor.execute(f"DROP VIEW IF EXISTS {temp_view_name}")
                except Exception:
                    logger.error(f"Failed to drop view: {temp_view_name}")
                    pass
                # trunk-ignore-end(bandit/B608)

    def create_view(
        self,
        view_name: str,
        logical_plan: LogicalPlan,
        ignore_if_exists: bool = True,
        description: Optional[str] = None,
    ) -> bool:
        """Create a new view in the current database."""
        with self.lock:
            view_identifier = TableIdentifier.from_string(view_name).enrich(
                self.get_current_catalog(),
                self.get_current_database())
            _verify_table_catalog(view_identifier)
            if view_identifier.is_db_name_equal(READ_ONLY_SYSTEM_SCHEMA_NAME):
                raise CatalogError(
                    f"Cannot create view '{view_identifier}' in read-only system database"
                )
            try:
                cursor = self.db_conn.cursor()
                if self._does_view_exist(cursor, view_identifier):
                    if ignore_if_exists:
                        return False
                    raise ValueError(f"View {view_name} already exists!")
                with DuckDBTransaction(cursor):
                    self.system_tables.save_view(
                        cursor, view_identifier.db, view_identifier.table, logical_plan, description)
                    return True
            except Exception as e:
                raise CatalogError(
                    f"Failed to create view: `{view_identifier.db}.{view_identifier.table}`"
                ) from e

    def set_view_description(self, view_name: str, description: Optional[str]) -> bool:
        """Set the description for a view."""
        with self.lock:
            view_identifier = TableIdentifier.from_string(view_name).enrich(
                self.get_current_catalog(),
                self.get_current_database())
            _verify_table_catalog(view_identifier)
            cursor = self.db_conn.cursor()
            if not self._does_view_exist(cursor, view_identifier):
                raise TableNotFoundError(view_identifier.table, view_identifier.db)
            try:
                self.system_tables.set_view_description(cursor, view_identifier.db, view_identifier.table, description)
                return True
            except Exception as e:
                raise CatalogError(
                    f"Failed to set description for view: `{view_identifier.db}.{view_identifier.table}`"
                ) from e


    def describe_tool(self, tool_name: str) -> Optional[UserDefinedTool]:
        """Get a tool's metadata from the system table."""
        cursor = self.db_conn.cursor()
        existing_tool = self.system_tables.describe_tool(cursor, tool_name)
        if not existing_tool:
            raise ToolNotFoundError(tool_name)
        return existing_tool

    def create_tool(
        self,
        tool_name: str,
        tool_description: str,
        tool_params: List[ToolParam],
        tool_query: "LogicalPlan",
        result_limit: int = 50,
        ignore_if_exists: bool = True
    ) -> bool:
        """Create a new tool in the current catalog."""
        # Ensure the tool is valid by resolving it.
        tool_definition = bind_tool(tool_name, tool_description, tool_params, result_limit, tool_query)
        cursor = self.db_conn.cursor()
        if self.system_tables.describe_tool(cursor, tool_name):
            if ignore_if_exists:
                return False
            raise ToolAlreadyExistsError(tool_name)
        self.system_tables.save_tool(cursor, tool_definition)
        return True

    def list_tools(self) -> List[UserDefinedTool]:
        """List all tools in the current catalog."""
        cursor = self.db_conn.cursor()
        return self.system_tables.list_tools(cursor)

    def drop_tool(self, tool_name: str, ignore_if_not_exists: bool = True) -> bool:
        """Drop a tool from the current catalog."""
        with self.lock:
            cursor = self.db_conn.cursor()
            if not self.system_tables.describe_tool(cursor, tool_name):
                if ignore_if_not_exists:
                    return False
                raise ToolNotFoundError(tool_name)
            return self.system_tables.delete_tool(cursor, tool_name)

    def write_df_to_table(self, df: pl.DataFrame, table_name: str, schema: Schema):
        """Write a Polars dataframe to a table in the current database."""
        temp_view_name = generate_unique_view_name()
        with self.lock:
            table_identifier = TableIdentifier.from_string(table_name).enrich(
                self.get_current_catalog(),
                self.get_current_database())
            _verify_table_catalog(table_identifier)

            if table_identifier.is_db_name_equal(READ_ONLY_SYSTEM_SCHEMA_NAME):
                raise CatalogError(
                    f"Cannot write to table '{table_identifier}' in read-only system database"
                )
            cursor = self.db_conn.cursor()
            try:
                # trunk-ignore-begin(bandit/B608)
                with DuckDBTransaction(cursor):
                    cursor.register(temp_view_name, df)
                    cursor.execute(
                        f"CREATE TABLE IF NOT EXISTS {table_identifier.build_qualified_table_name()} AS SELECT * FROM {temp_view_name}"
                    )
                    self.system_tables.save_table(
                        cursor, table_identifier.db, table_identifier.table, schema
                    )
            except Exception as e:
                raise CatalogError(
                    f"Failed to create table and write dataframe: `{table_identifier.db}.{table_identifier.table}`"
                ) from e
            finally:
                try:
                    cursor.execute(f"DROP VIEW IF EXISTS {temp_view_name}")
                except Exception:
                    logger.error(f"Failed to drop view: {temp_view_name}")
                    pass
            # trunk-ignore-end(bandit/B608)

    def insert_df_to_table(self, df: pl.DataFrame, table_name: str, schema: Schema):
        """Insert a Polars dataframe into a table in the current database."""
        temp_view_name = generate_unique_view_name()
        table_identifier = TableIdentifier.from_string(table_name).enrich(
            self.get_current_catalog(),
            self.get_current_database())
        _verify_table_catalog(table_identifier)
        if table_identifier.is_db_name_equal(READ_ONLY_SYSTEM_SCHEMA_NAME):
            raise CatalogError(
                f"Cannot insert into table '{table_identifier}' in read-only system database"
            )
        cursor = self.db_conn.cursor()
        if self._does_table_exist(cursor, table_identifier):
            existing_table_metadata = self.system_tables.get_table_metadata(cursor, table_identifier.db, table_identifier.table)
            existing_schema = existing_table_metadata.schema if existing_table_metadata else None
            if not existing_schema:
                raise InternalError(f"Schema for table '{table_name}' does not exist, but table exists.")
            if existing_schema != schema:
                raise CatalogError(
                    f"Table '{table_name}' already exists with a different schema!\n"
                    f"Existing schema: {existing_schema}\n"
                    f"New schema: {schema}\n"
                    "To replace the existing table, use mode='overwrite'."
                )
        try:
            # trunk-ignore-begin(bandit/B608)
            cursor.register(temp_view_name, df)
            cursor.execute(
                f"INSERT INTO {table_identifier.build_qualified_table_name()} SELECT * FROM {temp_view_name}"
            )
        except Exception as e:
            raise CatalogError(
                f"Failed to insert dataframe into table: `{table_identifier.db}.{table_identifier.table}`"
            ) from e
        finally:
            try:
                cursor.execute(f"DROP VIEW IF EXISTS {temp_view_name}")
            except Exception:
                logger.error(f"Failed to drop view: {temp_view_name}")
                pass
        # trunk-ignore-end(bandit/B608)

    def replace_table_with_df(self, df: pl.DataFrame, table_name: str, schema: Schema):
        """Replace a table in the current database with a Polars dataframe."""
        temp_view_name = generate_unique_view_name()
        table_identifier = TableIdentifier.from_string(table_name).enrich(
            self.get_current_catalog(),
            self.get_current_database())
        _verify_table_catalog(table_identifier)
        if table_identifier.is_db_name_equal(READ_ONLY_SYSTEM_SCHEMA_NAME):
            raise CatalogError(
                f"Cannot replace table '{table_identifier}' in read-only system database"
            )
        cursor = self.db_conn.cursor()
        try:
            # trunk-ignore-begin(bandit/B608)
            with DuckDBTransaction(cursor):
                cursor.register(temp_view_name, df)
                cursor.execute(
                    f"CREATE OR REPLACE TABLE {table_identifier.build_qualified_table_name()} AS SELECT * FROM {temp_view_name}"
                )
                self.system_tables.save_table(
                    cursor, table_identifier.db, table_identifier.table, schema
                )
        except Exception as e:
            raise CatalogError(
                f"Failed to overwrite table: `{table_identifier.db}.{table_identifier.table}`"
            ) from e
        finally:
            try:
                cursor.execute(f"DROP VIEW IF EXISTS {temp_view_name}")
            except Exception:
                logger.error(f"Failed to drop view: {temp_view_name}")
                pass
        # trunk-ignore-end(bandit/B608)

    def read_df_from_table(self, table_name: str) -> pl.DataFrame:
        """Read a Polars dataframe from a DuckDB table in the current database."""
        table_identifier = TableIdentifier.from_string(table_name).enrich(
            self.get_current_catalog(),
            self.get_current_database())
        _verify_table_catalog(table_identifier)
        try:
            # trunk-ignore-begin(bandit/B608)
            result = self.db_conn.cursor().execute(
                f"SELECT * FROM {table_identifier.build_qualified_table_name()}"
            ).pl()
            return apply_ingestion_coercions(result, coerce_array=False)
            # trunk-ignore-end(bandit/B608)
        except Exception as e:
            raise CatalogError(
                f"Failed to read dataframe from table: `{table_identifier.db}.{table_identifier.table}`"
            ) from e

    def insert_query_metrics(self, metrics: QueryMetrics) -> None:
        """Insert metrics into the metrics system read-only table."""
        self.system_tables.insert_query_metrics(self.db_conn.cursor(), metrics)

    def get_metrics_for_session(self, session_id: str) -> Dict[str, float]:
        """Get metrics for a specific session from the metrics system read-only table."""
        return self.system_tables.get_metrics_for_session(self.db_conn.cursor(), session_id)

    def _does_table_exist(self, cursor: duckdb.DuckDBPyConnection, table_identifier: TableIdentifier) -> bool:
        try:
            return cursor.execute(
                "SELECT 1 FROM information_schema.tables WHERE table_schema = ? AND table_name = ?",
                (table_identifier.db, table_identifier.table)
            ).fetchone() is not None
        except Exception as e:
            raise CatalogError(
                f"Failed to check if table: {table_identifier.db}.{table_identifier.table} exists"
            ) from e

    def _does_database_exist(self, cursor: duckdb.DuckDBPyConnection, database_name: str) -> bool:
        try:
            return cursor.execute(
                "SELECT * FROM duckdb_schemas() WHERE schema_name = ?",
                (database_name,)
            ).fetchone() is not None
        except Exception as e:
            raise CatalogError(
                f"Failed to check if database: {database_name} exists"
            ) from e

    def _does_view_exist(self, cursor: duckdb.DuckDBPyConnection, view_identifier: TableIdentifier) -> bool:
        try:
            views = self.system_tables.get_view(cursor, view_identifier.db, view_identifier.table)
            return views is not None
        except Exception as e:
            raise CatalogError(
                f"Failed to check if view: `{view_identifier.db}.{view_identifier.table}` exists"
            ) from e

def _verify_table_catalog(table_identifier: TableIdentifier) -> None:
    if not table_identifier.is_catalog_name_equal(DEFAULT_CATALOG_NAME):
        raise CatalogError(
            f"Invalid catalog name '{table_identifier.catalog}' in table name '{table_identifier.db}.{table_identifier.table}'. "
            f"Local execution mode only supports the default catalog '{DEFAULT_CATALOG_NAME}'. "
            f"Use table names like 'table', 'schema.table', or '{DEFAULT_CATALOG_NAME}.schema.table' instead."
        )

def _verify_db_catalog(db_identifier: DBIdentifier) -> None:
    if not db_identifier.is_catalog_name_equal(DEFAULT_CATALOG_NAME):
        raise CatalogError(
            f"Invalid catalog name '{db_identifier.catalog}' in database name '{db_identifier.db}'. "
            f"Local execution mode only supports the default catalog '{DEFAULT_CATALOG_NAME}'. "
            f"Use database names like 'database' or '{DEFAULT_CATALOG_NAME}.database' instead."
        )
