import logging
import threading
from typing import List

import duckdb
import polars as pl

from fenic._backends.local.schema_storage import (
    SYSTEM_SCHEMA_NAME,
    SchemaStorage,
)
from fenic._backends.utils.catalog_utils import (
    DBIdentifier,
    TableIdentifier,
    compare_object_names,
)
from fenic.core._interfaces.catalog import BaseCatalog
from fenic.core._utils.misc import generate_unique_arrow_view_name
from fenic.core._utils.schema import convert_custom_schema_to_polars_schema
from fenic.core.error import (
    CatalogError,
    DatabaseAlreadyExistsError,
    DatabaseNotFoundError,
    TableAlreadyExistsError,
    TableNotFoundError,
)
from fenic.core.types import (
    Schema,
)

logger = logging.getLogger(__name__)

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

    def __init__(self, connection):
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
    """A catalog for local execution mode. Implements the BaseCatalog - all table reads and writes should go through this class for unified table name canonicalization."""

    def __init__(self, connection: duckdb.DuckDBPyConnection):
        self.db_conn: duckdb.DuckDBPyConnection = connection
        self.lock = threading.RLock()
        self.create_database(DEFAULT_DATABASE_NAME)
        self.set_current_database(DEFAULT_DATABASE_NAME)
        self.schema_storage = SchemaStorage(self.db_conn)
        self.schema_storage.initialize_system_schema()

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
            try:
                schema = self.db_conn.execute(
                    "SELECT schema_name FROM duckdb_schemas() WHERE schema_name = ?;",
                    (db_identifier.db,),
                ).fetchone()
                return schema is not None
            except Exception as e:
                raise CatalogError(
                    f"Failed to check if database exists: {database_name}"
                ) from e

    def get_current_database(self) -> str:
        """Get the name of the current database in the current catalog."""
        with self.lock:
            try:
                return self.db_conn.execute("SELECT current_schema();").fetchone()[0]
            except Exception as e:
                raise CatalogError("Failed to get current database") from e

    def create_database(
        self, database_name: str, ignore_if_exists: bool = True
    ) -> bool:
        """Create a new database."""
        with self.lock:
            if self.does_database_exist(database_name):
                if ignore_if_exists:
                    return False
                raise DatabaseAlreadyExistsError(database_name)
            db_identifier = DBIdentifier.from_string(database_name).enrich(self.get_current_catalog())
            _verify_db_catalog(db_identifier)
            try:
                self.db_conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{db_identifier.db}";')
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
            if db_identifier.is_db_name_equal(self.get_current_database()):
                raise CatalogError(
                    f"Cannot drop the current database '{database_name}'. Switch to another database first."
                )
            if not self.does_database_exist(database_name):
                if not ignore_if_not_exists:
                    raise DatabaseNotFoundError(database_name)
                return False
            try:
                with DuckDBTransaction(self.db_conn):
                    if cascade:
                        self.db_conn.execute(
                            f'DROP SCHEMA IF EXISTS "{db_identifier.db}" CASCADE;'
                        )
                    else:
                        self.db_conn.execute(
                            f'DROP SCHEMA IF EXISTS "{db_identifier.db}";'
                        )
                    self.schema_storage.delete_database_schemas(db_identifier.db)
                return True
            except Exception as e:
                raise CatalogError(f"Failed to drop database: {database_name}") from e

    def list_databases(self) -> List[str]:
        """Get a list of all databases in the current catalog."""
        with self.lock:
            try:
                schemas = self.db_conn.execute(
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
            if not self.does_database_exist(database_name):
                raise DatabaseNotFoundError(database_name)
            try:
                self.db_conn.execute(f'USE "{database_name}";')
            except Exception as e:
                raise CatalogError("Failed to set current database") from e

    def does_table_exist(self, table_name: str) -> bool:
        """Checks if a table with the specified name exists."""
        with self.lock:
            table_identifier = TableIdentifier.from_string(table_name).enrich(
                self.get_current_catalog(),
                self.get_current_database())
            _verify_table_catalog(table_identifier)

            try:
                table = self.db_conn.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = ? AND table_name = ?;",
                    (table_identifier.db, table_identifier.table),
                ).fetchone()
                return table is not None
            except Exception as e:
                raise CatalogError(
                    f"Failed to check if table: `{table_identifier.db}.{table_identifier.table}` exists"
                ) from e

    def list_tables(self) -> List[str]:
        """Get a list of all tables in the current database."""
        with self.lock:
            try:
                result = self.db_conn.execute(
                    """
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = ? AND table_type = 'BASE TABLE'
                    """,
                    (self.get_current_database(),),
                )
                result_list = result.fetchall()

                if len(result_list) > 0:
                    return [str(element[0]) for element in result_list]
                return []
            except Exception as e:
                raise CatalogError(
                    f"Failed to list tables in database '{self.get_current_database()}'"
                ) from e

    def describe_table(self, table_name: str) -> Schema:
        """Get the schema of the specified table."""
        with self.lock:
            table_identifier = TableIdentifier.from_string(table_name).enrich(
                self.get_current_catalog(),
                self.get_current_database())
            _verify_table_catalog(table_identifier)

            maybe_schema = self.schema_storage.get_schema(
                table_identifier.db, table_identifier.table
            )
            if maybe_schema is None:
                raise TableNotFoundError(table_identifier.table, table_identifier.db)
            return maybe_schema

    def drop_table(self, table_name: str, ignore_if_not_exists: bool = True) -> bool:
        """Drop a table."""
        with self.lock:
            table_identifier = TableIdentifier.from_string(table_name).enrich(
                self.get_current_catalog(),
                self.get_current_database())
            _verify_table_catalog(table_identifier)
            if not self.does_table_exist(table_name):
                if not ignore_if_not_exists:
                    raise TableNotFoundError(table_identifier.table, table_identifier.db)
                return False
            try:
                with DuckDBTransaction(self.db_conn):
                    sql = f"DROP TABLE IF EXISTS {self._build_qualified_table_name(table_identifier)}"
                    self.db_conn.execute(sql)
                    self.schema_storage.delete_schema(table_identifier.db, table_identifier.table)
                return True
            except Exception as e:
                raise CatalogError(
                    f"Failed to drop table: `{table_identifier.db}.{table_identifier.table}`"
                ) from e

    def create_table(
        self, table_name: str, schema: Schema, ignore_if_exists: bool = True
    ) -> bool:
        """Create a new table."""
        temp_view_name = generate_unique_arrow_view_name()
        with self.lock:
            table_identifier = TableIdentifier.from_string(table_name).enrich(
                self.get_current_catalog(),
                self.get_current_database())
            _verify_table_catalog(table_identifier)
            if self.does_table_exist(table_name):
                if not ignore_if_exists:
                    raise TableAlreadyExistsError(table_identifier.table, table_identifier.db)
                return False
            polars_schema = convert_custom_schema_to_polars_schema(schema)
            try:
                with DuckDBTransaction(self.db_conn):
                    self.db_conn.register(
                        temp_view_name, pl.DataFrame(schema=polars_schema)
                    )
                    # trunk-ignore-begin(bandit/B608): No major risk of SQL injection here, because queries run on a client side DuckDB instance.
                    self.db_conn.execute(
                        f"CREATE TABLE IF NOT EXISTS {self._build_qualified_table_name(table_identifier)} AS SELECT * FROM {temp_view_name} WHERE 1=0"
                    )
                    self.schema_storage.save_schema(
                        table_identifier.db, table_identifier.table, schema
                    )
                return True
            except Exception as e:
                raise CatalogError(
                    f"Failed to create table: `{table_identifier.db}.{table_identifier.table}`"
                ) from e
            finally:
                try:
                    self.db_conn.execute(f"DROP VIEW IF EXISTS {temp_view_name}")
                except Exception:
                    logger.error(f"Failed to drop view: {temp_view_name}")
                    pass
                # trunk-ignore-end(bandit/B608)

    def write_df_to_table(self, df: pl.DataFrame, table_name: str, schema: Schema):
        """Write a Polars dataframe to a table in the current database."""
        temp_view_name = generate_unique_arrow_view_name()
        with self.lock:
            table_identifier = TableIdentifier.from_string(table_name).enrich(
                self.get_current_catalog(),
                self.get_current_database())
            _verify_table_catalog(table_identifier)
            try:
                # trunk-ignore-begin(bandit/B608)
                with DuckDBTransaction(self.db_conn):
                    self.db_conn.register(temp_view_name, df)
                    self.db_conn.execute(
                        f"CREATE TABLE IF NOT EXISTS {self._build_qualified_table_name(table_identifier)} AS SELECT * FROM {temp_view_name}"
                    )
                    self.schema_storage.save_schema(
                        table_identifier.db, table_identifier.table, schema
                    )
            except Exception as e:
                raise CatalogError(
                    f"Failed to create table and write dataframe: `{table_identifier.db}.{table_identifier.table}`"
                ) from e
            finally:
                try:
                    self.db_conn.execute(f"DROP VIEW IF EXISTS {temp_view_name}")
                except Exception:
                    logger.error(f"Failed to drop view: {temp_view_name}")
                    pass
            # trunk-ignore-end(bandit/B608)

    def insert_df_to_table(self, df: pl.DataFrame, table_name: str, schema: Schema):
        """Insert a Polars dataframe into a table in the current database."""
        temp_view_name = generate_unique_arrow_view_name()
        with self.lock:
            if self.does_table_exist(table_name):
                if self.describe_table(table_name) != schema:
                    raise CatalogError(
                        f"Table '{table_name}' already exists with a different schema!\n"
                        f"Existing schema: {self.describe_table(table_name)}\n"
                        f"New schema: {schema}\n"
                        "To replace the existing table, use mode='overwrite'."
                    )
            table_identifier = TableIdentifier.from_string(table_name).enrich(
                self.get_current_catalog(),
                self.get_current_database())
            _verify_table_catalog(table_identifier)
            try:
                # trunk-ignore-begin(bandit/B608)
                self.db_conn.register(temp_view_name, df)
                self.db_conn.execute(
                    f"INSERT INTO {self._build_qualified_table_name(table_identifier)} SELECT * FROM {temp_view_name}"
                )
            except Exception as e:
                raise CatalogError(
                    f"Failed to insert dataframe into table: `{table_identifier.db}.{table_identifier.table}`"
                ) from e
            finally:
                try:
                    self.db_conn.execute(f"DROP VIEW IF EXISTS {temp_view_name}")
                except Exception:
                    logger.error(f"Failed to drop view: {temp_view_name}")
                    pass
            # trunk-ignore-end(bandit/B608)

    def replace_table_with_df(self, df: pl.DataFrame, table_name: str, schema: Schema):
        """Replace a table in the current database with a Polars dataframe."""
        temp_view_name = generate_unique_arrow_view_name()
        with self.lock:
            table_identifier = TableIdentifier.from_string(table_name).enrich(
                self.get_current_catalog(),
                self.get_current_database())
            _verify_table_catalog(table_identifier)
            try:
                # trunk-ignore-begin(bandit/B608)
                with DuckDBTransaction(self.db_conn):
                    self.db_conn.register(temp_view_name, df)
                    self.db_conn.execute(
                        f"CREATE OR REPLACE TABLE {self._build_qualified_table_name(table_identifier)} AS SELECT * FROM {temp_view_name}"
                    )
                    self.schema_storage.save_schema(
                        table_identifier.db, table_identifier.table, schema
                    )
            except Exception as e:
                raise CatalogError(
                    f"Failed to overwrite table: `{table_identifier.db}.{table_identifier.table}`"
                ) from e
            finally:
                try:
                    self.db_conn.execute(f"DROP VIEW IF EXISTS {temp_view_name}")
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
            return self.db_conn.execute(
                f"SELECT * FROM {self._build_qualified_table_name(table_identifier)}"
            ).pl()
            # trunk-ignore-end(bandit/B608)
        except Exception as e:
            raise CatalogError(
                f"Failed to read dataframe from table: `{table_identifier.db}.{table_identifier.table}`"
            ) from e

    def _build_qualified_table_name(self, table_identifier: TableIdentifier,
    ) -> str:
        return f'"{table_identifier.db}"."{table_identifier.table}"'

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
