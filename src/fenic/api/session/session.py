"""Main session class for interacting with the DataFrame API."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pandas as pd
import polars as pl
import pyarrow as pa

from fenic._backends.local.manager import LocalSessionManager
from fenic._constants import SQL_PLACEHOLDER_RE
from fenic.api.dataframe import DataFrame
from fenic.api.io.reader import DataFrameReader
from fenic.core._interfaces.session_state import BaseSessionState
from fenic.core._logical_plan.plans import SQL, InMemorySource, TableSource

if TYPE_CHECKING:
    from fenic._backends.cloud.session_state import CloudSessionState
    from fenic._backends.local.session_state import LocalSessionState
from pydantic import ConfigDict, validate_call

from fenic._backends.utils.catalog_utils import validate_view
from fenic.api.catalog import Catalog
from fenic.api.session.config import SessionConfig
from fenic.core.error import CatalogError, PlanError, ValidationError
from fenic.core.types.query_result import DataLike


class Session:
    """The entry point to programming with the DataFrame API. Similar to PySpark's SparkSession.

    Example: Create a session with default configuration
        ```python
        session = Session.get_or_create(SessionConfig(app_name="my_app"))
        ```

    Example: Create a session with cloud configuration
        ```python
        config = SessionConfig(
            app_name="my_app",
            cloud=True,
            api_key="your_api_key"
        )
        session = Session.get_or_create(config)
        ```
    """

    app_name: str
    _session_state: BaseSessionState
    _reader: DataFrameReader

    def __new__(cls):
        """Create a new Session instance."""
        if cls is Session:
            raise ValidationError(
                "Direct construction of Session is not allowed. Use Session.get_or_create() to create a Session."
            )
        return super().__new__(cls)

    @classmethod
    def get_or_create(
        cls,
        config: SessionConfig,
    ) -> Session:
        """Gets an existing Session or creates a new one with the configured settings.

        Returns:
            A Session instance configured with the provided settings
        """
        if config.cloud:
            from fenic._backends.cloud.manager import CloudSessionManager

            cloud_session_manager = CloudSessionManager()
            if not cloud_session_manager.initialized:
                session_manager_dependencies = (
                    CloudSessionManager.create_global_session_dependencies()
                )
                cloud_session_manager.configure(session_manager_dependencies)
            future = asyncio.run_coroutine_threadsafe(
                cloud_session_manager.get_or_create_session_state(config),
                cloud_session_manager._asyncio_loop,
            )
            cloud_session_state = future.result()
            return Session._create_cloud_session(cloud_session_state)

        local_session_state: LocalSessionState = LocalSessionManager().get_or_create_session_state(config._to_resolved_config())
        return Session._create_local_session(local_session_state)

    @classmethod
    def _create_local_session(
        cls,
        session_state: LocalSessionState,
    ) -> Session:
        """Get or create a local session."""
        session = super().__new__(cls)
        session.app_name = session_state.app_name
        session._session_state = session_state
        session._reader = DataFrameReader(session._session_state)
        return session

    @classmethod
    def _create_cloud_session(
        cls,
        session_state: CloudSessionState,
    ) -> Session:
        """Create a cloud session."""
        session = super().__new__(cls)
        session.app_name = session_state.config.app_name
        session._session_state = session_state
        session._reader = DataFrameReader(session._session_state)
        return session

    @property
    def read(self) -> DataFrameReader:
        """Returns a DataFrameReader that can be used to read data in as a DataFrame.

        Returns:
            DataFrameReader: A reader interface to read data into DataFrame

        Raises:
            RuntimeError: If the session has been stopped
        """
        return self._reader

    @property
    def catalog(self) -> Catalog:
        """Interface for catalog operations on the Session."""
        return Catalog(self._session_state.catalog)

    def create_dataframe(
        self,
        data: DataLike,
    ) -> DataFrame:
        """Create a DataFrame from a variety of Python-native data formats.

        Args:
            data: Input data. Must be one of:
                - Polars DataFrame
                - Pandas DataFrame
                - dict of column_name -> list of values
                - list of dicts (each dict representing a row)
                - pyarrow Table

        Returns:
            A new DataFrame instance

        Raises:
            ValueError: If the input format is unsupported or inconsistent with provided column names.

        Example: Create from Polars DataFrame
            ```python
            import polars as pl
            df = pl.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
            session.create_dataframe(df)
            ```

        Example: Create from Pandas DataFrame
            ```python
            import pandas as pd
            df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
            session.create_dataframe(df)
            ```

        Example: Create from dictionary
            ```python
            session.create_dataframe({"col1": [1, 2], "col2": ["a", "b"]})
            ```

        Example: Create from list of dictionaries
            ```python
            session.create_dataframe([
                {"col1": 1, "col2": "a"},
                {"col1": 2, "col2": "b"}
            ])
            ```

        Example: Create from pyarrow Table
            ```python
            import pyarrow as pa
            table = pa.Table.from_pydict({"col1": [1, 2], "col2": ["a", "b"]})
            session.create_dataframe(table)
            ```
        """
        try:
            if isinstance(data, pl.DataFrame):
                pl_df = data
            elif isinstance(data, pd.DataFrame):
                pl_df = pl.from_pandas(data)
            elif isinstance(data, dict):
                pl_df = pl.DataFrame(data)
            elif isinstance(data, list):
                if not data:
                    raise ValidationError(
                        "Cannot create DataFrame from empty list. Provide a non-empty list of dictionaries, lists, or other supported data types."
                    )

                if not isinstance(data[0], dict):
                    raise ValidationError(
                        "Cannot create DataFrame from list of non-dict values. Provide a list of dictionaries."
                    )
                pl_df = pl.DataFrame(data)
            elif isinstance(data, pa.Table):
                pl_df = pl.from_arrow(data)

            else:
                raise ValidationError(
                    f"Unsupported data type: {type(data)}. Supported types are: Polars DataFrame, Pandas DataFrame, dict, or list."
                )

        except ValidationError:
            raise
        except Exception as e:
            raise PlanError(f"Failed to create DataFrame from {data}") from e

        return DataFrame._from_logical_plan(
            InMemorySource.from_session_state(pl_df, self._session_state),
            self._session_state,
        )

    def table(self, table_name: str) -> DataFrame:
        """Returns the specified table as a DataFrame.

        Args:
            table_name: Name of the table

        Returns:
            Table as a DataFrame

        Raises:
            ValueError: If the table does not exist

        Example: Load an existing table
            ```python
            df = session.table("my_table")
            ```
        """
        if not self._session_state.catalog.does_table_exist(table_name):
            raise ValueError(f"Table {table_name} does not exist")
        return DataFrame._from_logical_plan(
            TableSource.from_session_state(table_name, self._session_state),
            self._session_state,
        )

    def view(self, view_name: str) -> DataFrame:
        """Returns the specified view as a DataFrame.

        Args:
            view_name: Name of the view
        Returns:
            DataFrame: Dataframe with the given view
        """
        if not self._session_state.catalog.does_view_exist(view_name):
            raise CatalogError(f"View {view_name} does not exist")

        view_plan = self._session_state.catalog.get_view_plan(view_name)
        validate_view(view_name, view_plan, self._session_state)

        return DataFrame._from_logical_plan(
            view_plan,
            self._session_state,
        )

    def sql(self, query: str, /, **tables: DataFrame) -> DataFrame:
        """Execute a read-only SQL query against one or more DataFrames using named placeholders.

        This allows you to execute ad hoc SQL queries using familiar syntax when it's more convenient than the DataFrame API.
        Placeholders in the SQL string (e.g. `{df}`) should correspond to keyword arguments (e.g. `df=my_dataframe`).

        For supported SQL syntax and functions, refer to the DuckDB SQL documentation:
        https://duckdb.org/docs/sql/introduction.

        Args:
            query: A SQL query string with placeholders like `{df}`
            **tables: Keyword arguments mapping placeholder names to DataFrames

        Returns:
            A lazy DataFrame representing the result of the SQL query

        Raises:
            ValidationError: If a placeholder is used in the query but not passed
                as a keyword argument

        Example: Simple join between two DataFrames
            ```python
            df1 = session.create_dataframe({"id": [1, 2]})
            df2 = session.create_dataframe({"id": [2, 3]})
            result = session.sql(
                "SELECT * FROM {df1} JOIN {df2} USING (id)",
                df1=df1,
                df2=df2
            )
            ```

        Example: Complex query with multiple DataFrames
            ```python
            users = session.create_dataframe({"user_id": [1, 2], "name": ["Alice", "Bob"]})
            orders = session.create_dataframe({"order_id": [1, 2], "user_id": [1, 2]})
            products = session.create_dataframe({"product_id": [1, 2], "name": ["Widget", "Gadget"]})

            result = session.sql(\"\"\"
                SELECT u.name, p.name as product
                FROM {users} u
                JOIN {orders} o ON u.user_id = o.user_id
                JOIN {products} p ON o.product_id = p.product_id
            \"\"\", users=users, orders=orders, products=products)
            ```
        """
        query = query.strip()
        if not query:
            raise ValidationError("SQL query must not be empty.")

        placeholders = set(SQL_PLACEHOLDER_RE.findall(query))
        missing = placeholders - tables.keys()
        if missing:
            raise ValidationError(
                f"Missing DataFrames for placeholders in SQL query: {', '.join(sorted(missing))}. "
                f"Make sure to pass them as keyword arguments, e.g., sql(..., {next(iter(missing))}=df)."
            )

        logical_plans = []
        template_names = []
        input_session_states = []
        for name, table in tables.items():
            if name in placeholders:
                template_names.append(name)
                logical_plans.append(table._logical_plan)
                input_session_states.append(table._session_state)

        DataFrame._ensure_same_session(self._session_state, input_session_states)
        return DataFrame._from_logical_plan(
            SQL.from_session_state(logical_plans, template_names, query, self._session_state),
            self._session_state,
        )

    def stop(self, skip_usage_summary: bool = False):
        """Stops the session and closes all connections.

        Args:
            skip_usage_summary: Whether to skip printing the usage summary.

        Unless `skip_usage_summary` is set, a summary of your session's metrics will print once you stop your session.
        """
        self._session_state.stop(skip_usage_summary=skip_usage_summary)

Session.createDataFrame = Session.create_dataframe
Session.get_or_create = validate_call(config=ConfigDict(strict=True))(
    Session.get_or_create
)
Session.getOrCreate = Session.get_or_create
Session.table = validate_call(config=ConfigDict(strict=True))(Session.table)
Session.sql = validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))(Session.sql)
