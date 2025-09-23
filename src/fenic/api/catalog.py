"""Catalog API for managing database objects in Fenic."""

from typing import List, Optional

from pydantic import ConfigDict, validate_call

from fenic.api.dataframe.dataframe import DataFrame
from fenic.core._interfaces.catalog import BaseCatalog
from fenic.core.mcp.types import ToolParam, UserDefinedTool
from fenic.core.types import DatasetMetadata, Schema


class Catalog:
    """Entry point for catalog operations.

    Provides methods to manage catalogs, databases, and tables, as well as
    read-only access to system tables such as `fenic_system.query_metrics`.

    ### Catalog and Database Management
    Example:
        ```python
        # Create a catalog
        session.catalog.create_catalog("my_catalog")  # → True

        # Set active catalog
        session.catalog.set_current_catalog("my_catalog")

        # Create a database
        session.catalog.create_database("my_database")  # → True

        # Set active database
        session.catalog.set_current_database("my_database")

        # Create a table
        session.catalog.create_table(
            "my_table",
            Schema([ColumnField("id", IntegerType)])
        )  # → True
        ```

    ### Metrics Table (Local Sessions Only)

    Query metrics are recorded for each session and stored locally
    in `fenic_system.query_metrics`. Metrics can be loaded into a DataFrame
    for analysis.

    Example:
        ```python
        # Load all metrics for the current application
        metrics_df = session.table("fenic_system.query_metrics")

        # Show the 10 most recent queries in the application
        recent_queries = session.sql(\"\"\"
            SELECT *
            FROM {df}
            ORDER BY CAST(end_ts AS TIMESTAMP) DESC
            LIMIT 10
        \"\"\", df=metrics_df)
        recent_queries.show()

        # Find query metrics for a specific session with non-zero LM costs
        specific_session_queries = session.sql(\"\"\"
            SELECT *
            FROM {df}
            WHERE session_id = '9e7e256f-fad9-4cd9-844e-399d795aaea0'
                AND total_lm_cost > 0
            ORDER BY CAST(end_ts AS TIMESTAMP) ASC
        \"\"\", df=metrics_df)
        specific_session_queries.show()

        # Aggregate total LM costs and requests between a specific time window
        metrics_window = session.sql(\"\"\"
            SELECT
                CAST(SUM(total_lm_cost) AS DOUBLE) AS total_lm_cost_in_window,
                CAST(SUM(total_lm_requests) AS DOUBLE) AS total_lm_requests_in_window
            FROM {df}
            WHERE CAST(end_ts AS TIMESTAMP) BETWEEN
                CAST('2025-08-29 10:00:00' AS TIMESTAMP)
                AND CAST('2025-08-29 12:00:00' AS TIMESTAMP)
        \"\"\", df=metrics_df)

        metrics_window.show()
        ```
    """

    def __init__(self, catalog: BaseCatalog):
        """Initialize a Catalog instance.

        Args:
            catalog: The underlying catalog implementation.
        """
        self.catalog = catalog


    @validate_call(config=ConfigDict(strict=True))
    def does_catalog_exist(self, catalog_name: str) -> bool:
        """Checks if a catalog with the specified name exists.

        Args:
            catalog_name (str): Name of the catalog to check.

        Returns:
            bool: True if the catalog exists, False otherwise.

        Example: Check if a catalog exists
            ```python
            # Check if 'my_catalog' exists
            session.catalog.does_catalog_exist('my_catalog')
            # Returns: True
            ```
        """
        return self.catalog.does_catalog_exist(catalog_name)

    def get_current_catalog(self) -> str:
        """Returns the name of the current catalog.

        Returns:
            str: The name of the current catalog.

        Example: Get current catalog name
            ```python
            # Get the name of the current catalog
            session.catalog.get_current_catalog()
            # Returns: 'default'
            ```
        """
        return self.catalog.get_current_catalog()

    @validate_call(config=ConfigDict(strict=True))
    def set_current_catalog(self, catalog_name: str) -> None:
        """Sets the current catalog.

        Args:
            catalog_name (str): Name of the catalog to set as current.

        Raises:
            ValueError: If the specified catalog doesn't exist.

        Example: Set current catalog
            ```python
            # Set 'my_catalog' as the current catalog
            session.catalog.set_current_catalog('my_catalog')
            ```
        """
        self.catalog.set_current_catalog(catalog_name)

    def list_catalogs(self) -> List[str]:
        """Returns a list of available catalogs.

        Returns:
            List[str]: A list of catalog names available in the system.
            Returns an empty list if no catalogs are found.

        Example: List all catalogs
            ```python
            # Get all available catalogs
            session.catalog.list_catalogs()
            # Returns: ['default', 'my_catalog', 'other_catalog']
            ```
        """
        return self.catalog.list_catalogs()

    @validate_call(config=ConfigDict(strict=True))
    def create_catalog(self, catalog_name: str, ignore_if_exists: bool = True) -> bool:
        """Creates a new catalog.

        Args:
            catalog_name (str): Name of the catalog to create.
            ignore_if_exists (bool): If True, return False when the catalog already exists.
                If False, raise an error when the catalog already exists.
                Defaults to True.

        Raises:
            CatalogAlreadyExistsError: If the catalog already exists and ignore_if_exists is False.

        Returns:
            bool: True if the catalog was created successfully, False if the catalog
            already exists and ignore_if_exists is True.

        Example: Create a new catalog
            ```python
            # Create a new catalog named 'my_catalog'
            session.catalog.create_catalog('my_catalog')
            # Returns: True
            ```

        Example: Create an existing catalog with ignore_if_exists
            ```python
            # Try to create an existing catalog with ignore_if_exists=True
            session.catalog.create_catalog('my_catalog', ignore_if_exists=True)
            # Returns: False
            ```

        Example: Create an existing catalog without ignore_if_exists
            ```python
            # Try to create an existing catalog with ignore_if_exists=False
            session.catalog.create_catalog('my_catalog', ignore_if_exists=False)
            # Raises: CatalogAlreadyExistsError
            ```
        """
        return self.catalog.create_catalog(catalog_name, ignore_if_exists)

    @validate_call(config=ConfigDict(strict=True))
    def drop_catalog(
        self, catalog_name: str, ignore_if_not_exists: bool = True
    ) -> bool:
        """Drops a catalog.

        Args:
            catalog_name (str): Name of the catalog to drop.
            ignore_if_not_exists (bool): If True, silently return if the catalog doesn't exist.
                If False, raise an error if the catalog doesn't exist.
                Defaults to True.

        Raises:
            CatalogNotFoundError: If the catalog does not exist and ignore_if_not_exists is False

        Returns:
            bool: True if the catalog was dropped successfully, False if the catalog
            didn't exist and ignore_if_not_exists is True.

        Example: Drop a non-existent catalog
            ```python
            # Try to drop a non-existent catalog
            session.catalog.drop_catalog('my_catalog')
            # Returns: False
            ```

        Example: Drop a non-existent catalog without ignore_if_not_exists
            ```python
            # Try to drop a non-existent catalog with ignore_if_not_exists=False
            session.catalog.drop_catalog('my_catalog', ignore_if_not_exists=False)
            # Raises: CatalogNotFoundError
            ```
        """
        return self.catalog.drop_catalog(catalog_name, ignore_if_not_exists)

    @validate_call(config=ConfigDict(strict=True))
    def does_database_exist(self, database_name: str) -> bool:
        """Checks if a database with the specified name exists.

        Args:
            database_name (str): Fully qualified or relative database name to check.

        Returns:
            bool: True if the database exists, False otherwise.

        Example: Check if a database exists
            ```python
            # Check if 'my_database' exists
            session.catalog.does_database_exist('my_database')
            # Returns: True
            ```
        """
        return self.catalog.does_database_exist(database_name)

    def get_current_database(self) -> str:
        """Returns the name of the current database in the current catalog.

        Returns:
            str: The name of the current database.

        Example: Get current database name
            ```python
            # Get the name of the current database
            session.catalog.get_current_database()
            # Returns: 'default'
            ```
        """
        return self.catalog.get_current_database()

    @validate_call(config=ConfigDict(strict=True))
    def set_current_database(self, database_name: str) -> None:
        """Sets the current database.

        Args:
            database_name (str): Fully qualified or relative database name to set as current.

        Raises:
            DatabaseNotFoundError: If the specified database doesn't exist.

        Example: Set current database
            ```python
            # Set 'my_database' as the current database
            session.catalog.set_current_database('my_database')
            ```
        """
        self.catalog.set_current_database(database_name)

    def list_databases(self) -> List[str]:
        """Returns a list of databases in the current catalog.

        Returns:
            List[str]: A list of database names in the current catalog.
            Returns an empty list if no databases are found.

        Example: List all databases
            ```python
            # Get all databases in the current catalog
            session.catalog.list_databases()
            # Returns: ['default', 'my_database', 'other_database']
            ```
        """
        return self.catalog.list_databases()

    @validate_call(config=ConfigDict(strict=True))
    def create_database(
        self, database_name: str, ignore_if_exists: bool = True
    ) -> bool:
        """Creates a new database.

        Args:
            database_name (str): Fully qualified or relative database name to create.
            ignore_if_exists (bool): If True, return False when the database already exists.
                If False, raise an error when the database already exists.
                Defaults to True.

        Raises:
            DatabaseAlreadyExistsError: If the database already exists and ignore_if_exists is False.

        Returns:
            bool: True if the database was created successfully, False if the database
            already exists and ignore_if_exists is True.

        Example: Create a new database
            ```python
            # Create a new database named 'my_database'
            session.catalog.create_database('my_database')
            # Returns: True
            ```

        Example: Create an existing database with ignore_if_exists
            ```python
            # Try to create an existing database with ignore_if_exists=True
            session.catalog.create_database('my_database', ignore_if_exists=True)
            # Returns: False
            ```

        Example: Create an existing database without ignore_if_exists
            ```python
            # Try to create an existing database with ignore_if_exists=False
            session.catalog.create_database('my_database', ignore_if_exists=False)
            # Raises: DatabaseAlreadyExistsError
            ```
        """
        return self.catalog.create_database(database_name, ignore_if_exists)

    @validate_call(config=ConfigDict(strict=True))
    def drop_database(
        self,
        database_name: str,
        cascade: bool = False,
        ignore_if_not_exists: bool = True,
    ) -> bool:
        """Drops a database.

        Args:
            database_name (str): Fully qualified or relative database name to drop.
            cascade (bool): If True, drop all tables in the database.
                Defaults to False.
            ignore_if_not_exists (bool): If True, silently return if the database doesn't exist.
                If False, raise an error if the database doesn't exist.
                Defaults to True.

        Raises:
            DatabaseNotFoundError: If the database does not exist and ignore_if_not_exists is False
            CatalogError: If the current database is being dropped, if the database is not empty and cascade is False

        Returns:
            bool: True if the database was dropped successfully, False if the database
            didn't exist and ignore_if_not_exists is True.

        Example: Drop a non-existent database
            ```python
            # Try to drop a non-existent database
            session.catalog.drop_database('my_database')
            # Returns: False
            ```

        Example: Drop a non-existent database without ignore_if_not_exists
            ```python
            # Try to drop a non-existent database with ignore_if_not_exists=False
            session.catalog.drop_database('my_database', ignore_if_not_exists=False)
            # Raises: DatabaseNotFoundError
            ```
        """
        return self.catalog.drop_database(database_name, cascade, ignore_if_not_exists)

    @validate_call(config=ConfigDict(strict=True))
    def does_table_exist(self, table_name: str) -> bool:
        """Checks if a table with the specified name exists.

        Args:
            table_name (str): Fully qualified or relative table name to check.

        Returns:
            bool: True if the table exists, False otherwise.

        Example: Check if a table exists
            ```python
            # Check if 'my_table' exists
            session.catalog.does_table_exist('my_table')
            # Returns: True
            ```
        """
        return self.catalog.does_table_exist(table_name)

    def list_tables(self) -> List[str]:
        """Returns a list of tables stored in the current database.

        This method queries the current database to retrieve all available table names.

        Returns:
            List[str]: A list of table names stored in the database.
            Returns an empty list if no tables are found.

        Example: List all tables
            ```python
            # Get all tables in the current database
            session.catalog.list_tables()
            # Returns: ['table1', 'table2', 'table3']
            ```
        """
        return self.catalog.list_tables()

    @validate_call(config=ConfigDict(strict=True))
    def describe_table(self, table_name: str) -> DatasetMetadata:
        """Returns the schema of the specified table.

        Args:
            table_name (str): Fully qualified or relative table name to describe.

        Returns:
            DatasetMetadata: An object containing:
                schema: A schema object describing the table's structure with field names and types.
                description: A natural language description of the table's contents and uses.

        Raises:
            TableNotFoundError: If the table doesn't exist.

        Example: Describe a table's schema
            ```python
            # For a table created with: create_table('t1', Schema([ColumnField('id', IntegerType)]), description='My table description')
            session.catalog.describe_table('t1')
            # Returns: DatasetMetadata(schema=Schema([
            #     ColumnField('id', IntegerType),
            # ]), description="My table description")
            ```
        """
        return self.catalog.describe_table(table_name)

    @validate_call(config=ConfigDict(strict=True))
    def set_table_description(self, table_name: str, description: Optional[str] = None) -> None:
        """Set or unset the description for a table.

        Args:
            table_name: Fully qualified or relative table name to set the description for.
            description: The description to set for the table.

        Raises:
            TableNotFoundError: If the table doesn't exist.
        """
        self.catalog.set_table_description(table_name, description)

    @validate_call(config=ConfigDict(strict=True))
    def drop_table(self, table_name: str, ignore_if_not_exists: bool = True) -> bool:
        """Drops the specified table.

        By default this method will return False if the table doesn't exist.

        Args:
            table_name (str): Fully qualified or relative table name to drop.
            ignore_if_not_exists (bool): If True, return False when the table doesn't exist.
                If False, raise an error when the table doesn't exist.
                Defaults to True.

        Returns:
            bool: True if the table was dropped successfully, False if the table
            didn't exist and ignore_if_not_exist is True.

        Raises:
            TableNotFoundError: If the table doesn't exist and ignore_if_not_exists is False

        Example: Drop an existing table
            ```python
            # Drop an existing table 't1'
            session.catalog.drop_table('t1')
            # Returns: True
            ```

        Example: Drop a non-existent table with ignore_if_not_exists
            ```python
            # Try to drop a non-existent table with ignore_if_not_exists=True
            session.catalog.drop_table('t2', ignore_if_not_exists=True)
            # Returns: False
            ```

        Example: Drop a non-existent table without ignore_if_not_exists
            ```python
            # Try to drop a non-existent table with ignore_if_not_exists=False
            session.catalog.drop_table('t2', ignore_if_not_exists=False)
            # Raises: TableNotFoundError
            ```
        """
        return self.catalog.drop_table(table_name, ignore_if_not_exists)

    @validate_call(config=ConfigDict(strict=True))
    def create_table(
        self, table_name: str, schema: Schema, ignore_if_exists: bool = True, description: Optional[str] = None
    ) -> bool:
        """Creates a new table.

        Args:
            table_name (str): Fully qualified or relative table name to create.
            schema (Schema): Schema of the table to create.
            ignore_if_exists (bool): If True, return False when the table already exists.
                If False, raise an error when the table already exists.
                Defaults to True.
            description (Optional[str]): Description of the table to create.
                Defaults to None.

        Returns:
            bool: True if the table was created successfully, False if the table
            already exists and ignore_if_exists is True.

        Raises:
            TableAlreadyExistsError: If the table already exists and ignore_if_exists is False

        Example: Create a new table
            ```python
            # Create a new table with an integer column
            session.catalog.create_table('my_table', Schema([
                ColumnField('id', IntegerType),
            ]), description='My table description')
            # Returns: True
            ```

        Example: Create an existing table with ignore_if_exists
            ```python
            # Try to create an existing table with ignore_if_exists=True
            session.catalog.create_table('my_table', Schema([
                ColumnField('id', IntegerType),
            ]), ignore_if_exists=True, description='My table description')
            # Returns: False
            ```

        Example: Create an existing table without ignore_if_exists
            ```python
            # Try to create an existing table with ignore_if_exists=False
            session.catalog.create_table('my_table', Schema([
                ColumnField('id', IntegerType),
            ]), ignore_if_exists=False, description='My table description')
            # Raises: TableAlreadyExistsError
            ```
        """
        return self.catalog.create_table(table_name, schema, ignore_if_exists, description)

    def list_views(self) -> List[str]:
        """Returns a list of views stored in the current database.

        This method queries the current database to retrieve all available view names.

        Returns:
            List[str]: A list of view names stored in the database.
            Returns an empty list if no views are found.

        Example:
            >>> session.catalog.list_views()
            ['view1', 'view2', 'view3'].
        """
        return self.catalog.list_views()

    @validate_call(config=ConfigDict(strict=True))
    def describe_view(self, view_name: str) -> DatasetMetadata:
        """Returns the schema and description of the specified view.

        Args:
            view_name (str): Fully qualified or relative view name to describe.

        Returns:
            DatasetMetadata: An object containing:
                schema: A schema object describing the view's structure with field names and types.
                description: A natural language description of the view's contents and uses.

        Raises:
            TableNotFoundError: If the view doesn't exist.

        """
        return self.catalog.describe_view(view_name)

    @validate_call(config=ConfigDict(strict=True))
    def set_view_description(self, view_name: str, description: Optional[str] = None) -> None:
        """Set the description for a view.

        Args:
            view_name (str): Fully qualified or relative view name to set the description for.
            description (str): The description to set for the view.

        Raises:
            TableNotFoundError: If the view doesn't exist.
            ValidationError: If the description is empty.

        Example: Set a description for a view
            ```python
            # Set a description for a view 'v1'
            session.catalog.set_view_description('v1', 'My view description')
        """
        self.catalog.set_view_description(view_name, description)

    @validate_call(config=ConfigDict(strict=True))
    def does_view_exist(self, view_name: str) -> bool:
        """Checks if a view with the specified name exists.

        Args:
            view_name (str): Fully qualified or relative view name to check.

        Returns:
            bool: True if the view exists, False otherwise.

        Example:
            >>> session.catalog.does_view_exist('my_view')
            True.
        """
        return self.catalog.does_view_exist(view_name)

    @validate_call(config=ConfigDict(strict=True))
    def drop_view(self, view_name: str, ignore_if_not_exists: bool = True) -> bool:
        """Drops the specified view.

        By default this method will return False if the view doesn't exist.

        Args:
            view_name (str): Fully qualified or relative view name to drop.
            ignore_if_not_exists (bool, optional): If True, return False when the view
                doesn't exist. If False, raise an error when the view doesn't exist.
                Defaults to True.

        Returns:
            bool: True if the view was dropped successfully, False if the view
                didn't exist and ignore_if_not_exist is True.

        Raises:
            TableNotFoundError: If the view doesn't exist and ignore_if_not_exists is False
        Example:
            >>> # For an existing view 'v1'
            >>> session.catalog.drop_table('v1')
            True
            >>> # For a non-existent table 'v2'
            >>> session.catalog.drop_table('v2', ignore_if_not_exists=True)
            False
            >>> session.catalog.drop_table('v2', ignore_if_not_exists=False)
            # Raises TableNotFoundError.
        """
        return self.catalog.drop_view(view_name, ignore_if_not_exists)

    @validate_call(config=ConfigDict(strict=True))
    def describe_tool(self, tool_name: str) -> UserDefinedTool:
        """Returns the tool with the specified name from the current catalog.

        Args:
            tool_name (str): The name of the tool to get.

        Raises:
            ToolNotFoundError: If the tool doesn't exist.

        Returns:
            Tool: The tool with the specified name.
        """
        return self.catalog.describe_tool(tool_name)

    @validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
    def create_tool(
        self,
        tool_name: str,
        tool_description: str,
        tool_query: DataFrame,
        tool_params: List[ToolParam],
        result_limit: int = 50,
        ignore_if_exists: bool = True
    ) -> bool:
        """Creates a new tool in the current catalog.

        Args:
            tool_name (str): The name of the tool.
            tool_description (str): The description of the tool.
            tool_query (DataFrame): The query to execute when the tool is called.
            tool_params (Sequence[ToolParam]): The parameters of the tool.
            result_limit (int): The maximum number of rows to return from the tool.
            ignore_if_exists (bool): If True, return False when the tool already exists.
                If False, raise an error when the tool already exists.
                Defaults to True.

        Returns:
            bool: True if the tool was created successfully, False otherwise.

        Raises:
            ToolAlreadyExistsError: If the tool already exists.

        Examples:
            ```python
            # Create a new tool with a single parameter
            df = session.create_dataframe(...)

            session.catalog.create_tool(
                tool_name="my_tool",
                tool_description="A tool that does something",
                tool_query=df,
                result_limit=100,
                tool_params=[ToolParam(name="param1", description="A parameter", allowed_values=["value1", "value2"])],
            )
            # Returns: True
            ```
        """
        return self.catalog.create_tool(
            tool_name,
            tool_description,
            tool_params,
            tool_query._logical_plan,
            result_limit,
            ignore_if_exists,
        )

    @validate_call(config=ConfigDict(strict=True))
    def drop_tool(self, tool_name: str, ignore_if_not_exists: bool = True) -> bool:
        """Drops the specified tool from the current catalog.

        Args:
            tool_name (str): The name of the tool to drop.
            ignore_if_not_exists (bool): If True, return False when the tool doesn't exist.
                If False, raise an error when the tool doesn't exist.
                Defaults to True.

        Returns:
            bool: True if the tool was dropped successfully, False if the tool
                didn't exist and ignore_if_not_exists is True.

        Raises:
            ToolNotFoundError: If the tool doesn't exist and ignore_if_not_exists is False

        Example:
            >>> session.catalog.drop_tool('my_tool')
            True
            >>> session.catalog.drop_tool('my_tool', ignore_if_not_exists=True)
            False
            >>> session.catalog.drop_tool('my_tool', ignore_if_not_exists=False)
            # Raises ToolNotFoundError.
        """
        return self.catalog.drop_tool(tool_name, ignore_if_not_exists)

    def list_tools(self) -> List[UserDefinedTool]:
        """Lists the tools available in the current catalog."""
        return self.catalog.list_tools()

    # Spark-style camelCase aliases
    doesCatalogExist = does_catalog_exist
    getCurrentCatalog = get_current_catalog
    setCurrentCatalog = set_current_catalog
    listCatalogs = list_catalogs
    doesDatabaseExist = does_database_exist
    doesTableExist = does_table_exist
    getCurrentDatabase = get_current_database
    setCurrentDatabase = set_current_database
    listTables = list_tables
    describeTable = describe_table
    dropTable = drop_table
    createTable = create_table
    setTableDescription = set_table_description
    listViews = list_views
    doesViewExist = does_view_exist
    dropView = drop_view
    describeTool = describe_tool
    createTool = create_tool
    dropTool = drop_tool
