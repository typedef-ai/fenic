# fenic.api.session.session

Canonical HTML: https://docs.fenic.ai/latest/reference/fenic/api/session/session/

Main session class for interacting with the DataFrame API.

Classes:

- **`Session`**
  –

  The entry point to programming with the DataFrame API. Similar to PySpark's SparkSession.

## Session

The entry point to programming with the DataFrame API. Similar to PySpark's SparkSession.

Create a session with default configuration

```
session = Session.get_or_create(SessionConfig(app_name="my_app"))
```

Create a session with cloud configuration

```
config = SessionConfig(
    app_name="my_app",
    cloud=True,
    api_key="your_api_key"
)
session = Session.get_or_create(config)
```

Methods:

- **`create_dataframe`**
  –

  Create a DataFrame from a variety of Python-native data formats.
- **`get_or_create`**
  –

  Gets an existing Session or creates a new one with the configured settings.
- **`sql`**
  –

  Execute a read-only SQL query against one or more DataFrames using named placeholders.
- **`stop`**
  –

  Stops the session and closes all connections.
- **`table`**
  –

  Returns the specified table as a DataFrame.
- **`view`**
  –

  Returns the specified view as a DataFrame.

Attributes:

- **`catalog`**
  (`Catalog`)
  –

  Interface for catalog operations on the Session.
- **`read`**
  (`DataFrameReader`)
  –

  Returns a DataFrameReader that can be used to read data in as a DataFrame.

### catalog

```
catalog: Catalog
```

Interface for catalog operations on the Session.

### read

```
read: DataFrameReader
```

Returns a DataFrameReader that can be used to read data in as a DataFrame.

Returns:

- **`DataFrameReader`** ( `DataFrameReader`
  ) –

  A reader interface to read data into DataFrame

Raises:

- `RuntimeError`
  –

  If the session has been stopped

### create_dataframe

```
create_dataframe(data: DataLike, schema: Schema | None = None) -> DataFrame
```

Create a DataFrame from a variety of Python-native data formats.

Parameters:

- **`data`**
  (`DataLike`)
  –

  Input data. Must be one of:
  - Polars DataFrame
  - Pandas DataFrame
  - dict of column_name -> list of values
  - list of dicts (each dict representing a row)
  - pyarrow Table
- **`schema`**
  (`Schema | None`, default:
  `None`
  )
  –

  Optional complete top-level fenic schema. When provided,
  field names are authoritative, result columns are ordered to
  match the schema, values are physically coerced to the schema's
  Polars representation, and the logical DataFrame schema is
  preserved exactly. Use this for logical string-backed types
  such as JSON and Markdown, and for preserving fixed-size
  embedding arrays through local and cloud execution.

Returns:

- `DataFrame`
  –

  A new DataFrame instance

Raises:

- `ValidationError`
  –

  If the input format is unsupported or the provided
  columns do not match the schema.
- `PlanError`
  –

  If the input data cannot be coerced to the provided
  schema, or the schema is invalid for plan construction.

Create from Polars DataFrame

```
import polars as pl
df = pl.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
session.create_dataframe(df)
```

Create from Pandas DataFrame

```
import pandas as pd
df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
session.create_dataframe(df)
```

Create from dictionary

```
session.create_dataframe({"col1": [1, 2], "col2": ["a", "b"]})
```

Create from list of dictionaries

```
session.create_dataframe([
    {"col1": 1, "col2": "a"},
    {"col1": 2, "col2": "b"}
])
```

Create from pyarrow Table

```
import pyarrow as pa
table = pa.Table.from_pydict({"col1": [1, 2], "col2": ["a", "b"]})
session.create_dataframe(table)
```

Create with an explicit schema

```
import fenic as fc

schema = fc.Schema([
    fc.ColumnField("age", fc.IntegerType),
    fc.ColumnField("name", fc.StringType),
])
session.create_dataframe({"name": ["Alice"], "age": ["42"]}, schema=schema)
```

Source code in `src/fenic/api/session/session.py`

```
def create_dataframe(
    self,
    data: DataLike,
    schema: Schema | None = None,
) -> DataFrame:
    """Create a DataFrame from a variety of Python-native data formats.

    Args:
        data: Input data. Must be one of:
            - Polars DataFrame
            - Pandas DataFrame
            - dict of column_name -> list of values
            - list of dicts (each dict representing a row)
            - pyarrow Table
        schema: Optional complete top-level fenic schema. When provided,
            field names are authoritative, result columns are ordered to
            match the schema, values are physically coerced to the schema's
            Polars representation, and the logical DataFrame schema is
            preserved exactly. Use this for logical string-backed types
            such as JSON and Markdown, and for preserving fixed-size
            embedding arrays through local and cloud execution.

    Returns:
        A new DataFrame instance

    Raises:
        ValidationError: If the input format is unsupported or the provided
            columns do not match the schema.
        PlanError: If the input data cannot be coerced to the provided
            schema, or the schema is invalid for plan construction.

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

    Example: Create with an explicit schema
        ```python
        import fenic as fc

        schema = fc.Schema([
            fc.ColumnField("age", fc.IntegerType),
            fc.ColumnField("name", fc.StringType),
        ])
        session.create_dataframe({"name": ["Alice"], "age": ["42"]}, schema=schema)
        ```
    """
    pl_df, row_field_names = _normalize_data_like_to_polars(
        data,
        allow_empty_list=schema is not None,
        validate_all_rows=schema is not None,
    )
    if schema is None:
        return DataFrame._from_logical_plan(
            InMemorySource.from_session_state(pl_df, self._session_state),
            self._session_state,
        )

    coerced_pl_df = _coerce_to_schema(pl_df, schema, row_field_names=row_field_names)

    return DataFrame._from_logical_plan(
        InMemorySource.from_schema(coerced_pl_df, schema),
        self._session_state,
    )
```

### get_or_create

```
get_or_create(config: SessionConfig) -> Session
```

Gets an existing Session or creates a new one with the configured settings.

Returns:

- `Session`
  –

  A Session instance configured with the provided settings

Source code in `src/fenic/api/session/session.py`

```
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
```

### sql

```
sql(query: str, /, **tables: DataFrame) -> DataFrame
```

Execute a read-only SQL query against one or more DataFrames using named placeholders.

This allows you to execute ad hoc SQL queries using familiar syntax when it's more convenient than the DataFrame API.
Placeholders in the SQL string (e.g. `{df}`) should correspond to keyword arguments (e.g. `df=my_dataframe`).

For supported SQL syntax and functions, refer to the DuckDB SQL documentation:
https://duckdb.org/docs/sql/introduction.

Parameters:

- **`query`**
  (`str`)
  –

  A SQL query string with placeholders like `{df}`
- **`**tables`**
  (`DataFrame`, default:
  `{}`
  )
  –

  Keyword arguments mapping placeholder names to DataFrames

Returns:

- `DataFrame`
  –

  A lazy DataFrame representing the result of the SQL query

Raises:

- `ValidationError`
  –

  If a placeholder is used in the query but not passed
  as a keyword argument

Simple join between two DataFrames

```
df1 = session.create_dataframe({"id": [1, 2]})
df2 = session.create_dataframe({"id": [2, 3]})
result = session.sql(
    "SELECT * FROM {df1} JOIN {df2} USING (id)",
    df1=df1,
    df2=df2
)
```

Complex query with multiple DataFrames

```
users = session.create_dataframe({"user_id": [1, 2], "name": ["Alice", "Bob"]})
orders = session.create_dataframe({"order_id": [1, 2], "user_id": [1, 2]})
products = session.create_dataframe({"product_id": [1, 2], "name": ["Widget", "Gadget"]})

result = session.sql("""
    SELECT u.name, p.name as product
    FROM {users} u
    JOIN {orders} o ON u.user_id = o.user_id
    JOIN {products} p ON o.product_id = p.product_id
""", users=users, orders=orders, products=products)
```

Source code in `src/fenic/api/session/session.py`

```
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
```

### stop

```
stop(skip_usage_summary: bool = False)
```

Stops the session and closes all connections.

Parameters:

- **`skip_usage_summary`**
  (`bool`, default:
  `False`
  )
  –

  Whether to skip printing the usage summary.

Unless `skip_usage_summary` is set, a summary of your session's metrics will print once you stop your session.

Source code in `src/fenic/api/session/session.py`

```
def stop(self, skip_usage_summary: bool = False):
    """Stops the session and closes all connections.

    Args:
        skip_usage_summary: Whether to skip printing the usage summary.

    Unless `skip_usage_summary` is set, a summary of your session's metrics will print once you stop your session.
    """
    self._session_state.stop(skip_usage_summary=skip_usage_summary)
```

### table

```
table(table_name: str) -> DataFrame
```

Returns the specified table as a DataFrame.

Parameters:

- **`table_name`**
  (`str`)
  –

  Name of the table

Returns:

- `DataFrame`
  –

  Table as a DataFrame

Raises:

- `ValueError`
  –

  If the table does not exist

Load an existing table

```
df = session.table("my_table")
```

Source code in `src/fenic/api/session/session.py`

```
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
```

### view

```
view(view_name: str) -> DataFrame
```

Returns the specified view as a DataFrame.

Parameters:

- **`view_name`**
  (`str`)
  –

  Name of the view

Returns:
DataFrame: Dataframe with the given view

Source code in `src/fenic/api/session/session.py`

```
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
```
