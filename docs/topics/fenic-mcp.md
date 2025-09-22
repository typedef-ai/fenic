# Fenic MCP: Create and Serve Catalog Tools

This guide shows how to expose Fenic Paramaterized Tools via an MCP server. Paramaterized Tools are created by adding placeholder values to DataFrame operations
to be filled in at runtime, and managed in the Fenic Catalog. In most respects, these Paramaterized Tools are like SQL Macros. Just like one might create a macro as:

```SQL
CREATE MACRO get_users(user_name) AS TABLE
    SELECT * FROM users WHERE name = user_name;
```

One can create a Paramaterized Tool in Fenic:

```python
filter_tool_df = df.filter(fc.col("name") == fc.tool_param("user_name", StringType))
session.catalog.create_tool(
    "users_by_name",
    "Filter users by name",
    filter_tool_df,
    tool_params=[
        # If default values are provided, the parameters will be marked as `Optional` in the MCP API Spec.
        ToolParam(name="user_name", description="User's Name (Exact Match)")
    ]
)

```

## Prerequisites

- A working Fenic installation and a way to create a `Session` (optionally via a `SessionConfig` JSON file).
- Any required provider API keys available in your environment (e.g., `OPENAI_API_KEY`).

## Step 1: Create or augment tables with descriptions

You can set a table description when creating the table or for an existing table.

Create with description:

```python
from fenic import ColumnField, IntegerType, Schema, Session

session = Session.get_or_create()

session.catalog.create_table(
    "orders",
    Schema([ColumnField("order_id", IntegerType)]),
    description="Customer orders with line-item totals",
)
```

Create by writing a DataFrame, then set a description:

```python
df = session.create_dataframe({"order_id": [1, 2, 3]})
df.write.save_as_table("orders", mode="overwrite")
session.catalog.set_table_description("orders", "Customer orders with line-item totals")
```

Add or update description on an existing table:

```python
session.catalog.set_table_description("orders", "Customer orders with line-item totals")
```

You can confirm the description (and schema) via:

```python
meta = session.catalog.describe_table("orders")
print(meta.description)
print([f.name for f in meta.schema.column_fields])
```

## Step 2: Create catalog tools

Tools encapsulate a parameterized query and an optional row limit. Define inputs via `tool_param` placeholders in your query and register their schema via `ToolParam`, then save with `create_tool`.

```python
from fenic import Session
from fenic.api.functions import col
from fenic.core._logical_plan.expressions.basic import tool_param
from fenic.core.mcp.types import ToolParam

session = Session.get_or_create(fc.SessionConfig(
    app_name="mcp_example",
))

# Create a small users table and add a description
users_df = session.create_dataframe({
    "name": ["Alice", "Bob", "Charlie", "Diana"],
    "age": [25, 40, 31, 18],
})
users_df.write.save_as_table("users", mode="overwrite")
session.catalog.set_table_description("users", "User information")

users = session.table("users")

# Tool A: Filter users by optional age range. Uses coalesce to evaluate to True if the user does not pass in one side of the range.
optional_min = fc.coalesce(fc.col("age") >= tool_param("min_age", IntegerType), fc.lit(True))
optional_max = fc.coalesce(fc.col("age") <= tool_param("max_age", IntegerType), fc.lit(True))
core_filter = df.filter(optional_min & optional_max)
session.catalog.create_tool(
    "users_by_age_range",
    "Filter users by age",
    core_filter,
    tool_params=[
        # If default values are provided, the parameters will be marked as `Optional` in the MCP API Spec.
        ToolParam(name="min_age", description="Minimum age", has_default=True, default_value=None),
        ToolParam(name="max_age", description="Maximum age", has_default=True, default_value=None),
    ]
)

# Tool B: Case-sensitive regex search by name (use (?i) for case-insensitive).
name_search_query = users.filter(col("name").rlike(tool_param("name_regex")))

# If a default is not provided, the paramater will be marked as `required` in the MCP API Spec.
name_search_params = [
    ToolParam(
        name="name_regex",
        description="Search for users by name, using a regular expression. (use (?i) for case-insensitive)",
    )
]

session.catalog.create_tool(
    tool_name="users_by_name_regex",
    tool_description="Return users whose name matches the provided regex.",
    tool_query=name_search_query,
    tool_params=name_search_params,
    result_limit=100,
)
```

List, fetch, or drop tools:

```python
all_tools = session.catalog.list_tools()
age_tool = session.catalog.describe_tool("users_by_age_range")
search_tool = session.catalog.describe_tool("users_by_name_regex")
session.catalog.drop_tool("users_by_age_range", ignore_if_not_exists=True)
session.catalog.drop_tool("users_by_name_regex", ignore_if_not_exists=True)
```

### Step 2b: Create dynamic tools with `@fenic_tool`

Dynamic tools let you expose arbitrary Python logic as an MCP tool. They are defined with the `@fenic_tool` decorator and must return a Fenic `DataFrame`. Annotate parameters with `typing_extensions.Annotated` to provide per-argument descriptions in the tool schema. The server automatically adds `limit` and `table_format` keyword-only parameters for limiting the size of result sets and output formatting -- if the tool handles its own limiting, set `client_limit_parameter` to `False` to disable this behavior. The wrapped function can be async (recommended) or synchronous.

```python
from typing_extensions import Annotated

from fenic import Session
from fenic.api.dataframe.dataframe import DataFrame
from fenic.api.functions import col, coalesce, lit
from fenic.api.functions import sum as sum_
from fenic.api.mcp.tool_generation import fenic_tool

session = Session.get_or_create()

# Two base DataFrames
users = session.create_dataframe({
    "id": [1, 2, 3, 4],
    "name": ["Alice", "Bob", "Charlie", "Diana"],
    "age": [25, 40, 31, 18],
})

orders = session.create_dataframe({
    "order_id": [10, 11, 12, 13, 14],
    "user_id": [1, 1, 2, 3, 3],
    "amount": [50.0, 75.0, 20.0, 15.0, 120.0],
})

# Aggregate orders per user
orders_total = orders.group_by("user_id").agg(
    sum_(col("amount")).alias("total_amount")
)


@fenic_tool(
    tool_name="users_with_min_spend",
    tool_description="Users whose name matches regex (optional) and total order amount >= min_total",
    max_result_limit=100,
    default_table_format="markdown",
)
async def users_with_min_spend(
    name_regex: Annotated[Optional[str], "Regex for user name (use (?i) for case-insensitive)"] = None,
    min_total: Annotated[float, "Minimum total order amount"],
) -> DataFrame:
    joined = users.join(orders_total, left_on="id", right_on="user_id", how="left")
    pred_name = col("name").rlike(name_regex) if name_regex else fc.lit(True)
    pred_total = coalesce(col("total_amount"), lit(0.0)) >= min_total
    return joined.filter(pred_name & pred_total).select("id", "name", "age", "total_amount")
```

Notes:

- The decorated function must not use `*args` or `**kwargs` and must return a Fenic `DataFrame`.
- Use `Annotated[type, "description"]` for parameters to generate a clear MCP schema.
- Dynamic tools are not stored in the catalog; they exist only while your server process is running.
- Dynamic tools can be used to integrate your MCP server with external data sources or APIs to perform operations.

### Step 2c: Auto-generate system tools from catalog tables

You can generate a suite of reusable data tools (Schema, Profile, Read, Search Summary, Search Content, Analyze) directly from catalog tables and their descriptions. This is helpful for quickly exposing exploratory and read/query capabilities to MCP.

Requirements:

- Each table must exist and have a non-empty description (see Step 1).

Example:

```python
from fenic import Session
from fenic.api.mcp.server import create_mcp_server
from fenic.api.mcp.tool_generation import ToolGenerationConfig

session = Session.get_or_create()

server = create_mcp_server(
    session,
    server_name="Fenic MCP",
    automated_tool_generation=ToolGenerationConfig(
        table_names=["orders", "users"],
        tool_group_name="Dataset Exploration",
        sql_max_rows=200,
    ),
)
```

## Step 3a: Serve tools programmatically

Use the MCP server helpers to serve existing catalog tools. If you want all registered tools, call `list_tools()`. If you want a subset, fetch by name.

```python
from fenic import Session
from fenic.api.mcp.server import create_mcp_server, run_mcp_server_sync, run_mcp_server_async, run_mcp_server_asgi,

session = Session.get_or_create(fc.SessionConfig(
    app_name="mcp_example",
))

# Load all catalog tools
tools = session.catalog.list_tools()

server = create_mcp_server(session, server_name="Fenic MCP", parameterized_tools=tools)

# Run HTTP server (defaults shown); if additional configuration is required, any argument that can be passed to FastMCP `run` can be passed here
#
run_mcp_server_sync(
    server,
    transport="http",
    host="127.0.0.1",
    port=8000,
    stateless_http=True,
    path="/mcp",
)

# If already within an async context, the server can run inside that existing context instead of creating a new event loop
await run_mcp_server_async(
    server,
    transport="http",
    host="127.0.0.1",
    port=8000,
    stateless_http=True,
    path="/mcp",
)

# Finally, in production environments it might be necessary to configure the application with additional middleware, or serve the application from something other
# than uvicorn -- in that case, we expose `run_mcp_server_asgi`, which creates a Starlette ASGI application that can be plugged in to your existing stack

asgi_app = run_mcp_server_asgi(
    server,
    transport="http",
    host="127.0.0.1",
    port=8000,
    stateless_http=True,
    path="/mcp",
    # middleware = [...]
)

```

Include dynamic tools:

```python
from fenic.api.mcp.tool_generation import fenic_tool

# Assume `users_name_regex` is defined as in Step 2b
server = create_mcp_server(
    session,
    server_name="Fenic MCP",
    parameterized_tools=tools,
    dynamic_tools=[users_name_regex],
)
```

Enable automated tool generation (Schema/Profile/Read/Search/Analyze) from catalog tables:

```python
from fenic.api.mcp.tool_generation import ToolGenerationConfig

server = create_mcp_server(
    session,
    server_name="Fenic MCP",
    automated_tool_generation=ToolGenerationConfig(
        table_names=["orders", "users"],
        tool_group_name="Core Datasets",
        sql_max_rows=200,
    ),
)
```

## Step 3b: Serve tools via CLI (fenic-serve)

The CLI starts an MCP server directly from your catalog. By default, it serves all registered tools in the current database, using uvicorn.

Basic usage (serve all tools registered in catalog, using the existing `mcp_app`):

```bash
fenic-serve --app-name mcp_example --port 8000 --host 127.0.0.1
```

Serve specific tools only:

```bash
fenic-serve --app-name mcp_example --tools users_by_age_range users_by_name_regex --port 8000
```

Provide a session configuration via JSON file, and customize the path.

```bash
fenic-serve --config-file ./session.config.json --port 8000 --path /
```

Example `session.config.json` (minimal):

```json
{
  "app_name": "mcp_example"
}
```

Environment variables for model providers (if your tools use semantic operators) should be set in your shell, or via your runner (for example: `uv run --env-file .env ...`).

## Dynamic vs Parameterized tools

| Aspect                  | Dynamic tools (`@fenic_tool`)                                                                                  | Parameterized tools (catalog)                                                                          |
| ----------------------- | -------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| Flexibility             | Highest: arbitrary Python and Fenic ops; closures; any logic that returns a `DataFrame`.                       | Moderate: declarative `DataFrame` queries with `tool_param` placeholders.                              |
| Persistence/Portability | Not persisted; exist only while the server process is running; require source code at runtime.                 | Persisted in the catalog; portable across sessions and environments without access to original code.   |
| Discoverability         | Not listed in the catalog; visible only via the MCP server that registers them.                                | First-class catalog objects: `list_tools()`, `get_tool()`, `drop_tool()`.                              |
| Parameters/Schema       | From function signature using `Annotated[type, "description"]`. `limit` and `table_format` are auto-added.     | From `ToolParam` definitions bound to `tool_param(...)` in the plan. Defaults mark params as optional. |
| Execution context       | Executes the returned logical plan from your function; can capture `DataFrame`s or use session access in code. | Executes a stored logical plan with bound parameters from the catalog.                                 |
| Result formatting       | `table_format` supports `markdown` or `structured`; `limit` caps rows (capped by `max_result_limit` if set).   | Same.                                                                                                  |
| Best for                | Custom logic, semantic/procedural transforms, external data integration, mixing multiple data sources in code. | Reusable, shareable queries/macros that outlive the application process.                               |

## Troubleshooting

- No tools found: ensure you have created tools in the current database (`session.catalog.list_tools()`).
- Table descriptions: recommended for documentation; set via `create_table(..., description=...)` or `set_table_description()`.
- HTTP path: The default path for the mcp server is `/mcp`.
- SessionConfig exposes `to_json` for converting an existing SessionConfig to a jsonified version.
