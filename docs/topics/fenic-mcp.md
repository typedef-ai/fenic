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
age_tool = session.catalog.get_tool("users_by_age_range")
search_tool = session.catalog.get_tool("users_by_name_regex")
session.catalog.drop_tool("users_by_age_range", ignore_if_not_exists=True)
session.catalog.drop_tool("users_by_name_regex", ignore_if_not_exists=True)
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

server = create_mcp_server(session, server_name="Fenic MCP", tools=tools)

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

## Troubleshooting

- No tools found: ensure you have created tools in the current database (`session.catalog.list_tools()`).
- Table descriptions: recommended for documentation; set via `create_table(..., description=...)` or `set_table_description()`.
- HTTP path: The default path for the mcp server is `/mcp`.
- SessionConfig exposes `to_json` for converting an existing SessionConfig to a jsonified version.
