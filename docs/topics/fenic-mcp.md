# Fenic MCP: Create and Serve Catalog Tools

This guide shows how to expose User Defined fenic tools via an MCP server. User Defined Tools are created by adding placeholder values to DataFrame operations
to be filled in at runtime, and managed in the Fenic Catalog. In most respects, these tools are like SQL Macros or parameterized views. Just like one might create a macro as:

```SQL
CREATE MACRO get_users(user_name) AS TABLE
    SELECT * FROM users WHERE name = user_name;
```

One can create a Tool in Fenic:

```python
import fenic as fc

filter_tool_df = df.filter(fc.col("name") == fc.tool_param("user_name", fc.StringType))
session.catalog.create_tool(
    "users_by_name",
    "Filter users by name",
    filter_tool_df,
    tool_params=[
        # If default values are provided, the parameters will be marked as `Optional` in the MCP API Spec.
        fc.ToolParam(name="user_name", description="User's Name (Exact Match)")
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
import fenic as fc
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
optional_min = fc.coalesce(fc.col("age") >= fc.tool_param("min_age", fc.IntegerType), fc.lit(True))
optional_max = fc.coalesce(fc.col("age") <= fc.tool_param("max_age", fc.IntegerType), fc.lit(True))
core_filter = df.filter(optional_min & optional_max)
session.catalog.create_tool(
    "users_by_age_range",
    "Filter users by age",
    core_filter,
    tool_params=[
        # If default values are provided, the parameters will be marked as `Optional` in the MCP API Spec.
        fc.ToolParam(name="min_age", description="Minimum age", has_default=True, default_value=None),
        fc.ToolParam(name="max_age", description="Maximum age", has_default=True, default_value=None),
    ]
)

# Tool B: Case-sensitive regex search by name (use (?i) for case-insensitive).
name_search_query = users.filter(fc.col("name").rlike(fc.tool_param("name_regex")))

# If a default is not provided, the parameter will be marked as `required` in the MCP API Spec.
name_search_params = [
    fc.ToolParam(
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

### Step 2a: Auto-generate system tools from catalog tables

You can generate a suite of reusable data tools (Schema, Profile, Read, Search Summary, Search Content, Analyze) directly from catalog tables and their descriptions.
This is helpful for quickly exposing exploratory and read/query capabilities to MCP. Available tools include:

- Schema: list columns/types for any or all tables
- Profile: column statistics (counts, basic numeric analysis [min, max, mean, etc.], contextual information for text columns [average_length, etc.])
- Read: read a selection of rows from a single table. These rows can be paged over, filtered and can use column projections.
- Search Summary: regex search across all text columns in all tables -- returns back dataframe names with result counts.
- Search Content: regex search across a single table, specifying one or more text columns to search across -- returns back rows corresponding to the query.
- Analyze: Write raw SQL to perform complex analysis on one or more tables.

Requirements:

- Each table must exist and have a non-empty description (see Step 1).

Example:

```python
from fenic import Session
from fenic.api.mcp.server import create_mcp_server
from fenic.api.mcp.tools import SystemToolConfig

session = Session.get_or_create(...)
server = create_mcp_server(
    session,
    server_name="Fenic MCP",
    system_tools=SystemToolConfig(
        table_names=session.catalog.list_tables(),
        tool_namespace="Dataset Exploration",
        max_result_rows=200,
    ),
)
```

## Step 3a: Serve tools programmatically

Use the MCP server helpers to serve existing catalog tools. To use all catalog tools in the MCP server,
pass `session.catalog.list_tools` to `create_mcp_server`:

```python
from fenic import Session,SessionConfig
from fenic.api.mcp.server import create_mcp_server, run_mcp_server_sync, run_mcp_server_async, run_mcp_server_asgi

session = Session.get_or_create(SessionConfig(
    app_name="mcp_example",
    ...
))
server = create_mcp_server(session, server_name="Fenic MCP", user_defined_tools=session.catalog.list_tools())

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
  "app_name": "mcp_demo",
  "semantic": {
    "language_models": {
      "gpt-4.1-nano": {
        "model_name": "gpt-4.1-nano",
        "rpm": 2500,
        "tpm": 2000000
      }
    },
    "default_language_model": "gpt-4.1-nano"
  }
}
```

Environment variables for model providers (if your tools use semantic operators) should be set in your shell, or via your runner (for example: `uv run --env-file .env ...`).

## Troubleshooting

- No tools found: ensure you have created tools in the current database (`session.catalog.list_tools()`).
- Table descriptions: recommended for documentation; set via `create_table(..., description=...)` or `set_table_description()`.
- HTTP path: The default path for the mcp server is `/mcp`.
- SessionConfig exposes `to_json` for converting an existing SessionConfig to a jsonified version.
