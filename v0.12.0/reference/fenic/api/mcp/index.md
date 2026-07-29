# fenic.api.mcp

Canonical HTML: https://docs.fenic.ai/latest/reference/fenic/api/mcp/

MCP Tool Creation/Server Management API.

Classes:

- **`SystemToolConfig`**
  –

  Configuration for canonical system tools.

Functions:

- **`create_mcp_server`**
  –

  Create an MCP server from datasets and tools.
- **`run_mcp_server_asgi`**
  –

  Run an MCP server as a Starlette ASGI app.
- **`run_mcp_server_async`**
  –

  Run an MCP server asynchronously.
- **`run_mcp_server_sync`**
  –

  Run an MCP server synchronously.

## SystemToolConfig

```
SystemToolConfig(table_names: list[str], tool_namespace: Optional[str] = None, max_result_rows: int = 100)
```

Configuration for canonical system tools.

fenic can automatically generate a set of canonical tools for operating on one or more fenic tables.

- Schema: list columns/types for any or all tables
- Profile: column statistics (counts, basic numeric analysis [min, max, mean, etc.], contextual information for text columns [average_length, etc.])
- Read: read a selection of rows from a single table. These rows can be paged over, filtered and can use column projections.
- Search Summary: literal or regex search across all text columns in all tables -- returns back dataframe names with result counts. Use `search_mode="literal"` for plain substring search or `search_mode="regex"` for regular expressions.
- Search Content: literal or regex search across a single table, specifying one or more text columns to search across -- returns back rows corresponding to the query. Use `search_mode="literal"` for plain substring search or `search_mode="regex"` for regular expressions.
- Analyze: Write raw SQL to perform complex analysis on one or more tables.

Attributes:

- **`table_names`**
  (`list[str]`)
  –

  List of the fenic table names the tools should be able to access. To allow access to all tables, pass `session.catalog.list_tables()`
- **`tool_namespace`**
  (`Optional[str]`)
  –

  If provided, will prefix the names of the generated tools with this namespace value.
  For example, by default the generated tools will be named `read`, `profile`, etc. With multiple fenic
  MCP servers, these tool names will clash, which can be confusing. In order to disambiguate, the `tool_namespace`
  is prefixed to the tool name (in snake case), so a `tool_namespace` of `fenic` would create the tools `fenic_read`,
  `fenic_profile`, etc.
- **`max_result_rows`**
  (`int`)
  –

  Maximum number of rows to be returned from Read/Analyze tools.

Example:

```
    from fenic import SystemToolConfig
    from fenic.api.mcp.tools import SystemToolConfig
    from fenic.api.mcp.server import create_mcp_server
    from fenic.api.session.session import Session
    session = Session.get_or_create(...)
    df = session.create_dataframe({
        "c1": [1, 2, 3],
        "c2": [4, 5, 6]
    })
    df.write.save_as_table("table1", mode="overwrite")
    session.catalog.set_table_description("table1", "Table 1 Description")
    server = create_mcp_server(session, "Test Server", system_tools=SystemToolConfig(
        table_names=["table1"],
        tool_namespace="Auto",
        max_result_rows=100
    ))
```

Example: Allow generated tools to access all tables in the catalog.

```
    from fenic import SystemToolConfig
    from fenic.api.mcp.tools import SystemToolConfig
    from fenic.api.mcp.server import create_mcp_server
    from fenic.api.session.session import Session
    session = Session.get_or_create(...)
    # Assuming you already have one or more tables saved to the catalog, with descriptions.
    server = create_mcp_server(session, "Test Server", system_tools=SystemToolConfig(
        table_names=session.catalog.list_tables()
        tool_namespace="Auto",
        max_result_rows=100
    ))
```

## create_mcp_server

```
create_mcp_server(session: Session, server_name: str, *, user_defined_tools: Optional[List[UserDefinedTool]] = None, system_tools: Optional[SystemToolConfig] = None, concurrency_limit: int = 8) -> FenicMCPServer
```

Create an MCP server from datasets and tools.

Parameters:

- **`session`**
  (`Session`)
  –

  Fenic session used to execute tools.
- **`server_name`**
  (`str`)
  –

  Name of the MCP server.
- **`user_defined_tools`**
  (`Optional[List[UserDefinedTool]]`, default:
  `None`
  )
  –

  User defined tools to register with the MCP server.
- **`system_tools`**
  (`Optional[SystemToolConfig]`, default:
  `None`
  )
  –

  Configuration for automatically created system tools.
- **`concurrency_limit`**
  (`int`, default:
  `8`
  )
  –

  Maximum number of concurrent tool executions.

Source code in `src/fenic/api/mcp/server.py`

```
@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def create_mcp_server(
    session: Session,
    server_name: str,
    *,
    user_defined_tools: Optional[List[UserDefinedTool]] = None,
    system_tools: Optional[SystemToolConfig] = None,
    concurrency_limit: int = 8,
) -> FenicMCPServer:
    """Create an MCP server from datasets and tools.

    Args:
        session: Fenic session used to execute tools.
        server_name: Name of the MCP server.
        user_defined_tools: User defined tools to register with the MCP server.
        system_tools: Configuration for automatically created system tools.
        concurrency_limit: Maximum number of concurrent tool executions.
    """
    generated_system_tools = []
    user_defined_tools = user_defined_tools or []
    if system_tools:
        generated_system_tools.extend(
            auto_generate_system_tools_from_tables(
                system_tools.table_names,
                session,
                tool_namespace=system_tools.tool_namespace,
                max_result_limit=system_tools.max_result_rows
            )
        )
    if not (user_defined_tools or system_tools):
        raise ConfigurationError("No tools provided. Either provide `user_defined_tools` or set `system_tools` to create system tools for catalog tables.")
    return FenicMCPServer(session._session_state, user_defined_tools, generated_system_tools, server_name, concurrency_limit)
```

## run_mcp_server_asgi

```
run_mcp_server_asgi(server: FenicMCPServer, *, stateless_http: bool = True, transport: Literal['streamable-http', 'sse'] = 'streamable-http', path: Optional[str] = '/mcp', **kwargs)
```

Run an MCP server as a Starlette ASGI app.

Returns a Starlette ASGI app that can be integrated into any ASGI server.
This is useful for running the MCP server in a production environment, or running the MCP server as part of a larger application.

Parameters:

- **`server`**
  (`FenicMCPServer`)
  –

  MCP server to run.
- **`stateless_http`**
  (`bool`, default:
  `True`
  )
  –

  If True, use stateless HTTP.
- **`transport`**
  (`Literal['streamable-http', 'sse']`, default:
  `'streamable-http'`
  )
  –

  Transport protocol to use (streamable-http, sse).
- **`path`**
  (`Optional[str]`, default:
  `'/mcp'`
  )
  –

  Path to listen on.
- **`kwargs`**
  –

  Additional starlette-specific arguments to pass to FastMCP.

Notes

Additional keyword arguments:
- `middleware`: A list of Starlette `ASGIMiddleware` middleware to apply to the app.

Source code in `src/fenic/api/mcp/server.py`

```
@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def run_mcp_server_asgi(
    server: FenicMCPServer,
    *,
    stateless_http: bool = True,
    transport: Literal['streamable-http', 'sse'] = 'streamable-http',
    path: Optional[str] = "/mcp",
    **kwargs,
):
    """Run an MCP server as a Starlette ASGI app.

    Returns a Starlette ASGI app that can be integrated into any ASGI server.
    This is useful for running the MCP server in a production environment, or running the MCP server as part of a larger application.

    Args:
        server: MCP server to run.
        stateless_http: If True, use stateless HTTP.
        transport: Transport protocol to use (streamable-http, sse).
        path: Path to listen on.
        kwargs: Additional starlette-specific arguments to pass to FastMCP.

    Notes:
        Additional keyword arguments:
        - `middleware`: A list of Starlette `ASGIMiddleware` middleware to apply to the app.
    """
    return server.http_app(stateless_http=stateless_http, transport=transport, path=path, **kwargs)
```

## run_mcp_server_async

```
run_mcp_server_async(server: FenicMCPServer, *, transport: MCPTransport = 'http', stateless_http: bool = True, port: Optional[int] = None, host: Optional[str] = None, path: Optional[str] = '/mcp', **kwargs)
```

Run an MCP server asynchronously.

Use this when calling from asynchronous code. This does not create a new event loop.

Parameters:

- **`server`**
  (`FenicMCPServer`)
  –

  MCP server to run.
- **`transport`**
  (`MCPTransport`, default:
  `'http'`
  )
  –

  Transport protocol (http, stdio).
- **`stateless_http`**
  (`bool`, default:
  `True`
  )
  –

  If True, use stateless HTTP.
- **`port`**
  (`Optional[int]`, default:
  `None`
  )
  –

  Port to listen on.
- **`host`**
  (`Optional[str]`, default:
  `None`
  )
  –

  Host to listen on.
- **`path`**
  (`Optional[str]`, default:
  `'/mcp'`
  )
  –

  Path to listen on.
- **`kwargs`**
  –

  Additional transport-specific arguments to pass to FastMCP.

Source code in `src/fenic/api/mcp/server.py`

```
@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
async def run_mcp_server_async(
    server: FenicMCPServer,
    *,
    transport: MCPTransport = "http",
    stateless_http: bool = True,
    port: Optional[int] = None,
    host: Optional[str] = None,
    path: Optional[str] = "/mcp",
    **kwargs,
):
    """Run an MCP server asynchronously.

    Use this when calling from asynchronous code. This does not create a new event loop.

    Args:
        server: MCP server to run.
        transport: Transport protocol (http, stdio).
        stateless_http: If True, use stateless HTTP.
        port: Port to listen on.
        host: Host to listen on.
        path: Path to listen on.
        kwargs: Additional transport-specific arguments to pass to FastMCP.
    """
    await server.run_async(transport=transport, stateless_http=stateless_http, port=port, host=host, path=path, **kwargs)
```

## run_mcp_server_sync

```
run_mcp_server_sync(server: FenicMCPServer, *, transport: MCPTransport = 'http', stateless_http: bool = True, port: Optional[int] = None, host: Optional[str] = None, path: Optional[str] = '/mcp', **kwargs)
```

Run an MCP server synchronously.

Use this when calling from synchronous code. This creates a new event loop and runs the server in it.

Parameters:

- **`server`**
  (`FenicMCPServer`)
  –

  MCP server to run.
- **`transport`**
  (`MCPTransport`, default:
  `'http'`
  )
  –

  Transport protocol (http, stdio).
- **`stateless_http`**
  (`bool`, default:
  `True`
  )
  –

  If True, use stateless HTTP.
- **`port`**
  (`Optional[int]`, default:
  `None`
  )
  –

  Port to listen on.
- **`host`**
  (`Optional[str]`, default:
  `None`
  )
  –

  Host to listen on.
- **`path`**
  (`Optional[str]`, default:
  `'/mcp'`
  )
  –

  Path to listen on.
- **`kwargs`**
  –

  Additional transport-specific arguments to pass to FastMCP.

Source code in `src/fenic/api/mcp/server.py`

```
@validate_call(config=ConfigDict(strict=True, arbitrary_types_allowed=True))
def run_mcp_server_sync(
    server: FenicMCPServer,
    *,
    transport: MCPTransport = "http",
    stateless_http: bool = True,
    port: Optional[int] = None,
    host: Optional[str] = None,
    path: Optional[str] = "/mcp",
    **kwargs,
):
    """Run an MCP server synchronously.

    Use this when calling from synchronous code. This creates a new event loop and runs the server in it.

    Args:
        server: MCP server to run.
        transport: Transport protocol (http, stdio).
        stateless_http: If True, use stateless HTTP.
        port: Port to listen on.
        host: Host to listen on.
        path: Path to listen on.
        kwargs: Additional transport-specific arguments to pass to FastMCP.
    """
    server.run(transport=transport, stateless_http=stateless_http, port=port, host=host, path=path, **kwargs)
```
