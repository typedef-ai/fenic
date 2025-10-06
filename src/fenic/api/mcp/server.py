"""Create MCP servers using Fenic DataFrames.

This module exposes helpers to:
- Build a Fenic-backed MCP server from datasets and tools
- Run the server synchronously or asynchronously
"""
from typing import List, Literal, Optional

from pydantic import ConfigDict, validate_call

from fenic.api.mcp._tool_generation_utils import auto_generate_system_tools_from_tables
from fenic.api.mcp.tools import (
    SystemToolConfig,
)
from fenic.api.session.session import Session
from fenic.core.error import ConfigurationError
from fenic.core.mcp._server import FenicMCPServer, MCPTransport
from fenic.core.mcp.types import UserDefinedTool


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