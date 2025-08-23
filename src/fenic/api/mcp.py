"""Create MCP servers using Fenic DataFrames.

This module exposes helpers to:
- Build a Fenic-backed MCP server from datasets and tools
- Run the server synchronously or asynchronously
"""

from typing import List, Optional, Union

from fenic.api.session.session import Session
from fenic.api.tools import DatasetSpec, auto_generate_core_tools
from fenic.core._logical_plan.tools import DynamicTool, ResolvedTool
from fenic.core.error import ConfigurationError
from fenic.core.mcp.generator import FenicMCPServer, MCPTransport


def create_mcp_server(
    session: Session,
    server_name: str,
    *,
    tools: Optional[List[Union[ResolvedTool, DynamicTool]]] = None,
    generate_automated_tools: Optional[bool] = None,
    datasets: Optional[List[DatasetSpec]] = None,
    tool_group_name: Optional[str] = None,
    sql_max_rows: int = 100,
) -> FenicMCPServer:
    """Create an MCP server from datasets and tools.

    Args:
        session: Fenic session used to execute tools.
        server_name: Name of the MCP server.
        tools: Additional tools to register (optional).
        generate_automated_tools: If True, generate Schema/Describe/Analyze tools for datasets.
        datasets: Datasets exposed to the tools (names, descriptions, DataFrames).
        tool_group_name: Prefix for auto-generated tool names.
        sql_max_rows: Maximum rows for the auto-generated SQL tool.
    """
    if datasets is None:
        datasets = []
    if tools is None:
        tools: List[Union[ResolvedTool, DynamicTool]] = []
    if generate_automated_tools and datasets and tool_group_name:
        tools.extend(auto_generate_core_tools(datasets, session, tool_group_name=tool_group_name, sql_max_rows=sql_max_rows))
    if not tools:
        raise ConfigurationError("No tools provided. Either provide tools or set generate_automated_tools=True and provide datasets.")
    return FenicMCPServer(session._session_state, tools, server_name)

def run_mcp_server_sync(
    server: FenicMCPServer,
    *,
    transport: MCPTransport = "http",
    stateless_http: bool = True,
    port: Optional[int] = None,
    host: Optional[str] = None,
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
        kwargs: Additional transport-specific arguments to pass to FastMCP.
    """
    server.run(transport=transport, stateless_http=stateless_http, port=port, host=host, **kwargs)


async def run_mcp_server_async(
    server: FenicMCPServer,
    *,
    transport: MCPTransport = "http",
    stateless_http: bool = True,
    port: Optional[int] = None,
    host: Optional[str] = None,
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
        kwargs: Additional transport-specific arguments to pass to FastMCP.
    """
    await server.run_async(transport=transport, stateless_http=stateless_http, port=port, host=host, **kwargs)