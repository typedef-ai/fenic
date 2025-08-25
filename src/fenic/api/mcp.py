"""Create MCP servers using Fenic DataFrames.

This module exposes helpers to:
- Build a Fenic-backed MCP server from datasets and tools
- Run the server synchronously or asynchronously
"""
from dataclasses import dataclass
from typing import List, Optional, Union

from fenic.api.session.session import Session
from fenic.api.tools import DatasetSpec, auto_generate_core_tools
from fenic.core._logical_plan.tools import DynamicTool, ResolvedTool
from fenic.core.error import ConfigurationError
from fenic.core.mcp.generator import FenicMCPServer, MCPTransport


@dataclass
class ToolGenerationConfig:
    """Configuration for automated tool generation.

    Args:
        datasets: List of DatasetSpec objects.
        tool_group_name: Name of the tool group.
        sql_max_rows: Maximum number of rows to be returned from SQL queries.
    """

    datasets: List[DatasetSpec]
    tool_group_name: str
    sql_max_rows: int = 100

def create_mcp_server(
    session: Session,
    server_name: str,
    *,
    tools: Optional[List[Union[ResolvedTool, DynamicTool]]] = None,
    automated_tool_generation: Optional[ToolGenerationConfig] = None,
) -> FenicMCPServer:
    """Create an MCP server from datasets and tools.

    Args:
        session: Fenic session used to execute tools.
        server_name: Name of the MCP server.
        tools: Additional tools to register (optional).
        automated_tool_generation: Generate automated tools for one or more Dataframes
    """
    if tools is None:
        tools: List[Union[ResolvedTool, DynamicTool]] = []
    if automated_tool_generation:
        tools.extend(auto_generate_core_tools(
            automated_tool_generation.datasets,
            session,
            tool_group_name=automated_tool_generation.tool_group_name,
            sql_max_rows=automated_tool_generation.sql_max_rows)
        )
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