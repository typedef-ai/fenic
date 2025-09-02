"""MCP server and tool integration for Fenic.

This module exposes a small wrapper that registers Tool Definitions
with FastMCP, collects results, and formats them as table-like output
for model consumption. FastMCP is an optional dependency.

Install with:
    pip install "fenic[mcp]"
    # or
    pip install fastmcp
"""
import asyncio
import logging
import re
from typing import Any, Dict, List, Union

from pydantic import BaseModel
from typing_extensions import Literal

from fenic.core._interfaces.session_state import BaseSessionState
from fenic.core._utils.structured_outputs import (
    convert_pydantic_model_to_key_descriptions,
)
from fenic.core.error import ConfigurationError
from fenic.core.mcp._binder import bind_parameters
from fenic.core.mcp._tools import (
    create_pydantic_model_for_tool,
)
from fenic.core.mcp.types import (
    ParameterizedToolDefinition,
    TableFormat,
)
from fenic.logging import configure_logging

logger = logging.getLogger(__name__)


class MCPResultSet(BaseModel):
    """Structured result returned to the MCP client."""
    table_schema: List[Dict[str, Any]]
    rows: Union[List[Dict[str, Any]], str]
    row_count: int

MCPTransport = Literal["http", "stdio"]

class FenicMCPServer:
    """Register Fenic tools and serve them via FastMCP."""

    def __init__(
        self,
        session_state: BaseSessionState,
        paramaterized_tools: list[ParameterizedToolDefinition],
        server_name: str = "Fenic Views",
        concurrency_limit: int = 8
    ):
        """Initialize the server with a Fenic session state and tool list.

        Args:
            session_state: Fenic session state to use for tool execution.
            paramaterized_tools: List of user-created tools
            server_name: Name of the MCP server.
            concurrency_limit: Maximum number of concurrent tool executions.
        """
        self.session_state = session_state
        self.server_name = server_name
        self.paramaterized_tools = paramaterized_tools
        self._collect_semaphore = asyncio.Semaphore(concurrency_limit)
        if not paramaterized_tools:
            raise ConfigurationError("No tools provided to MCP server.")
        try:
            from fastmcp import FastMCP
            from mcp.types import ToolAnnotations
        except ImportError:
            raise ImportError(
                "To use fenic MCP server generation, install the 'mcp' extra: pip install \"fenic[mcp]\""
            ) from None
        self.mcp = FastMCP(self.server_name)
        for tool in self.paramaterized_tools:
            tool_fn = self._build_parameterized_tool(tool)
            self.mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))(tool_fn)

    async def run_async(self, transport: MCPTransport = "http", **kwargs):
        """Run the MCP server asynchronously.

        Args:
            transport: Transport protocol to use (http, stdio).
            kwargs: Additional transport-specific arguments to pass to FastMCP.
        """
        # Ensure Fenic logs are visible unless host app already configured logging
        configure_logging()
        await self.mcp.run_async(transport=transport, show_banner=False, **kwargs)

    def run(self, transport: MCPTransport = "http", **kwargs):
        """Run the MCP server. This is a synchronous function.

        Args:
            transport: Transport protocol to use (http, stdio).
            kwargs: Additional transport-specific arguments to pass to FastMCP.
        """
        # Ensure Fenic logs are visible unless host app already configured logging
        configure_logging()
        self.mcp.run(transport=transport, show_banner=False, **kwargs)

    def http_app(self, **kwargs):
        """Create a Starlette ASGI app for the MCP server."""
        return self.mcp.http_app(**kwargs)

    def _build_parameterized_tool(self, tool: ParameterizedToolDefinition):
        """Build a Pydantic single-parameter tool function for ResolvedTool."""
        ParamsModel = create_pydantic_model_for_tool(tool)

        async def tool_fn(params: Union[ParamsModel, str]) -> MCPResultSet:  # type: ignore[name-defined]
            # https://github.com/anthropics/claude-code/issues/3084
            # Claude Code will pass structured json parameters as a string
            if isinstance(params, str):
                params = ParamsModel.model_validate_json(params)
            payload = params.model_dump(exclude_none=True)
            table_format: TableFormat = payload.pop("table_format", "markdown")
            requested_limit = payload.pop("limit", None)
            effective_limit: int = tool.result_limit if requested_limit is None else min(int(requested_limit), tool.result_limit)
            try:
                bound_plan = bind_parameters(tool._parameterized_view, payload, tool.params)
                async with self._collect_semaphore:
                    pl_df, metrics = await asyncio.to_thread(lambda: self.session_state.execution.collect(bound_plan, n=effective_limit))
                    logger.info(f"Completed query for {tool.name} in {metrics.execution_time_ms:.0f}ms with {metrics.num_output_rows} result rows.")
                    logger.debug(f"Query Details: {params.model_dump_json()}")

                rows_list = pl_df.to_dicts()

                schema_fields = [{"name": name, "type": str(dtype)} for name, dtype in pl_df.schema.items()]
                result_set = MCPResultSet(
                    table_schema=schema_fields,
                    rows=rows_list,
                    row_count=len(rows_list),
                )
                if table_format == "markdown":
                    result_set.rows = _render_markdown_preview(rows_list)
                return result_set
            except Exception as e:
                from fastmcp.exceptions import ToolError
                raise ToolError(f"Fenic server failed to execute tool {tool.name}. Underlying error: {e}") from e

        tool_fn.__name__ = _to_snake_case(tool.name)
        pydantic_schema_description = convert_pydantic_model_to_key_descriptions(ParamsModel)
        tool_fn.__doc__ = "\n\n".join([tool.description, pydantic_schema_description])
        return tool_fn


def _to_snake_case(name: str) -> str:
    result = name
    return "_".join(
        re.sub(
            "([A-Z][a-z]+)",
            r" \1",
            re.sub("([A-Z]+)", r" \1", result.replace("-", " ")),
        ).split()
    ).lower()

def _render_markdown_preview(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "No rows."
    columns = list(rows[0].keys())
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)
