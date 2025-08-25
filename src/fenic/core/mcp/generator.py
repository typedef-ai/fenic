"""MCP server and tool integration for Fenic.

This module exposes a small wrapper that registers DynamicTool and ResolvedTool
instances with FastMCP, collects results, and formats them as table-like output
for model consumption. FastMCP is an optional dependency.

Install with:
    pip install "fenic[mcp]"
    # or
    pip install fastmcp
"""
import asyncio
import logging
import re
from functools import wraps
from typing import Any, Dict, List, Union

from fastmcp.exceptions import ToolError
from pydantic import BaseModel
from typing_extensions import Literal

from fenic.core._interfaces.session_state import BaseSessionState
from fenic.core._logical_plan.binder import bind_parameters
from fenic.core._logical_plan.tools import (
    DynamicTool,
    ResolvedTool,
    TableFormat,
    create_pydantic_model_for_tool,
)
from fenic.core._utils.structured_outputs import (
    convert_pydantic_model_to_key_descriptions,
)

logger = logging.getLogger(__name__)

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


class MCPResultSet(BaseModel):
    """Structured result returned to the MCP client."""
    table_schema: List[Dict[str, Any]]
    rows: Union[List[Dict[str, Any]], str]
    row_count: int

MCPTransport = Literal["http", "stdio"]

class FenicMCPServer:
    """Register Fenic tools and serve them via FastMCP."""

    def __init__(self, session_state: BaseSessionState, tools: List[Union[ResolvedTool, DynamicTool]], server_name: str = "Fenic Views"):
        """Initialize the server with a Fenic session state and tool list.

        Args:
            session_state: Fenic session state to use for tool execution.
            tools: List of tools to register.
            server_name: Name of the MCP server.
        """
        self.session_state = session_state
        self.server_name = server_name
        self.tools = tools
        self._collect_semaphore = asyncio.Semaphore(8)
        if not self.tools:
            raise ValueError("No tools provided")
        try:
            from fastmcp import FastMCP
        except ImportError:
            raise ImportError(
                "To use fenic MCP server generation, install the 'mcp' extra: pip install \"fenic[mcp]\""
            ) from None
        self.mcp = FastMCP(self.server_name)
        for tool in self.tools:
            tool_fn = self._build_tool(tool)
            self.mcp.tool(name=self._to_snake_case(tool.name), description=tool.description)(tool_fn)

    async def run_async(self, transport: MCPTransport = "http", **kwargs):
        """Run the MCP server asynchronously.

        Args:
            transport: Transport protocol to use (http, stdio).
            kwargs: Additional transport-specific arguments to pass to FastMCP.
        """
        await self.mcp.run_async(transport=transport, **kwargs)

    def run(self, transport: MCPTransport = "http", **kwargs):
        """Run the MCP server. This is a synchronous function.

        Args:
            transport: Transport protocol to use (http, stdio).
            kwargs: Additional transport-specific arguments to pass to FastMCP.
        """
        self.mcp.run(transport=transport, **kwargs)


    def _build_tool(self, tool: Union[ResolvedTool, DynamicTool]):
        """Create a FastMCP tool from a Fenic tool definition."""
        if isinstance(tool, ResolvedTool):
            return self._build_resolved_pydantic(tool)
        return self._register_dynamic_callable(tool)

    def _build_resolved_pydantic(self, tool: ResolvedTool):
        """Build a Pydantic single-parameter tool function for ResolvedTool."""
        ParamsModel = create_pydantic_model_for_tool(tool)

        async def tool_fn(params: Union[ParamsModel, str]) -> MCPResultSet:  # type: ignore[name-defined]
            # https://github.com/anthropics/claude-code/issues/3084
            # Claude Code will pass structured json parameters as a string
            if isinstance(params, str):
                params = ParamsModel.model_validate_json(params)
            payload = params.model_dump(exclude_none=True)
            table_format: TableFormat = payload.pop("table_format", "structured")
            requested_limit = payload.pop("limit", None)
            effective_limit: int = tool.result_limit if requested_limit is None else min(int(requested_limit), tool.result_limit)
            try:
                bound_plan = bind_parameters(tool.query, payload, tool.params)
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
                raise ToolError(f"Failed to execute tool {tool.name}. Underlying error: {e}") from e

        tool_fn.__name__ = self._to_snake_case(tool.name)
        pydantic_schema_description = convert_pydantic_model_to_key_descriptions(ParamsModel)
        tool_fn.__doc__ = "\n\n".join([tool.description, pydantic_schema_description])
        return tool_fn

    def _register_dynamic_callable(self, tool: DynamicTool):
        # Dynamic function must return a LogicalPlan. This registrar wraps the callable so
        # that we (a) execute/collect the plan off the event loop, (b) limit concurrency,
        # and (c) format the results into an MCPResultSet for FastMCP.
        #
        # Important: We intentionally use a two-layer wrapper pattern below.
        # - `wrapper` performs the actual execution/collection/formatting work.
        # - `wrapped` is decorated with `@wraps(tool.func)` so FastMCP can introspect the
        #   original function signature for tool schema generation.

        async def wrapper(*args, **kwargs) -> MCPResultSet:
            # Obtain the plan by invoking the dynamic tool. No session is injected here;
            # the callable is expected to derive any context it needs from inputs.
            bound_plan = tool.func(*args, **kwargs)
            n_rows = tool.result_limit
            # Collect on a thread to avoid blocking the event loop, and gate concurrent
            # collections with a semaphore to protect the backend executor.
            async with self._collect_semaphore:
                pl_df, _metrics = await asyncio.to_thread(
                    lambda: self.session_state.execution.collect(bound_plan, n=n_rows)
                )
            rows_list = pl_df.to_dicts()
            schema_fields = [{"name": name, "type": str(dtype)} for name, dtype in pl_df.schema.items()]
            table_format = "structured"
            out = MCPResultSet(table_schema=schema_fields, rows=rows_list, row_count=len(rows_list))
            if table_format == "markdown":
                out.rows = _render_markdown_preview(rows_list)
            return out

        @wraps(tool.func)
        async def wrapped(*args, **kwargs):
            # Delegate to the inner wrapper; @wraps preserves the original signature so
            # FastMCP can generate a clean tool schema from annotations.
            return await wrapper(*args, **kwargs)

        return wrapped


    def _to_snake_case(self, name: str) -> str:
        result = name
        return "_".join(
            re.sub(
                "([A-Z][a-z]+)",
                r" \1",
                re.sub("([A-Z]+)", r" \1", result.replace("-", " ")),
            ).split()
        ).lower()