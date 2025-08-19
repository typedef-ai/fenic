"""MCP tool generator for Fenic parameterized views (no exec, no **kwargs).

This module integrates Fenic's core parameterized view primitives with FastMCP by
creating tools that accept a single Pydantic model (params). The model provides
full schema for MCP, strong typing, and default limit handling.

FastMCP is an optional dependency. If it's not installed, attempts to generate
an MCP server will raise a helpful ImportError. Install with:

    pip install "fenic[mcp]"
    # or
    pip install fastmcp
"""
import re
from typing import List

from typing_extensions import Literal

import fenic as fc
from fenic.api.dataframe.dataframe import DataFrame
from fenic.core._logical_plan.binder import bind_parameters
from fenic.core._logical_plan.tools import ResolvedTool, create_pydantic_model_for_tool


class MCPGenerator:
    """Generate MCP tools from tools using a single Pydantic params argument."""

    def __init__(self, session: fc.Session, tools: List[ResolvedTool], server_name: str = "Fenic Views"):
        """Initialize the generator with a Fenic session and server name."""
        self.session = session
        self.server_name = server_name
        self.tools = tools
        if not self.tools:
            raise ValueError("No tools provided")
        try:
            from fastmcp import FastMCP
        except ImportError:
            raise ImportError("FastMCP is not installed. Install the 'mcp' extra: pip install \"fenic[mcp]\" "
                "or install fastmcp directly.") from None
        self.mcp = FastMCP(self.server_name)
        for tool in self.tools:
            tool_fn = self._build_tool(tool)
            self.mcp.tool()(tool_fn)

    def run(self, transport: Literal["http", "stdio"] = "http", **kwargs):
        """Run the MCP server."""
        self.mcp.run(transport=transport, **kwargs)

    def _build_tool(self, tool: ResolvedTool):
        """Create a FastMCP tool with signature (params: ViewParamsModel) -> str."""
        ParamsModel = create_pydantic_model_for_tool(tool)

        def tool_fn(params: ParamsModel) -> str:  # type: ignore[name-defined]
            payload = params.model_dump(exclude_none=True)
            limit: int = tool.result_limit
            try:
                bound_plan = bind_parameters(tool.query, payload, tool.params)
                result_df = DataFrame._from_logical_plan(bound_plan, self.session._session_state)
                preview_df = result_df.limit(limit)
                rows_list = preview_df.to_pylist()
                if not rows_list:
                    return f"No results for {tool.name}."
                lines: List[str] = [f"Showing {len(rows_list)} rows (limit {limit})\n"]
                first = rows_list[0]
                columns = list(first.keys())
                for i, row in enumerate(rows_list, 1):
                    pairs = ", ".join(f"{col}: {row.get(col)}" for col in columns)
                    lines.append(f"{i}. {pairs}")
                return "\n".join(lines)
            except Exception as e:
                return f"Error executing {tool.name}: {e}"

        tool_fn.__name__ = self._to_snake_case(tool.name)
        tool_fn.__doc__ = tool.description
        return tool_fn

    def _to_snake_case(self, name: str) -> str:
        result = name
        return '_'.join(
            re.sub('([A-Z][a-z]+)', r' \1',
                re.sub('([A-Z]+)', r' \1',
                    result.replace('-', ' '))).split()).lower()