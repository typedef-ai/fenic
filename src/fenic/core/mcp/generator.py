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
import asyncio
import copy
import re
from typing import Any, Dict, List, Union

from fastmcp.exceptions import ToolError
from pydantic import BaseModel
from typing_extensions import Literal

import fenic as fc
from fenic.api.dataframe.dataframe import DataFrame
from fenic.core._logical_plan.binder import bind_parameters
from fenic.core._logical_plan.tools import ResolvedTool, create_pydantic_model_for_tool, TableFormat
from fenic.core._utils.structured_outputs import (
    convert_pydantic_model_to_key_descriptions,
)


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
    """A result set from an MCP tool."""
    table_schema: List[Dict[str, Any]]
    rows: Union[List[Dict[str, Any]], str]
    row_count: int


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
            raise ImportError(
                "FastMCP is not installed. Install the 'mcp' extra: pip install \"fenic[mcp]\" or install fastmcp directly."
            ) from None
        self.mcp = FastMCP(self.server_name)
        for tool in self.tools:
            tool_fn = self._build_tool(tool)
            self.mcp.tool()(tool_fn)

    async def run(self, transport: Literal["http", "stdio"] = "http", **kwargs):
        """Run the MCP server."""
        await self.mcp.run_async(transport=transport, **kwargs)

    def _build_tool(self, tool: ResolvedTool):
        """Create a FastMCP tool with signature (params: ViewParamsModel) -> str."""
        ParamsModel = create_pydantic_model_for_tool(tool)

        async def tool_fn(params: ParamsModel) -> MCPResultSet:  # type: ignore[name-defined]
            payload = params.model_dump(exclude_none=True)
            table_format: TableFormat = payload.pop("table_format", "structured")
            limit: int = payload.pop("limit", tool.result_limit)
            try:
                mutable_plan_copy = copy.deepcopy(tool.query)
                bound_plan = bind_parameters(mutable_plan_copy, payload, tool.params)
                result_df = DataFrame._from_logical_plan(bound_plan, self.session._session_state)
                preview_df = result_df.limit(limit)
                rows_list = await asyncio.to_thread(preview_df.to_pylist)

                # Build schema array without mapping types to simple strings yet
                schema_fields = [
                    {"name": f.name, "type": str(f.data_type)}  # NOTE: consider mapping to simple strings later
                    for f in result_df.schema.column_fields
                ]
                result_set = MCPResultSet(
                    table_schema=schema_fields,
                    rows=rows_list,
                    row_count=len(rows_list))
                if table_format == "markdown":
                    result_set.rows = _render_markdown_preview(rows_list)
                return result_set
            except Exception as e:
                raise ToolError(f"Failed to execute tool {tool.name}") from e

        tool_fn.__name__ = self._to_snake_case(tool.name)  # mcp tools must not have spaces
        pydantic_schema_description = convert_pydantic_model_to_key_descriptions(ParamsModel)
        tool_fn.__doc__ = "\n\n".join([tool.description, pydantic_schema_description])
        return tool_fn

    def _to_snake_case(self, name: str) -> str:
        result = name
        return "_".join(
            re.sub(
                "([A-Z][a-z]+)",
                r" \1",
                re.sub("([A-Z]+)", r" \1", result.replace("-", " ")),
            ).split()
        ).lower()