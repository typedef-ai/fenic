"""MCP tool generator for Fenic parameterized views (no exec, no **kwargs).

This module integrates Fenic's core parameterized view primitives with FastMCP by
creating tools that accept a single Pydantic model (params). The model provides
full schema for MCP, strong typing, and default limit handling.
"""

from typing import Any, Dict, List, Optional, Type, get_origin

from fastmcp import FastMCP
from pydantic import BaseModel, Field, create_model
from typing_extensions import Literal

import fenic as fc
from fenic.core._logical_plan.parameterized_views import ParamaterizedQuery

DEFAULT_TOOL_LIMIT = 50


class AutoMCPGenerator:
    """Generate MCP tools from views using a single Pydantic params argument."""

    def __init__(self, session: fc.Session, server_name: str = "Fenic Views"):
        """Initialize the generator with a Fenic session and server name."""
        self.session = session
        self.server_name = server_name
        self.views: Dict[str, ParamaterizedQuery] = {}

    def register_view(self, view: ParamaterizedQuery) -> None:
        """Register a parameterized view to expose as a tool."""
        self.views[view.name] = view

    def generate_server(self) -> FastMCP:
        """Build and return a FastMCP server containing all registered tools."""
        mcp = FastMCP(self.server_name)
        for view in self.views.values():
            tool_fn = self._build_tool(view)
            mcp.tool()(tool_fn)
        return mcp

    def _build_tool(self, view: ParamaterizedQuery):
        """Create a FastMCP tool with signature (params: ViewParamsModel) -> str."""
        ParamsModel = self._build_params_model(view)

        def tool(params: ParamsModel) -> str:  # type: ignore[name-defined]
            payload = params.model_dump(exclude_none=True)
            limit: int = int(payload.pop("limit", DEFAULT_TOOL_LIMIT))
            try:
                result_df = view.execute(self.session, **payload)
                preview_df = result_df.limit(limit)
                rows_list = preview_df.to_pylist()
                if not rows_list:
                    return f"No results for {view.name}."
                lines: List[str] = [f"Showing {len(rows_list)} rows (limit {limit})\n"]
                first = rows_list[0]
                if isinstance(first, dict):
                    # list[dict]
                    columns = list(first.keys())
                    for i, row in enumerate(rows_list, 1):
                        pairs = ", ".join(f"{col}: {row.get(col)}" for col in columns)
                        lines.append(f"{i}. {pairs}")
                else:
                    # list of sequences
                    for i, row in enumerate(rows_list, 1):
                        pairs = ", ".join(str(v) for v in row)
                        lines.append(f"{i}. {pairs}")
                return "\n".join(lines)
            except Exception as e:
                return f"Error executing {view.name}: {e}"

        tool.__name__ = view.name
        tool.__doc__ = view.description
        tool.__annotations__ = {"params": ParamsModel, "return": str}
        return tool

    def _build_params_model(self, view: ParamaterizedQuery) -> Type[BaseModel]:
        """Create a Pydantic model type describing the view's parameters."""
        fields: Dict[str, tuple] = {}
        for name, param in view.parameters.items():
            origin = get_origin(param.type)
            # Map base type
            if origin is list or origin is List:
                base_ann: Any = str
                ann: Any = List[base_ann]
                if param.enum_values:
                    lit = Literal[tuple(param.enum_values)]  # type: ignore[index]
                    ann = List[lit]  # type: ignore[valid-type]
            else:
                if param.enum_values:
                    ann = Literal[tuple(param.enum_values)]  # type: ignore[index]
                else:
                    if param.type in (str, int, float, bool):
                        ann = param.type
                    else:
                        ann = Any

            # Handle required vs optional
            if param.required and param.default is None:
                default = ...
            else:
                default = param.default

            fields[name] = (
                ann,
                Field(default, description=param.description),
            )

        # Always include a default limit for tool output preview
        fields["limit"] = (
            Optional[int],
            Field(DEFAULT_TOOL_LIMIT, description="Maximum number of rows to display"),
        )

        model_name = f"{view.name}_Params"
        return create_model(model_name, **fields)  # type: ignore[arg-type]


# Convenience function

def create_mcp_server_from_views(
    views: List[ParamaterizedQuery],
    session: fc.Session,
    server_name: str = "Fenic Views",
) -> FastMCP:
    """Create an MCP server from a list of parameterized views."""
    generator = AutoMCPGenerator(session, server_name)
    for view in views:
        generator.register_view(view)
    return generator.generate_server()
