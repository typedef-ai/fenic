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
import inspect
import logging
from functools import wraps
from typing import Any, Callable, Coroutine, Dict, List, Optional, Union

import polars as pl
from pydantic import BaseModel, ConfigDict
from typing_extensions import Annotated, Literal

from fenic.core._interfaces.session_state import BaseSessionState
from fenic.core._logical_plan import LogicalPlan
from fenic.core._utils.misc import to_snake_case
from fenic.core._utils.structured_outputs import (
    convert_pydantic_model_to_key_descriptions,
)
from fenic.core._utils.type_inference import infer_pytype_from_dtype
from fenic.core.error import ConfigurationError
from fenic.core.mcp._binder import bind_parameters
from fenic.core.mcp._tools import (
    LIMIT_DESCRIPTION,
    TABLE_FORMAT_DESCRIPTION,
    BoundToolParam,
    create_pydantic_model_for_tool,
)
from fenic.core.mcp.types import (
    SystemTool,
    TableFormat,
    UserDefinedTool,
)
from fenic.core.types.datatypes import ArrayType
from fenic.logging import configure_logging

logger = logging.getLogger(__name__)

class MCPResultSet(BaseModel):
    """Structured result returned to the MCP client."""
    model_config = ConfigDict(extra="forbid")

    table_schema: Optional[List[Dict[str, Any]]]
    rows: Union[List[Dict[str, Any]], str]
    returned_result_count: int
    total_result_count: int

MCPTransport = Literal["http", "stdio"]

class FenicMCPServer:
    """Register Fenic tools and serve them via FastMCP."""

    def __init__(
        self,
        session_state: BaseSessionState,
        user_defined_tools: list[UserDefinedTool],
        system_tools: list[SystemTool],
        server_name: str = "Fenic Views",
        concurrency_limit: int = 8
    ):
        """Initialize the server with a Fenic session state and tool list.

        Args:
            session_state: Fenic session state to use for tool execution.
            user_defined_tools: List of user-created tools
            system_tools: List of auto-generated tools
            server_name: Name of the MCP server.
            concurrency_limit: Maximum number of concurrent tool executions.
        """
        self.session_state = session_state
        self.server_name = server_name
        self.user_defined_tools = user_defined_tools
        self.system_tools = system_tools
        self._collect_semaphore = asyncio.Semaphore(concurrency_limit)
        if not (user_defined_tools or system_tools):
            raise ConfigurationError("No tools provided to MCP server.")
        try:
            from fastmcp import FastMCP
            from mcp.types import ToolAnnotations
        except ImportError:
            raise ImportError(
                "To use fenic MCP server generation, install the 'mcp' extra: pip install \"fenic[mcp]\""
            ) from None
        self.mcp = FastMCP(self.server_name)
        for tool in self.user_defined_tools:
            tool_fn = self._build_user_defined_tool(tool)
            self.mcp.tool(
                annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False),
                name=to_snake_case(tool.name),
                title=tool.name
            )(tool_fn)

        for tool in self.system_tools:
            tool_fn = self._build_system_tool(tool)
            self.mcp.tool(
                annotations=ToolAnnotations(
                    readOnlyHint=tool.read_only,
                    openWorldHint=tool.open_world,
                    destructiveHint=tool.destructive,
                    idempotentHint=tool.idempotent,
                ),
                name=to_snake_case(tool.name),
                title=tool.name,
                description=tool.description
            )(tool_fn)

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

    def _handle_result_set(
        self,
        plan: LogicalPlan,
        pl_df: pl.DataFrame,
        effective_limit: Optional[int],
        table_format: TableFormat
    ) -> MCPResultSet:
        """Handle the result set from a logical plan."""
        original_result_count = len(pl_df)
        if effective_limit and original_result_count > effective_limit:
            pl_df = pl_df.limit(effective_limit)
        schema_fields = [{"name": field.name, "type": str(field.data_type)} for field in plan.schema().column_fields]
        rows_list = pl_df.to_dicts()
        returned_result_count = len(rows_list)
        if table_format == "structured":
            result_set = MCPResultSet(
                table_schema=schema_fields,
                rows=rows_list,
                returned_result_count=returned_result_count,
                total_result_count=original_result_count,
            )
        else:
            rows = _render_markdown_preview(rows_list)
            result_set = MCPResultSet(
                table_schema=schema_fields,
                rows=rows,
                returned_result_count=returned_result_count,
                total_result_count=original_result_count,
            )
        return result_set

    def _build_user_defined_tool(
        self,
        tool_definition: UserDefinedTool
    ) -> Callable[..., Coroutine[Any, Any, MCPResultSet]]:
        """Build a keyword-argument tool function with per-field schema for FastMCP.

        We still validate/coerce using a generated Pydantic model under the hood,
        but expose individual keyword-only parameters in the function signature so
        FastMCP generates a clean per-argument schema (no single params model).
        """
        ParamsModel = create_pydantic_model_for_tool(tool_definition)

        # Names of bound parameters for filtering out helper args
        param_names = {p.name for p in tool_definition.params}

        async def tool_fn_wrapper(*args, **kwargs) -> MCPResultSet:
            # Extract UI/runtime-only flags
            table_format: TableFormat = kwargs.pop("table_format", "markdown")
            # Compute effective limit: cap by tool.result_limit when provided
            effective_limit = _calculate_effective_limit(tool_definition, kwargs.pop("limit", None))

            # Validate/coerce only the bound parameters using the ParamsModel
            payload_only_tool_params = {k: v for k, v in kwargs.items() if k in param_names}
            params_obj = ParamsModel.model_validate(payload_only_tool_params, strict=False)
            payload = params_obj.model_dump(exclude_none=True)

            try:
                bound_plan = bind_parameters(tool_definition._parameterized_view, payload, tool_definition.params)
                async with self._collect_semaphore:
                    pl_df, metrics = await asyncio.to_thread(
                        lambda: self.session_state.execution.collect(bound_plan)
                    )
                    logger.info(f"Completed query for {tool_definition.name}")
                    logger.info(metrics.get_summary())
                    logger.debug(f"Query Details: {params_obj.model_dump_json()}")
                    return self._handle_result_set(bound_plan, pl_df, effective_limit, table_format)
            except Exception as e:
                from fastmcp.exceptions import ToolError
                raise ToolError(f"Fenic server failed to execute tool {tool_definition.name}. Underlying error: {e}") from e


        try:
            params: list[inspect.Parameter] = []
            annotations: Dict[str, object] = {}
            # Add one keyword-only parameter per tool param
            for param in tool_definition.params:
                param_type = _type_for_param(param)
                param_annotation = _annotate_with_description(param_type, param.description)
                default_value = param.default_value if param.has_default else inspect._empty
                params.append(
                    inspect.Parameter(
                        name=param.name,
                        kind=inspect.Parameter.KEYWORD_ONLY,
                        default=default_value,
                        annotation=param_annotation,
                    )
                )
                annotations[param.name] = param_annotation

            # Add table_format and limit just like system tools
            tf_ann = Annotated[TableFormat, (
                TABLE_FORMAT_DESCRIPTION
            )]
            lim_ann = Annotated[Optional[Union[str, int]], LIMIT_DESCRIPTION]
            params.append(
                inspect.Parameter(
                    name="table_format",
                    kind=inspect.Parameter.KEYWORD_ONLY,
                    default="markdown",
                    annotation=tf_ann,
                )
            )
            params.append(
                inspect.Parameter(
                    name="limit",
                    kind=inspect.Parameter.KEYWORD_ONLY,
                    default=tool_definition.max_result_limit,
                    annotation=lim_ann,
                )
            )
            annotations["table_format"] = tf_ann
            annotations["limit"] = lim_ann

            tool_fn_wrapper.__signature__ = inspect.Signature(parameters=params, return_annotation=MCPResultSet)  # type: ignore[attr-defined]
            # Update annotations for Pydantic
            existing = getattr(tool_fn_wrapper, "__annotations__", {}) or {}
            existing.update(annotations)
            tool_fn_wrapper.__annotations__ = existing
        except (ValueError, TypeError) as e:
            raise ConfigurationError(f"Failed to add parameters to function {tool_fn_wrapper.__name__}") from e

        # Docstring includes schema from the Pydantic model
        pydantic_schema_description = convert_pydantic_model_to_key_descriptions(ParamsModel)
        tool_fn_wrapper.__doc__ = "\n\n".join([tool_definition.description, pydantic_schema_description])
        return tool_fn_wrapper

    def _build_system_tool(
        self,
        tool_definition: SystemTool
    ) -> Callable[[...], Coroutine[Any, Any, MCPResultSet]]:
        # Dynamic function must return a LogicalPlan. This registrar wraps the callable so
        # that we (a) execute/collect the plan off the event loop, (b) limit concurrency,
        # and (c) format the results into an MCPResultSet for FastMCP.
        #
        # Important: We intentionally use a two-layer wrapper pattern below.
        # - `wrapper` performs the actual execution/collection/formatting work.
        # - `wrapped` is decorated with `@wraps(tool.func)` so FastMCP can introspect the
        #   original function signature for tool schema generation.

        async def wrapper(*args, **kwargs) -> MCPResultSet:
            # Extract table_format from kwargs if provided, otherwise use tool default
            table_format = kwargs.pop("table_format", tool_definition.default_table_format)
            # Extract optional limit and compute effective_limit against tool.result_limit
            effective_limit = _calculate_effective_limit(tool_definition, kwargs.pop("limit", None))
            # Obtain the plan by invoking the system tool. No session is injected here;
            # the callable is expected to derive any context it needs from inputs.
            try:
                # Collect on a thread to avoid blocking the event loop, and gate concurrent
                # collections with a semaphore to protect the backend executor.
                async with self._collect_semaphore:
                    bound_plan = await asyncio.to_thread(
                        lambda: tool_definition.func(*args, **kwargs)
                    )
                    pl_df, metrics = await asyncio.to_thread(
                        lambda: self.session_state.execution.collect(bound_plan)
                    )
                    logger.info(f"Completed query for {tool_definition.name}")
                    logger.info(metrics.get_summary())
                    logger.debug(f"Query Details: {args if args else kwargs}")
                    return self._handle_result_set(bound_plan, pl_df, effective_limit, table_format)
            except Exception as e:
                from fastmcp.exceptions import ToolError
                raise ToolError(f"Fenic server failed to execute tool {tool_definition.name}. Underlying error: {e}") from e

        @wraps(tool_definition.func)
        async def wrapped(*args, **kwargs):
            # Delegate to the inner wrapper; @wraps preserves the original signature so
            # FastMCP can generate a clean tool schema from annotations.
            return await wrapper(*args, **kwargs)

        # Expose `table_format` and `limit` to the schema without requiring user functions to accept them
        _expose_keyword_param(
            "table_format",
            wrapped=wrapped,
            default_value=tool_definition.default_table_format,
            py_type=TableFormat,
            description=(
                TABLE_FORMAT_DESCRIPTION
            ),
        )
        if tool_definition.max_result_limit and tool_definition.add_limit_parameter:
            _expose_keyword_param(
                "limit",
                wrapped=wrapped,
                default_value=tool_definition.max_result_limit,
                py_type=Optional[Union[int, str]],
                description=LIMIT_DESCRIPTION,
            )

        return wrapped

# Helper to expose extra keyword-only parameters to FastMCP without changing the
# user function's implementation surface area. Ensures both signature and
# annotations are updated for Pydantic schema generation.
def _expose_keyword_param(
    param_name: str,
    wrapped: Callable,
    py_type: object,
    description: str | None = None,
    *,
    default_value: object,
) -> None:
    try:
        original_sig = inspect.signature(wrapped)
        if param_name not in original_sig.parameters:
            # Build final annotation, augmenting with a description when provided
            ann = Annotated[py_type, description] if description is not None else py_type
            new_params = list(original_sig.parameters.values())
            new_params.append(
                inspect.Parameter(
                    name=param_name,
                    kind=inspect.Parameter.KEYWORD_ONLY,
                    default=default_value,
                    annotation=ann,
                )
            )
            wrapped.__signature__ = original_sig.replace(parameters=new_params)  # type: ignore[attr-defined]
            # Ensure Pydantic can resolve the added parameter by updating annotations
            annotations = getattr(wrapped, "__annotations__", {}) or {}
            if param_name not in annotations:
                annotations[param_name] = ann
                wrapped.__annotations__ = annotations
    except (ValueError, TypeError) as e:
        raise ConfigurationError(f"Failed to add parameter {param_name} to function {wrapped.__name__}") from e

# Build a synthetic signature and annotations for FastMCP schema generation
def _type_for_param(p: BoundToolParam) -> type:
    # allowed_values -> Literal[...] (or list[Literal[...]] for arrays)
    if p.allowed_values:
        literal_values = tuple(p.allowed_values)
        literal_type = Literal[literal_values]  # type: ignore[valid-type]
        if isinstance(p.data_type, ArrayType):
            return list[literal_type]  # type: ignore[valid-type]
        if p.has_default:
            literal_type = Optional[literal_type]
        return literal_type
    # Otherwise infer py type and wrap list for arrays
    if isinstance(p.data_type, ArrayType):
        base_py = list[infer_pytype_from_dtype(p.data_type.element_type)]  # type: ignore[valid-type]
    else:
        base_py = infer_pytype_from_dtype(p.data_type)
    if p.has_default:
        base_py = Optional[base_py]
    return base_py

def _annotate_with_description(base_ann: type, description: Optional[str] = None):
    if description:
        return Annotated[base_ann, description]
    return base_ann

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

def _calculate_effective_limit(
    tool: Union[UserDefinedTool, SystemTool],
    requested_limit: Optional[Union[int, str]]
) -> Optional[int]:
    if requested_limit:
        if isinstance(requested_limit, str):
            requested_limit = int(requested_limit)
        if tool.max_result_limit:
            effective_limit = min(int(requested_limit), tool.max_result_limit)
        else:
            effective_limit = int(requested_limit)
    else:
        effective_limit = tool.max_result_limit
    return effective_limit