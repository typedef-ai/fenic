"""API-layer generators for automatic MCP tools from DataFrames.

These helpers generate System Tool Definitions for:
- Schema: dataset column names and types
- Profile: per-column statistics (counts, numeric summaries, simple string summaries)
- Analyze: DuckDB SQL across one or more datasets.

All generated tools return LogicalPlan objects. The MCP server wrapper handles
execution and result formatting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    List,
)

from fenic.api.mcp._tool_generation_utils import (
    auto_generate_system_tools,
    build_datasets_from_tables,
)
from fenic.api.session.session import Session
from fenic.core.mcp.types import SystemTool


@dataclass
class ToolGenerationConfig:
    """Configuration for automated tool generation.

    Attributes:
        table_names: List of table names.
        tool_group_name: Name of the tool group.
        max_result_rows: Maximum number of rows to be returned from Read/Analyze tools.
    """

    table_names: List[str]
    tool_group_name: str
    max_result_rows: int = 100


def auto_generate_system_tools_from_tables(
    table_names: List[str],
    session: Session,
    *,
    tool_group_name: str,
    max_result_limit: int = 100,
) -> List[SystemTool]:
    """Generate Schema/Profile/Read/Search/Analyze tools from catalog tables.

    Validates that each table exists and has a non-empty description in catalog metadata.
    """
    datasets = build_datasets_from_tables(table_names, session)
    return auto_generate_system_tools(
        datasets,
        session,
        tool_group_name=tool_group_name,
        max_result_limit=max_result_limit,
    )
