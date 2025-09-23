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
    Optional,
)


@dataclass
class SystemToolConfig:
    """Configuration for canonical system tools.

    fenic can automatically generate a set of canonical tools for operating on one or more fenic tables.

    - Schema: list columns/types for any or all tables
    - Profile: column statistics (counts, basic numeric analysis [min, max, mean, etc.], contextual information for text columns [average_length, etc.])
    - Read: read a selection of rows from a single table. These rows can be paged over, filtered and can use column projections.
    - Search Summary: regex search across all text columns in all tables -- returns back dataframe names with result counts.
    - Search Content: regex search across a single table, specifying one or more text columns to search across -- returns back rows corresponding to the query.
    - Analyze: Write raw SQL to perform complex analysis on one or more tables.

    Attributes:
        table_names: List of the fenic table names the tools should be able to access. To allow access to all tables, pass `session.catalog.list_tables()`
        tool_namespace: If provided, will prefix the names of the generated tools with this namespace value.
            For example, by default the generated tools will be named `read`, `profile`, etc. With multiple fenic
            MCP servers, these tool names will clash, which can be confusing. In order to disambiguate, the `tool_namespace`
            is prefixed to the tool name (in snake case), so a `tool_namespace` of `fenic` would create the tools `fenic_read`,
            `fenic_profile`, etc.
        max_result_rows: Maximum number of rows to be returned from Read/Analyze tools.

    Example:
    ```python
        from fenic import SystemToolConfig
        from fenic.api.mcp.tools import SystemToolConfig
        from fenic.api.mcp.server import create_mcp_server
        from fenic.api.session.session import Session
        session = Session.get_or_create(...)
        df = session.create_dataframe({
            "c1": [1, 2, 3],
            "c2": [4, 5, 6]
        })
        df.write.save_as_table("table1", mode="overwrite")
        session.catalog.set_table_description("table1", "Table 1 Description")
        server = create_mcp_server(session, "Test Server", system_tools=SystemToolConfig(
            table_names=["table1"],
            tool_namespace="Auto",
            max_result_rows=100
        ))
    ```

    Example: Allow generated tools to access all tables in the catalog.
    ```python
        from fenic import SystemToolConfig
        from fenic.api.mcp.tools import SystemToolConfig
        from fenic.api.mcp.server import create_mcp_server
        from fenic.api.session.session import Session
        session = Session.get_or_create(...)
        # Assuming you already have one or more tables saved to the catalog, with descriptions.
        server = create_mcp_server(session, "Test Server", system_tools=SystemToolConfig(
            table_names=session.catalog.list_tables()
            tool_namespace="Auto",
            max_result_rows=100
        ))
    ```
    """

    table_names: list[str]
    tool_namespace: Optional[str] = None
    max_result_rows: int = 100