"""CLI for running a Fenic-backed MCP Server.

This script launches an MCP server using tables or pre-registered tools from the catalog.

Examples:
  - Run with environment variables from .env file:
      uv run --env-file .env fenic-serve

  - Run with all existing catalog tools and all defaults (default when --tools omitted):
      fenic-serve

  - Run with specific tools from catalog:
      fenic-serve \
        --tools sales_by_product sales_by_customer

  - Run with automated tools from tables (descriptions required in catalog metadata):
      fenic-serve \
        --tables orders customers \
        --tool_namespace "sales" \
        --sql-max-rows 100

  - Provide a session configuration via JSON file:
      fenic-serve \
        --config-file ./session.config.json \
        --tables orders customers

Notes:
  - A SessionConfig file only needs to be specified if your tools use semantic operations that depend on model definitions.
  - Environment variables (e.g., OPENAI_API_KEY) should be provided by your shell or with
    "uv run --env-file .env".
  - The configuration file must be JSON matching fenic.api.session.config.SessionConfig.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from fenic.api.mcp.server import create_mcp_server, run_mcp_server_sync
from fenic.api.mcp.tools import SystemToolConfig
from fenic.api.session.config import SessionConfig
from fenic.api.session.session import Session
from fenic.core.error import ConfigurationError, ToolNotFoundError


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Fenic MCP server from catalog tables or tools.")

    # Session config
    parser.add_argument("--config-file", type=str, default=None, help="Path to JSON file for SessionConfig.")
    parser.add_argument("--app-name", type=str, default="fenic_mcp",
                        help="SessionConfig.app_name if no config file provided.")
    parser.add_argument("--db-path", type=str, default=None, help="SessionConfig.db_path if no config file provided.")

    # Server
    parser.add_argument("--server-name", type=str, default="Fenic MCP", help="Name for the MCP server.")
    parser.add_argument("--concurrency-limit", type=int, default=8, help="Maximum number of concurrent tool executions.")
    parser.add_argument("--transport", type=str, choices=["http", "stdio"], default="http", help="Transport protocol.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host for HTTP transport.")
    parser.add_argument("--port", type=int, default=8000, help="Port for HTTP transport.")
    parser.add_argument("--path", type=str, default="/mcp", help="Path for HTTP MCP Server endpoint.")
    parser.add_argument("--stateful-http", action="store_true",
                        help="Run the MCP server in stateful mode -- clients are assigned a session id and can use it to persist state across requests (default: stateless).")

    # Inputs: tools or tables
    parser.add_argument("--tools", nargs="*", default=None, help="Catalog tool names to load.")
    parser.add_argument("--tables", nargs="*", default=None, help="Catalog table names for automated tool generation.")
    parser.add_argument("--tool_namespace", type=str, default=None,
                        help="Tool namespace for generated tools. Will be converted to `snake_case`. "
                             "Example: If provided with a value of `dataset_exploration`, tools will be generated as `dataset_exploration_schema`, `dataset_exploration_read` etc.")
    parser.add_argument("--sql-max-rows", type=int, default=100, help="Row limit for Analyze/Read/Search tools.")

    return parser.parse_args()


def _load_session_config(args: argparse.Namespace) -> SessionConfig:
    if args.config_file:
        path = Path(args.config_file)
        if not path.exists():
            raise ConfigurationError(f"Session config file not found: {path}")
        try:
            config_data = json.loads(path.read_text())
        except Exception as e:
            raise ConfigurationError(f"Failed to parse SessionConfig JSON: {e}") from e
        return SessionConfig.model_validate(config_data)

    # Minimal config if none provided
    db_path: Optional[Path] = Path(args.db_path) if args.db_path else None
    return SessionConfig(app_name=args.app_name, db_path=db_path)


def main() -> None:
    """Main entry point for the fenic_mcp script."""
    args = _parse_args()
    session_config = _load_session_config(args)
    session = Session.get_or_create(session_config)

    # Tools: if none specified, load all registered catalog tools
    tools = []
    if args.tools:
        for tool_name in args.tools:
            try:
                tools.append(session.catalog.describe_tool(tool_name))
            except ToolNotFoundError as e:
                raise ConfigurationError(f"Tool {tool_name} not found in the catalog.") from e
    else:
        tools = session.catalog.list_tools()

    auto_cfg: Optional[SystemToolConfig] = None
    if args.tables:
        auto_cfg = SystemToolConfig(
            table_names=args.tables,
            tool_namespace=args.tool_namespace,
            max_result_rows=args.sql_max_rows,
        )

    # If neither tables nor any tools resolved, error out with guidance
    if not args.tables and not tools:
        raise ConfigurationError(
            "No tools or tables provided, and no tools registered in the catalog. Provide --tools/--tables or register tools.")

    server = create_mcp_server(
        session,
        server_name=args.server_name,
        user_defined_tools=tools if tools else None,
        system_tools=auto_cfg,
        concurrency_limit=args.concurrency_limit,
    )

    run_mcp_server_sync(
        server,
        transport=args.transport,
        stateless_http=not args.stateful_http,
        host=args.host,
        port=args.port,
        path=args.path,
    )


if __name__ == "__main__":
    main()
