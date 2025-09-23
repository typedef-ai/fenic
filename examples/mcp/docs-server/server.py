"""MCP server for Fenic using native Fenic MCP capabilities.

This server uses Fenic's built-in MCP support instead of FastMCP.
"""
import logging
import os
import sys

import fenic as fc
from fenic.api.mcp import create_mcp_server
from fenic.core.mcp.types import ToolParam
from fenic.core.types.datatypes import StringType

fc.logging.configure_logging()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def setup_session():
    """Setup Fenic session with proper configuration."""
    work_dir = os.environ.get("FENIC_WORK_DIR", os.path.expanduser("~/.fenic"))
    os.makedirs(work_dir, exist_ok=True)
    os.chdir(work_dir)
    logger.info(f"Working directory: {work_dir}")
    
    # Configure fenic session
    config = fc.SessionConfig(
        app_name="docs",  # Must match the app_name used in populate_tables.py
    )
    
    return fc.Session.get_or_create(config)


def register_search_tools(session: fc.Session):
    """Register search-related tools in the catalog."""
    logger.info("Registering tools in catalog...")
    
    # Tool 1: Search API documentation
    search_query = (
        session.table("api_df")
        .filter(
            (fc.col("is_public")) 
            & (~fc.col("qualified_name").rlike(r"(^|\.)_"))
            & (
                fc.col("name").rlike(fc.tool_param("query", StringType))
                | fc.col("qualified_name").rlike(fc.tool_param("query", StringType))
                | (fc.col("docstring").is_not_null() & fc.col("docstring").rlike(fc.tool_param("query", StringType)))
            )
        )
        .select(
            "type",
            "name", 
            "qualified_name",
            "docstring",
            "parameters",
            "returns",
            "parent_class"
        )
    )
    
    session.catalog.create_tool(
        tool_name="search_fenic_api",
        tool_description="Search Fenic API documentation using regex patterns. Returns matching API elements including functions, classes, methods, and their docstrings.",
        tool_query=search_query,
        tool_params=[
            ToolParam(
                name="query",
                description="Regex pattern to search (e.g., 'semantic.*extract')",
            ),
        ],
        result_limit=50,  # Use result_limit instead of dynamic limit
        ignore_if_exists=True
    )
    
    # Tool 2: Get entity by qualified name
    entity_query = (
        session.table("api_df")
        .filter(
            (fc.col("is_public"))
            & (fc.col("qualified_name") == fc.tool_param("qualified_name", StringType))
        )
        .select(
            "type",
            "name",
            "qualified_name",
            "docstring",
            "annotation",
            "returns",
            "parameters",
            "parent_class",
            "line_start",
            "line_end",
            "filepath"
        )
    )
    
    session.catalog.create_tool(
        tool_name="get_entity",
        tool_description="Get detailed information about a specific API entity by its fully qualified name. Returns complete documentation including parameters, returns, and implementation details.",
        tool_query=entity_query,
        tool_params=[
            ToolParam(
                name="qualified_name",
                description="Fully qualified name (e.g., 'fenic.api.dataframe.DataFrame.select')"
            ),
        ],
        result_limit=1,
        ignore_if_exists=True
    )
    
    # Tool 3: Get project overview (no parameters)
    overview_query = session.table("fenic_summary").select("project_summary")
    
    session.catalog.create_tool(
        tool_name="get_project_overview",
        tool_description="Get a comprehensive overview of the Fenic project, including its purpose, key features, architecture, and main API components.",
        tool_query=overview_query,
        tool_params=[],
        result_limit=1,
        ignore_if_exists=True
    )
    
    # Tool 4: Get API tree
    tree_query = (
        session.table("hierarchy_df")
        .filter(
            (fc.col("is_public"))
            & (fc.col("type") != "attribute")
            & (~fc.col("name").starts_with("_"))
        )
        .select("qualified_name", "name", "type", "depth", "path_parts")
        .order_by([fc.col("depth"), fc.col("type"), fc.col("name")])
    )
    
    session.catalog.create_tool(
        tool_name="get_api_tree",
        tool_description="Get the hierarchical tree structure of Fenic's public API, showing modules, classes, functions, and methods in their organizational hierarchy.",
        tool_query=tree_query,
        tool_params=[],
        result_limit=5000,
        ignore_if_exists=True
    )
    
    # Tool 5: Search by type
    type_search_query = (
        session.table("api_df")
        .filter(
            (fc.col("is_public"))
            & (fc.col("type") == fc.tool_param("element_type", StringType))
            & (
                fc.col("name").rlike(fc.tool_param("pattern", StringType))
                | fc.col("qualified_name").rlike(fc.tool_param("pattern", StringType))
            )
        )
        .select("type", "name", "qualified_name", "docstring")
    )
    
    session.catalog.create_tool(
        tool_name="search_by_type",
        tool_description="Search for specific types of API elements (class, function, method, module) with optional pattern matching. Useful for finding all classes, all methods of a certain type, etc.",
        tool_query=type_search_query,
        tool_params=[
            ToolParam(
                name="element_type",
                description="Type of element: 'class', 'function', 'method', 'module'",
                allowed_values=["class", "function", "method", "module", "attribute"]
            ),
            ToolParam(
                name="pattern",
                description="Regex pattern to match",
                default_value=".*",
                has_default=True
            ),
        ],
        result_limit=50,  # Use result_limit instead of dynamic limit
        ignore_if_exists=True
    )
    
    logger.info("Successfully registered 5 tools in catalog")


def main():
    """Main entry point for the native Fenic MCP server."""
    try:
        # Setup session
        session = setup_session()
        
        # Check if required tables exist
        required_tables = ["api_df", "hierarchy_df", "fenic_summary"]
        missing_tables = []
        for table in required_tables:
            if not session.catalog.does_table_exist(table):
                missing_tables.append(table)
        
        if missing_tables:
            logger.error(
                f"Missing required tables: {missing_tables}\n"
                "Please run 'python populate_tables.py' to set up the documentation database."
            )
            sys.exit(1)
        
        # Register tools in catalog
        register_search_tools(session)
        
        # Get all tools from catalog
        tools = session.catalog.list_tools()
        logger.info(f"Found {len(tools)} tools in catalog")
        for tool in tools:
            logger.info(f"  - {tool.name}: {tool.description}")
        
        # Create native MCP server
        server = create_mcp_server(
            session=session,
            server_name="Fenic Documentation Server",
            user_defined_tools=tools,
            concurrency_limit=8
        )
        
        # Run the server using HTTP transport
        logger.info("Starting Fenic MCP server on HTTP port 8000...")
        server.run(
            transport="http",
            host="127.0.0.1",
            port=8000,
            stateless_http=True
        )
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Failed to start MCP server: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
