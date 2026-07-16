"""This is a MCP server for the Fenic project.

It is used to search the Fenic codebase and provide documentation for the Fenic API.
"""

import asyncio
import faulthandler
import json
import logging
import os
import sys
import uuid
from dataclasses import dataclass
from typing import Literal

import fenic as fc
import structlog
from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError, ValidationError
from fastmcp.resources import TextResource
from fenic_mcp.server.search import get_entity_by_qualified_name, search_api_docs
from fenic_mcp.server.utils.session import create_session, log_fenic_version
from fenic_mcp.server.utils.tree_operations import build_tree, tree_to_string
from fenic_mcp.server.utils.validation import validate_and_sanitize_regex
from starlette.requests import Request
from starlette.responses import JSONResponse
from typing_extensions import Annotated

structlog.configure(
    processors=[
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)
# Configure the standard library logging to output to stdout
logging.basicConfig(
    format="%(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = structlog.get_logger(__name__)


@dataclass
class ProjectInformation:
    api_tree: str
    project_overview: str


def _generate_project_overview(session: fc.Session) -> ProjectInformation:
    logger.info("Getting project overview", endpoint="/get_project_overview")
    overview_df = session.table("fenic_summary").select("project_summary")
    api_tree_df = (
        session.table("hierarchy_df")
        .filter(
            (fc.col("is_public"))
            & (fc.col("type") != "attribute")
            & (~fc.col("name").starts_with("_"))
        )
        .select("qualified_name", "name", "type", "depth", "path_parts")
    )

    overview = overview_df.to_pydict()["project_summary"][0]
    structure = api_tree_df.to_pydict()
    api_tree_formatted = tree_to_string(build_tree(structure))
    project_overview_formatted = f"## Fenic Project Overview\n\n{overview}\n\n## API Tree\n\n{api_tree_formatted}"
    return ProjectInformation(
        api_tree=api_tree_formatted, project_overview=project_overview_formatted
    )


class FenicMCP:
    def __init__(self, max_concurrency: int = 16):
        self.session = create_session()
        self.project_information = _generate_project_overview(self.session)
        self.semaphore = asyncio.Semaphore(max_concurrency)

    def generate_server(self) -> FastMCP:
        mcp = FastMCP(
            name="fenic-docs",
            instructions="To establish initial context, use the `get_project_overview` and `get_api_tree` tools before performing any other actions.",
        )
        # Add resources (getting started, overview, API tree)
        try:
            # Get the directory where this script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            guide_path = os.path.join(script_dir, "fenic_getting_started_guide.md")
            with open(guide_path, "r", encoding="utf-8") as guide_file:
                guide_text = guide_file.read()
                mcp.add_resource(
                    TextResource(
                        uri="resource://fenic_getting_started_guide",
                        name="Fenic Getting Started Guide",
                        mime_type="text/markdown",
                        text=guide_text,
                    )
                )
            # Add in-memory resources for project overview and API tree for agents to consume
            if self.project_information and self.project_information.project_overview:
                mcp.add_resource(
                    TextResource(
                        uri="resource://fenic_project_overview",
                        name="Fenic Project Overview",
                        mime_type="text/markdown",
                        text=self.project_information.project_overview,
                    )
                )
            if self.project_information and self.project_information.api_tree:
                mcp.add_resource(
                    TextResource(
                        uri="resource://fenic_api_tree",
                        name="Fenic API Tree",
                        mime_type="text/markdown",
                        text=self.project_information.api_tree,
                    )
                )
        except FileNotFoundError:
            logger.warning(
                "fenic_getting_started_guide.md not found, skipping resource"
            )
        except Exception as e:
            logger.error(f"Error reading getting started guide: {e}")

        @mcp.tool()
        async def search(
            ctx: Context,
            query: Annotated[
                str,
                "Regular expression pattern to match against names, qualified paths, and docstrings.",
            ],
            max_results: Annotated[
                int, "Maximum number of results to return (default: 30)"
            ] = 30,
            output_format: Annotated[
                Literal["markdown", "json"],
                "The format of the output. (default: markdown)",
            ] = "markdown",
        ) -> str:
            r"""Search Fenic API/docs using a regular expression (REQUIRED).

            Use this to find functions, classes, methods, and other API elements by name,
            qualified path, or text in their docstrings/signatures. Queries are treated as
            case-insensitive regular expressions.

            Requirements:
            - The `query` MUST be a regular expression string.
            - Do NOT wrap in `/.../` or add inline flags like `(?i)`; case-insensitive is applied automatically.
            - Supported constructs include `.`, `.*`, `?`, `+`, `{m,n}`, character classes `[]`,
              alternation `|`, grouping `()`, and non-capturing `(?:...)`.
            - Disallowed: backreferences (\1), lookbehind, exotic inline constructs. Length ≤ 256 chars.

            Args:
                query: Regular expression pattern to match against names, qualified paths, and docstrings.
                max_results: Maximum number of results to return (default: 30)
                output_format: The format of the output. (default: markdown)

            Returns:
                Search results grouped by type with name, qualified path, and doc excerpt.

            Examples (use regex ONLY):
                - Basic: "semantic.*extract"
                - Alternation: "semantic.*(extract|join)"
                - Anchored path: "^fenic\\.api\\.(dataframe|column)\\."
                - Optional suffix: "join(?:_with)?"
            """
            query_id = uuid.uuid4()
            try:
                sanitized_query = validate_and_sanitize_regex(query)
            except ValidationError as e:
                error_message = f"Error validating regex query: {e}"
                logger.error(error_message)
                await ctx.error(error_message)
                raise ToolError(error_message) from e

            try:
                logger.info(
                    f"Performing search for query: {sanitized_query} ({query_id})",
                    query_id=query_id,
                    query=sanitized_query,
                    max_results=max_results,
                    output_format=output_format,
                    endpoint="/search",
                )
                search_df = search_api_docs(self.session, sanitized_query)
                search_df = search_df.order_by(
                    [fc.col("score").desc(), fc.col("type"), fc.col("name")]
                ).limit(max_results)

                # Collect API results
                async with self.semaphore:
                    api_rows = await asyncio.to_thread(lambda: search_df.to_pylist())

                # Format output
                total_results = len(api_rows)

                if output_format.lower() == "json":
                    payload = {
                        "query": query,
                        "sanitized_query": sanitized_query,
                        "total_results": total_results,
                        "results": [
                            {
                                "type": r.get("type"),
                                "name": r.get("name"),
                                "qualified_name": r.get("qualified_name"),
                                "score": r.get("score"),
                                "docstring": r.get("docstring"),
                            }
                            for r in api_rows
                        ],
                    }
                    logger.info(
                        "Search completed",
                        query_id=query_id,
                        total_results=total_results,
                        format="json",
                    )
                    return json.dumps(payload)

                # Default: markdown output
                output = f"# Search Results for: `{query}`\n\n"
                output += f"Found {total_results} matches\n\n"

                if total_results == 0:
                    output += "No results found. Try (regex required):\n"
                    output += "- Use alternation: '(extract|join|merge)'\n"
                    output += "- Use wildcards/anchors: '^fenic\\.api\\..*DataFrame', 'join.*semantic'\n"
                    logger.info(
                        "Search completed with 0 results",
                        query_id=query_id,
                        format="markdown",
                    )
                    return output

                output += "## 📖 API Documentation\n"
                current_type = None
                for row in api_rows:
                    if row["type"] != current_type:
                        current_type = row["type"]
                        output += f"\n### {current_type.capitalize()}s\n"

                    output += f"\n**`{row['name']}`** - `{row['qualified_name']}`\n"
                    doc_full = row.get("docstring")
                    if doc_full:
                        output += f"  {doc_full}\n"

                logger.info(
                    "Search completed",
                    query_id=query_id,
                    total_results=total_results,
                    format="markdown",
                )
                return output
            except Exception as e:
                error_message = "An exception occurred while performing search"
                logger.error(error_message, exc_info=e, query=query, query_id=query_id)
                await ctx.error(error_message)
                raise ToolError(error_message) from e

        @mcp.tool()
        async def get_project_overview() -> str:
            """Get a high-level overview of the Fenic project. This should be the starting point for figuring out where to look next for specific questions."""
            logger.info("Getting Project Overview", endpoint="/get_project_overview")
            return self.project_information.project_overview

        @mcp.tool()
        async def get_api_tree() -> str:
            """Get the API tree of the Fenic project."""
            logger.info("Getting API tree", endpoint="/get_api_tree")
            return self.project_information.api_tree

        @mcp.tool()
        async def get_entity(
            ctx: Context,
            qualified_name: str,
            output_format: str = "markdown",
        ) -> str:
            """Fetch detailed documentation for a single API entity by its qualified name.

            Args:
                qualified_name: Fully-qualified API path (e.g., "fenic.api.dataframe.dataframe.DataFrame.select")
                output_format: "markdown" or "json" (default: markdown)
            """
            try:
                logger.info(
                    "Fetching entity",
                    endpoint="/get_entity",
                    qualified_name=qualified_name,
                )
                df = get_entity_by_qualified_name(self.session, qualified_name)
                async with self.semaphore:
                    rows = await asyncio.to_thread(lambda: df.to_pylist())
                if not rows:
                    msg = f"No entity found for qualified name: {qualified_name}"
                    await ctx.error(msg)
                    raise ToolError(msg)

                row = rows[0]

                if output_format.lower() == "json":
                    payload = {
                        "type": row.get("type"),
                        "name": row.get("name"),
                        "qualified_name": row.get("qualified_name"),
                        "docstring": row.get("docstring"),
                        "annotation": row.get("annotation"),
                        "returns": row.get("returns"),
                    }
                    return json.dumps(payload)

                # Markdown output
                title = f"# `{row.get('qualified_name')}`\n\n"
                meta = f"**Type**: {row.get('type')}\n\n"
                doc = row.get("docstring") or ""
                annotation = row.get("annotation")
                returns = row.get("returns")
                parts = [title, meta]
                if doc:
                    parts.append(doc.strip() + "\n\n")
                if annotation:
                    parts.append(f"### Annotation\n{annotation}\n\n")
                if returns:
                    parts.append(f"### Returns\n{returns}\n\n")
                return "".join(parts)
            except Exception as e:
                error_message = "An exception occurred while fetching entity"
                logger.error(error_message, exc_info=e, qualified_name=qualified_name)
                await ctx.error(error_message)
                raise ToolError(error_message) from e

        @mcp.custom_route("/healthz", methods=["GET"])
        async def health_check(_: Request):
            """
            Basic health check that the server is running.
            """
            return JSONResponse({"status": "healthy"})

        return mcp


def main():
    """Main entry point for the MCP server."""
    faulthandler.enable(all_threads=True)
    try:
        logger.info("Starting MCP server...")
        log_fenic_version()
        session = create_session()
        logger.info("Fenic session initialized successfully")
        # Check if required tables exist
        required_tables = ["api_df", "hierarchy_df", "fenic_summary"]
        missing_tables = []
        for table in required_tables:
            if not session.catalog.does_table_exist(table):
                missing_tables.append(table)

        if missing_tables:
            logger.error(
                f"Missing required tables: {missing_tables}\n"
                "Please run 'python populate_tables.py' to set up the documentation database.\n"
                "This will extract and index the Fenic API documentation."
            )
            import sys

            sys.exit(1)

        logger.info("All required tables found, starting MCP server on 0.0.0.0:8000")

        # Start the server - this should block and keep running
        try:
            mcp = FenicMCP().generate_server()
            mcp.run(
                transport="http",
                host="0.0.0.0",  # nosec B104
                stateless_http=True,
                path="/",
            )
        except KeyboardInterrupt:
            logger.info("Server stopped by user")
        except Exception as server_error:
            logger.error(f"Server error: {server_error}")
            import traceback

            logger.error(f"Server traceback: {traceback.format_exc()}")
            raise

    except Exception as e:
        logger.error(f"Failed to start MCP server: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    main()
