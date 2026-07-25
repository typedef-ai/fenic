"""Fenic-native MCP server for the Fenic documentation catalog."""

import logging

from starlette.requests import Request
from starlette.responses import JSONResponse

import fenic as fc
from fenic_mcp.server.utils.session import create_session, log_fenic_version

logger = logging.getLogger(__name__)

_ELEMENT_TYPES = ["all", "class", "function", "method", "module", "attribute"]
_TOOL_NAMES = {
    "search_fenic_api",
    "get_entity",
    "get_entities",
    "get_project_overview",
    "get_api_tree",
}


def register_docs_tools(session: fc.Session) -> None:
    """Register the documentation queries as parameterized Fenic tools."""
    for tool_name in _TOOL_NAMES:
        session.catalog.drop_tool(tool_name, ignore_if_not_exists=True)

    query = fc.tool_param("query", fc.StringType)
    element_type = fc.tool_param("element_type", fc.StringType)
    normalized_query = fc.text.lower(fc.text.trim(query))
    query_tokens = fc.arr.distinct(
        fc.text.split(
            fc.text.trim(
                fc.text.regexp_replace(normalized_query, r"[^a-z0-9]+", " ")
            ),
            r"\s+",
        )
    )
    docs = session.table("api_df").filter(
        fc.col("is_public") & ~fc.col("qualified_name").rlike(r"(^|\.)_")
    )

    search_tokens = fc.arr.distinct(
        fc.text.split(
            fc.text.trim(
                fc.text.regexp_replace(
                    fc.text.lower(
                        fc.text.concat_ws(
                            " ",
                            "name",
                            "qualified_name",
                            fc.coalesce(fc.col("docstring"), fc.lit("")),
                            fc.coalesce(fc.col("annotation"), fc.lit("")),
                            fc.coalesce(fc.col("returns"), fc.lit("")),
                        )
                    ),
                    r"[^a-z0-9]+",
                    " ",
                )
            ),
            r"\s+",
        )
    )
    matched_keyword_count = fc.arr.size(fc.arr.intersect(search_tokens, query_tokens))
    exact_name_match = fc.text.lower(fc.col("name")).contains(normalized_query)
    exact_path_match = fc.text.lower(fc.col("qualified_name")).contains(
        normalized_query
    )

    search_query = (
        docs.filter(
            ((element_type == fc.lit("all")) | (fc.col("type") == element_type))
            & (fc.arr.size(query_tokens) > fc.lit(0))
            & (matched_keyword_count > fc.lit(0))
        )
        .select(
            "type",
            "name",
            "qualified_name",
            matched_keyword_count.alias("matched_keywords"),
            fc.arr.size(query_tokens).alias("query_keywords"),
            (
                matched_keyword_count
                + fc.when(exact_name_match, fc.lit(12)).otherwise(fc.lit(0))
                + fc.when(exact_path_match, fc.lit(6)).otherwise(fc.lit(0))
            ).alias("score"),
        )
        .order_by(
            [
                fc.col("score").desc(),
                fc.col("matched_keywords").desc(),
                "type",
                "name",
            ]
        )
    )
    session.catalog.create_tool(
        tool_name="search_fenic_api",
        tool_description=(
            "Search Fenic's public Python API with one or more natural-language "
            "keywords. Punctuation and whitespace separate literal keywords; records "
            "matching more keywords rank higher. Returns compact discovery results "
            "without full docstrings. Call get_entity with a returned qualified_name "
            "for complete documentation."
        ),
        tool_query=search_query,
        tool_params=[
            fc.ToolParam(
                name="query",
                description=(
                    "One or more concepts or API names, such as "
                    "'recursive chunk word count' or 'semantic sim join'."
                ),
            ),
            fc.ToolParam(
                name="element_type",
                description="Optionally restrict results to one API element type.",
                allowed_values=_ELEMENT_TYPES,
                default_value="all",
            ),
        ],
        result_limit=50,
        ignore_if_exists=False,
    )

    entity_query = docs.filter(
        fc.col("qualified_name")
        == fc.tool_param("qualified_name", fc.StringType)
    ).select(
        "type",
        "name",
        "qualified_name",
        "docstring",
        "annotation",
        "returns",
        "parameters",
        "parent_class",
    )
    session.catalog.create_tool(
        tool_name="get_entity",
        tool_description=(
            "Fetch complete documentation for one public Fenic API entity. Pass a "
            "qualified_name returned by search_fenic_api."
        ),
        tool_query=entity_query,
        tool_params=[
            fc.ToolParam(
                name="qualified_name",
                description="Exact fully-qualified Python API name.",
            )
        ],
        result_limit=1,
        ignore_if_exists=False,
    )

    entities_query = docs.filter(
        fc.col("qualified_name").is_in(
            fc.tool_param(
                "qualified_names",
                fc.ArrayType(element_type=fc.StringType),
            )
        )
    ).select(
        "type",
        "name",
        "qualified_name",
        "docstring",
        "annotation",
        "returns",
        "parameters",
        "parent_class",
    )
    session.catalog.create_tool(
        tool_name="get_entities",
        tool_description=(
            "Fetch complete documentation for several public Fenic API entities in "
            "one call. Pass qualified_name values returned by search_fenic_api."
        ),
        tool_query=entities_query,
        tool_params=[
            fc.ToolParam(
                name="qualified_names",
                description="Exact fully-qualified Python API names (maximum 10).",
            )
        ],
        result_limit=10,
        ignore_if_exists=False,
    )

    context = session.table("fenic_project_context")
    session.catalog.create_tool(
        tool_name="get_project_overview",
        tool_description=(
            "Get a concise overview of Fenic and its public API tree. Use this when "
            "first orienting to the library."
        ),
        tool_query=context.select("project_overview"),
        tool_params=[],
        result_limit=1,
        ignore_if_exists=False,
    )
    session.catalog.create_tool(
        tool_name="get_api_tree",
        tool_description=(
            "Get a precomputed, compact tree of Fenic's public modules, classes, "
            "functions, and methods."
        ),
        tool_query=context.select("api_tree"),
        tool_params=[],
        result_limit=1,
        ignore_if_exists=False,
    )


def create_native_server(
    session: fc.Session | None = None, *, refresh_tools: bool = False
):
    """Create the Fenic documentation MCP server."""
    session = session or create_session()
    missing_tables = [
        table_name
        for table_name in ("api_df", "fenic_project_context")
        if not session.catalog.does_table_exist(table_name)
    ]
    if missing_tables:
        raise RuntimeError(
            f"Missing required documentation tables: {', '.join(missing_tables)}"
        )

    if refresh_tools:
        register_docs_tools(session)
    installed_tool_names = {tool.name for tool in session.catalog.list_tools()}
    missing_tools = _TOOL_NAMES - installed_tool_names
    if missing_tools:
        raise RuntimeError(
            "Missing required documentation tools: "
            f"{', '.join(sorted(missing_tools))}. Run data preparation first."
        )
    tools = [
        tool for tool in session.catalog.list_tools() if tool.name in _TOOL_NAMES
    ]
    server = fc.create_mcp_server(
        session=session,
        server_name="fenic-docs",
        user_defined_tools=tools,
        concurrency_limit=16,
    )

    @server.mcp.custom_route("/healthz", methods=["GET"])
    async def health_check(_: Request) -> JSONResponse:
        return JSONResponse({"status": "healthy"})

    return server


def main() -> None:
    """Run the native server locally over streamable HTTP."""
    log_fenic_version()
    server = create_native_server(refresh_tools=True)
    server.run(
        transport="http",
        host="127.0.0.1",
        port=8000,
        path="/mcp",
        stateless_http=True,
    )


if __name__ == "__main__":
    main()
