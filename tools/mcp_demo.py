"""Demo: Generate and run an MCP server from Fenic parameterized views.

This script:
- Creates a sample `users` view
- Defines a parameterized view with a few filters
- Generates an MCP server using fenic.core.mcp.generator
- Runs the server (Ctrl+C to stop)

Prereqs:
- Install MCP extra: `pip install "fenic[mcp]"` (or `pip install fastmcp`)
- Run with: `uv run python tools/mcp_demo.py`
"""

import textwrap
import fenic as fc
from fenic import IntegerType, OpenAILanguageModel, SemanticConfig, StringType
from fenic.api.functions import tool_param
from fenic.core._logical_plan.tools import ToolParam
from fenic.core.mcp.generator import MCPGenerator


def main() -> None:
    fc.configure_logging()
    local_session = fc.Session.get_or_create(fc.SessionConfig(
        app_name="mcp_demo",
        semantic=SemanticConfig(
            language_models={
                "nano": OpenAILanguageModel(
                    model_name="gpt-4.1-nano",
                    rpm=2500,
                    tpm=1_000_000
                )
            }
        )
    ))
    data = local_session.create_dataframe(
        {
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
            "age": [25, 30, 35, 28, 32],
            "department": ["engineering", "marketing", "engineering", "sales", "engineering"],
            "status": ["active", "inactive", "pending", "active", "active"],
            "bio": [
                "Alice is a software engineer at Google. She enjoys hiking and reading.",
                "Bob is a marketing manager at Apple. He enjoys playing tennis and cooking.",
                "Charlie is a software engineer at Facebook. He enjoys playing basketball and reading.",
                "Diana is a sales representative at Amazon. She enjoys playing video games and listening to podcasts.",
                "Eve is a software engineer at Microsoft. She enjoys hiking and reading.",
            ],
        }
    )
    # Optional min/max bounds via coalesce: if param is absent (default None), predicate becomes True
    # TODO(bc): can this be generated dynamically from the view? We have the schema and can use the types to infer which operators to support and prefixes
    optional_min = fc.coalesce(fc.col("age") >= tool_param("min_age", IntegerType), fc.lit(True))
    optional_max = fc.coalesce(fc.col("age") <= tool_param("max_age", IntegerType), fc.lit(True))
    optional_department = fc.coalesce(fc.col("department") == tool_param("department", StringType), fc.lit(True))
    optional_status = fc.coalesce(fc.col("status") == tool_param("status", StringType), fc.lit(True))
    optional_bio_contains = fc.coalesce(fc.col("bio").contains(tool_param("bio_contains", StringType)), fc.lit(True))
    core_filter = data.filter(
        optional_min & optional_max & optional_department & optional_status & optional_bio_contains)


    # TODO(bc): this too, could likely be automatically generated from a fenic dataframe schema,
    # just like we know the prefixes and suffixes for the names, we can generate descriptions that are
    # this simple, and the model is easily able to interpret what they mean.
    local_session.catalog.create_tool(
        "users_filter",
        "Filter users by all available columns (age, department, status, bio)",
        core_filter,
        tool_params=[
            ToolParam(name="min_age", description="Minimum age", has_default=True, default_value=None),
            ToolParam(name="max_age", description="Maximum age", has_default=True, default_value=None),
            ToolParam(name="department", description="Department equals", has_default=True, default_value=None),
            ToolParam(name="status", description="Status equals", has_default=True, default_value=None),
            ToolParam(name="bio_contains", description="Bio contains", has_default=True, default_value=None)
        ]
    )

    # TODO(bc): now what would be really cool is automatically generating this or something else like it (perhaps using embeddings for semantic similarity search?)
    # as then you'd have ready to go semantic search on any dataframe. if you wanted to do this in general, it would be very useful to have some sort of `nest(struct_name)`
    # dataframe function to return a dataframe with all of the fields 
    freestyle_semantic_bio = data.filter(fc.semantic.predicate(
        textwrap.dedent("""\
        The following is a search query in a database of user information.
        QUERY: {{query}}
        Determine if the user data shown below matches the query.
        USER DATA:
        NAME: {{name}}
        AGE: {{age}}
        DEPARTMENT: {{department}}
        STATUS: {{status}}
        BIO: {{bio}}
        """),
        query=tool_param("query", StringType),
        bio=fc.col("bio"),
        name=fc.col("name"),
        age=fc.col("age"),
        department=fc.col("department"),
        status=fc.col("status"),
    ))

    local_session.catalog.create_tool("semantic_query",
                                      "Describe in natural language the users you are looking for, by name, age, department, status, or semantically similar information from their provided bio.",
                                      freestyle_semantic_bio,
                                      tool_params=[
                                          ToolParam(
                                              name="query",
                                              description="Natural language query",
                                          )])

    tools = local_session.catalog.list_tools()
    mcp_generator = MCPGenerator(local_session, tools, "FenicMCP")
    mcp_generator.run(stateless_http=True)


if __name__ == "__main__":
    main()
