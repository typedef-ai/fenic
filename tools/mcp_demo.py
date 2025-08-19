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

import fenic as fc
from fenic import IntegerType, OpenAILanguageModel, SemanticConfig, StringType, lit
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
            "status": ["active", "inactive", "active", "pending", "active"],
            "bio": [
                "Alice is a software engineer at Google. She enjoys hiking and reading.",
                "Bob is a marketing manager at Apple. He enjoys playing tennis and cooking.",
                "Charlie is a software engineer at Facebook. He enjoys playing basketball and reading.",
                "Diana is a sales representative at Amazon. She enjoys playing video games and listening to podcasts.",
                "Eve is a software engineer at Microsoft. She enjoys hiking and reading.",
            ],
        }
    )
    age_filter = data.filter(
        fc.when(
            tool_param("min_age", IntegerType).is_not_null(),
            (fc.col("age") >= tool_param("min_age", IntegerType))
        ).otherwise(lit(True))).filter(
        fc.when(
            tool_param("max_age", IntegerType).is_not_null(),
            (fc.col("age") <= tool_param("max_age", IntegerType))
        ).otherwise(lit(True))
    )
    department_filter = data.filter(fc.col("department") == tool_param("department", StringType))
    status_filter = data.filter(fc.col("status") == tool_param("status", StringType))
    bio_semantic_filter = data.filter(fc.semantic.predicate(
        "Based on the user's bio, would they be interested in {{activity}}?"
        "User Information:"
        "{{bio}}",
        bio=fc.col("bio"),
        activity=tool_param("activity", StringType),
    ))
    local_session.catalog.create_tool(
        "age_filter",
        "Filter users by age",
        age_filter,
        tool_params=[
            ToolParam(name="min_age", description="Minimum age", has_default=True, default_value=None),
            ToolParam(name="max_age", description="Maximum age", has_default=True, default_value=None),
        ]
    )
    local_session.catalog.create_tool(
        "department_filter",
        "Filter users by department",
        department_filter,
        tool_params=[
            ToolParam(name="department", description="Department equals")
        ]
    )

    local_session.catalog.create_tool(
        "status_filter",
        "Filter users by status",
        status_filter,
        tool_params=[ToolParam(
            name="status",
            description="Status equals",
            allowed_values=["active", "inactive", "pending"],
        )],
    )

    local_session.catalog.create_tool("bio_semantic_filter",
                                      "Describe one or more activities in natural language, This will return users that are interested in that activity",
                                      bio_semantic_filter,
                                      tool_params=[ToolParam(
                                          name="activity",
                                          description="One or more activities in natural language",
                                      )])

    tools = local_session.catalog.list_tools()
    mcp_generator = MCPGenerator(local_session, tools, "FenicMCP")
    mcp_generator.run(stateless_http=True)


if __name__ == "__main__":
    main()
