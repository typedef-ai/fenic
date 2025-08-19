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
from fenic import OpenAILanguageModel, SemanticConfig
from fenic.core import (
    ParamaterizedQuery,
    ViewFilters,
    enum_param,
    string_param,
)
from fenic.core._logical_plan.parameterized_views import FilterContext, QueryParameter
from fenic.core.mcp.generator import MCPGenerator


def _seed_sample_users(session: fc.Session) -> None:
    """Create a small `users` view for demo purposes (idempotent)."""
    data = session.create_dataframe(
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
    if session.catalog.does_view_exist("users"):
        session.catalog.drop_view("users")
    data.write.save_as_view("users")


def semantic_filter(df: fc.DataFrame, value: str, ctx: FilterContext) -> fc.DataFrame:
    template = textwrap.dedent(f"""
                Based on the user's bio, is the user interested in {value}?
                """
                """
                BIO: {{ bio }}
                """
                )
    return df.filter(fc.semantic.predicate(template, bio=fc.col(ctx.column)))


def build_user_search_view() -> ParamaterizedQuery:
    """Define a parameterized view over the `users` view."""
    return ParamaterizedQuery(
        name="search_users",
        description="Search users by department, status, or age range.",
        base_view="users",
        parameters={
            # string equality match on department
            "department": string_param("department", "Department equals", ViewFilters.equals, required=False),
            # enum list: allows multiple statuses
            "status": enum_param("status", "Filter status", ["active", "inactive", "pending"], required=False),
            # simple range on age (e.g., "25-32", ">=30")
            "min_age": QueryParameter("min_age", int, "Minimum age", ViewFilters.greater_equal, required=False),
            "max_age": QueryParameter("max_age", int, "Maximum age", ViewFilters.less_equal, required=False),
            "interest": QueryParameter("interest", str,
                                      "Based on their biography, would the user enjoy the provided interest?",
                                      semantic_filter, required=False),
        },
        parameter_mapping={
            "min_age": "age",
            "max_age": "age",
            "interest": "bio"
        },
    )


def main() -> None:
    fc.configure_logging()
    session = fc.Session.get_or_create(fc.SessionConfig(
        app_name="mcp_demo",
        semantic=SemanticConfig(
            language_models={
                "nano" : OpenAILanguageModel(
                    model_name="gpt-4.1-nano",
                    rpm=2500,
                    tpm=1_000_000
                )
            }
        )
    ))

    # Seed demo data
    _seed_sample_users(session)

    # Build parameterized views
    user_view = build_user_search_view()

    # Generate MCP server
    gen = MCPGenerator(session, server_name="Fenic Demo Views")
    gen.register_view(user_view)

    try:
        mcp = gen.generate_server()
    except ImportError as e:
        print(str(e))
        print("\nInstall MCP extra: pip install \"fenic[mcp]\" (or install fastmcp)")
        return

    print("\nMCP server is starting. Available tools:")
    print("- search_users(params): Filter users with department, status, age_range, limit")
    print("\nExamples:")
    print("- search_users(params={\"department\": \"engineering\", \"limit\": 3})")
    print("- search_users(params={\"status\": [\"active\"]})")
    print("- search_users(params={\"age_range\": \"25-30\"})")

    # Start MCP server (Ctrl+C to stop)
    mcp.run(transport="http", stateless_http=True)


if __name__ == "__main__":
    main()
