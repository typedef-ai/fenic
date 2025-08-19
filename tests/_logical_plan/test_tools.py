from typing import Literal

from fenic import IntegerType, Session, SessionConfig, StringType, col
from fenic.api.functions import param, semantic
from fenic.core._logical_plan.tools import ToolParam
from fenic.core.mcp.generator import MCPGenerator


def test_tool_creation(local_session_config: SessionConfig) -> None:
    """Create a tool from the system table.
    Raises:
    """
    local_session = Session.get_or_create(local_session_config)
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

    age_filter = data.filter((col("age") >= param("min_age", IntegerType)) & (col("age") <= param("max_age", IntegerType)))
    department_filter = data.filter(col("department") == param("department", StringType))
    status_filter = data.filter(col("status") == param("status", StringType))
    bio_semantic_filter = data.filter(semantic.predicate(
        "Based on the user's bio, would they be interested in {{activity}}?"
        "User Information:"
        "{{bio}}",
        bio=col("bio"),
        activity=param("activity", StringType),
    ))
    local_session.catalog.create_tool(
        "Age Filter",
        "Filter users by age",
        age_filter,
        tool_params=[
            ToolParam(name="min_age", description="Minimum age",
                      type=int, required=False, default=18),
            ToolParam(name="max_age", description="Maximum age",
                      type=int, required=False, default=65),
        ]
    )
    local_session.catalog.create_tool(
        "Department Filter",
        "Filter users by department",
        department_filter,
        params=[
            ToolParam(name="department", type=str, description="Department equals")
        ]
    )

    local_session.catalog.create_tool(
        "Status Filter",
        "Filter users by status",
        status_filter,
        params=[ToolParam(
            name="status",
            description="Status equals",
            type=Literal["active", "inactive", "pending"],
            required=True
        )],
    )

    local_session.catalog.create_tool("Bio Semantic Filter",
                                      "Describe one or more activities in natural language, This will return users that are interested in that activity",
                                      bio_semantic_filter,
                                      params=[ToolParam(
                                          name="activity",
                                          description="One or more activities in natural language",
                                          type=str
                                      )])

    tools = local_session.catalog.list_tools()
    assert len(tools) == 4
    mcp_generator = MCPGenerator(local_session, tools, "Test views")
    mcp_generator.run()
