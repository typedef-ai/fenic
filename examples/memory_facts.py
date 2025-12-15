"""Memory — Facts: Extract, embed, and semantic recall.

This example demonstrates how to add structure to free-form chats by extracting
them into typed facts and recalling them semantically so agents don't carry
giant histories.

Tools exposed: `mem_recall` (semantic query), plus system tools over `preferences`
"""

from pydantic import BaseModel, Field

import fenic as fc
from fenic import SemanticConfig, OpenAILanguageModel, OpenAIEmbeddingModel
from fenic.api.mcp._tool_generation_utils import auto_generate_system_tools_from_tables
from fenic.core.mcp._server import FenicMCPServer
from fenic.core.mcp.types import SystemTool


class Preference(BaseModel):
    category: str = Field(description="The category of the preference")
    value: str = Field(description="The value of the preference")


def main():
    session = fc.Session.get_or_create(
        fc.SessionConfig(
            app_name="mem_facts",
            semantic=SemanticConfig(
                language_models={
                    "gpt4": OpenAILanguageModel(
                        model_name="gpt-4.1-nano", rpm=100, tpm=100_000
                    )
                },
                embedding_models={
                    "embed": OpenAIEmbeddingModel(
                        model_name="text-embedding-3-small", rpm=100, tpm=100_000
                    )
                },
                default_embedding_model="embed",
            ),
        )
    )

    msgs = session.create_dataframe(
        [
            {"user_id": "user123", "message": "I'm vegetarian and allergic to nuts."},
            {"user_id": "user123", "message": "I prefer morning meetings."},
        ]
    )

    prefs = msgs.select(
        fc.col("user_id"),
        fc.semantic.extract(fc.col("message"), Preference).alias("pref"),
        fc.semantic.embed(fc.col("message")).alias("vec"),
    ).unnest("pref")
    prefs.write.save_as_table("preferences", mode="overwrite")
    session.catalog.set_table_description(
        "preferences", "User preferences extracted from messages with embeddings"
    )

    async def mem_recall(user_id: str, query: str, k: int = 3):
        user_prefs = session.table("preferences").filter(
            fc.col("user_id") == fc.lit(user_id)
        )
        q = session.create_dataframe([{"q": query}])
        res = q.semantic.sim_join(
            user_prefs,
            left_on=fc.semantic.embed(fc.col("q")),
            right_on=fc.col("vec"),
            k=k,
            similarity_score_column="relevance",
        ).select("category", "value", "relevance")
        return res._plan

    generated_system_tools = auto_generate_system_tools_from_tables(
        ["preferences"], session, tool_namespace="mem", max_result_limit=100
    )

    server = FenicMCPServer(
        session._session_state,
        user_defined_tools=[],
        system_tools=[
            SystemTool(
                name="mem_recall",
                description="Semantic memory recall",
                max_result_limit=100,
                func=mem_recall,
            ),
            *generated_system_tools,
        ],
        server_name="Memory (Facts)",
    )

    print("Memory Facts example completed. Preferences table created.")
    print("MCP server configured with mem_recall tool and system tools.")
    return server


if __name__ == "__main__":
    main()
