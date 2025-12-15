"""Memory — Blocks & Episodes: Profile + timeline.

This example demonstrates how to maintain a profile block alongside a recent
event timeline and return scoped snapshots.

Tools exposed: `get_user_context` (profile + last N events), plus system tools
"""

from datetime import datetime
from typing import Optional

import fenic as fc
from fenic import SemanticConfig, OpenAILanguageModel
from fenic.api.mcp._tool_generation_utils import auto_generate_system_tools_from_tables
from fenic.core.mcp._server import FenicMCPServer
from fenic.core.mcp.types import SystemTool
from pydantic import BaseModel, Field


class MemoryBlock(BaseModel):
    block_name: str = Field(description="Name of the memory block")
    content: str = Field(description="Content stored in the memory block")
    last_updated: str = Field(description="Timestamp of last update")


class AccountEvent(BaseModel):
    event_type: str = Field(description="Type of account event")
    amount: Optional[float] = Field(default=None, description="Amount involved in the event")
    status: Optional[str] = Field(default=None, description="Status of the event")
    description: Optional[str] = Field(default=None, description="Description of the event")


def main():
    session = fc.Session.get_or_create(
        fc.SessionConfig(
            app_name="mem_blocks",
            semantic=SemanticConfig(
                language_models={
                    "gpt4": OpenAILanguageModel(
                        model_name="gpt-4.1-nano", rpm=100, tpm=100_000
                    )
                }
            ),
        )
    )

    blocks = session.create_dataframe(
        [
            {
                "user_id": "user123",
                "block_name": "profile",
                "content": "Name: Taylor; Dept: Finance",
                "last_updated": datetime.now().isoformat(),
            }
        ]
    )
    blocks.write.save_as_table("memory_blocks", mode="overwrite")

    ev = session.create_dataframe(
        [
            {
                "user_id": "user123",
                "event": "Failed transaction of $99.99",
                "timestamp": "2025-01-01",
            },
            {
                "user_id": "user123",
                "event": "Card expired",
                "timestamp": "2025-01-05",
            },
            {
                "user_id": "user123",
                "event": "Account suspended",
                "timestamp": "2025-01-06",
            },
        ]
    )
    timeline = ev.select(
        fc.col("user_id"),
        fc.col("timestamp"),
        fc.semantic.extract(fc.col("event"), AccountEvent).alias("data"),
    ).unnest("data")
    timeline.write.save_as_table("account_timeline", mode="overwrite")
    session.catalog.set_table_description(
        "memory_blocks", "User memory blocks containing profile and other persistent data"
    )
    session.catalog.set_table_description(
        "account_timeline", "Timeline of account events per user"
    )

    async def get_user_context(user_id: str, last_n: int = 3):
        profile = (
            session.table("memory_blocks")
            .filter(
                (fc.col("user_id") == fc.lit(user_id))
                & (fc.col("block_name") == fc.lit("profile"))
            )
            .select("block_name", "content", "last_updated")
        )
        recent = (
            session.table("account_timeline")
            .filter(fc.col("user_id") == fc.lit(user_id))
            .sort(fc.col("timestamp").desc())
            .limit(last_n)
            .select("timestamp", "event_type", "status", "amount", "description")
        )
        return {"profile": profile._plan, "recent_events": recent._plan}

    generated_system_tools = auto_generate_system_tools_from_tables(
        ["memory_blocks", "account_timeline"],
        session,
        tool_namespace="memctx",
        max_result_limit=100,
    )

    server = FenicMCPServer(
        session._session_state,
        user_defined_tools=[],
        system_tools=[
            SystemTool(
                name="get_user_context",
                description="Profile + recent events",
                max_result_limit=100,
                func=get_user_context,
            ),
            *generated_system_tools,
        ],
        server_name="Memory (Blocks & Episodes)",
    )

    print("Memory Blocks & Episodes example completed.")
    print("Tables created: memory_blocks, account_timeline")
    print("MCP server configured with get_user_context tool and system tools.")
    return server


if __name__ == "__main__":
    main()
