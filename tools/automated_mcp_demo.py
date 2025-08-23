import asyncio
from typing import Union

import fenic as fc
from fenic import OpenAILanguageModel, SemanticConfig
from fenic.api.tools import (
    auto_generate_core_tools,
)
from fenic.core._logical_plan.tools import DynamicTool, ResolvedTool
from fenic.core.mcp.generator import MCPGenerator


async def main():
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

    conversations_df = local_session.read.parquet("./data/clean_conversations.parquet")
    enriched_profiles_df = local_session.read.parquet("./data/enriched_profiles.parquet").select(
        "profile_id", "full_name", "age", "gender", "location", "looking_for", "pets", "occupation", "hobbies",
        "ideal_partner", "bio")
    moderation_report_df = local_session.read.parquet("./data/moderation_report.parquet")

    tools: list[Union[DynamicTool, ResolvedTool]] = []
    tools.extend(auto_generate_core_tools(
        conversations_df,
        "conversations",
        "Raw Conversations taking place between users on a dating app.",
        sql_max_rows=25
    ))
    tools.extend(auto_generate_core_tools(
        enriched_profiles_df,
        "profiles",
        "Profiles of users in the dating app, contains demographic and self-written biographic information about each user."))
    tools.extend(auto_generate_core_tools(
        moderation_report_df,
        "moderation_report",
        "Curated Report detailing findings by the moderation team on analysis of the `Dating App Conversations`. Contains descriptions of bad actor behavior/explanations"
    ))
    mcp_generator = MCPGenerator(local_session._session_state, tools, "Content Moderation MCP")
    await mcp_generator.run(stateless_http=True)

    print("Made it to here!")


if __name__ == "__main__":
    asyncio.run(main())
