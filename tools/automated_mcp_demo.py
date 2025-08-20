import asyncio

import fenic as fc
from fenic import OpenAILanguageModel, SemanticConfig
from fenic.api.tools import auto_generate_filter_tool, auto_generate_semantic_tool
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

    # Generate tools for the conversations dataset
    semantic_tool = auto_generate_semantic_tool(conversations_df, "conversations_semantic_filter",
                                                "Semantic query tool for the raw conversations dataset, this contains the raw conversations between users on the dating app.")
    filter_tool = auto_generate_filter_tool(conversations_df, "conversations_filter",
                                            "Filter conversations tool for the conversations dataset, this contains the raw conversations between users on the dating app.")
    tools = [semantic_tool, filter_tool]

    # Generate tools for the enriched_profiles dataset
    semantic_tool = auto_generate_semantic_tool(enriched_profiles_df, "user_profiles_semantic_filter",
                                                "Semantic query tool for the Profiles dataset. This contains the profile information for the dating app users.")
    filter_tool = auto_generate_filter_tool(enriched_profiles_df, "user_profiles_filter",
                                            "Filter enriched_profiles tool for the enriched_profiles dataset. This contains the profile information for the dating app users.")
    tools.append(semantic_tool)
    tools.append(filter_tool)

    # Generate tools for the moderation_report dataset
    semantic_tool = auto_generate_semantic_tool(moderation_report_df, "moderation_report_semantic_filter",
                                                "Semantic query tool for the moderation_report dataset. This contains the moderation report for the dating app users.")
    filter_tool = auto_generate_filter_tool(moderation_report_df, "moderation_report_filter",
                                            "Filter moderation_report tool for the moderation_report dataset. This contains the moderation report for the dating app users.")
    tools.append(semantic_tool)
    tools.append(filter_tool)
    mcp_generator = MCPGenerator(local_session, tools, "Content Moderation MCP")
    await mcp_generator.run(stateless_http=True)

    print("Made it to here!")


if __name__ == "__main__":
    asyncio.run(main())
