import fenic as fc
from fenic import OpenAILanguageModel, SemanticConfig
from fenic.api.mcp import create_mcp_server, run_mcp_server_sync, ToolGenerationConfig
from fenic.api.tools import DatasetSpec


def main():
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

    dataset_specs = [
        DatasetSpec(
            name="conversations",
            description="Raw conversations between users on a dating app.",
            df=conversations_df,
        ),
        DatasetSpec(
            name="profiles",
            description=(
                "Profiles of users in the dating app, containing demographic and self-written biographic information."
            ),
            df=enriched_profiles_df,
        ),
        DatasetSpec(
            name="moderation_report",
            description=(
                "Curated report with moderation analysis of the dating app conversations; includes descriptions of bad-actor behaviors/explanations."
            ),
            df=moderation_report_df,
        ),
    ]
    mcp_generator = create_mcp_server(
        local_session,
        "Dating App Moderation Demo",
        automated_tool_generation=ToolGenerationConfig(
            datasets=dataset_specs,
            tool_group_name="Dating App Moderation",
            sql_max_rows=50
        )
    )
    run_mcp_server_sync(mcp_generator)


if __name__ == "__main__":
    main()
