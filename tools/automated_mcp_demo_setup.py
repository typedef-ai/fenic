
import fenic as fc
from fenic import OpenAILanguageModel, SemanticConfig


def main():
    fc.configure_logging()
    local_session = fc.Session.get_or_create(fc.SessionConfig(
        app_name="automated_mcp_demo",
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

    conversations_df = local_session.read.parquet("s3://typedef-assets/demo/mcp/clean_conversations.parquet")
    enriched_profiles_df = local_session.read.parquet("s3://typedef-assets/demo/mcp/enriched_profiles.parquet").select(
        "profile_id", "full_name", "age", "gender", "location", "looking_for", "pets", "occupation", "hobbies",
        "ideal_partner", "bio")
    moderation_report_df = local_session.read.parquet("s3://typedef-assets/demo/mcp/moderation_report.parquet")
    conversations_df.write.save_as_table(table_name="conversations", mode="overwrite")
    local_session.catalog.set_table_description("conversations", "Raw conversations between users on a dating app.")
    enriched_profiles_df.write.save_as_table(
        table_name="enriched_profiles",
        mode="overwrite",
    )
    local_session.catalog.set_table_description(
        "enriched_profiles",
        description="Profiles of users in the dating app, containing demographic and self-written biographic information."
    )
    moderation_report_df.write.save_as_table(
        table_name="moderation_report",
        mode="overwrite",
    )
    local_session.catalog.set_table_description(
        "moderation_report",
        description="Curated report with moderation analysis of the dating app conversations; includes descriptions of bad-actor behaviors/explanations.",
    )
    mcp_generator = fc.create_mcp_server(
        local_session,
        "Dating App Moderation Demo",
        system_tools=fc.SystemToolConfig(
            table_names=["conversations", "enriched_profiles", "moderation_report"],
            tool_namespace="Dating App Moderation",
            max_result_rows=100
        )
    )
    fc.run_mcp_server_sync(mcp_generator, port=8001)


if __name__ == "__main__":
    main()
