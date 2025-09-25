"""MCP configuration for evaluation - modified to not run the server."""

import fenic as fc
from fenic.core.error import TableNotFoundError
from fenic.core.mcp._server import FenicMCPServer


def setup_mcp_for_evaluation(local_session: fc.Session) -> FenicMCPServer:
    """Set up MCP server configuration for evaluation without running it."""
    try:
        local_session.catalog.describe_table("conversations")
    except TableNotFoundError:
        conversations_df = local_session.read.parquet("s3://typedef-assets/demo/mcp/clean_conversations.parquet")
        conversations_df.write.save_as_table(table_name="conversations", mode="overwrite")
        local_session.catalog.set_table_description("conversations", "Raw conversations between users on a dating app.")

    try:
        local_session.catalog.describe_table("enriched_profiles")
    except TableNotFoundError:
        enriched_profiles_df = local_session.read.parquet("s3://typedef-assets/demo/mcp/enriched_profiles.parquet").select(
        "profile_id", "full_name", "age", "gender", "location", "looking_for", "pets", "occupation", "hobbies",
        "ideal_partner", "bio")
        enriched_profiles_df.write.save_as_table(
            table_name="enriched_profiles",
            mode="overwrite",
        )
        local_session.catalog.set_table_description("enriched_profiles", "Profiles of users in the dating app, containing demographic and self-written biographic information.")
    try:
        local_session.catalog.describe_table("moderation_report")
    except TableNotFoundError:
        moderation_report_df = local_session.read.parquet("s3://typedef-assets/demo/mcp/moderation_report.parquet").select(
            "conversation_id",
            "user1_id",
            "user2_id",
            "conversation_summary",
            "primary_concern",
            "secondary_concerns",
            "behavior_severity",
            "escalation_observed",
            "recommended_action",
            "primary_bad_actor",
            "explanation"
        )
        moderation_report_df.write.save_as_table(
            table_name="moderation_report",
            mode="overwrite",
        )
        local_session.catalog.set_table_description("moderation_report", "Curated report with moderation analysis of the dating app conversations; includes descriptions of bad-actor behaviors/explanations.")

    return fc.create_mcp_server(
        local_session,
        "Dating App Moderation Demo",
        system_tools=fc.SystemToolConfig(
            table_names=["conversations", "enriched_profiles", "moderation_report"],
            tool_namespace="Dating App",
            max_result_rows=100
        )
    )
