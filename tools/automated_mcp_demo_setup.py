from textwrap import dedent
from typing import Annotated

import fenic as fc
from fenic import OpenAILanguageModel, SemanticConfig
from fenic.api.mcp.tool_generation import fenic_tool


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

    @fenic_tool(
        tool_name="semantic_profile_search",
        tool_description="Search User Profile information using a natural language query.",
        default_table_format="structured",
        max_result_limit=100,
    )
    def semantic_profile_search(
        query: Annotated[str, "Natural Language Query"]
    ):
        predicate = f"""\
            Is the following query true for the provided row of data?

            QUERY: {query}
            DATA:
                Profile Id: {{{{profile_id}}}}
                Full Name: {{{{full_name}}}}
                Age: {{{{age}}}}
                Gender: {{{{gender}}}}
                Location: {{{{location}}}}
                Looking for: {{{{looking_for}}}}
                Pets: {{{{pets}}}}
                Occupation: {{{{occupation}}}}
                Hobbies: {{{{hobbies}}}}
                Ideal Partner: {{{{ideal_partner}}}}
                Bio: {{{{bio}}}}
        """

        return enriched_profiles_df.filter(
            fc.semantic.predicate(
                predicate,
                profile_id=fc.col("profile_id"),
                full_name=fc.col("full_name"),
                age=fc.col("age"),
                gender=fc.col("gender"),
                location=fc.col("location"),
                looking_for=fc.col("looking_for"),
                pets=fc.col("pets"),
                occupation=fc.col("occupation"),
                hobbies=fc.col("hobbies"),
                ideal_partner=fc.col("ideal_partner"),
                bio=fc.col("bio"),
            )
        )

    @fenic_tool(
        tool_name="user_activity_report",
        tool_description="Generate a markdown report summarizing a user's profile and conversation topics",
        default_table_format="markdown",
        max_result_limit=1,
    )
    def user_activity_report(
        profile_id: Annotated[int | str, "Profile ID to generate the report for"],
    ):
        profiles = local_session.table("enriched_profiles")
        conversations = local_session.table("conversations")
        # Normalize input profile_id to int when possible (dataset uses Int64)
        try:
            profile_id_val = int(profile_id)  # handles int and numeric strings
        except Exception:
            profile_id_val = profile_id

        # Profile summary (aggregate to guarantee a single row)
        profile_filtered = profiles.filter(fc.col("profile_id") == fc.lit(profile_id_val)).limit(1)
        profile_summary = profile_filtered.with_column(
            "profile_summary",
            fc.text.jinja(
                dedent("""\
                Profile Summary for {{profile_id}}

                Full Name: {{full_name}}
                Age: {{age}}
                Gender: {{gender}}
                Location: {{location}}
                Looking for: {{looking_for}}
                Hobbies: {{hobbies}}
                Occupation: {{occupation}}
                Ideal Partner: {{ideal_partner}}
                Bio: {{bio}}
                """),
                profile_id=fc.col("profile_id"),
                full_name=fc.col("full_name"),
                age=fc.col("age"),
                gender=fc.col("gender"),
                location=fc.col("location"),
                looking_for=fc.col("looking_for"),
                hobbies=fc.col("hobbies"),
                occupation=fc.col("occupation"),
                ideal_partner=fc.col("ideal_partner"),
                bio=fc.col("bio"),
            )
        ).select(
            fc.lit(profile_id_val).alias("profile_id"),
            fc.col("profile_summary"),
        )

        profile_summary.show()

        # Conversations for this user: user participates as user1 or user2
        user_messages = conversations.filter(
            (fc.col("user1_id") == fc.lit(profile_id_val)) | (fc.col("user2_id") == fc.lit(profile_id_val))
        ).with_column(
            "conversation_doc",
            fc.text.jinja(
                dedent("""\
                Target User for Summarization Id: {{profile_id}}
                Conversation ID: {{conversation_id}}
                User1 Profile ID: {{user1_id}}
                User2 Profile ID: {{user2_id}}
                Conversation Text:

                {{conversation_text}}
                """),
                conversation_id=fc.col("conversation_id"),
                user1_id=fc.col("user1_id"),
                user2_id=fc.col("user2_id"),
                conversation_text=fc.col("conversation_text"),
                profile_id=fc.lit(profile_id_val),
            )
        )

        # Conversation summary
        conv_reduce = fc.semantic.reduce(
            dedent("""
            Summarize this user's conversation activity: main topics, tone, and any notable patterns
            (e.g., icebreakers used, follow-up behavior, response style). Be specific but concise.
            Keep in mind that conversations are between two users, you only want to summarize the messages from the Target User.
            """),
            fc.col("conversation_doc"),
        )
        conversation_summary = user_messages.agg(conv_reduce.alias("conversation_summary")).select(
            fc.lit(profile_id_val).alias("profile_id"),
            fc.col("conversation_summary"),
        )
        # Compose final report
        parts = (
            profile_summary
            .join(conversation_summary, on="profile_id", how="left")
        )
        report = parts.select(
            fc.col("profile_id"),
            fc.semantic.map(
                dedent("""
                User Activity Report (Profile {{profile_id}})

                Generate a natural language report about the user based on the information provided. The report should be concise, but include all relevant information
                to allow for customization based on the user's preferences. This report will be used to customize the user's experience using an agentic dating assistant.

                ## Profile Summary
                {{profile_summary}}

                ## Conversation Summary
                {{conversation_summary}}
                """),
                profile_id=fc.col("profile_id"),
                profile_summary=fc.col("profile_summary"),
                conversation_summary=fc.col("conversation_summary"),
                max_output_tokens=512,
                strict=False,
            ).alias("report"),
        )
        return report

    mcp_generator = fc.create_mcp_server(
        local_session,
        "Dating App Moderation Demo",
        dynamic_tools=[semantic_profile_search, user_activity_report],
        automated_tool_generation=fc.ToolGenerationConfig(
            table_names=["conversations", "enriched_profiles", "moderation_report"],
            tool_group_name="Dating App Moderation",
            max_result_rows=100
        )
    )
    fc.run_mcp_server_sync(mcp_generator)


if __name__ == "__main__":
    main()
