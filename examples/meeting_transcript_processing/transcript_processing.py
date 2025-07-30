"""Meeting transcript processing example using fenic semantic operations.

This example demonstrates how to work with transcripts in fenic using its native
transcript processing capabilities, including format detection, parsing, and semantic
extraction of structured information from unstructured meeting content.
"""

from typing import List, Optional

from pydantic import BaseModel, Field

import fenic as fc

# Data

# Engineering Architecture Review Transcript
architecture_review_transcript = """Sarah (00:02:15)
  Alright, so we're here to discuss the user service redesign. Um, Mike can you walk us through the current bottlenecks?

Mike (00:02:28)
  Sure, so right now we're seeing about 2-second response times on the /users endpoint during peak hours. The main issue is we're hitting the PostgreSQL database directly for every request.

David (00:02:45)
  Right, and I think we discussed adding Redis caching last sprint but didn't get to it.

Sarah (00:03:01)
  OK so action item for Mike - investigate Redis implementation. What about the authentication service dependency?

Mike (00:03:18)
  Yeah that's another bottleneck. Every user request has to validate the JWT token with the auth service. We could cache those validations too.

David (00:03:35)
  Actually, we had that incident last week where auth service went down and took out the whole user flow. Incident #INC-2024-007.

Sarah (00:03:48)
  Good point. So we need resilience there. Mike, can you also look into circuit breaker patterns? I'm thinking we implement that by end of Q1.

Mike (00:04:05)
  Yep, I'll research both Redis caching and circuit breakers. Should have a design doc ready by next Friday.

Sarah (00:04:15)
  Perfect. David, anything else on the database side?

David (00:04:22)
  We should consider read replicas too. I've been seeing high CPU on the primary during reports generation.

Sarah (00:04:35)
  OK, let's add that to the backlog. Decision: we're moving forward with the caching and circuit breaker approach for user service optimization."""

# Incident Post-Mortem Transcript
incident_postmortem_transcript = """Alex (00:01:05)
  OK everyone, this is the post-mortem for yesterday's outage. Incident #INC-2024-012. We had approximately 45 minutes of downtime starting at 14:30 UTC.

Jordan (00:01:20)
  The root cause was the payment service running out of memory. We saw heap size spike to 8GB before the JVM crashed.

Sam (00:01:35)
  Yeah, I was monitoring the dashboards. CPU was normal but memory kept climbing. No garbage collection could keep up.

Alex (00:01:48)
  What triggered it? Any recent deployments?

Jordan (00:01:55)
  We deployed the new batch processing feature Tuesday morning. I think there's a memory leak in the transaction processing loop.

Sam (00:02:10)
  Action item for me - I'll review the batch processing code and look for leaked objects or unclosed resources.

Alex (00:02:20)
  Good. Jordan, can you increase the heap size as a temporary mitigation? Maybe 12GB instead of 8?

Jordan (00:02:30)
  Already done. I bumped it to 12GB and added memory alerts at 80% usage.

Alex (00:02:40)
  Perfect. Sam, when can you have the code review done?

Sam (00:02:45)
  I'll have findings by tomorrow EOD. If it's a simple fix, we can hotfix Friday.

Alex (00:02:55)
  Decision: temporary mitigation in place, code review by Thursday, hotfix target Friday if feasible."""

# Sprint Planning Transcript
sprint_planning_transcript = """Emma (00:00:30)
  Alright team, let's plan Sprint 23. We have 40 story points available this sprint. What's our priorities?

Ryan (00:00:45)
  The user authentication refactor is our biggest priority. That's been blocking the mobile team for two weeks.

Lisa (00:00:58)
  Right, and we estimated that at 13 story points. We also need to address the API rate limiting issues.

Emma (00:01:12)
  Good point. Ryan, can you take the auth refactor? And Lisa, would you handle the rate limiting?

Ryan (00:01:20)
  Yeah, I can do the auth work. I'll need to coordinate with the mobile team on the JWT token format changes.

Lisa (00:01:35)
  Sure, I'll take rate limiting. I'm thinking we implement token bucket algorithm with Redis backend.

Emma (00:01:50)
  Perfect. What about the database migration for the user profiles table?

Ryan (00:02:05)
  That's risky. We're adding three new columns and need to backfill data for 2 million users.

Lisa (00:02:18)
  Action item - let's create a migration plan with zero-downtime strategy. I can draft that by Wednesday.

Emma (00:02:35)
  Great. Decision: Sprint 23 priorities are auth refactor, rate limiting, and migration planning. Ryan and Lisa are the leads."""

# Create DataFrame with meeting transcripts


def main(config: Optional[fc.SessionConfig] = None):
    """Process meeting transcripts to extract insights using fenic's semantic operations."""
    # 1. Configure session with semantic capabilities
    config = config or fc.SessionConfig(
        app_name="meeting_transcript_processing",
        semantic=fc.SemanticConfig(
            language_models={
                "mini" : fc.OpenAILanguageModel(
                    model_name="gpt-4o-mini",
                    rpm=500,
                    tpm=200_000,
                )
            }
        ),
    )

    # Create session
    session = fc.Session.get_or_create(config)

    transcripts_data = [
        {
            "meeting_id": "ARCH-2024-001",
            "meeting_type": "Architecture Review",
            "transcript": architecture_review_transcript,
        },
        {
            "meeting_id": "INC-2024-012",
            "meeting_type": "Incident Post-Mortem",
            "transcript": incident_postmortem_transcript,
        },
        {
            "meeting_id": "SPRINT-23",
            "meeting_type": "Sprint Planning",
            "transcript": sprint_planning_transcript,
        },
    ]

    transcripts_df = session.create_dataframe(transcripts_data)

    print("Meeting transcripts loaded:")
    transcripts_df.select(fc.col("meeting_id"), fc.col("meeting_type")).show()

    # Step 1: Use fenic's native transcript processing
    print("\n=== Step 1: Native Transcript Parsing ===")

    # Parse transcripts into structured format
    # generic is a commonly found format for transcripts that follows the format:
    # speaker (00:00:00)
    # content
    # speaker (00:00:00)
    parsed_transcripts_df = transcripts_df.with_column(
        "structured_transcript",
        fc.text.parse_transcript(fc.col("transcript"), "generic"),
    )

    print("\nParsed transcript structure sample:")
    sample_parsed = parsed_transcripts_df.select(
        fc.col("meeting_id"), fc.col("structured_transcript")
    ).limit(1)
    sample_parsed.show()

    # Step 2: Explode structured transcript into individual segments
    print("\n=== Step 2: Extract Individual Speaking Segments ===")
    segments_df = (
        parsed_transcripts_df.explode("structured_transcript")
        .unnest("structured_transcript")
        .select(
            fc.col("meeting_id"),
            fc.col("meeting_type"),
            fc.col("speaker"),
            fc.col("start_time"),
            fc.col("content"),
        )
    )

    print("Individual speaking segments:")
    segments_df.show(5)

    # Step 3: Semantic extraction schemas
    print("\\n=== Step 3: Define Semantic Extraction Schemas ===")

    # Technical entities schema

    class TechnicalEntitiesSchema(BaseModel):
        services: List[str] = Field(
            description="Technical services or systems mentioned (e.g., user-service, auth-service, payment-service)"
        )
        technologies: List[str] = Field(
            description="Technologies, databases, or tools mentioned (e.g., Redis, PostgreSQL, JWT, JVM)"
        )
        metrics: List[str] = Field(
            description="Performance metrics, numbers, or measurements mentioned "
                        "(e.g., response times, memory usage)"
        )
        incident_references: List[str] = Field(
            description="Incident IDs, ticket numbers, or reference numbers mentioned"
        )


    # Action items schema
    class ActionItemSchema(BaseModel):
        has_action_item: str = Field(
            description="Whether this segment contains an action item (yes/no)"
        )
        assignee: Optional[str] = Field(
            default=None, description="Person assigned to the action item (if any)"
        )
        task_description: str = Field(
            description="Description of the task or action to be completed"
        )
        deadline: str = Field(
            default=None, description="When the task should be completed (if mentioned)"
        )

    # Decisions schema
    class DecisionSchema(BaseModel):
        has_decision: str = Field(
            description="Whether this segment contains a decision (yes/no)"
        )
        decision_summary: str = Field(description="Summary of the decision made")
        decision_rationale: Optional[str] = Field(
            default=None, description="Why this decision was made (if mentioned)"
        )

    print("Created schemas for:")
    print("- Technical entities (services, technologies, metrics, incidents)")
    print("- Action items (assignee, task, deadline)")
    print("- Decisions (summary, rationale)")

    # Step 4: Apply semantic extraction to each segment
    print("\n=== Step 4: Semantic Extraction from Speaking Segments ===")

    # Extract technical entities
    enriched_df = (
        segments_df.with_column(
            "technical_entities",
            fc.semantic.extract(fc.col("content"), TechnicalEntitiesSchema),
        )
        .with_column(
            "action_items", fc.semantic.extract(fc.col("content"), ActionItemSchema)
        )
        .with_column(
            "decisions", fc.semantic.extract(fc.col("content"), DecisionSchema)
        )
        .cache()
    )

    print("Applied semantic extraction to all segments")

    # Step 5: Extract and structure insights
    print("\n=== Step 5: Extract Structured Insights ===")
    insights_df = (
        enriched_df.unnest("technical_entities")
        .unnest("action_items")
        .unnest("decisions")
        .select(
            fc.col("meeting_id"),
            fc.col("meeting_type"),
            fc.col("speaker"),
            fc.col("start_time"),
            fc.col("content"),
            fc.col("services"),
            fc.col("technologies"),
            fc.col("metrics"),
            fc.col("incident_references"),
            fc.col("has_action_item"),
            fc.col("assignee"),
            fc.col("task_description"),
            fc.col("deadline"),
            fc.col("has_decision"),
            fc.col("decision_summary"),
            fc.col("decision_rationale"),
        )
    )

    print("Structured insights sample:")
    insights_df.select("*").show(5)

    # Step 6: Aggregate insights by meeting
    print("\n=== Step 6: Meeting-Level Insights and Analytics ===")

    # Extract action items
    action_items_summary = insights_df.filter(
        fc.col("has_action_item") == "yes"
    ).select(
        fc.col("meeting_id"),
        fc.col("meeting_type"),
        fc.col("assignee"),
        fc.col("task_description"),
        fc.col("deadline"),
    )
    print("Action Items Summary:")
    action_items_summary.show()

    # Extract decisions
    decisions_summary = insights_df.filter(fc.col("has_decision") == "yes").select(
        fc.col("meeting_id"),
        fc.col("meeting_type"),
        fc.col("decision_summary"),
        fc.col("decision_rationale"),
    )

    print("\nDecisions Summary:")
    decisions_summary.show()

    # Technical entities mentioned across meetings
    all_services = (
        insights_df.select(fc.col("meeting_id"), fc.col("services"))
        .explode("services")
        .filter(fc.col("services").is_not_null() & (fc.col("services") != ""))
        .group_by("services")
        .agg(fc.count(fc.col("meeting_id")).alias("mention_count"))
        .sort("mention_count", ascending=False)
    )

    print("\nMost Mentioned Services:")
    all_services.show()

    all_technologies = (
        insights_df.select(fc.col("meeting_id"), fc.col("technologies"))
        .explode("technologies")
        .filter(fc.col("technologies").is_not_null() & (fc.col("technologies") != ""))
        .group_by("technologies")
        .agg(fc.count(fc.col("meeting_id")).alias("mention_count"))
        .sort("mention_count", ascending=False)
    )

    print("\nMost Mentioned Technologies:")
    all_technologies.show()

    # Calculate all metrics in a single aggregation using when()
    meeting_stats = insights_df.group_by("meeting_id", "meeting_type").agg(
        fc.count(fc.col("speaker")).alias("total_segments"),
        fc.sum(
            fc.when(fc.col("has_action_item") == "yes", fc.lit(1)).otherwise(fc.lit(0))
        ).alias("action_items_count"),
        fc.sum(
            fc.when(fc.col("has_decision") == "yes", fc.lit(1)).otherwise(fc.lit(0))
        ).alias("decisions_count"),
        fc.sum(
            fc.when(fc.col("services").is_not_null(), fc.lit(1)).otherwise(fc.lit(0))
        ).alias("technical_mentions"),
    )

    print("\nMeeting Productivity Metrics:")
    meeting_stats.show()

    # Step 7: Generate actionable outputs
    print("\n=== Step 7: Actionable Knowledge Outputs ===")

    # Count unique assignees and their workload
    assignee_workload = (
        insights_df.filter(fc.col("has_action_item") == "yes")
        .group_by("assignee")
        .agg(fc.count("*").alias("assigned_tasks"))
        .order_by(fc.col("assigned_tasks").desc())
    )

    print("Team Member Workload (Action Items):")
    assignee_workload.show()

    # Timeline of action items
    action_timeline = (
        insights_df.filter(
            (fc.col("has_action_item") == "yes") & (fc.col("deadline").is_not_null())
        )
        .select(
            fc.col("meeting_id"),
            fc.col("assignee"),
            fc.col("task_description"),
            fc.col("deadline"),
        )
        .order_by("deadline")
    )

    print("\nAction Items Timeline:")
    action_timeline.show()

    print("\n=== Processing Complete ===")
    print("\nThis pipeline demonstrates:")
    print("✅ Native transcript parsing with fenic's built-in functions")
    print("✅ Semantic extraction of technical entities, action items, and decisions")
    print("✅ Structured data processing on unstructured meeting content")
    print("✅ Automated knowledge capture for engineering teams")
    print("✅ Actionable insights for project management and team coordination")


if __name__ == "__main__":
    main()
