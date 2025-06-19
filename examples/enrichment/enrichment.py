from typing import Optional

from pydantic import BaseModel, Field

import fenic as fc


def main(config: Optional[fc.SessionConfig] = None):
    # Configure session with semantic capabilities
    config = config or fc.SessionConfig(
        app_name="log_enrichment",
        semantic=fc.SemanticConfig(
            language_models= {
                "mini": fc.OpenAIModelConfig(model_name="gpt-4o-mini", rpm=500, tpm=200_000)
            }
        )
    )

    # Create session
    session = fc.Session.get_or_create(config)

    # Raw application logs with different text formats
    raw_logs_data = [
        "2024-01-15 14:32:01 [ERROR] payment-api: Connection timeout to db-primary.internal:5432 after 30s retries=3 connection_id=conn_789",
        "2024-01-15 14:32:15 [WARN] user-service: Rate limit exceeded client_ip=192.168.1.100 requests=1205/min endpoint=/api/v1/users",
        "2024-01-15 14:33:02 [ERROR] order-processor: Payment validation failed order_id=12345 payment_method=credit_card error_code=INVALID_CVV",
        "2024-01-15 14:35:22 [INFO] auth-service: User login successful user_id=user_67890 session_id=sess_abc123 ip=10.0.1.55",
        "2024-01-15 14:36:45 [ERROR] notification-service: Failed to send email to user@example.com smtp_error=Connection_refused retries=2",
        "2024-01-15 14:37:12 [WARN] inventory-service: Low stock alert product_id=SKU_9876 current_stock=5 threshold=10",
        "2024-01-15 14:38:33 [ERROR] payment-api: Database connection pool exhausted max_connections=50 active=50 pending=15",
        "2024-01-15 14:39:01 [CRITICAL] order-processor: Circuit breaker opened for payment-gateway failure_rate=85% threshold=80%",
        "2024-01-15 14:40:15 [ERROR] user-service: Authentication failed user_id=user_12345 reason=invalid_token attempts=3",
        "2024-01-15 14:41:22 [WARN] cache-service: Redis connection latency high avg_latency=250ms threshold=100ms",
        "2024-01-15 14:42:33 [ERROR] file-service: Disk space critical mount=/data/uploads usage=95% available=2GB",
        "2024-01-15 14:43:44 [INFO] metrics-service: Health check passed response_time=45ms status=healthy",
        "2024-01-15 14:44:55 [ERROR] search-service: Elasticsearch cluster unhealthy nodes_down=2 total_nodes=5",
        "2024-01-15 14:45:10 [WARN] api-gateway: Request timeout to upstream service=user-service timeout=10s endpoint=/api/v1/profile",
        "2024-01-15 14:46:20 [ERROR] backup-service: S3 upload failed file=backup_20240115.tar.gz error=AccessDenied bucket=prod-backups"
    ]

    # Service metadata for classical enrichment
    service_metadata_data = [
        {"service_name": "payment-api", "team_owner": "payments-team", "criticality": "critical", "on_call_channel": "#payments-oncall"},
        {"service_name": "user-service", "team_owner": "identity-team", "criticality": "high", "on_call_channel": "#identity-alerts"},
        {"service_name": "order-processor", "team_owner": "commerce-team", "criticality": "critical", "on_call_channel": "#commerce-oncall"},
        {"service_name": "auth-service", "team_owner": "identity-team", "criticality": "critical", "on_call_channel": "#identity-alerts"},
        {"service_name": "notification-service", "team_owner": "platform-team", "criticality": "medium", "on_call_channel": "#platform-alerts"},
        {"service_name": "inventory-service", "team_owner": "commerce-team", "criticality": "high", "on_call_channel": "#commerce-oncall"},
        {"service_name": "cache-service", "team_owner": "platform-team", "criticality": "high", "on_call_channel": "#platform-alerts"},
        {"service_name": "file-service", "team_owner": "platform-team", "criticality": "medium", "on_call_channel": "#platform-alerts"},
        {"service_name": "metrics-service", "team_owner": "observability-team", "criticality": "medium", "on_call_channel": "#observability"},
        {"service_name": "search-service", "team_owner": "data-team", "criticality": "high", "on_call_channel": "#data-oncall"},
        {"service_name": "api-gateway", "team_owner": "platform-team", "criticality": "critical", "on_call_channel": "#platform-alerts"},
        {"service_name": "backup-service", "team_owner": "platform-team", "criticality": "medium", "on_call_channel": "#platform-alerts"}
    ]

    # Create DataFrames
    logs_df = session.create_dataframe({"raw_message": raw_logs_data})
    metadata_df = session.create_dataframe(service_metadata_data)

    print("🚀 Log Enrichment Pipeline")
    print("=" * 70)
    print(f"Processing {logs_df.count()} log entries with {metadata_df.count()} service metadata records\n")

    # Stage 1: Parse unstructured logs using template extraction
    print("🔍 Stage 1: Parsing unstructured log messages...")
    log_template = "${timestamp:none} [${level:none}] ${service:none}: ${message:none}"

    parsed_df = logs_df.select(
        fc.text.extract("raw_message", log_template).alias("parsed")
    ).select(
        fc.col("parsed").get_item("timestamp").alias("timestamp"),
        fc.col("parsed").get_item("level").alias("level"),
        fc.col("parsed").get_item("service").alias("service"),
        fc.col("parsed").get_item("message").alias("message")
    ).filter(
        fc.col("timestamp").is_not_null()
    )

    print("Sample parsed logs:")
    parsed_df.select("timestamp", "level", "service").show(3)

    # Stage 2: Classical enrichment with service metadata
    print("\n🔗 Stage 2: Enriching with service metadata...")
    # Rename service_name to service using select with alias
    metadata_df_renamed = metadata_df.select(
        fc.col("service_name").alias("service"),
        "team_owner",
        "criticality",
        "on_call_channel"
    )
    enriched_df = parsed_df.join(
        metadata_df_renamed,
        on="service",
        how="left"
    ).select(
        "timestamp",
        "level",
        "service",
        "message",
        "team_owner",
        "criticality",
        "on_call_channel"
    )

    print("Sample enriched logs:")
    enriched_df.select("service", "timestamp", "team_owner", "criticality").show(3)

    # Stage 3: Semantic enrichment using LLM operations
    print("\n🧠 Stage 3: Applying semantic enrichment with LLMs...")
    print("This may take a few moments as we process logs with language models...")

    # Define the Pydantic model for semantic error extraction
    class ErrorAnalysis(BaseModel):
        """Pydantic model for semantic error extraction."""
        error_category: str = Field(..., description="Main category of the error (e.g., database, network, authentication, resource)")
        affected_component: str = Field(..., description="Specific component or resource affected")
        potential_cause: str = Field(..., description="Most likely root cause of the issue")
    # Semantic extraction for error analysis using Pydantic model
    final_df = enriched_df.select(
        "timestamp",
        "level",
        "service",
        "message",
        "team_owner",
        "criticality",
        "on_call_channel",
        # Extract error analysis information using Pydantic model
        fc.semantic.extract("message", ErrorAnalysis).alias("analysis"),
        # Classify incident severity based on message and service criticality
        fc.semantic.classify(
            fc.text.concat(fc.col("message"), fc.lit(" (criticality: "), fc.col("criticality"), fc.lit(")")),
            ["low", "medium", "high", "critical"]
        ).alias("incident_severity"),
        # Generate remediation steps
        fc.semantic.map(
            "Generate 2-3 specific remediation steps that the on-call team should take to resolve this issue: {message} | Service: {service} | Team: {team_owner}"
        ).alias("remediation_steps")
    )

    # Create readable final output with extracted fields
    final_readable = final_df.select(
        "timestamp",
        "level",
        "service",
        "message",
        "team_owner",
        "criticality",
        "on_call_channel",
        fc.col("analysis").get_item("error_category").alias("error_category"),
        fc.col("analysis").get_item("affected_component").alias("affected_component"),
        fc.col("analysis").get_item("potential_cause").alias("potential_cause"),
        "incident_severity",
        "remediation_steps"
    )

    # Display results
    print("\n✅ Pipeline Complete! Final enriched logs:")
    print("-" * 70)
    final_readable.show()

    # Analytics examples
    print("\n📈 Analytics Examples:")
    print("-" * 70)

    # Error category distribution
    print("\nError Category Distribution:")
    final_readable.group_by("error_category").agg(fc.count("*").alias("count")).show()

    # Severity by service criticality
    print("\nIncident Severity by Service Criticality:")
    final_readable.group_by("criticality", "incident_severity").agg(fc.count("*").alias("count")).show()

    # High-priority incidents requiring immediate attention
    print("\nHigh-Priority Incidents (Critical/High severity):")
    print("-" * 70)
    critical_incidents = final_readable.filter(
        (final_readable.incident_severity == "critical") | (final_readable.incident_severity == "high")
    ).select(
        "service",
        "team_owner",
        "incident_severity",
        "on_call_channel",
        "remediation_steps"
    )
    critical_incidents.show()

    critical_count = critical_incidents.count()
    print(f"\n🚨 Found {critical_count} high-priority incidents requiring immediate attention")

    # Clean up
    session.stop()

    print("\nAnalysis complete!")
    print("\nNext steps:")
    print("   - Integrate with real log streams (Kafka, Elasticsearch)")
    print("   - Set up automated alerting for critical incidents")
    print("   - Build historical trend analysis")
    print("   - Create auto-generated incident reports")


if __name__ == "__main__":
    # Note: Ensure you have set your OpenAI API key:
    # export OPENAI_API_KEY="your-api-key-here"
    main()
