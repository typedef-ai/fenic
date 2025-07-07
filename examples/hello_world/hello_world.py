from typing import Optional

from pydantic import BaseModel, Field

import fenic as fc


# Create Pydantic models for extracting error analysis information
class ErrorAnalysis(BaseModel):
    root_cause: str = Field(description="The root cause of this error")
    fix_recommendation: str = Field(description="How to fix this error")

class ErrorPattern(BaseModel):
    error_type: str = Field(description="Type of error (e.g., NullPointer, Timeout, ConnectionRefused)")
    component: str = Field(description="Affected component or system")

def main(config: Optional[fc.SessionConfig] = None):
    # 1. Configure session with semantic capabilities
    config = config or fc.SessionConfig(
        app_name="hello_debug",
        semantic=fc.SemanticConfig(
            language_models= {
                "mini": fc.OpenAIModelConfig(
                    model_name="gpt-4o-mini",  # Fast and effective for log analysis
                    rpm=500,
                    tpm=200_000
                )
            }
        )
    )

    # Create session
    session = fc.Session.get_or_create(config)

    # 2. Create sample error logs - the kind developers see every day
    error_logs = [
        {
            "timestamp": "2024-01-20 14:23:45",
            "service": "api-gateway",
            "error_log": """
ERROR: NullPointerException in UserService.getProfile()
    at com.app.UserService.getProfile(UserService.java:45)
    at com.app.ApiController.handleRequest(ApiController.java:123)
    at java.base/java.lang.Thread.run(Thread.java:834)

User ID: 12345 was not found in cache, attempted DB lookup returned null
            """
        },
        {
            "timestamp": "2024-01-20 14:24:12",
            "service": "auth-service",
            "error_log": """
node:internal/process/promises:288
            triggerUncaughtException(err, true /* fromPromise */);
            ^

Error: connect ECONNREFUSED 127.0.0.1:6379
    at TCPConnectWrap.afterConnect [as oncomplete] (node:net:1494:16)
    at Protocol._enqueue (/app/node_modules/redis/lib/redis.js:458:48)
    at Protocol._write (/app/node_modules/redis/lib/redis.js:326:10)

Redis connection failed during session validation
            """
        },
        {
            "timestamp": "2024-01-20 14:25:33",
            "service": "payment-processor",
            "error_log": """
Traceback (most recent call last):
  File "/app/payment/processor.py", line 89, in process_payment
    response = stripe.Charge.create(
  File "/usr/local/lib/python3.9/site-packages/stripe/api_resources/charge.py", line 45, in create
    return cls._static_request("post", cls.class_url(), params=params)
  File "/usr/local/lib/python3.9/site-packages/stripe/api_requestor.py", line 234, in request
    raise error.APIConnectionError(msg)
stripe.error.APIConnectionError: Connection error: timeout after 30s

Payment processing failed for order_id: ORD-789456
            """
        },
        {
            "timestamp": "2024-01-20 14:26:01",
            "service": "data-pipeline",
            "error_log": """
django.db.utils.OperationalError: could not connect to server: Connection refused
    Is the server running on host "db.prod.internal" (10.0.1.50) and accepting
    TCP/IP connections on port 5432?

FATAL: Batch job 'daily_analytics' failed after 3 retries
Table 'user_metrics' has 2.3M pending records
            """
        },
        {
            "timestamp": "2024-01-20 14:27:15",
            "service": "frontend",
            "error_log": """
TypeError: Cannot read property 'map' of undefined
    at ProfileList (ProfileList.jsx:34:19)
    at renderWithHooks (react-dom.development.js:14985:18)
    at updateFunctionComponent (react-dom.development.js:17356:20)

API response was: {"error": "rate_limit_exceeded", "retry_after": 60}
Component tried to render before data loaded
            """
        },
        {
            "timestamp": "2024-01-20 14:28:03",
            "service": "api-gateway",
            "error_log": """
WARN: Slow query detected in UserService.searchUsers()
Query took 2.3 seconds to complete
SELECT * FROM users WHERE name LIKE '%john%' ORDER BY created_at DESC
Consider adding an index on the name column for better performance
            """
        },
        {
            "timestamp": "2024-01-20 14:28:45",
            "service": "cache-service",
            "error_log": """
INFO: Cache miss for key 'user_preferences_12345'
Fetching data from primary database
Cache hit ratio: 87% (normal range: 85-95%)
No action required
            """
        },
        {
            "timestamp": "2024-01-20 14:29:12",
            "service": "notification-service",
            "error_log": """
WARN: Email delivery delayed for notification_id: notify_789
SMTP server response: 450 Requested mail action not taken: mailbox unavailable
Will retry in 5 minutes (attempt 2/3)
            """
        },
        {
            "timestamp": "2024-01-20 14:29:33",
            "service": "analytics",
            "error_log": """
DEBUG: Processing batch of 1,250 events
Memory usage: 45MB (limit: 512MB)
Processing time: 1.2s
All events processed successfully
            """
        }
    ]

    # 3. Create DataFrame from the error logs
    df = session.create_dataframe(error_logs)

    print("Hello World! Error Log Analyzer")
    print("=" * 70)
    print(f"Found {df.count()} log entries to analyze\n")

    # 4. Analyze errors using semantic operations
    df_analyzed = df.select(
        "timestamp",
        "service",
        # Classify error severity
        fc.semantic.classify("error_log", ["low", "medium", "high", "critical"]).alias("severity"),
        # Extract key debugging information
        fc.semantic.extract(
            "error_log",
            ErrorAnalysis
        ).alias("analysis")
    )

    # Show analysis with extracted fields
    df_analysis_readable = df_analyzed.select(
        "timestamp",
        "service",
        "severity",
        df_analyzed.analysis.root_cause.alias("root_cause"),
        df_analyzed.analysis.fix_recommendation.alias("fix_recommendation")
    )

    print("Error Analysis Results:")
    print("-" * 70)
    df_analysis_readable.show()

    # 5. Focus on critical errors
    print("\nCritical Errors Requiring Immediate Attention:")
    print("-" * 70)
    critical_errors = df_analyzed.filter(
        (df_analyzed["severity"] == "critical") | (df_analyzed["severity"] == "high")
    ).select(
        "timestamp",
        "service",
        df_analyzed.analysis.root_cause.alias("root_cause"),
        df_analyzed.analysis.fix_recommendation.alias("fix_recommendation")
    )
    critical_errors.show()

    # 6. Extract specific error patterns
    df_patterns = df.select(
        "service",
        fc.semantic.extract(
            "error_log",
            ErrorPattern
        ).alias("patterns")
    )

    print("\nError Patterns Detected:")
    print("-" * 70)

    # Show pattern details
    df_pattern_details = df_patterns.select(
        "service",
        df_patterns.patterns.error_type.alias("error_type"),
        df_patterns.patterns.component.alias("component")
    )
    df_pattern_details.show()

    # Clean up
    session.stop()

    print("\nAnalysis complete!")
    print("\nNext steps:")
    print("   - Try adding your own error logs")
    print("   - Extract specific fields like error codes or user IDs")
    print("   - Build alerts for critical errors")
    print("   - Create auto-generated runbooks")


if __name__ == "__main__":
    # Note: Ensure you have set your OpenAI API key:
    # export OPENAI_API_KEY="your-api-key-here"
    main()
