"""Customer feedback clustering and analysis example using fenic.

This example demonstrates how to use semantic.with_cluster_labels() and semantic.reduce()
to automatically cluster customer feedback into themes and generate summaries
for each discovered category.
"""

from typing import Optional

import fenic as fc


def main(config: Optional[fc.SessionConfig] = None):
    """Analyze customer feedback using semantic clustering and summarization."""
    # Configure session with both language models and embedding models
    config = config or fc.SessionConfig(
        app_name="feedback_clustering",
        semantic=fc.SemanticConfig(
            language_models={
                "mini": fc.OpenAILanguageModel(
                    model_name="gpt-4o-mini",
                    rpm=500,
                    tpm=200_000,
                )
            },
            embedding_models={
                "small": fc.OpenAIEmbeddingModel(
                    model_name="text-embedding-3-small",
                    rpm=3000,
                    tpm=1_000_000
                )
            }
        ),
    )

    # Create session
    session = fc.Session.get_or_create(config)

    print("Customer Feedback Clustering & Analysis")
    print("=" * 50)
    print("Demonstrating semantic.with_cluster_labels() and semantic.reduce()")
    print()

    # Sample customer feedback data with various themes
    feedback_data = [
        {
            "feedback_id": "fb_001",
            "customer_name": "Alice Johnson",
            "feedback": "The mobile app crashes every time I try to upload a photo. Very frustrating experience!",
            "rating": 1,
            "timestamp": "2024-01-15"
        },
        {
            "feedback_id": "fb_002",
            "customer_name": "Bob Smith",
            "feedback": "Love the new dark mode feature! Much easier on the eyes during night time use.",
            "rating": 5,
            "timestamp": "2024-01-16"
        },
        {
            "feedback_id": "fb_003",
            "customer_name": "Carol Davis",
            "feedback": "The app is way too slow when loading my dashboard. Takes over 30 seconds every time.",
            "rating": 2,
            "timestamp": "2024-01-17"
        },
        {
            "feedback_id": "fb_004",
            "customer_name": "David Wilson",
            "feedback": "Please add a feature to export data to Excel. Really need this for my monthly reports.",
            "rating": 3,
            "timestamp": "2024-01-18"
        },
        {
            "feedback_id": "fb_005",
            "customer_name": "Emma Brown",
            "feedback": "The checkout process is so confusing. Too many steps to complete a simple purchase.",
            "rating": 2,
            "timestamp": "2024-01-19"
        },
        {
            "feedback_id": "fb_006",
            "customer_name": "Frank Miller",
            "feedback": "Amazing customer support team! They solved my billing issue in just minutes.",
            "rating": 5,
            "timestamp": "2024-01-20"
        },
        {
            "feedback_id": "fb_007",
            "customer_name": "Grace Lee",
            "feedback": "Button layouts are inconsistent across different screens. Looks unprofessional.",
            "rating": 2,
            "timestamp": "2024-01-21"
        },
        {
            "feedback_id": "fb_008",
            "customer_name": "Henry Clark",
            "feedback": "Would love to see integration with Google Calendar for appointment scheduling.",
            "rating": 4,
            "timestamp": "2024-01-22"
        },
        {
            "feedback_id": "fb_009",
            "customer_name": "Ivy Martinez",
            "feedback": "App constantly freezes when I try to edit my profile information. Please fix!",
            "rating": 1,
            "timestamp": "2024-01-23"
        },
        {
            "feedback_id": "fb_010",
            "customer_name": "Jack Taylor",
            "feedback": "The search functionality is excellent! Found exactly what I needed quickly.",
            "rating": 5,
            "timestamp": "2024-01-24"
        },
        {
            "feedback_id": "fb_011",
            "customer_name": "Karen White",
            "feedback": "Loading times are terrible. Sometimes the app doesn't respond for minutes.",
            "rating": 1,
            "timestamp": "2024-01-25"
        },
        {
            "feedback_id": "fb_012",
            "customer_name": "Leo Garcia",
            "feedback": "Add offline mode please! Need to access my data when traveling without internet.",
            "rating": 3,
            "timestamp": "2024-01-26"
        }
    ]

    # Create DataFrame
    feedback_df = session.create_dataframe(feedback_data)

    print(f"Loaded {feedback_df.count()} customer feedback entries:")
    feedback_df.select("customer_name", "feedback", "rating").show()
    print()

    # Step 1: Create embeddings for feedback text
    print("Step 1: Creating embeddings for feedback analysis...")
    print("-" * 50)

    # Generate embeddings from the feedback text
    feedback_with_embeddings = feedback_df.select(
        "*",
        fc.semantic.embed(fc.col("feedback")).alias("feedback_embeddings")
    )

    print("Embeddings created successfully!")
    print("Ready for semantic clustering...")
    print()

    # Step 2: Cluster feedback into semantic themes
    print("Step 2: Clustering feedback into themes using semantic.with_cluster_labels()...")
    print("-" * 60)

    # Use semantic group_by to cluster feedback into 4 thematic groups
    # and apply semantic.reduce directly in the aggregation
    feedback_clusters = feedback_with_embeddings.semantic.with_cluster_labels(
        fc.col("feedback_embeddings"),
        4  # Number of clusters - expecting themes like bugs, performance, features, praise
    ).group_by(
        "cluster_label"
    ).agg(
        fc.count("*").alias("feedback_count"),
        fc.avg("rating").alias("avg_rating"),
        fc.collect_list("customer_name").alias("customer_names"),
        fc.semantic.reduce(
            (
                "Analyze this cluster of customer feedback and provide a concise summary of the main theme, "
                "common issues, and sentiment."
            ),
            column=fc.col("feedback")
        ).alias("theme_summary")
    )

    print("Feedback clustered and summarized!")
    print("Theme Analysis Results:")
    print("=" * 70)

    # Display detailed analysis for each cluster
    feedback_clusters.select(
        "cluster_label",
        "feedback_count",
        "avg_rating",
        "theme_summary"
    ).sort("cluster_label").show()
    print()

    # Clean up
    session.stop()
    print("Analysis complete!")


if __name__ == "__main__":
    main()
