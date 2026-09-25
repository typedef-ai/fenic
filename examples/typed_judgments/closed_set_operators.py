"""Use a decision provider with existing closed-set DataFrame operations."""

import fenic as fc


def main(config: fc.SessionConfig | None = None) -> None:
    """Classify, filter, join, and inspect sentiment on the local backend."""
    session = fc.Session.get_or_create(
        config
        or fc.SessionConfig(
            app_name="closed_set_operators",
            semantic=fc.SemanticConfig(
                language_models={
                    "decisions": fc.TypeSafeLanguageModel(
                        model_name="jev-1.13.0", rpm=60, tpm=64_000
                    )
                }
            ),
        )
    )
    try:
        messages = session.create_dataframe(
            {"text": ["Please explain this invoice.", "Thanks for the quick reply."]}
        )
        tagged = messages.with_column(
            "topic",
            fc.semantic.classify(
                "text",
                [
                    fc.ClassDefinition(
                        label="Billing", description="Charges or invoices"
                    ),
                    fc.ClassDefinition(label="Other", description="Other requests"),
                ],
            ),
        ).with_column("sentiment", fc.semantic.analyze_sentiment("text"))
        tagged.show()
        messages.filter(
            fc.semantic.predicate(
                "Does this discuss billing? {{ message }}", message=fc.col("text")
            )
        ).show()
        topics = session.create_dataframe({"category": ["Billing", "Account access"]})
        messages.semantic.join(
            topics,
            "Does {{ left_on }} concern {{ right_on }}?",
            left_on=fc.col("text"),
            right_on=fc.col("category"),
        ).show()
    finally:
        session.stop()


if __name__ == "__main__":
    main()
