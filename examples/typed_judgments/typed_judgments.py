"""Inspect closed-set probabilities before choosing a downstream threshold."""

import fenic as fc


def main(config: fc.SessionConfig | None = None) -> None:
    """Run a small native judgment pipeline with the configured provider key."""
    session = fc.Session.get_or_create(
        config
        or fc.SessionConfig(
            app_name="typed_judgments",
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
        question = fc.JudgeQuestion.noul(
            name="billing",
            instructions="Does the text describe a billing problem?",
        )
        messages = session.create_dataframe(
            {"text": ["My invoice includes a duplicate charge."]}
        )
        scored = messages.with_column(
            "judgment",
            fc.semantic.judge(state="text", questions=[question], model_alias="decisions"),
        ).unnest("judgment")
        scored.filter(fc.col("billing_p") >= 0.8).show()
    finally:
        session.stop()


if __name__ == "__main__":
    main()
