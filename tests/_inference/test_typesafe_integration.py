"""Opt-in live-provider coverage for the native judgment operation."""

import os

import pytest

import fenic as fc


@pytest.mark.requires_provider_key
@pytest.mark.skipif(
    not os.environ.get("TYPESAFE_API_KEY"),
    reason="TYPESAFE_API_KEY is required for the live TypeSafe integration test",
)
def test_native_judge_live(tmp_path):
    pytest.importorskip("typesafe_sdk")
    session = fc.Session.get_or_create(
        fc.SessionConfig(
            app_name="typesafe_integration",
            db_path=tmp_path,
            semantic=fc.SemanticConfig(
                language_models={
                    "judge": fc.TypeSafeLanguageModel(
                        model_name="jev-1.13.0", rpm=60, tpm=64_000
                    ),
                }
            ),
        )
    )
    try:
        result = (
            session.create_dataframe({"text": ["Two plus two equals four."]})
            .select(
                fc.semantic.judge(
                    state="text",
                    questions=[
                        fc.JudgeQuestion.noul(
                            name="correct", instructions="Is the statement correct?"
                        )
                    ],
                    request_timeout=30,
                ).alias("judgment"),
            )
            .unnest("judgment")
            .to_polars()
        )
        probability = result["correct_p"][0]
        assert probability is not None
        assert 0 <= probability <= 1
    finally:
        session.stop(skip_usage_summary=True)
