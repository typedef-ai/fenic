import polars as pl

from fenic._backends.local.semantic_operators.analyze_sentiment import (
    EXAMPLES,
    AnalyzeSentiment,
)


class TestAnalyzeSentiment:
    """Test cases for the AnalyzeSentiment operator."""

    def test_build_prompts(self, local_session):
        input = pl.Series(
            "input",
            [
                "Customer support was rude and unhelpful. I waited days for a response and never got my issue resolved.",
                "Absolutely loved the new user interface! It's clean, intuitive, and way faster than before.",
                "Not impressed. The food was cold, overpriced, and the service was painfully slow.",
                None,
            ],
        )
        analyze_sentiment = AnalyzeSentiment(input=input, model=local_session._session_state.get_language_model(), temperature=0)
        prefix = [
            {
                "content": analyze_sentiment.SYSTEM_PROMPT,
                "role": "system",
            },
            {
                "content": EXAMPLES.examples[0].input,
                "role": "user",
            },
            {"content": '{"output":"positive"}', "role": "assistant"},
            {
                "content": EXAMPLES.examples[1].input,
                "role": "user",
            },
            {"content": '{"output":"negative"}', "role": "assistant"},
            {
                "content": EXAMPLES.examples[2].input,
                "role": "user",
            },
            {"content": '{"output":"neutral"}', "role": "assistant"},
            {
                "content": EXAMPLES.examples[3].input,
                "role": "user",
            },
            {"content": '{"output":"positive"}', "role": "assistant"},
            {
                "content": EXAMPLES.examples[4].input,
                "role": "user",
            },
            {"content": '{"output":"negative"}', "role": "assistant"},
            {
                "content": EXAMPLES.examples[5].input,
                "role": "user",
            },
            {"content": '{"output":"neutral"}', "role": "assistant"},
            {
                "content": EXAMPLES.examples[6].input,
                "role": "user",
            },
            {"content": '{"output":"positive"}', "role": "assistant"},
            {
                "content": EXAMPLES.examples[7].input,
                "role": "user",
            },
            {"content": '{"output":"negative"}', "role": "assistant"},
            {
                "content": EXAMPLES.examples[8].input,
                "role": "user",
            },
            {"content": '{"output":"neutral"}', "role": "assistant"},
        ]
        expected = [
            prefix
            + [
                {
                    "content": "Customer support was rude and unhelpful. I waited days for a response and never got my issue resolved.",
                    "role": "user",
                }
            ],
            prefix
            + [
                {
                    "content": "Absolutely loved the new user interface! It's clean, intuitive, "
                    "and way faster than before.",
                    "role": "user",
                }
            ],
            prefix
            + [
                {
                    "content": "Not impressed. The food was cold, overpriced, and the service was painfully slow.",
                    "role": "user",
                }
            ],
            None,
        ]

        result = list(
            map(
                lambda x: x.to_message_list() if x else None,
                analyze_sentiment.build_request_messages_batch(),
            )
        )
        assert result == expected
