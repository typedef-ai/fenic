from enum import Enum

import polars as pl

from fenic import ClassifyExample, ClassifyExampleCollection
from fenic._backends.local.semantic_operators.classify import Classify


class TestClassify:
    """Test cases for the Classify operator."""

    def test_build_prompts(self, local_session):
        input = pl.Series(
            "input",
            [
                "Apple is set to release a new version of the iPhone this fall, featuring improved AI capabilities and better battery life.",
                "A recent study shows that walking just 30 minutes a day can significantly reduce the risk of heart disease.",
                "The latest Marvel movie topped the box office this weekend, pulling in over $200 million globally.",
                None,
            ],
        )
        examples = ClassifyExampleCollection(
            examples=[
                ClassifyExample(
                    input="A new VR headset promises to revolutionize gaming with ultra-low latency and full-body motion tracking.",
                    output="Technology",
                ),
                ClassifyExample(
                    input="The CDC has issued new guidelines encouraging people to get updated flu shots ahead of the winter season.",
                    output="Health",
                ),
                ClassifyExample(
                    input="Taylor Swift announced an international tour next year, with tickets expected to sell out within minutes.",
                    output="Entertainment",
                ),
            ]
        )
        labels = Enum(
            "output",
            [
                ("TECHNOLOGY", "Technology"),
                ("HEALTH", "Health"),
                ("ENTERTAINMENT", "Entertainment"),
            ],
        )
        classify = Classify(
            input=input,
            labels=labels,
            examples=examples,
            model=local_session._session_state.get_language_model(),
            temperature=0,
        )
        prefix = [
            {
                "content": "You are a text classification expert. Classify the following document into one of the following labels: Technology, Health, Entertainment. Respond with *only* the predicted label.",
                "role": "system",
            },
            {
                "content": examples.examples[0].input,
                "role": "user",
            },
            {"content": '{"output":"Technology"}', "role": "assistant"},
            {
                "content": examples.examples[1].input,
                "role": "user",
            },
            {"content": '{"output":"Health"}', "role": "assistant"},
            {
                "content": examples.examples[2].input,
                "role": "user",
            },
            {"content": '{"output":"Entertainment"}', "role": "assistant"},
        ]
        expected = [
            prefix
            + [
                {
                    "content": "Apple is set to release a new version of the iPhone this fall, featuring improved AI capabilities and better battery life.",
                    "role": "user",
                }
            ],
            prefix
            + [
                {
                    "content": "A recent study shows that walking just 30 minutes a day can significantly reduce the risk of heart disease.",
                    "role": "user",
                }
            ],
            prefix
            + [
                {
                    "content": "The latest Marvel movie topped the box office this weekend, pulling in over $200 million globally.",
                    "role": "user",
                }
            ],
            None,
        ]

        result = list(
            map(
                lambda x: x.to_message_list() if x else None,
                classify.build_request_messages_batch(),
            )
        )
        assert result == expected

    def test_classify_no_examples(self, local_session):
        input = pl.Series(
            "input",
            [
                "A new VR headset promises to revolutionize gaming with ultra-low latency and full-body motion tracking.",
            ],
        )
        labels = Enum(
            "output",
            [
                ("TECHNOLOGY", "Technology"),
                ("HEALTH", "Health"),
                ("ENTERTAINMENT", "Entertainment"),
                ("OTHER", "Other"),
            ],
        )
        classify = Classify(input=input, labels=labels, model=local_session._session_state.get_language_model(), temperature=0)
        expected = [
            [
                {
                    "role": "system",
                    "content": "You are a text classification expert. Classify the following document into one of the following labels: Technology, Health, Entertainment, Other. Respond with *only* the predicted label.",
                },
                {
                    "role": "user",
                    "content": "A new VR headset promises to revolutionize gaming with ultra-low latency and full-body motion tracking.",
                },
            ]
        ]
        result = list(
            map(
                lambda x: x.to_message_list() if x else None,
                classify.build_request_messages_batch(),
            )
        )
        assert result == expected
