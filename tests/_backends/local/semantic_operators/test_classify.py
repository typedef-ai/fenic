from textwrap import dedent

import polars as pl

from fenic import ClassifyExample, ClassifyExampleCollection
from fenic._backends.local.semantic_operators.classify import Classify
from fenic.core._logical_plan.expressions import ResolvedClassDefinition


class TestClassify:
    """Test cases for the Classify operator."""
    classes_with_descriptions = [
        ResolvedClassDefinition(label="Technology", description="Content related to software, hardware, the internet, gadgets, programming, AI, or emerging tech trends."),
        ResolvedClassDefinition(label="Health", description="Content related to health, fitness, medicine, or medical research."),
        ResolvedClassDefinition(label="Entertainment", description="Content related to movies, TV shows, music, or other forms of entertainment."),
    ]
    system_message_with_descriptions = dedent("""\
        You are a text classification expert. Classify the following document into one of the following labels:
        - Technology: Content related to software, hardware, the internet, gadgets, programming, AI, or emerging tech trends.
        - Health: Content related to health, fitness, medicine, or medical research.
        - Entertainment: Content related to movies, TV shows, music, or other forms of entertainment.
        Respond with *only* the predicted label.""")

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
        classify = Classify(
            input=input,
            classes=self.classes_with_descriptions,
            examples=examples,
            model=local_session._session_state.get_language_model(),
            temperature=0,
        )
        prefix = [
            {
                "content": self.system_message_with_descriptions,
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

    def test_classify_without_examples(self, local_session):
        input = pl.Series(
            "input",
            [
                "A new VR headset promises to revolutionize gaming with ultra-low latency and full-body motion tracking.",
            ],
        )
        classify = Classify(input=input, classes=self.classes_with_descriptions, model=local_session._session_state.get_language_model(), temperature=0)
        expected = [
            [
                {
                    "role": "system",
                    "content": self.system_message_with_descriptions,
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

    def test_build_prompts_without_descriptions(self, local_session):
        classes = [
            ResolvedClassDefinition(label="Technology"),
            ResolvedClassDefinition(label="Health"),
            ResolvedClassDefinition(label="Entertainment"),
        ]
        system_message = dedent("""\
            You are a text classification expert. Classify the following document into one of the following labels:
            - Technology
            - Health
            - Entertainment
            Respond with *only* the predicted label.""")
        input = pl.Series(
            "input",
            [
                "Apple is set to release a new version of the iPhone this fall, featuring improved AI capabilities and better battery life.",
            ],
        )
        classify = Classify(input=input, classes=classes, model=local_session._session_state.get_language_model(), temperature=0)
        expected = [
            [
                {
                    "role": "system",
                    "content": system_message,
                },
                {
                    "role": "user",
                    "content": "Apple is set to release a new version of the iPhone this fall, featuring improved AI capabilities and better battery life.",
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
