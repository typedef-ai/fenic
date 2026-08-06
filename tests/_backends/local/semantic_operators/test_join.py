from textwrap import dedent

import jinja2
import polars as pl

from fenic import JoinExample, JoinExampleCollection
from fenic._backends.local.semantic_operators.join import (
    LEFT_ID_KEY,
    RENDERED_INSTRUCTION_KEY,
    RIGHT_ID_KEY,
    Join,
)
from fenic._backends.local.semantic_operators.predicate import Predicate


class TestJoin:
    """Test cases for the Join operator."""

    TEMPLATE = dedent("""
    Movie: {{ left_on }}

    Claim: This movie is a good recommendation for someone who enjoys {{ right_on }} films.

    Evaluate the claim based on the following criteria:
    1. Does the movie belong to or strongly align with the {{ right_on }} category?
    2. Would the plot and themes likely appeal to typical {{ right_on }} fans?
    3. Does the tone match what {{ right_on }} enthusiasts generally expect?
    4. Are there elements that might disappoint someone specifically seeking {{ right_on }} content?""").strip()

    GOOD_WILL_HUNTING = "Good Will Hunting is a 1997 American drama film directed by Gus Van Sant and written by Ben Affleck and Matt Damon. It stars Robin Williams, Damon, Affleck, Stellan Skarsgård and Minnie Driver. The film tells the story of janitor Will Hunting, whose mathematical genius is discovered by a professor at MIT."
    SPIDER_MAN = "Spider-Man is a 2002 American superhero film based on the Marvel Comics character Spider-Man. Directed by Sam Raimi from a screenplay by David Koepp, it is the first installment in Raimi's Spider-Man trilogy."

    GOOD_WILL_HUNTING_RENDERED_TEMPLATE = jinja2.Template(TEMPLATE).render(
        left_on=GOOD_WILL_HUNTING,
        right_on="Drama",
    )
    SPIDER_MAN_RENDERED_TEMPLATE = jinja2.Template(TEMPLATE).render(
        left_on=SPIDER_MAN,
        right_on="Action",
    )

    left_df = pl.DataFrame(
        {
            "left_on": [
                "Good Will Hunting is a 1997 American drama film directed by Gus Van Sant and written by Ben Affleck and Matt Damon. It stars Robin Williams, Damon, Affleck, Stellan Skarsgård and Minnie Driver. The film tells the story of janitor Will Hunting, whose mathematical genius is discovered by a professor at MIT.",
                None,
                "Spider-Man is a 2002 American superhero film based on the Marvel Comics character Spider-Man. Directed by Sam Raimi from a screenplay by David Koepp, it is the first installment in Raimi's Spider-Man trilogy.",
            ],
            "movie_title": ["Good Will Hunting", "The Dark Knight", "Spider-Man"],
        }
    )
    right_df = pl.DataFrame(
        {
            "right_on": ["Drama", "Horror", "Action", None],
            "user_id": [1, 2, 3, 4],
        }
    )

    def test_build_join_pairs_strict(self, local_session):
        sem_join = Join(
            left_df=self.left_df,
            right_df=self.right_df,
            strict=True,
            jinja_template=self.TEMPLATE,
            model=local_session._session_state.get_language_model(),
            temperature=0,
        )
        left_documents, right_documents = sem_join._join_documents()
        df = sem_join._build_join_pair_block(left_documents, right_documents).select(LEFT_ID_KEY, RIGHT_ID_KEY, RENDERED_INSTRUCTION_KEY)
        assert df[LEFT_ID_KEY].to_list() == [0, 0, 0, 2, 2, 2]
        assert df[RIGHT_ID_KEY].to_list() == [0, 1, 2, 0, 1, 2]
        assert df[RENDERED_INSTRUCTION_KEY].to_list() == [
            jinja2.Template(self.TEMPLATE).render(left_on=self.GOOD_WILL_HUNTING, right_on="Drama"),
            jinja2.Template(self.TEMPLATE).render(left_on=self.GOOD_WILL_HUNTING, right_on="Horror"),
            jinja2.Template(self.TEMPLATE).render(left_on=self.GOOD_WILL_HUNTING, right_on="Action"),
            jinja2.Template(self.TEMPLATE).render(left_on=self.SPIDER_MAN, right_on="Drama"),
            jinja2.Template(self.TEMPLATE).render(left_on=self.SPIDER_MAN, right_on="Horror"),
            jinja2.Template(self.TEMPLATE).render(left_on=self.SPIDER_MAN, right_on="Action"),
        ]

    def test_build_join_pairs_non_strict(self, local_session):
        sem_join = Join(
            left_df=self.left_df,
            right_df=self.right_df,
            strict=False,
            jinja_template=self.TEMPLATE,
            model=local_session._session_state.get_language_model(),
            temperature=0,
        )
        left_documents, right_documents = sem_join._join_documents()
        df = sem_join._build_join_pair_block(left_documents, right_documents).select(LEFT_ID_KEY, RIGHT_ID_KEY, RENDERED_INSTRUCTION_KEY)
        assert df[LEFT_ID_KEY].to_list() == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
        assert df[RIGHT_ID_KEY].to_list() == [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
        assert df[RENDERED_INSTRUCTION_KEY].to_list() == [
            jinja2.Template(self.TEMPLATE).render(left_on=self.GOOD_WILL_HUNTING, right_on="Drama"),
            jinja2.Template(self.TEMPLATE).render(left_on=self.GOOD_WILL_HUNTING, right_on="Horror"),
            jinja2.Template(self.TEMPLATE).render(left_on=self.GOOD_WILL_HUNTING, right_on="Action"),
            jinja2.Template(self.TEMPLATE).render(left_on=self.GOOD_WILL_HUNTING, right_on="none"), # we use "none" instead of None here because Rust jinja2 renders None as "none" instead of "None"
            jinja2.Template(self.TEMPLATE).render(left_on="none", right_on="Drama"),
            jinja2.Template(self.TEMPLATE).render(left_on="none", right_on="Horror"),
            jinja2.Template(self.TEMPLATE).render(left_on="none", right_on="Action"),
            jinja2.Template(self.TEMPLATE).render(left_on="none", right_on="none"),
            jinja2.Template(self.TEMPLATE).render(left_on=self.SPIDER_MAN, right_on="Drama"),
            jinja2.Template(self.TEMPLATE).render(left_on=self.SPIDER_MAN, right_on="Horror"),
            jinja2.Template(self.TEMPLATE).render(left_on=self.SPIDER_MAN, right_on="Action"),
            jinja2.Template(self.TEMPLATE).render(left_on=self.SPIDER_MAN, right_on="none"),
        ]

    def test_convert_examples(self, local_session):
        join_examples = JoinExampleCollection(
            examples=[
                JoinExample(
                    left_on="Dune (titled on-screen as Dune: Part One) is a 2021 American epic space opera film directed and co-produced by Denis Villeneuve, who co-wrote the screenplay with Jon Spaihts and Eric Roth. ",
                    right_on="Romantic Comedy",
                    output=False,
                )
            ]
        )
        sem_join = Join(
            left_df=self.left_df,
            right_df=self.right_df,
            jinja_template=self.TEMPLATE,
            strict=True,
            model=local_session._session_state.get_language_model(),
            examples=join_examples,
            temperature=0,
        )
        predicate_examples = sem_join._convert_examples().examples
        assert len(predicate_examples) == 1
        assert (
            predicate_examples[0].input["left_on"]
            == "Dune (titled on-screen as Dune: Part One) is a 2021 American epic space opera film directed and co-produced by Denis Villeneuve, who co-wrote the screenplay with Jon Spaihts and Eric Roth. "
        )
        assert predicate_examples[0].input["right_on"] == "Romantic Comedy"
        assert predicate_examples[0].output is False

    def test_execute_bounds_predicate_blocks_without_losing_or_duplicating_pairs(
        self, local_session, monkeypatch
    ):
        observed_blocks = []

        def fake_execute(predicate):
            rendered = predicate.input.to_list()
            observed_blocks.append(rendered)
            return pl.Series(["keep" in prompt for prompt in rendered])

        monkeypatch.setattr(Predicate, "execute", fake_execute)

        sem_join = Join(
            left_df=pl.DataFrame(
                {
                    "left_on": ["left-0", "left-1", "left-2"],
                    "left_payload": [0, 1, 2],
                }
            ),
            right_df=pl.DataFrame(
                {
                    "right_on": ["keep", "skip", "keep-too"],
                    "right_payload": [10, 11, 12],
                }
            ),
            jinja_template="{{ left_on }} {{ right_on }}",
            strict=True,
            model=local_session._session_state.get_language_model(),
            temperature=0,
            pair_block_size=2,
        )

        result = sem_join.execute().sort(["left_payload", "right_payload"])

        assert [len(block) for block in observed_blocks] == [2, 1, 2, 1, 2, 1]
        assert all(len(block) <= 2 for block in observed_blocks)
        assert result.select(["left_payload", "right_payload"]).to_dicts() == [
            {"left_payload": left, "right_payload": right}
            for left in range(3)
            for right in (10, 12)
        ]

    def test_execute_splits_pair_blocks_to_the_rendered_token_budget(
        self, local_session, monkeypatch
    ):
        observed_blocks = []

        def fake_execute(predicate):
            observed_blocks.append(predicate.input.to_list())
            return pl.Series([False] * len(predicate.input))

        monkeypatch.setattr(Predicate, "execute", fake_execute)
        sem_join = Join(
            left_df=pl.DataFrame({"left_on": ["left"]}),
            right_df=pl.DataFrame({"right_on": ["one", "two", "three"]}),
            jinja_template="{{ left_on }} {{ right_on }}",
            strict=True,
            model=local_session._session_state.get_language_model(),
            temperature=0,
            pair_block_size=3,
            block_token_budget=4,
        )
        counted_prompts = []
        monkeypatch.setattr(
            sem_join.model,
            "count_tokens",
            lambda prompt: counted_prompts.append(prompt) or 2,
        )

        sem_join.execute()

        assert [len(block) for block in observed_blocks] == [1, 2]
        assert all(2 * len(block) <= 4 for block in observed_blocks)
        assert set(counted_prompts) == {"left one", "left two", "left three"}
