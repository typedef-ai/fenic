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
        df = sem_join._build_join_pairs_df().select(LEFT_ID_KEY, RIGHT_ID_KEY, RENDERED_INSTRUCTION_KEY)
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
        df = sem_join._build_join_pairs_df().select(LEFT_ID_KEY, RIGHT_ID_KEY, RENDERED_INSTRUCTION_KEY)
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
