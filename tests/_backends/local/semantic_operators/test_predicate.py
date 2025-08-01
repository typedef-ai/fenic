from textwrap import dedent

import jinja2
import polars as pl

from fenic import PredicateExample, PredicateExampleCollection
from fenic._backends.local.semantic_operators.predicate import Predicate


class TestPredicate:
    """Test cases for the Predicate operator."""

    TEMPLATE = dedent("""
    Movie: {{ description }}

    Claim: This movie is a good recommendation for someone who enjoys {{ genre }} films.

    Evaluate the claim based on the following criteria:
    1. Does the movie belong to or strongly align with the {{ genre }} category?
    2. Would the plot and themes likely appeal to typical {{ genre }} fans?
    3. Does the tone match what {{ genre }} enthusiasts generally expect?
    4. Are there elements that might disappoint someone specifically seeking {{ genre }} content?""").strip()

    RENDERED_TEMPLATE = jinja2.Template(TEMPLATE).render(
        description="Good Will Hunting is a 1997 American drama film directed by Gus Van Sant and written by Ben Affleck and Matt Damon. It stars Robin Williams, Damon, Affleck, Stellan Skarsgård and Minnie Driver. The film tells the story of janitor Will Hunting, whose mathematical genius is discovered by a professor at MIT.",
        genre="drama",
    )

    def test_build_prompts(self, local_session):
        examples = PredicateExampleCollection(
            examples=[
                PredicateExample(
                    input={
                        "description": "A Beautiful Mind is a 2001 American biographical drama film about the mathematician John Nash, a Nobel Laureate in Economics, played by Russell Crowe. The film is directed by Ron Howard based on a screenplay by Akiva Goldsman, who adapted the 1998 biography by Sylvia Nasar.",
                        "genre": "horror",
                    },
                    output=False,
                ),
                PredicateExample(
                    input={
                        "description": "Spider-Man is a 2002 American superhero film based on the Marvel Comics character Spider-Man. Directed by Sam Raimi from a screenplay by David Koepp, it is the first installment in Raimi's Spider-Man trilogy.",
                        "genre": "action",
                    },
                    output=True,
                ),
            ]
        )
        sem_predicate = Predicate(
            input=pl.Series(name="movie", values=[self.RENDERED_TEMPLATE]),
            jinja_template=self.TEMPLATE,
            examples=examples,
            model=local_session._session_state.get_language_model(),
            temperature=0,
        )

        result = list(
            map(
                lambda x: x.to_message_list() if x else None,
                sem_predicate.build_request_messages_batch(),
            )
        )

        expected = [
            [
                {
                    "content": sem_predicate.SYSTEM_PROMPT,
                    "role": "system",
                },
                {
                    "content": dedent("""\
                    Movie: A Beautiful Mind is a 2001 American biographical drama film about the mathematician John Nash, a Nobel Laureate in Economics, played by Russell Crowe. The film is directed by Ron Howard based on a screenplay by Akiva Goldsman, who adapted the 1998 biography by Sylvia Nasar.

                    Claim: This movie is a good recommendation for someone who enjoys horror films.

                    Evaluate the claim based on the following criteria:
                    1. Does the movie belong to or strongly align with the horror category?
                    2. Would the plot and themes likely appeal to typical horror fans?
                    3. Does the tone match what horror enthusiasts generally expect?
                    4. Are there elements that might disappoint someone specifically seeking horror content?"""),
                    "role": "user",
                },
                {"content": '{"output":false}', "role": "assistant"},
                {
                    "content": dedent("""\
                    Movie: Spider-Man is a 2002 American superhero film based on the Marvel Comics character Spider-Man. Directed by Sam Raimi from a screenplay by David Koepp, it is the first installment in Raimi's Spider-Man trilogy.

                    Claim: This movie is a good recommendation for someone who enjoys action films.

                    Evaluate the claim based on the following criteria:
                    1. Does the movie belong to or strongly align with the action category?
                    2. Would the plot and themes likely appeal to typical action fans?
                    3. Does the tone match what action enthusiasts generally expect?
                    4. Are there elements that might disappoint someone specifically seeking action content?"""),
                    "role": "user",
                },
                {"content": '{"output":true}', "role": "assistant"},
                {
                    "content": dedent("""\
                    Movie: Good Will Hunting is a 1997 American drama film directed by Gus Van Sant and written by Ben Affleck and Matt Damon. It stars Robin Williams, Damon, Affleck, Stellan Skarsgård and Minnie Driver. The film tells the story of janitor Will Hunting, whose mathematical genius is discovered by a professor at MIT.

                    Claim: This movie is a good recommendation for someone who enjoys drama films.

                    Evaluate the claim based on the following criteria:
                    1. Does the movie belong to or strongly align with the drama category?
                    2. Would the plot and themes likely appeal to typical drama fans?
                    3. Does the tone match what drama enthusiasts generally expect?
                    4. Are there elements that might disappoint someone specifically seeking drama content?"""),
                    "role": "user",
                },
            ]
        ]
        assert result == expected

    def test_predicate_without_examples(self, local_session):
        sem_predicate = Predicate(
            input=pl.Series(name="movie", values=[self.RENDERED_TEMPLATE]),
            jinja_template=self.TEMPLATE,
            model=local_session._session_state.get_language_model(),
            temperature=0,
        )
        prompts = list(
            map(
                lambda x: x.to_message_list() if x else None,
                sem_predicate.build_request_messages_batch(),
            )
        )
        expected = [
            [
                {
                    "content": sem_predicate.SYSTEM_PROMPT,
                    "role": "system",
                },
                {
                    "content": dedent("""\
                    Movie: Good Will Hunting is a 1997 American drama film directed by Gus Van Sant and written by Ben Affleck and Matt Damon. It stars Robin Williams, Damon, Affleck, Stellan Skarsgård and Minnie Driver. The film tells the story of janitor Will Hunting, whose mathematical genius is discovered by a professor at MIT.

                    Claim: This movie is a good recommendation for someone who enjoys drama films.

                    Evaluate the claim based on the following criteria:
                    1. Does the movie belong to or strongly align with the drama category?
                    2. Would the plot and themes likely appeal to typical drama fans?
                    3. Does the tone match what drama enthusiasts generally expect?
                    4. Are there elements that might disappoint someone specifically seeking drama content?"""),
                    "role": "user",
                },
            ]
        ]
        assert prompts == expected
