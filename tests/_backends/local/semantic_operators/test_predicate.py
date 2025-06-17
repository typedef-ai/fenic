import polars as pl

from fenic import PredicateExample, PredicateExampleCollection
from fenic._backends.local.semantic_operators.predicate import Predicate


class TestPredicate:
    """Test cases for the Predicate operator."""

    def test_build_prompts(self, local_session):
        instruction = "Based on the movie's description: {description} and the user's preferred genre: {genre}, the user will enjoy this movie."
        source_df = pl.DataFrame(
            {
                "description": [
                    "Good Will Hunting is a 1997 American drama film directed by Gus Van Sant and written by Ben Affleck and Matt Damon. It stars Robin Williams, Damon, Affleck, Stellan Skarsgård and Minnie Driver. The film tells the story of janitor Will Hunting, whose mathematical genius is discovered by a professor at MIT.",
                ],
                "genre": ["Drama"],
            }
        )
        examples = PredicateExampleCollection(
            examples=[
                PredicateExample(
                    input={
                        "description": "A Beautiful Mind is a 2001 American biographical drama film about the mathematician John Nash, a Nobel Laureate in Economics, played by Russell Crowe. The film is directed by Ron Howard based on a screenplay by Akiva Goldsman, who adapted the 1998 biography by Sylvia Nasar.",
                        "genre": "Horror",
                    },
                    output=False,
                ),
                PredicateExample(
                    input={
                        "description": "Spider-Man is a 2002 American superhero film based on the Marvel Comics character Spider-Man. Directed by Sam Raimi from a screenplay by David Koepp, it is the first installment in Raimi's Spider-Man trilogy.",
                        "genre": "Action",
                    },
                    output=True,
                ),
            ]
        )
        sem_predicate = Predicate(
            input=source_df,
            user_instruction=instruction,
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
                    "content": "### Claim\n"
                    "Based on the movie's description: [DESCRIPTION] and the user's "
                    "preferred genre: [GENRE], the user will enjoy this movie.\n"
                    "\n"
                    "### Context\n"
                    "[DESCRIPTION]: «A Beautiful Mind is a 2001 American "
                    "biographical drama film about the mathematician John Nash, a "
                    "Nobel Laureate in Economics, played by Russell Crowe. The film "
                    "is directed by Ron Howard based on a screenplay by Akiva "
                    "Goldsman, who adapted the 1998 biography by Sylvia Nasar.»\n"
                    "[GENRE]: «Horror»",
                    "role": "user",
                },
                {"content": '{"output":false}', "role": "assistant"},
                {
                    "content": "### Claim\n"
                    "Based on the movie's description: [DESCRIPTION] and the user's "
                    "preferred genre: [GENRE], the user will enjoy this movie.\n"
                    "\n"
                    "### Context\n"
                    "[DESCRIPTION]: «Spider-Man is a 2002 American superhero film "
                    "based on the Marvel Comics character Spider-Man. Directed by "
                    "Sam Raimi from a screenplay by David Koepp, it is the first "
                    "installment in Raimi's Spider-Man trilogy.»\n"
                    "[GENRE]: «Action»",
                    "role": "user",
                },
                {"content": '{"output":true}', "role": "assistant"},
                {
                    "content": "### Claim\n"
                    "Based on the movie's description: [DESCRIPTION] and the user's "
                    "preferred genre: [GENRE], the user will enjoy this movie.\n"
                    "\n"
                    "### Context\n"
                    "[DESCRIPTION]: «Good Will Hunting is a 1997 American drama film "
                    "directed by Gus Van Sant and written by Ben Affleck and Matt "
                    "Damon. It stars Robin Williams, Damon, Affleck, Stellan "
                    "Skarsgård and Minnie Driver. The film tells the story of "
                    "janitor Will Hunting, whose mathematical genius is discovered "
                    "by a professor at MIT.»\n"
                    "[GENRE]: «Drama»",
                    "role": "user",
                },
            ]
        ]
        assert result == expected

    def test_predicate_without_examples(self, local_session):
        instruction = "Based on the movie's description: {description} and the user's preferred genre: {genre}, the user will enjoy this movie."
        source_df = pl.DataFrame(
            {
                "description": [
                    "A Beautiful Mind is a 2001 American biographical drama film about the mathematician John Nash, a Nobel Laureate in Economics, played by Russell Crowe. The film is directed by Ron Howard based on a screenplay by Akiva Goldsman, who adapted the 1998 biography by Sylvia Nasar.",
                ],
                "genre": [
                    "Drama",
                ],
            }
        )
        sem_predicate = Predicate(
            input=source_df,
            user_instruction=instruction,
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
                    "content": "### Claim\n"
                    "Based on the movie's description: [DESCRIPTION] and the user's "
                    "preferred genre: [GENRE], the user will enjoy this movie.\n"
                    "\n"
                    "### Context\n"
                    "[DESCRIPTION]: «A Beautiful Mind is a 2001 American "
                    "biographical drama film about the mathematician John Nash, a "
                    "Nobel Laureate in Economics, played by Russell Crowe. The film "
                    "is directed by Ron Howard based on a screenplay by Akiva "
                    "Goldsman, who adapted the 1998 biography by Sylvia Nasar.»\n"
                    "[GENRE]: «Drama»",
                    "role": "user",
                },
            ]
        ]
        assert prompts == expected
