import polars as pl

from fenic import JoinExample, JoinExampleCollection
from fenic._backends.local.semantic_operators.join import Join


class TestJoin:
    """Test cases for the Join operator."""

    instruction = "Based on the movie's description: {description:left} and the user's preferred genre: {genre:right}, the user will enjoy this movie."
    left_df = pl.DataFrame(
        {
            "description": [
                "Good Will Hunting is a 1997 American drama film directed by Gus Van Sant and written by Ben Affleck and Matt Damon. It stars Robin Williams, Damon, Affleck, Stellan Skarsgård and Minnie Driver. The film tells the story of janitor Will Hunting, whose mathematical genius is discovered by a professor at MIT.",
                None,
                "Spider-Man is a 2002 American superhero film based on the Marvel Comics character Spider-Man. Directed by Sam Raimi from a screenplay by David Koepp, it is the first installment in Raimi's Spider-Man trilogy.",
            ],
            "movie_title": ["Good Will Hunting", "The Dark Knight", "Spider-Man"],
        }
    )
    right_df = pl.DataFrame(
        {
            "genre": ["Drama", "Horror", "Action", None],
            "user_id": [1, 2, 3, 4],
        }
    )

    def test_build_join_pairs(self, local_session):
        sem_join = Join(
            left_df=self.left_df,
            right_df=self.right_df,
            left_on="description",
            right_on="genre",
            join_instruction=self.instruction,
            model=local_session._session_state.get_language_model(),
            temperature=0,
        )
        df = sem_join._build_join_pairs_df().select("_left_id", "_right_id")
        assert df["_left_id"].to_list() == [0, 0, 0, 2, 2, 2]
        assert df["_right_id"].to_list() == [0, 1, 2, 0, 1, 2]

    def test_convert_examples(self, local_session):
        join_examples = JoinExampleCollection(
            examples=[
                JoinExample(
                    left="Dune (titled on-screen as Dune: Part One) is a 2021 American epic space opera film directed and co-produced by Denis Villeneuve, who co-wrote the screenplay with Jon Spaihts and Eric Roth. ",
                    right="Romantic Comedy",
                    output=False,
                )
            ]
        )
        sem_join = Join(
            left_df=self.left_df,
            right_df=self.right_df,
            left_on="description",
            right_on="genre",
            join_instruction=self.instruction,
            model=local_session._session_state.get_language_model(),
            examples=join_examples,
            temperature=0,
        )
        predicate_examples = sem_join._convert_examples().examples
        assert len(predicate_examples) == 1
        assert (
            predicate_examples[0].input["description"]
            == "Dune (titled on-screen as Dune: Part One) is a 2021 American epic space opera film directed and co-produced by Denis Villeneuve, who co-wrote the screenplay with Jon Spaihts and Eric Roth. "
        )
        assert predicate_examples[0].input["genre"] == "Romantic Comedy"
        assert predicate_examples[0].output is False
