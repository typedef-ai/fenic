from textwrap import dedent

import jinja2
import polars as pl
from pydantic import BaseModel, Field

from fenic import MapExample, MapExampleCollection
from fenic._backends.local.semantic_operators.map import Map
from fenic.core._logical_plan.resolved_types import ResolvedResponseFormat


class TestMap:
    """Test cases for the Map operator."""

    TEMPLATE = dedent("""\
        Analyze this movie based on its description and the critic review.

        Movie: {{ description }}
        Review: {{ review }}

        Write a concise summary highlighting the good qualities in this movie that led to a favorable rating.""")

    RENDERED_TEMPLATE = jinja2.Template(TEMPLATE).render(
        description="Good Will Hunting is a 1997 American drama film directed by Gus Van Sant and written by Ben Affleck and Matt Damon. It stars Robin Williams, Damon, Affleck, Stellan Skarsgård and Minnie Driver. The film tells the story of janitor Will Hunting, whose mathematical genius is discovered by a professor at MIT.",
        review="When people tumble into love – in Hollywood movies, that is – intelligence is rarely the motivating force that brings them together. Being adorable or eccentric, or having an amazing head of hair – these are the usual qualities that make one flavor of the month hot for another.But in the wonderfully original 'Good Will Hunting', Matt Damon's appeal doesn't spring from good looks, sculpted locks or cover-boy ubiquitousness – although certainly those qualities should haul in the crowds. What counts is his thinking organ. When Will Hunting (Damon) meets Skylar (Minnie Driver), a highly intelligent Harvard student, they waltz on a mental plateau that Julia Roberts and Brad Pitt couldn't reach by cable car.",
    )

    def test_build_prompts(self, local_session):
        example = MapExample(
            input={
                "description": "A Beautiful Mind is a 2001 American biographical drama film about the mathematician John Nash, a Nobel Laureate in Economics, played by Russell Crowe. The film is directed by Ron Howard based on a screenplay by Akiva Goldsman, who adapted the 1998 biography by Sylvia Nasar.",
                "review": "So many things come together so beautifully in this movie based on the life of John Forbes Nash Jr. that you're likely to find yourself willing to benignly overlook its occasional biographical lapses and narrative sweetening.",
            },
            output="The movie's favorable rating is due to how many elements come together so beautifully, creating a compelling and engaging portrayal of John Nash's life. The film's effective storytelling, strong performances, and overall craftsmanship contribute to its positive reception, making viewers willing to overlook minor flaws.",
        )
        sem_map = Map(
            input=pl.Series(name="movie", values=[self.RENDERED_TEMPLATE]),
            jinja_template=self.TEMPLATE,
            model=local_session._session_state.get_language_model(),
            examples=MapExampleCollection(examples=[example]),
            temperature=0,
            max_tokens=512,
        )

        result = list(
            map(
                lambda x: x.to_message_list() if x else None,
                sem_map.build_request_messages_batch(),
            )
        )
        expected = [
            [
                {
                    "content": dedent("""\
                        Follow the user's instruction exactly and generate only the requested output.

                        Requirements:
                        1. Follow the instruction exactly as written
                        2. Output only what is requested - no explanations, no prefixes, no metadata
                        3. Be concise and direct
                        4. Do not add formatting or structure unless explicitly requested"""),
                    "role": "system",
                },
                {
                    "content": dedent("""\
                        Analyze this movie based on its description and the critic review.

                        Movie: A Beautiful Mind is a 2001 American biographical drama film about the mathematician John Nash, a Nobel Laureate in Economics, played by Russell Crowe. The film is directed by Ron Howard based on a screenplay by Akiva Goldsman, who adapted the 1998 biography by Sylvia Nasar.
                        Review: So many things come together so beautifully in this movie based on the life of John Forbes Nash Jr. that you're likely to find yourself willing to benignly overlook its occasional biographical lapses and narrative sweetening.

                        Write a concise summary highlighting the good qualities in this movie that led to a favorable rating."""),
                    "role": "user",
                },
                {
                    "content": example.output,
                    "role": "assistant",
                },
                {
                    "content": dedent("""\
                        Analyze this movie based on its description and the critic review.

                        Movie: Good Will Hunting is a 1997 American drama film directed by Gus Van Sant and written by Ben Affleck and Matt Damon. It stars Robin Williams, Damon, Affleck, Stellan Skarsgård and Minnie Driver. The film tells the story of janitor Will Hunting, whose mathematical genius is discovered by a professor at MIT.
                        Review: When people tumble into love – in Hollywood movies, that is – intelligence is rarely the motivating force that brings them together. Being adorable or eccentric, or having an amazing head of hair – these are the usual qualities that make one flavor of the month hot for another.But in the wonderfully original 'Good Will Hunting', Matt Damon's appeal doesn't spring from good looks, sculpted locks or cover-boy ubiquitousness – although certainly those qualities should haul in the crowds. What counts is his thinking organ. When Will Hunting (Damon) meets Skylar (Minnie Driver), a highly intelligent Harvard student, they waltz on a mental plateau that Julia Roberts and Brad Pitt couldn't reach by cable car.

                        Write a concise summary highlighting the good qualities in this movie that led to a favorable rating."""),

                    "role": "user",
                },
            ]
        ]
        assert result == expected

    def test_map_without_examples(self, local_session):
        sem_map = Map(
            input=pl.Series(name="movie", values=[self.RENDERED_TEMPLATE]),
            jinja_template=self.TEMPLATE,
            model=local_session._session_state.get_language_model(),
            temperature=0,
            max_tokens=512,
        )
        prompts = list(
            map(
                lambda x: x.to_message_list() if x else None,
                sem_map.build_request_messages_batch(),
            )
        )
        expected = [
            [
                {
                    "content": dedent("""\
                        Follow the user's instruction exactly and generate only the requested output.

                        Requirements:
                        1. Follow the instruction exactly as written
                        2. Output only what is requested - no explanations, no prefixes, no metadata
                        3. Be concise and direct
                        4. Do not add formatting or structure unless explicitly requested"""),
                    "role": "system",
                },
                {
                    "content": dedent("""\
                        Analyze this movie based on its description and the critic review.

                        Movie: Good Will Hunting is a 1997 American drama film directed by Gus Van Sant and written by Ben Affleck and Matt Damon. It stars Robin Williams, Damon, Affleck, Stellan Skarsgård and Minnie Driver. The film tells the story of janitor Will Hunting, whose mathematical genius is discovered by a professor at MIT.
                        Review: When people tumble into love – in Hollywood movies, that is – intelligence is rarely the motivating force that brings them together. Being adorable or eccentric, or having an amazing head of hair – these are the usual qualities that make one flavor of the month hot for another.But in the wonderfully original 'Good Will Hunting', Matt Damon's appeal doesn't spring from good looks, sculpted locks or cover-boy ubiquitousness – although certainly those qualities should haul in the crowds. What counts is his thinking organ. When Will Hunting (Damon) meets Skylar (Minnie Driver), a highly intelligent Harvard student, they waltz on a mental plateau that Julia Roberts and Brad Pitt couldn't reach by cable car.

                        Write a concise summary highlighting the good qualities in this movie that led to a favorable rating."""),
                    "role": "user",
                },
            ]
        ]
        assert prompts == expected

    def test_map_with_schema(self, local_session):
        class MovieReviewAnalysis(BaseModel):
            expected_review_score: int = Field(description="1-10 rating of the movie, where 10 is the highest rating")
            movie_themes: list[str] = Field(description="Themes of the movie")
            expected_plot: str = Field(description="Plot summary of the movie")

        template = dedent("""\
            Analyze this movie based on its description and the critic review.

            Movie: {{ description }}
            Review: {{ review }}

            Generate a view analysis report.
        """)

        rendered_template = jinja2.Template(template).render(
            description="Good Will Hunting is a 1997 American drama film directed by Gus Van Sant and written by Ben Affleck and Matt Damon. It stars Robin Williams, Damon, Affleck, Stellan Skarsgård and Minnie Driver. The film tells the story of janitor Will Hunting, whose mathematical genius is discovered by a professor at MIT.",
            review="When people tumble into love – in Hollywood movies, that is – intelligence is rarely the motivating force that brings them together. Being adorable or eccentric, or having an amazing head of hair – these are the usual qualities that make one flavor of the month hot for another.But in the wonderfully original 'Good Will Hunting', Matt Damon's appeal doesn't spring from good looks, sculpted locks or cover-boy ubiquitousness – although certainly those qualities should haul in the crowds. What counts is his thinking organ. When Will Hunting (Damon) meets Skylar (Minnie Driver), a highly intelligent Harvard student, they waltz on a mental plateau that Julia Roberts and Brad Pitt couldn't reach by cable car.",
        )

        sem_map = Map(
            input=pl.Series(name="movie", values=[rendered_template]),
            jinja_template=template,
            model=local_session._session_state.get_language_model(),
            temperature=0,
            max_tokens=512,
            response_format=ResolvedResponseFormat.from_pydantic_model(MovieReviewAnalysis)
        )
        expected = [
            [
                {
                    "content": dedent("""\
                    Follow the user's instruction exactly and generate output according to the user's schema.

                    Output Schema:
                    expected_review_score (int): 1-10 rating of the movie, where 10 is the highest rating
                    movie_themes (list of str): Themes of the movie
                    expected_plot (str): Plot summary of the movie

                    How to read the output schema:
                    - Nested fields are expressed using dot notation (e.g., 'organization.name' means 'name' is a subfield of 'organization')
                    - Lists are denoted using 'list of [type]' (e.g., 'employees' is a list of str)
                    - For lists: 'fieldname[item].subfield' means each item in the list has that subfield
                    - Type annotations are shown in parentheses (e.g., string, integer, boolean, date)
                    - Fields marked (optional) can be omitted if not applicable

                    Requirements:
                    1. Follow the instruction exactly as written
                    2. Generate output that matches the provided schema exactly
                    3. Include all required fields - no extra fields, no missing fields
                    4. Each field's content must match its description precisely"""),
                    "role": "system",
                },
                {
                    "content" : dedent("""\
                        Analyze this movie based on its description and the critic review.

                        Movie: Good Will Hunting is a 1997 American drama film directed by Gus Van Sant and written by Ben Affleck and Matt Damon. It stars Robin Williams, Damon, Affleck, Stellan Skarsgård and Minnie Driver. The film tells the story of janitor Will Hunting, whose mathematical genius is discovered by a professor at MIT.
                        Review: When people tumble into love – in Hollywood movies, that is – intelligence is rarely the motivating force that brings them together. Being adorable or eccentric, or having an amazing head of hair – these are the usual qualities that make one flavor of the month hot for another.But in the wonderfully original 'Good Will Hunting', Matt Damon's appeal doesn't spring from good looks, sculpted locks or cover-boy ubiquitousness – although certainly those qualities should haul in the crowds. What counts is his thinking organ. When Will Hunting (Damon) meets Skylar (Minnie Driver), a highly intelligent Harvard student, they waltz on a mental plateau that Julia Roberts and Brad Pitt couldn't reach by cable car.

                        Generate a view analysis report."""),
                    "role": "user",
                },
            ]
        ]
        prompts = list(
            map(
                lambda x: x.to_message_list() if x else None,
                sem_map.build_request_messages_batch(),
            )
        )
        assert prompts == expected
