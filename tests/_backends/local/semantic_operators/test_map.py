import polars as pl

from fenic import MapExample, MapExampleCollection
from fenic._backends.local.semantic_operators.map import Map


class TestMap:
    """Test cases for the Map operator."""

    def test_build_prompts(self, local_session):
        instruction = "Given a movie's description: {description} and the critic review: {review}, summarize the good qualities in this movie that led to a favorable rating."
        source_df = pl.DataFrame(
            {
                "description": [
                    "Good Will Hunting is a 1997 American drama film directed by Gus Van Sant and written by Ben Affleck and Matt Damon. It stars Robin Williams, Damon, Affleck, Stellan Skarsgård and Minnie Driver. The film tells the story of janitor Will Hunting, whose mathematical genius is discovered by a professor at MIT.",
                ],
                "review": [
                    "When people tumble into love – in Hollywood movies, that is – intelligence is rarely the motivating force that brings them together. Being adorable or eccentric, or having an amazing head of hair – these are the usual qualities that make one flavor of the month hot for another."
                    "But in the wonderfully original 'Good Will Hunting', Matt Damon's appeal doesn't spring from good looks, sculpted locks or cover-boy ubiquitousness – although certainly those qualities should haul in the crowds. What counts is his thinking organ. When Will Hunting (Damon) meets Skylar (Minnie Driver), a highly intelligent Harvard student, they waltz on a mental plateau that Julia Roberts and Brad Pitt couldn't reach by cable car.",
                ],
            }
        )
        example = MapExample(
            input={
                "description": "A Beautiful Mind is a 2001 American biographical drama film about the mathematician John Nash, a Nobel Laureate in Economics, played by Russell Crowe. The film is directed by Ron Howard based on a screenplay by Akiva Goldsman, who adapted the 1998 biography by Sylvia Nasar.",
                "review": "So many things come together so beautifully in this movie based on the life of John Forbes Nash Jr. that you're likely to find yourself willing to benignly overlook its occasional biographical lapses and narrative sweetening.",
            },
            output="The movie's favorable rating is due to how many elements come together so beautifully, creating a compelling and engaging portrayal of John Nash's life. The film's effective storytelling, strong performances, and overall craftsmanship contribute to its positive reception, making viewers willing to overlook minor flaws.",
        )
        sem_map = Map(
            input=source_df,
            user_instruction=instruction,
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
                    "content": sem_map.SYSTEM_PROMPT,
                    "role": "system",
                },
                {
                    "content": f"### Instruction\n"
                    "Given a movie's description: [DESCRIPTION] and the critic "
                    "review: [REVIEW], summarize the good qualities in "
                    "this movie that led to a favorable rating.\n"
                    "\n"
                    "### Context\n"
                    f"[DESCRIPTION]: «{example.input['description']}»\n"
                    f"[REVIEW]: «{example.input['review']}»",
                    "role": "user",
                },
                {
                    "content": example.output,
                    "role": "assistant",
                },
                {
                    "content": "### Instruction\n"
                    "Given a movie's description: [DESCRIPTION] and the critic "
                    "review: [REVIEW], summarize the good qualities in "
                    "this movie that led to a favorable rating.\n"
                    "\n"
                    "### Context\n"
                    "[DESCRIPTION]: «Good Will Hunting is a 1997 American drama film "
                    "directed by Gus Van Sant and written by Ben Affleck and Matt "
                    "Damon. It stars Robin Williams, Damon, Affleck, Stellan "
                    "Skarsgård and Minnie Driver. The film tells the story of "
                    "janitor Will Hunting, whose mathematical genius is discovered "
                    "by a professor at MIT.»\n"
                    "[REVIEW]: «When people tumble into love – in Hollywood movies, "
                    "that is – intelligence is rarely the motivating force that "
                    "brings them together. Being adorable or eccentric, or having an "
                    "amazing head of hair – these are the usual qualities that make "
                    "one flavor of the month hot for another.But in the wonderfully "
                    "original 'Good Will Hunting', Matt Damon's appeal doesn't "
                    "spring from good looks, sculpted locks or cover-boy "
                    "ubiquitousness – although certainly those qualities should haul "
                    "in the crowds. What counts is his thinking organ. When Will "
                    "Hunting (Damon) meets Skylar (Minnie Driver), a highly "
                    "intelligent Harvard student, they waltz on a mental plateau "
                    "that Julia Roberts and Brad Pitt couldn't reach by cable car.»",
                    "role": "user",
                },
            ]
        ]
        assert result == expected

    def test_map_without_examples(self, local_session):
        instruction = "Given a movie's description: {description} and the critic review: {review}, summarize the good qualities in this movie that led to a favorable rating."
        source_df = pl.DataFrame(
            {
                "description": [
                    "A Beautiful Mind is a 2001 American biographical drama film about the mathematician John Nash, a Nobel Laureate in Economics, played by Russell Crowe. The film is directed by Ron Howard based on a screenplay by Akiva Goldsman, who adapted the 1998 biography by Sylvia Nasar.",
                ],
                "review": [
                    "So many things come together so beautifully in this movie based on the life of John Forbes Nash Jr. that you're likely to find yourself willing to benignly overlook its occasional biographical lapses and narrative sweetening.",
                ],
            }
        )
        sem_map = Map(
            input=source_df,
            user_instruction=instruction,
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
                    "content": sem_map.SYSTEM_PROMPT,
                    "role": "system",
                },
                {
                    "content": "### Instruction\n"
                    "Given a movie's description: [DESCRIPTION] and the critic "
                    "review: [REVIEW], summarize the good qualities in "
                    "this movie that led to a favorable rating.\n"
                    "\n"
                    "### Context\n"
                    "[DESCRIPTION]: «A Beautiful Mind is a 2001 American "
                    "biographical drama film about the mathematician John Nash, a "
                    "Nobel Laureate in Economics, played by Russell Crowe. The film "
                    "is directed by Ron Howard based on a screenplay by Akiva "
                    "Goldsman, who adapted the 1998 biography by Sylvia Nasar.»\n"
                    "[REVIEW]: «So many things come together so beautifully in this "
                    "movie based on the life of John Forbes Nash Jr. that you're "
                    "likely to find yourself willing to benignly overlook its "
                    "occasional biographical lapses and narrative sweetening.»",
                    "role": "user",
                },
            ]
        ]
        assert prompts == expected
