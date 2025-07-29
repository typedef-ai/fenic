from textwrap import dedent

import polars as pl
from pydantic import BaseModel, Field

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
                    "content": "You are an AI assistant designed to follow instructions. "
                               "Your task is to generate responses based on instructions that reference one or more "
                               "context fields. Each input message will have two sections:\n1. An instruction labeled "
                               "with the prefix: ###Instruction\n2. One or more context fields labeled with the prefix:"
                               " ###Context\nThe instruction will reference the context fields using square brackets [LIKE_THIS]. "
                               "Each context field will be labeled with its name in square brackets, matching the references in the instruction. "
                               "Your response should fulfill the instruction by appropriately integrating each of the referenced context fields without using any external information. "
                               "Your response should not include unnecessary preamble or explanation.\n",
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

    def test_map_with_schema(self, local_session):
        instruction = "Given a movie's description: {description} and the critic review: {review}, generate a review analysis report using the schema provided."
        class MovieReviewAnalysis(BaseModel):
            expected_review_score: int = Field(description="Based on the provided review, what score would the reviewer likely leave out of 10", ge=0, le=10)
            movie_themes: list[str] = Field(description="Based on the provided description and review, what are the probable themes of the film?")
            expected_plot: str = Field(description="Based on the provided description and review, what is a probable plot summary of the film?")

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
            response_format=MovieReviewAnalysis
        )
        expected = [
            [
                {
                    "content": str.strip(dedent("""\
                    You are an AI assistant designed to follow instructions. Your task is to generate structured responses using the provided schema, based on instructions that reference one or more context fields. Each input message will have two sections:
                    1. An instruction labeled with the prefix: ###Instruction
                    2. One or more context fields labeled with the prefix: ###Context
                    The instruction will reference the context fields using square brackets [LIKE_THIS]. Each context field will be labeled with its name in square brackets, matching the references in the instruction. Your response should fulfill the instruction by appropriately integrating each of the referenced context fields without using any external information. Your response should not include unnecessary preamble or explanation.
                    Output Guidelines:
                    1. Generate output that matches the field descriptions exactly.
                    2. For list fields, include all relevant items that match the field description.
                    3. Ensure all field names in your structured output exactly match the field schema.
                    4. Use the field descriptions as guidance for what content to generate for each field.
                    
                    How to read the field schema:
                    - Nested fields are expressed using dot notation (e.g., 'organization.name' means 'name' is a subfield of 'organization')
                    - Lists are denoted using 'list of [type]' (e.g., 'employees' is a list of [string])
                    - Type annotations are shown in parentheses (e.g., string, integer, boolean, date)
                    
                    Field Schema:
                    expected_review_score (int): Based on the provided review, what score would the reviewer likely leave out of 10
                    movie_themes (list of str): Based on the provided description and review, what are the probable themes of the film?
                    expected_plot (str): Based on the provided description and review, what is a probable plot summary of the film?
                    """)),
                    "role": "system",
                },
                {
                    "content" : str.strip(dedent("""
                    ### Instruction
                    Given a movie's description: [DESCRIPTION] and the critic review: [REVIEW], generate a review analysis report using the schema provided.

                    ### Context
                    [DESCRIPTION]: «A Beautiful Mind is a 2001 American biographical drama film about the mathematician John Nash, a Nobel Laureate in Economics, played by Russell Crowe. The film is directed by Ron Howard based on a screenplay by Akiva Goldsman, who adapted the 1998 biography by Sylvia Nasar.»
                    [REVIEW]: «So many things come together so beautifully in this movie based on the life of John Forbes Nash Jr. that you're likely to find yourself willing to benignly overlook its occasional biographical lapses and narrative sweetening.»
                    """)),
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
                    "content": "You are an AI assistant designed to follow instructions. "
                               "Your task is to generate responses based on instructions that reference one or more "
                               "context fields. Each input message will have two sections:\n1. An instruction labeled "
                               "with the prefix: ###Instruction\n2. One or more context fields labeled with the prefix:"
                               " ###Context\nThe instruction will reference the context fields using square brackets [LIKE_THIS]. "
                               "Each context field will be labeled with its name in square brackets, matching the references in the instruction. "
                               "Your response should fulfill the instruction by appropriately integrating each of the referenced context fields without using any external information. "
                               "Your response should not include unnecessary preamble or explanation.\n",
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
