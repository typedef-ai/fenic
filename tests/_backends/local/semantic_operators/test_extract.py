from textwrap import dedent
from typing import List, Literal, Optional

import polars as pl
from pydantic import BaseModel, Field

from fenic._backends.local.semantic_operators.extract import Extract
from fenic.core._logical_plan.resolved_types import ResolvedResponseFormat


class WorkExperience(BaseModel):
    company: str = Field(description="Name of the company")
    title: str = Field(description="Job title held")
    start_year: int = Field(description="Year the job started")
    end_year: Optional[int] = Field(None, description="Year the job ended (if applicable)")
    description: Optional[str] = Field(None, description="Short description of responsibilities")
    employment_type: Optional[Literal["full-time", "part-time", "contractor"]] = Field(
        None, description="Type of employment"
    )

class Education(BaseModel):
    institution: str = Field(description="Name of the educational institution")
    major: str = Field(description="Field of study")
    graduation_year: Optional[int] = Field(None, description="Year of graduation")

class Resume(BaseModel):
    name: str = Field(description="Full name of the candidate")
    work_experience: List[WorkExperience] = Field(description="List of work experiences")
    skills: List[str] = Field(description="List of individual skills mentioned")
    education: Optional[Education] = Field(None, description="Education details")

class TestExtract:
    """Test cases for the Extract operator."""

    def test_build_prompts(self, local_session):
        jane_resume = dedent(
            """\
            Jane Doe is a software engineer with over 6 years of experience. She worked at OpenAI from 2021 to 2024 as a full-time Machine Learning Engineer, focusing on NLP research.
            Before that, she was at Google as a full-time Software Engineer from 2018 to 2021, building distributed systems.

            She is skilled in Python, PyTorch, and distributed computing. Also familiar with Rust and Kubernetes.

            She graduated from MIT in 2017 with a degree in Computer Science."""
        ).strip()

        input = pl.Series("input", [jane_resume])

        extract = Extract(
            input=input,
            response_format=ResolvedResponseFormat.from_pydantic_model(Resume),
            model=local_session._session_state.get_language_model(),
            max_output_tokens=1024,
            temperature=0,
        )

        result = list(
            map(
                lambda x: x.to_message_list() if x else None,
                extract.build_request_messages_batch(),
            )
        )

        expected = [[
            {
                "role": "system",
                "content": dedent("""\
                    Extract information from the document according to the output schema.

                    Output Schema:
                    name (str): Full name of the candidate
                    work_experience (list of objects): List of work experiences
                    work_experience[item].company (str): Name of the company
                    work_experience[item].title (str): Job title held
                    work_experience[item].start_year (int): Year the job started
                    work_experience[item].end_year (int (optional)): Year the job ended (if applicable)
                    work_experience[item].description (str (optional)): Short description of responsibilities
                    work_experience[item].employment_type ('full-time' or 'part-time' or 'contractor' (optional)): Type of employment
                    skills (list of str): List of individual skills mentioned
                    education (object (optional)): Education details
                    education.institution (str): Name of the educational institution
                    education.major (str): Field of study
                    education.graduation_year (int (optional)): Year of graduation

                    How to read the output schema:
                    - Nested fields are expressed using dot notation (e.g., 'organization.name' means 'name' is a subfield of 'organization')
                    - Lists are denoted using 'list of [type]' (e.g., 'employees' is a list of str)
                    - For lists: 'fieldname[item].subfield' means each item in the list has that subfield
                    - Type annotations are shown in parentheses (e.g., string, integer, boolean, date)
                    - Fields marked (optional) can be omitted if not applicable

                    Requirements:
                    1. Extract only information explicitly stated in the document
                    2. Do not infer, guess, or generate information not present
                    3. Include all required fields - no extra fields, no missing fields
                    4. For list fields, extract all items that match the field description
                    5. Be thorough and precise - capture all relevant content without changing meaning""").strip(),
            },
            {
                "role": "user",
                "content": dedent(
                    """\
                    Jane Doe is a software engineer with over 6 years of experience. She worked at OpenAI from 2021 to 2024 as a full-time Machine Learning Engineer, focusing on NLP research.
                    Before that, she was at Google as a full-time Software Engineer from 2018 to 2021, building distributed systems.

                    She is skilled in Python, PyTorch, and distributed computing. Also familiar with Rust and Kubernetes.

                    She graduated from MIT in 2017 with a degree in Computer Science."""
                ),
            },
        ]]

        assert result == expected
