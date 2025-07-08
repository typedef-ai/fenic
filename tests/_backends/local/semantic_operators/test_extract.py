from textwrap import dedent
from typing import List, Literal, Optional

import polars as pl
from pydantic import BaseModel, Field

from fenic._backends.local.semantic_operators.extract import Extract


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
            schema=Resume,
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
                "content": (
                    "You are an expert at structured data extraction. Your task is to extract relevant information from a given document using only the information explicitly stated in the text. You must adhere strictly to the provided field definitions. Do not infer or generate information that is not directly supported by the document.\n\n"
                    "The field schema below defines the structure of the information you are expected to extract.\n\n"
                    "How to read the field schema:\n"
                    "- Nested fields are expressed using dot notation (e.g., 'organization.name' means 'name' is a subfield of 'organization')\n"
                    "- Lists are denoted using 'list of [type]' (e.g., 'employees' is a list of [string])\n"
                    "- Type annotations are shown in parentheses (e.g., string, integer, boolean, date)\n\n"
                    "Extraction Guidelines:\n"
                    "1. Extract only what is explicitly present or clearly supported in the document—do not guess or extrapolate.\n"
                    "2. For list fields, extract all items that match the field description.\n"
                    "3. If a field is not found in the document, return null for single values and [] for lists.\n"
                    "4. Ensure all field names in your structured output exactly match the field schema.\n"
                    "5. Be thorough and precise—capture all relevant content without changing or omitting meaning.\n\n"
                    "Field Schema:\n"
                    "name (str): Full name of the candidate\n"
                    "work_experience (list of objects): List of work experiences\n"
                    "work_experience[item].company (str): Name of the company\n"
                    "work_experience[item].title (str): Job title held\n"
                    "work_experience[item].start_year (int): Year the job started\n"
                    "work_experience[item].end_year (int (optional)): Year the job ended (if applicable)\n"
                    "work_experience[item].description (str (optional)): Short description of responsibilities\n"
                    "work_experience[item].employment_type ('full-time' or 'part-time' or 'contractor' (optional)): Type of employment\n"
                    "skills (list of str): List of individual skills mentioned\n"
                    "education (object (optional)): Education details\n"
                    "education.institution (str): Name of the educational institution\n"
                    "education.major (str): Field of study\n"
                    "education.graduation_year (int (optional)): Year of graduation"
                ),
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
