"""Demo: Candidate hunting with Fenic + MCP.

Scenario:
- Dataset contains free-form resumes and optional cover letters
- We pre-extract a structured profile from each resume using semantic.extract
- Tools:
  1) candidates_for_job_description(job_description): predicate over structured profile to find good-fit candidates
  2) create_outreach_for_candidate(candidate_id, tone, company, job_title, recruiter_name, why_join, instructions?):
     generate a personalized outreach email using resume + cover letter context

Run:
- pip install "fenic[mcp]"  # or: pip install fastmcp
- uv run python tools/mcp_demo.py
"""

import textwrap
from typing import Annotated, Optional

from pydantic import BaseModel, Field

import fenic as fc
from fenic import IntegerType, OpenAILanguageModel, SemanticConfig, StringType
from fenic.api.functions import tool_param
from fenic.api.mcp import ToolGenerationConfig, create_mcp_server, run_mcp_server_sync
from fenic.api.tools import DatasetSpec, dynamic_tool_from_df
from fenic.core._logical_plan.tools import ToolParam
from fenic.core.error import PlanError


def main() -> None:
    fc.configure_logging()
    local_session = fc.Session.get_or_create(fc.SessionConfig(
        app_name="mcp_demo",
        semantic=SemanticConfig(
            language_models={
                "gpt-4.1-nano": OpenAILanguageModel(
                    model_name="gpt-4.1-nano",
                    rpm=2500,
                    tpm=2_000_000
                ),
                "gpt-4.1-mini": OpenAILanguageModel(
                    model_name="gpt-4.1-mini",
                    rpm=2500,
                    tpm=2_000_000
                ),
                "gpt-5-nano": OpenAILanguageModel(
                    model_name="gpt-5-nano",
                    rpm=2500,
                    tpm=2_000_000,
                    profiles={"default" : OpenAILanguageModel.Profile(
                        reasoning_effort="minimal"
                    )}
                ),

                "gpt-5-mini": OpenAILanguageModel(
                    model_name="gpt-5-mini",
                    rpm=2500,
                    tpm=2_000_000,
                    profiles={"default": OpenAILanguageModel.Profile(
                        reasoning_effort="minimal"
                    )}
                )
            },
            default_language_model="gpt-4.1-nano",
        )
    ))

    try:
        candidates_df = local_session.read.parquet("./data/candidates.parquet")
    except PlanError:
        # Synthetic candidate dataset: (candidate_id, candidate_resume)
        candidates_df = local_session.read.csv("./data/resume_raw.csv").limit(1000)

        class CandidateProfile(BaseModel):
            first_name: str = Field( description="Candidate's first name.")
            last_name: str = Field( description="Candidate's last name.")
            title: Optional[str] = Field( description="Candidate's title. (if provided).")
            pronouns: Optional[str] = Field( description="Candidate's pronouns. (if provided).")
            education: str = Field(description="Degrees or programs")
            seniority: str = Field(description="Likely seniority level, e.g., junior/senior/staff/principal")
            skills: str = Field(description="Notable technical or domain skills")
            experience: str = Field(description="Summary of candidate's work history, with companies, durations, and ")

        candidates_df = candidates_df.with_column(
            "profile",
            fc.semantic.extract("candidate_resume", CandidateProfile, max_output_tokens=4096)
        )
        # Materialize the enriched dataframe
        candidates_df.write.parquet("./data/candidates.parquet")

    # Tool 1: candidates_for_job_description — filter by free-form job description
    # We evaluate candidates by referencing structured profile fields in a predicate.
    fit_pred = fc.semantic.predicate(
        textwrap.dedent(
            """\
            Job Description
            {{job}}
            Candidate Profile

            Seniority Level: {{profile.seniority}}
            Skills: {{profile.skills}}
            Education: {{profile.education}}
            Experience: {{profile.experience}}

            Evaluation Instructions
            Assess this candidate's fit for the role based on the following criteria:
            1. Skills Match

            Does the candidate possess the required technical skills?
            Are their transferable skills relevant to the role requirements?
            What skill gaps exist, if any?

            2. Experience Relevance

            Is their work experience directly applicable to this position?
            Have they handled similar responsibilities or projects?
            Does their career progression align with the role's expectations?

            3. Education Alignment

            Does their educational background meet the minimum requirements?
            Are there any preferred qualifications they possess?
            Do certifications or continuing education demonstrate commitment to the field?

            4. Seniority Level Compatibility

            Is the candidate's current level appropriate for this role?
            Critical consideration: Senior-level candidates are unlikely to accept roles significantly below their current level unless there are compelling reasons (career change, work-life balance, geographic preferences, company prestige, etc.)
            Would this represent a step up, lateral move, or step down for the candidate?

            5. Overall Assessment
            Based on the above factors, determine:

            Recommendation: Should we pursue this candidate?
            """
        ),
        job=tool_param("job_description", StringType),
        profile=fc.col("profile"),
        strict=False,
        model_alias="gpt-4.1-mini",
    )
    candidates_for_job = candidates_df.filter(fit_pred).select(
        fc.col("candidate_id"),
        fc.col("candidate_resume"),
        fc.col("profile"),
    )
    local_session.catalog.create_tool(
        "candidates_for_job_description",
        "Find candidates who are a good fit for a free-form job description using structured profiles.",
        candidates_for_job,
        tool_params=[
            ToolParam(name="job_description",
                      description="Free-form job description text to match candidates against."),
        ],
    )

    # Tool 2: create_outreach_for_candidate — personalize a recruiting email at runtime
    # Include resume + optional cover letter as rich context for personalization.
    outreach_email = candidates_df.filter(
        fc.col("candidate_id").is_in(tool_param("candidate_ids", fc.ArrayType(element_type=IntegerType))),
    ).select(
        fc.col("candidate_id"),
        fc.semantic.map(
            textwrap.dedent(
                """\
                You are a recruiter writing to {{candidate_id}}.
                Use the candidate's resume (and cover letter if present) to personalize the email.
                Company: {{company}}
                Job Title: {{job_title}}
                Job Description: {{job_description}}
                Recruiter: {{recruiter_name}}
                Why Join: {{why_join}}
                Tone: {{tone}}
                Extra Instructions: {{instructions}}

                Candidate Resume:\n{{resume}}
                \n\n
                Candidate Profile:\n {{profile}}

                Write the email with a short subject line and a body under ~150 words.
                Avoid generic phrasing; reference specific details from the resume.
                """
            ),
            candidate_id=fc.col("candidate_id"),
            resume=fc.col("candidate_resume"),
            profile=fc.col("profile"),
            company=tool_param("company", StringType),
            job_title=tool_param("job_title", StringType),
            job_description=tool_param("job_description", StringType),
            recruiter_name=tool_param("recruiter_name", StringType),
            why_join=tool_param("why_join", StringType),
            instructions=tool_param("instructions", StringType),
            tone=tool_param("tone", StringType),
            strict=False,
            temperature=0.8,
            max_output_tokens=320,
            model_alias="gpt-5-mini"
        ).alias("email"),
    )
    # Filter to a single candidate_id at runtime
    local_session.catalog.create_tool(
        "create_outreach_for_candidate",
        "Create a personalized recruiting email for a candidate using resume and cover letter context.",
        outreach_email,
        tool_params=[
            ToolParam(name="candidate_ids", description="IDs of the candidate(s) for which to generate outreach emails, e.g., [123456, 423512]"),
            ToolParam(
                name="tone",
                description="One word writing tone to use (e.g., friendly, formal, concise).",
                has_default=True,
                default_value="friendly",
            ),
            ToolParam(name="company", description="Your company name."),
            ToolParam(name="job_title", description="The job title being offered."),
            ToolParam(name="job_description", description="The job description being offered."),
            ToolParam(name="recruiter_name", description="Your name for the signature."),
            ToolParam(name="why_join", description="A sentence about why the candidate should join."),
            ToolParam(
                name="instructions",
                description="Optional style/formatting instructions.",
                has_default=True,
                default_value="",
            ),
        ],
    )

    # Launch MCP server with our custom tools, along with the auto-generated tools.
    server = create_mcp_server(
        session=local_session,
        server_name="Fenic Semantic Demo",
        tools=local_session.catalog.list_tools(),
        automated_tool_generation=ToolGenerationConfig(
            datasets=[
                DatasetSpec(df=candidates_df, name="candidates", description="The candidates in the hiring pipeline")],
            tool_group_name="Candidate Information"
        )
    )
    run_mcp_server_sync(server)



if __name__ == "__main__":
    main()
