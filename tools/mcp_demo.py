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

from pydantic import BaseModel, Field

import fenic as fc
from fenic import OpenAILanguageModel, SemanticConfig, StringType
from fenic.api.functions import tool_param
from fenic.api.mcp import create_mcp_server, run_mcp_server_sync
from fenic.core._logical_plan.tools import ToolParam


def main() -> None:
    fc.configure_logging()
    local_session = fc.Session.get_or_create(fc.SessionConfig(
        app_name="mcp_demo",
        semantic=SemanticConfig(
            language_models={
                "nano": OpenAILanguageModel(
                    model_name="gpt-4.1-nano",
                    rpm=10_000,
                    tpm=10_000_000
                )
            }
        )
    ))
    # Synthetic candidate dataset: (candidate_id, candidate_resume)
    candidates_df = local_session.read.csv("./data/resume_raw.csv")

    # Extract structured profile from resumes for filtering and routing
    class Experience(BaseModel):
        company: str = Field(description="Company name")
        title: str = Field(description="Job title")
        highlights: list[str] = Field(description="Key bullet points or notable work")

    class CandidateProfile(BaseModel):
        education: list[str] = Field(description="Degrees or programs")
        seniority: str = Field(description="Likely seniority level, e.g., junior/senior/staff/principal")
        fit_roles: list[str] = Field(description="Suitable role families, e.g., 'platform', 'ml', 'frontend'")
        skills: list[str] = Field(description="Notable technical or domain skills")
        experience: list[Experience] = Field(description="Work history", max_length=5)

    enriched = candidates_df.select(
        fc.col("candidate_id"),
        fc.col("candidate_resume"),
        fc.semantic.extract("candidate_resume", CandidateProfile, max_output_tokens=4096).alias("profile"),
    ).cache()

    # Materialize the enriched dataframe
    enriched_profile_count = enriched.count()
    print(f"Enriched profile count: {enriched_profile_count}")

    # Tool 1: candidates_for_job_description — filter by free-form job description
    # We evaluate candidates by referencing structured profile fields in a predicate.
    fit_pred = fc.semantic.predicate(
        textwrap.dedent(
            """\
            Job Description: {{job}}
            Candidate Profile:
              Seniority: {{profile.seniority}}
              Fit Roles: {{profile.fit_roles}}
              Skills: {{profile.skills}}
              Education: {{profile.education}}
              Experience: {{profile.experience}}
            This candidate is a good fit for the job description.
            """
        ),
        job=tool_param("job_description", StringType),
        profile=fc.col("profile"),
    )
    candidates_for_job = enriched.filter(fit_pred).select(
        fc.col("candidate_id"),
        fc.col("candidate_resume"),
        fc.col("profile"),
    )
    local_session.catalog.create_tool(
        "candidates_for_job_description",
        "Find candidates who are a good fit for a free-form job description using structured profiles.",
        candidates_for_job,
        tool_params=[
            ToolParam(name="job_description", description="Free-form job description text to match candidates against."),
        ],
    )

    # Tool 2: create_outreach_for_candidate — personalize a recruiting email at runtime
    # Include resume + optional cover letter as rich context for personalization.
    outreach_plan = enriched.select(
        fc.col("candidate_id"),
        fc.semantic.map(
            textwrap.dedent(
                """\
                You are a recruiter writing to {{candidate_id}}.
                Use the candidate's resume (and cover letter if present) to personalize the email.
                Company: {{company}}
                Job Title: {{job_title}}
                Recruiter: {{recruiter_name}}
                Why Join: {{why_join}}
                Tone: {{tone}}
                Extra Instructions: {{instructions}}

                Candidate Resume:\n{{resume}}
                Candidate Cover Letter (may be empty):\n{{cover_letter}}

                Write the email with a short subject line and a body under ~150 words.
                Avoid generic phrasing; reference specific details from the resume/letter.
                """
            ),
            candidate_id=fc.col("candidate_id"),
            resume=fc.col("resume"),
            cover_letter=fc.col("cover_letter"),
            company=tool_param("company", StringType),
            job_title=tool_param("job_title", StringType),
            recruiter_name=tool_param("recruiter_name", StringType),
            why_join=tool_param("why_join", StringType),
            instructions=tool_param("instructions", StringType),
            tone=tool_param("tone", StringType),
            temperature=0.7,
            max_output_tokens=320,
        ).alias("email"),
    )
    # Filter to a single candidate_id at runtime
    outreach_filtered = outreach_plan.filter(
        fc.col("candidate_id") == tool_param("candidate_id", StringType)
    )
    local_session.catalog.create_tool(
        "create_outreach_for_candidate",
        "Create a personalized recruiting email for a candidate using resume and cover letter context.",
        outreach_filtered,
        tool_params=[
            ToolParam(name="candidate_id", description="ID of the candidate, e.g., CAND-001"),
            ToolParam(
                name="tone",
                description="Writing tone to use (e.g., friendly, formal, concise).",
                has_default=True,
                default_value="friendly",
                allowed_values=["friendly", "formal", "concise"],
            ),
            ToolParam(name="company", description="Your company name."),
            ToolParam(name="job_title", description="The job title being offered."),
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

    # Launch MCP server with only our custom tools
    tools = local_session.catalog.list_tools()
    server = create_mcp_server(
        session=local_session,
        server_name="Fenic Semantic Demo",
        tools=tools,
    )
    run_mcp_server_sync(server)


if __name__ == "__main__":
    main()
