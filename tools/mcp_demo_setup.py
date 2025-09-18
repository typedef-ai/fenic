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
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

import fenic as fc
from fenic import IntegerType, OpenAILanguageModel, SemanticConfig, StringType
from fenic.api.functions.core import tool_param
from fenic.core.mcp.types import ToolParam


def main() -> None:
    fc.configure_logging()
    session_config = fc.SessionConfig(
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
    )
    local_session = fc.Session.get_or_create(session_config)
    job_category_list = [
                "Engineering",
                "Software & IT",
                "Data & AI",
                "Sales",
                "Marketing",
                "Customer Success",
                "HR & People Ops",
                "Finance & Accounting",
                "Operations and Supply Chain",
                "Legal and Compliance",
                "Administrative",
                "Executive and General Management",
                "Other"]
    try:
        candidates_df = local_session.read.parquet("s3://typedef-assets/demo/mcp/candidates.parquet")
    except Exception:
        # Synthetic candidate dataset: (candidate_id, candidate_resume)
        raw_candidates = local_session.read.parquet("s3://typedef-assets/demo/mcp/raw_resumes.parquet").limit(1000)

        class CandidateProfile(BaseModel):
            first_name: Optional[str] = Field( description="Candidate's first name.")
            last_name: Optional[str] = Field( description="Candidate's last name.")
            title: Optional[str] = Field( description="Candidate's title. (if provided).")
            pronouns: Optional[str] = Field( description="Candidate's pronouns. (if provided).")
            education: str = Field(description="Degrees or programs")
            seniority: str = Field(description="Likely seniority level, e.g., junior/senior/staff/principal")
            skills: str = Field(description="Notable technical or domain skills")
            experience: str = Field(description="Summary of candidate's work history, with companies, durations, and notable achievements.")
            job_category: List[Literal[
                "Engineering",
                "Software & IT",
                "Data & AI",
                "Sales",
                "Marketing",
                "Customer Success",
                "HR & People Ops",
                "Finance & Accounting",
                "Operations and Supply Chain",
                "Legal and Compliance",
                "Administrative",
                "Executive and General Management",
                "Other"]] = Field(description="Pick between 1 and 3 job categories that describe the candidate's work history.", max_length=3)

        candidates_df = raw_candidates.with_column(
            "profile",
            fc.semantic.extract("candidate_resume", CandidateProfile, max_output_tokens=4096)
        ).unnest("profile").cache()

    candidates_df.write.save_as_table("candidates", mode="overwrite")
    local_session.catalog.set_table_description("candidates", "Resumes for all candidates in our hiring pipeline")
    candidates_df = local_session.table("candidates")
    # Tool 1: search_candidates -- perform a regex search over the candidate education, seniority, skills, experience
    education_match = fc.coalesce(fc.col("education").rlike(tool_param("education_query", StringType)), fc.lit(True))
    seniority_match = fc.coalesce(fc.col("seniority").rlike(tool_param("seniority_query", StringType)), fc.lit(True))
    skills_match = fc.coalesce(fc.col("skills").rlike(tool_param("skills_query", StringType)), fc.lit(True))
    experience_match = fc.coalesce(fc.col("experience").rlike(tool_param("experience_query", StringType)), fc.lit(True))
    job_category_match = fc.coalesce(fc.tool_param("job_category_query", StringType).is_in(fc.col("job_category")), fc.lit(True))
    merged_filter = education_match & seniority_match & skills_match & experience_match & job_category_match
    search_candidates = candidates_df.filter(merged_filter).select("candidate_id", "first_name", "last_name", "education", "seniority", "skills", "experience", "job_category")
    if local_session.catalog.describe_tool("search_candidates"):
        local_session.catalog.drop_tool("search_candidates")
    local_session.catalog.create_tool(
        "search_candidates",
        "Search candidates by education, seniority, skills, and experience using regex patterns.",
        search_candidates,
        tool_params=[
            ToolParam(name="education_query", description="Regex pattern to match against education.", has_default=True, default_value=None),
            ToolParam(name="seniority_query", description="Regex pattern to match against seniority.", has_default=True, default_value=None),
            ToolParam(name="skills_query", description="Regex pattern to match against skills.", has_default=True, default_value=None),
            ToolParam(name="experience_query", description="Regex pattern to match against experience.", has_default=True, default_value=None),
            ToolParam(name="job_category_query", description="Filter candidates by job category.", has_default=True, default_value=None, allowed_values=job_category_list),
        ],
        result_limit=100,
    )

    # Tool 2: candidate_resumes_by_candidate_ids -- given a list of candidate ids, return the raw resumes for each candidate
    candidate_resumes = candidates_df.filter(fc.col("candidate_id").is_in(fc.tool_param("candidate_ids", fc.ArrayType(element_type=IntegerType)))).select("candidate_id", "candidate_resume")
    if local_session.catalog.describe_tool("candidate_resumes_by_candidate_ids"):
        local_session.catalog.drop_tool("candidate_resumes_by_candidate_ids")
    local_session.catalog.create_tool(
        "candidate_resumes_by_candidate_ids",
        "Return the raw resumes for a list of candidate ids.",
        candidate_resumes,
        tool_params=[ToolParam(name="candidate_ids", description="List of candidate ids to return the resumes for.")],
        result_limit=100,
    )

    # Tool 3: candidates_for_job_description — filter by free-form job description
    # We evaluate candidates by referencing structured profile fields in a predicate.
    fit_pred = fc.semantic.predicate(
        textwrap.dedent(
            """\
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
            
            JOB DESCRIPTION
            {{job}}
            
            CANDIDATE PROFILE
            Seniority Level: {{seniority}}
            Skills: {{skills}}
            Education: {{education}}
            Experience: {{experience}}
            Job Category: {{job_category}}
            """
        ),
        job=fc.tool_param("job_description", StringType),
        seniority=fc.col("seniority"),
        skills=fc.col("skills"),
        education=fc.col("education"),
        experience=fc.col("experience"),
        job_category=fc.col("job_category"),
        strict=False,
        model_alias="gpt-4.1-mini",
    )
    job_category_match = fc.coalesce(fc.tool_param("job_category_query", StringType).is_in(fc.col("job_category")), fc.lit(True))
    candidates_for_job = candidates_df.filter(job_category_match).filter(fit_pred).select("candidate_id", "first_name", "last_name", "education", "seniority", "skills", "experience", "job_category")
    if local_session.catalog.describe_tool("candidates_for_job_description"):
        local_session.catalog.drop_tool("candidates_for_job_description")
    local_session.catalog.create_tool(
        "candidates_for_job_description",
        "Find candidates who are a good fit for a free-form job description using structured profiles.",
        candidates_for_job,
        tool_params=[
            ToolParam(name="job_description",
                      description="Free-form job description text to match candidates against."),
            ToolParam(name="job_category_query", description="Filter candidates by job category.", has_default=True, default_value=None, allowed_values=job_category_list),
        ],
        result_limit=100,
    )

    # Tool 4: create_outreach_for_candidate — personalize a recruiting email at runtime
    # Include resume + optional cover letter as rich context for personalization.
    outreach_email = candidates_df.filter(
        fc.col("candidate_id").is_in(fc.tool_param("candidate_ids", fc.ArrayType(element_type=IntegerType))),
    ).select(
        fc.col("candidate_id"),
        fc.semantic.map(
            textwrap.dedent(
                """\
                You are a recruiter writing to {{candidate_id}}.
                Use the candidate's resume (and cover letter if present) to personalize the email with empathy and understanding of the candidate's background and motivations.
                Company: {{company}}
                Job Title: {{job_title}}
                Job Description: {{job_description}}
                Recruiter: {{recruiter_name}}
                Why Join: {{why_join}}
                Style/Tone Instructions: {{instructions}}

                CANDIDATE RESUME:\n{{resume}}
                \n\n
                CANDIDATE PROFILE:\n
                Name: {{first_name}} {{last_name}}
                Pronouns: {{pronouns}}
                Title: {{title}}
                Seniority Level: {{seniority}}
                Skills: {{skills}}
                Education: {{education}}
                Experience: {{experience}}
                Job Category: {{job_category}}

                Write the email with a short subject line and a body under ~150 words.
                Avoid generic phrasing; reference specific details from the resume.
                """
            ),
            candidate_id=fc.col("candidate_id"),
            resume=fc.col("candidate_resume"),
            first_name=fc.col("first_name"),
            last_name=fc.col("last_name"),
            pronouns=fc.col("pronouns"),
            seniority=fc.col("seniority"),
            skills=fc.col("skills"),
            education=fc.col("education"),
            experience=fc.col("experience"),
            job_category=fc.col("job_category"),
            title=fc.col("title"),
            company=fc.tool_param("company", StringType),
            job_title=fc.tool_param("job_title", StringType),
            job_description=fc.tool_param("job_description", StringType),
            recruiter_name=fc.tool_param("recruiter_name", StringType),
            why_join=fc.tool_param("why_join", StringType),
            instructions=fc.tool_param("instructions", StringType),
            strict=False,
            max_output_tokens=320,
            model_alias="gpt-5-mini"
        ).alias("email"),
    )
    # Filter to a single candidate_id at runtime
    if local_session.catalog.describe_tool("create_outreach_for_candidate"):
        local_session.catalog.drop_tool("create_outreach_for_candidate")
    local_session.catalog.create_tool(
        "create_outreach_for_candidate",
        "Create a personalized recruiting email for a candidate using resume and cover letter context.",
        outreach_email,
        tool_params=[
            ToolParam(name="candidate_ids", description="IDs of the candidate(s) for which to generate outreach emails, e.g., [123456, 423512]"),
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
            )
        ],
    )

    # Launch MCP server with our custom tools, along with the auto-generated tools.
    # server = fc.create_mcp_server(
    #     session=local_session,
    #     server_name="Fenic Semantic Demo",
    #     tools=local_session.catalog.list_tools(),
    #     automated_tool_generation=fc.ToolGenerationConfig(
    #        table_names=["candidates"],
    #        tool_group_name="Candidate Information"
    #     )
    # )
    # fc.run_mcp_server_sync(server)

    # Testing an issue with event loop shutdown
    local_session.stop()



if __name__ == "__main__":
    main()
