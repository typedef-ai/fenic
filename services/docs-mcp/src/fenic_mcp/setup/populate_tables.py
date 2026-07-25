#!/usr/bin/env python
"""Populate the documentation tables required for the MCP server.

This script creates the API, hierarchy, summary, serving context, and release
provenance tables.
"""

import logging
import os
import textwrap
from importlib.metadata import version as package_version
from typing import Any, Dict, List

import griffe

import fenic as fc
from fenic.api.dataframe import DataFrame
from fenic_mcp.server.native import register_docs_tools
from fenic_mcp.server.utils.session import log_fenic_version
from fenic_mcp.server.utils.tree_operations import build_tree, tree_to_string

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _setup_session(work_dir: str) -> fc.Session:
    # Use the same directory setup as the MCP server
    logger.info("Setting up session...")
    os.chdir(work_dir)
    # Configure fenic session (same as MCP server)
    config = fc.SessionConfig(
        app_name="docs",
        semantic=fc.SemanticConfig(
            llm_response_cache=fc.LLMResponseCacheConfig(ttl="30d"),
            language_models={
                "mini": fc.OpenAILanguageModel(
                    model_name="gpt-5.4-mini",
                    rpm=2000,
                    tpm=10_000_000,
                ),
            },
            default_language_model="mini",
        ),
    )

    session = fc.Session.get_or_create(config)
    return session


def _load_fenic_api():
    """Load the Fenic API using Griffe."""
    logger.info("Loading Fenic API with Griffe...")
    # Load fenic API using Griffe
    loader = griffe.GriffeLoader()
    fenic_api = loader.load("fenic")
    return fenic_api


def _extract_api_elements(
    module: griffe.Module, parent_path: str = ""
) -> List[Dict[str, Any]]:
    """Extract API elements from a module recursively."""
    elements: List[Dict[str, Any]] = []
    current_path = f"{parent_path}.{module.name}" if parent_path else module.name

    elements.append(
        {
            "type": "module",
            "name": module.name,
            "qualified_name": current_path,
            "docstring": module.docstring.value if module.docstring else None,
            "filepath": str(module.filepath) if module.filepath else None,
            "is_public": module.is_public,
            "is_private": module.is_private,
            "line_start": module.lineno,
            "line_end": module.endlineno,
            "annotation": None,
            "returns": None,
        }
    )

    for member in module.members.values():
        if isinstance(member, griffe.Module):
            elements.extend(_extract_api_elements(member, current_path))
        elif isinstance(member, griffe.Class):
            elements.append(
                {
                    "type": "class",
                    "name": member.name,
                    "qualified_name": f"{current_path}.{member.name}",
                    "docstring": member.docstring.value if member.docstring else None,
                    "bases": [str(b) for b in member.bases],
                    "is_public": member.is_public,
                    "is_private": member.is_private,
                    "line_start": member.lineno,
                    "line_end": member.endlineno,
                    "annotation": None,
                    "returns": None,
                }
            )
            for func in member.members.values():
                if isinstance(func, griffe.Function):
                    elements.append(
                        {
                            "type": "method",
                            "name": func.name,
                            "qualified_name": f"{current_path}.{member.name}.{func.name}",
                            "parent_class": member.name,
                            "docstring": (
                                func.docstring.value if func.docstring else None
                            ),
                            "is_public": func.is_public,
                            "is_private": func.is_private,
                            "parameters": [p.name for p in func.parameters],
                            "returns": str(func.returns) if func.returns else None,
                            "line_start": func.lineno,
                            "line_end": func.endlineno,
                            "annotation": None,
                        }
                    )
        elif isinstance(member, griffe.Function):
            elements.append(
                {
                    "type": "function",
                    "name": member.name,
                    "qualified_name": f"{current_path}.{member.name}",
                    "docstring": member.docstring.value if member.docstring else None,
                    "is_public": member.is_public,
                    "is_private": member.is_private,
                    "parameters": [p.name for p in member.parameters],
                    "returns": str(member.returns) if member.returns else None,
                    "line_start": member.lineno,
                    "line_end": member.endlineno,
                    "annotation": None,
                }
            )
        elif isinstance(member, griffe.Attribute):
            elements.append(
                {
                    "type": "attribute",
                    "name": member.name,
                    "qualified_name": f"{current_path}.{member.name}",
                    "docstring": member.docstring.value if member.docstring else None,
                    "value": str(member.value) if member.value else None,
                    "annotation": str(member.annotation) if member.annotation else None,
                    "is_public": member.is_public,
                    "is_private": member.is_private,
                    "line_start": member.lineno,
                    "line_end": member.endlineno,
                    "returns": None,
                }
            )

    return elements


def _populate_api_df(
    session: fc.Session, fenic_api: griffe.Module
) -> DataFrame:
    """Populate the api_df table."""
    logger.info("Extracting API elements...")
    # Extract all API elements
    api_elements = _extract_api_elements(fenic_api)
    logger.info(f"Extracted {len(api_elements)} API elements")
    summarization_template = textwrap.dedent(
        """\
                                Type: {{type}}
                                Member Name: {{name}}
                                Qualified Name: {{qualified_name}}
                                Docstring: {{docstring}}
                                Value: {{value}}
                                Annotation: {{annotation}}
                                is Public? : {{is_public}}
                                is Private? : {{is_private}}
                                Parameters: {{parameters}}
                                Returns: {{returns}}
                                Parent Class: {{parent_class}}
                                """
    )
    # Create api_df DataFrame
    api_df = session.create_dataframe(api_elements)
    api_df = api_df.with_column(
        "api_element_summary",
        fc.text.jinja(
            summarization_template,
            strict=False,
            type=fc.col("type"),
            name=fc.col("name"),
            qualified_name=fc.col("qualified_name"),
            docstring=fc.col("docstring"),
            value=fc.col("value"),
            annotation=fc.col("annotation"),
            is_public=fc.col("is_public"),
            is_private=fc.col("is_private"),
            parameters=fc.col("parameters"),
            returns=fc.col("returns"),
            parent_class=fc.col("parent_class"),
        ),
    )
    # Save api_df table
    logger.info("Saving api_df table...")
    api_df.write.save_as_table("api_df", mode="overwrite")
    return api_df


def _populate_hierarchy_df(api_df: DataFrame) -> DataFrame:
    """Populate the hierarchy_df table."""
    # Split the qualified name into parts
    # Create hierarchy_df with depth and path information
    logger.info("Creating hierarchy_df...")
    hierarchy_df = api_df.select(
        "*",
        # Split the qualified name into parts
        fc.text.split(fc.col("qualified_name"), r"\.").alias("path_parts"),
        # Get the depth (number of dots + 1)
        (
            fc.text.length(fc.col("qualified_name"))
            - fc.text.length(
                fc.text.regexp_replace(fc.col("qualified_name"), r"\.", "")
            )
            + 1
        ).alias("depth"),
    )

    # Save hierarchy_df table
    logger.info("Saving hierarchy_df table...")
    hierarchy_df.write.save_as_table("hierarchy_df", mode="overwrite")
    return hierarchy_df


def _populate_fenic_summary(api_df: DataFrame) -> DataFrame:
    # Create fenic_summary - aggregate module summaries
    logger.info("Creating module summaries...")

    # Filter to public modules only
    public_modules = api_df.filter(
        (fc.col("type") == "module")
        & (fc.col("is_public"))
        & (~fc.col("name").starts_with("_"))
    )

    # Create module summaries based on docstrings
    module_summaries = public_modules.select(
        fc.col("name").alias("module"),
        fc.coalesce(fc.col("docstring"), fc.lit("No description available")).alias(
            "summary"
        ),
    ).with_column(
        "module_name_and_summary",
        fc.text.jinja(
            "Module: {{module_name}} Summary: {{summary}}",
            module_name=fc.col("module"),
            summary=fc.col("summary"),
        ),
    )

    # Create a project summary by aggregating module information
    logger.info("Creating project summary...")
    project_summary_df = module_summaries.agg(
        fc.semantic.reduce(
            "Create a comprehensive summary of the Fenic project based on these module descriptions. "
            "The summary should explain what Fenic is, its main features, and key capabilities. "
            "Focus on user facing features -- for example, users do not need to understand how serde works, as that is an implementation detail. "
            "Finally, in the context of fenic, MCP stands for Model Context Protocol.",
            model_alias="mini",
            column=fc.col("module_name_and_summary"),
            max_output_tokens=4096,
        ).alias("project_summary")
    ).cache()

    # Save fenic_summary table
    logger.info("Saving fenic_summary table...")
    logger.info(
        f"Generated Summary: {project_summary_df.to_pylist()[0]['project_summary']}"
    )
    project_summary_df.write.save_as_table("fenic_summary", mode="overwrite")
    return project_summary_df


def _populate_project_context(
    session: fc.Session, hierarchy_df: DataFrame, summary_df: DataFrame
) -> None:
    """Precompute the small, serving-time context returned by zero-arg tools."""
    public_hierarchy = hierarchy_df.filter(
        fc.col("is_public")
        & (fc.col("type") != "attribute")
        & ~fc.col("name").starts_with("_")
    ).select("qualified_name", "name", "type", "depth", "path_parts")
    api_tree = tree_to_string(build_tree(public_hierarchy.to_pydict()))
    project_summary = summary_df.to_pylist()[0]["project_summary"]
    project_overview = (
        f"## Fenic Project Overview\n\n{project_summary}\n\n"
        f"## API Tree\n\n{api_tree}"
    )
    session.create_dataframe(
        {
            "api_tree": [api_tree],
            "project_overview": [project_overview],
        }
    ).write.save_as_table("fenic_project_context", mode="overwrite")


def _populate_release_metadata(
    session: fc.Session,
    fenic_version: str,
    source_sha: str,
) -> None:
    """Persist provenance for the documentation catalog."""
    session.create_dataframe(
        {
            "fenic_version": [fenic_version],
            "source_sha": [source_sha],
        }
    ).write.save_as_table("fenic_release_metadata", mode="overwrite")


def populate_tables(data_dir: str = "./data") -> None:
    """Build and persist every table required by the documentation server."""
    log_fenic_version()
    fenic_version = package_version("fenic")
    expected_version = os.environ.get("FENIC_VERSION")
    if expected_version and fenic_version != expected_version:
        raise RuntimeError(
            f"Expected fenic {expected_version}, but data prep loaded {fenic_version}"
        )
    session = _setup_session(data_dir)
    fenic_api = _load_fenic_api()
    api_df = _populate_api_df(session, fenic_api)
    hierarchy_df = _populate_hierarchy_df(api_df)
    summary_df = _populate_fenic_summary(api_df)
    _populate_project_context(session, hierarchy_df, summary_df)
    _populate_release_metadata(
        session,
        fenic_version,
        os.environ.get("FENIC_SOURCE_SHA", "local"),
    )
    register_docs_tools(session)

    logger.info("\nSuccessfully created all required tables:")
    logger.info("- api_df: Contains all API elements with metadata")
    logger.info(
        "- hierarchy_df: Contains hierarchy information with depth and path parts"
    )
    logger.info("- fenic_summary: Contains project overview")
    logger.info("- fenic_project_context: Contains precomputed overview and API tree")
    logger.info("- fenic_release_metadata: Contains release version and source commit")
    logger.info("- MCP tools: Contains native parameterized documentation queries")
    session.stop()


if __name__ == "__main__":
    populate_tables()
