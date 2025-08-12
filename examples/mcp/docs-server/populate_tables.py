#!/usr/bin/env python
"""Populate the documentation tables required for the MCP server.

This script creates three tables: api_df, hierarchy_df, and fenic_summary.
"""

import logging
import os
from typing import Any, Dict, List

import griffe

import fenic as fc
from fenic.api.dataframe import DataFrame
from fenic.core.error import ValidationError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _setup_session() -> fc.Session:
    # Use the same directory setup as the MCP server
    logger.info("Setting up session...")

    if not os.environ.get("GEMINI_API_KEY"):
        logger.error("GEMINI_API_KEY is not set")
        raise ValueError("A Gemini API key is required to populate the Fenic summary tables.")

    work_dir = os.environ.get("FENIC_WORK_DIR", os.path.expanduser("~/.fenic"))
    os.makedirs(work_dir, exist_ok=True)
    os.chdir(work_dir)
    logger.info(f"Creating documentation tables in: {work_dir}")

    # Configure fenic session - In this case we need a Gemini API key
    # as we will use the semantic functions to create the project summary.
    config = fc.SessionConfig(
        app_name="docs",
        semantic=fc.SemanticConfig(
            language_models={
                "flash": fc.GoogleDeveloperLanguageModel(
                    model_name="gemini-2.0-flash",
                    rpm=2_000,
                    tpm=4_000_000,
                ),
            },
            default_language_model="flash",
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


def _extract_api_elements(module: griffe.Module, parent_path: str = "") -> List[Dict[str, Any]]:
    """Extract API elements from a module recursively."""
    elements: List[Dict[str, Any]] = []
    current_path = f"{parent_path}.{module.name}" if parent_path else module.name

    elements.append({
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
    })

    for member in module.members.values():
        if isinstance(member, griffe.Module):
            elements.extend(_extract_api_elements(member, current_path))
        elif isinstance(member, griffe.Class):
            elements.append({
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
            })
            for func in member.members.values():
                if isinstance(func, griffe.Function):
                    elements.append({
                        "type": "method",
                        "name": func.name,
                        "qualified_name": f"{current_path}.{member.name}.{func.name}",
                        "parent_class": member.name,
                        "docstring": func.docstring.value if func.docstring else None,
                        "is_public": func.is_public,
                        "is_private": func.is_private,
                        "parameters": [p.name for p in func.parameters],
                        "returns": str(func.returns) if func.returns else None,
                        "line_start": func.lineno,
                        "line_end": func.endlineno,
                        "annotation": None,
                    })
        elif isinstance(member, griffe.Function):
            elements.append({
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
            })
        elif isinstance(member, griffe.Attribute):
            elements.append({
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
            })

    return elements


def _populate_api_df(session: fc.Session, fenic_api: griffe.Module) -> DataFrame:
    """Populate the api_df table."""
    logger.info("Extracting API elements...")
    # Extract all API elements
    api_elements = _extract_api_elements(fenic_api)
    logger.info(f"Extracted {len(api_elements)} API elements")

    # Create api_df DataFrame
    api_df = session.createDataFrame(api_elements)

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
        (fc.text.length(fc.col("qualified_name")) -
        fc.text.length(fc.text.regexp_replace(fc.col("qualified_name"), r"\.", "")) + 1).alias("depth")
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
        (fc.col("type") == "module") &
        (fc.col("is_public")) &
        (~fc.col("name").starts_with("_"))
    )

    # Create module summaries based on docstrings
    module_summaries = public_modules.select(
        fc.col("name").alias("module"),
        fc.coalesce(fc.col("docstring"), fc.lit("No description available")).alias("description")
    )

    module_summaries = module_summaries.with_column(
        "module_summary",
        fc.text.jinja(
        (
            "Module: {{module}} Description: {{description}}"
        ),
        module=fc.col("module"),
        description=fc.col("description")))

    # Create a project summary by aggregating module information
    logger.info("Creating project summary...")
    project_summary_df = module_summaries.agg(
        fc.semantic.reduce(
            """Create a comprehensive summary of the Fenic project based on the modules and their descriptions.
            The summary should explain what Fenic is, list its main features,
            and have a brief explanation of its key capabilities.""",
            model_alias="flash",
            column=fc.col("module_summary"),
        ).alias("project_summary")
    )

    # Save fenic_summary table
    logger.info("Saving fenic_summary table...")
    project_summary_df.write.save_as_table("fenic_summary", mode="overwrite")
    return project_summary_df


def _verify_tables(session: fc.Session):
    """Verify that the tables were created successfully."""
    # Verify tables were created
    logger.info("\nVerifying tables...")
    for table_name in ["api_df", "hierarchy_df", "fenic_summary"]:
        if session.catalog.does_table_exist(table_name):
            count = session.table(table_name).count()
            logger.info(f"✓ {table_name}: {count} rows")
        else:
            logger.error(f"✗ {table_name}: NOT FOUND")
            raise ValidationError(f"Table {table_name} not found")


def main():
    """Main function to populate the tables."""
    session = _setup_session()
    fenic_api = _load_fenic_api()
    api_df = _populate_api_df(session, fenic_api)
    _ = _populate_hierarchy_df(api_df)
    _ = _populate_fenic_summary(api_df)
    _verify_tables(session)

    logger.info("\nSuccessfully created all required tables:")
    logger.info("- api_df: Contains all API elements with metadata")
    logger.info("- hierarchy_df: Contains hierarchy information with depth and path parts")
    logger.info("- fenic_summary: Contains project overview")


if __name__ == "__main__":
    main()
