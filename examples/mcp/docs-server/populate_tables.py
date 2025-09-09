#!/usr/bin/env python
"""Populate the documentation tables required for the MCP server.

This script creates three tables: api_df, hierarchy_df, and fenic_summary.
"""

import logging
import os
import textwrap
from typing import Any, Dict, List

import griffe

import fenic as fc
from fenic.api.dataframe import DataFrame

fc.logging.configure_logging()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _setup_session(work_dir: str) -> fc.Session:
    """Setup Fenic session with proper configuration."""
    logger.info("Setting up session...")
    os.makedirs(work_dir, exist_ok=True)
    os.chdir(work_dir)
    logger.info(f"Working directory: {work_dir}")
    
    # Configure fenic session
    config = fc.SessionConfig(
        app_name="docs",
        semantic=fc.SemanticConfig(
            language_models={
                "flash": fc.GoogleDeveloperLanguageModel(
                    api_key=os.environ.get("GEMINI_API_KEY"),
                    model_name="gemini-2.0-flash-exp",
                    rpm=2000,
                    tpm=4_000_000,
                ),
            },
            default_language_model="flash",
        ) if os.environ.get("GEMINI_API_KEY") else None,
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
            "parameters": None,
            "parent_class": None,
            "value": None,
            "bases": None,
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
                    "parameters": None,
                    "parent_class": None,
                    "filepath": str(member.filepath) if member.filepath else None,
                    "value": None,
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
                            "filepath": str(func.filepath) if func.filepath else None,
                            "value": None,
                            "bases": None,
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
                    "filepath": str(member.filepath) if member.filepath else None,
                    "parent_class": None,
                    "value": None,
                    "bases": None,
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
                    "filepath": str(member.filepath) if member.filepath else None,
                    "parameters": None,
                    "parent_class": None,
                    "bases": None,
                }
            )

    return elements


def _populate_api_df(session: fc.Session, fenic_api: griffe.Module) -> DataFrame:
    """Populate the api_df table with enhanced summaries."""
    logger.info("Extracting API elements...")
    # Extract all API elements
    api_elements = _extract_api_elements(fenic_api)
    logger.info(f"Extracted {len(api_elements)} API elements")
    
    # Define comprehensive summary template
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
    
    # Create api_df DataFrame with enhanced summary
    api_df = session.create_dataframe(api_elements)
    api_df = (
        api_df.with_column(
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
        .cache()
    )
    
    # Save api_df table
    logger.info("Saving api_df table...")
    api_df.write.save_as_table("api_df", mode="overwrite")
    return api_df


def _populate_hierarchy_df(api_df: DataFrame) -> DataFrame:
    """Populate the hierarchy_df table with path and depth information."""
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
    """Create project summary using semantic reduction."""
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
            "The summary should explain what Fenic is, its main features, and key capabilities.",
            model_alias="flash",
            column=fc.col("module_name_and_summary"),
            max_output_tokens=4096,
        ).alias("project_summary")
    ).cache()

    # Save fenic_summary table
    logger.info("Saving fenic_summary table...")
    summary_text = project_summary_df.to_pylist()[0]['project_summary']
    logger.info(f"Generated Summary Length: {len(summary_text)} characters")
    
    project_summary_df.write.save_as_table("fenic_summary", mode="overwrite")
    return project_summary_df


def _verify_tables(session: fc.Session):
    """Verify that all tables were created successfully."""
    logger.info("\nVerifying tables...")
    
    required_tables = ["api_df", "hierarchy_df", "fenic_summary"]
    for table_name in required_tables:
        if session.catalog.does_table_exist(table_name):
            count = session.table(table_name).count()
            logger.info(f"✓ {table_name}: {count} rows")
            
            # Check for required columns in api_df
            if table_name == "api_df":
                schema = session.table(table_name).schema
                if "api_element_summary" in str(schema):
                    logger.info("  - api_element_summary column present")
                else:
                    logger.warning("  - api_element_summary column missing")
        else:
            logger.error(f"✗ {table_name}: NOT FOUND")
            raise ValueError(f"Table {table_name} not found")


def populate_tables(data_dir: str = None) -> None:
    """Main function to populate all documentation tables.
    
    Args:
        data_dir: Directory to store the tables. If None, uses FENIC_WORK_DIR 
                 environment variable or defaults to ~/.fenic
    """
    # Determine working directory
    if data_dir is None:
        data_dir = os.environ.get("FENIC_WORK_DIR", os.path.expanduser("~/.fenic"))
    
    # Check for API key
    if not os.environ.get("GEMINI_API_KEY"):
        logger.warning("GEMINI_API_KEY is not set. Semantic features may be limited.")
    
    # Setup and populate tables
    session = _setup_session(data_dir)
    fenic_api = _load_fenic_api()
    api_df = _populate_api_df(session, fenic_api)
    _ = _populate_hierarchy_df(api_df)
    _ = _populate_fenic_summary(api_df)
    
    # Verify tables
    _verify_tables(session)

    logger.info("\nSuccessfully created all required tables:")
    logger.info("- api_df: Contains all API elements with enhanced summaries")
    logger.info("- hierarchy_df: Contains hierarchy information with depth and path parts")
    logger.info("- fenic_summary: Contains comprehensive project overview")
    logger.info(f"\nTables stored in: {data_dir}")


if __name__ == "__main__":
    populate_tables()