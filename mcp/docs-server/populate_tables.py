#!/usr/bin/env python
"""
Populate the documentation tables required for the MCP server.
This script creates three tables: api_df, hierarchy_df, and fenic_summary.
"""

import os
import tempfile
import fenic as fc
import griffe
from typing import List, Dict, Any

# Use the same directory setup as the MCP server
work_dir = os.environ.get("FENIC_WORK_DIR", os.path.expanduser("~/.fenic"))
os.makedirs(work_dir, exist_ok=True)
os.chdir(work_dir)

print(f"Creating documentation tables in: {work_dir}")

# Configure fenic session (same as MCP server)
config = fc.SessionConfig(
    app_name="docs",
    semantic=fc.SemanticConfig(
        language_models={
            "flash": fc.GoogleGLAModelConfig(
                model_name="gemini-2.0-flash",
                rpm=2000,
                tpm=4_000_000,
            ),
        },
        default_language_model="flash",
    ),
)

session = fc.Session.get_or_create(config)

print("Loading Fenic API with Griffe...")

# Load fenic API using Griffe
loader = griffe.GriffeLoader()
fenic_api = loader.load("fenic")

def extract_api_elements(module: griffe.Module, parent_path: str = "") -> List[Dict[str, Any]]:
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
            elements.extend(extract_api_elements(member, current_path))
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

print("Extracting API elements...")

# Extract all API elements
api_elements = extract_api_elements(fenic_api)
print(f"Extracted {len(api_elements)} API elements")

# Create api_df DataFrame
api_df = session.createDataFrame(api_elements)

# Save api_df table
print("Saving api_df table...")
api_df.write.save_as_table("api_df", mode="overwrite")

# Create hierarchy_df with depth and path information
print("Creating hierarchy_df...")
hierarchy_df = api_df.select(
    "*",
    # Split the qualified name into parts
    fc.text.split(fc.col("qualified_name"), r"\.").alias("path_parts"),
    # Get the depth (number of dots + 1)
    (fc.text.length(fc.col("qualified_name")) -
     fc.text.length(fc.text.regexp_replace(fc.col("qualified_name"), r"\.", "")) + 1).alias("depth")
)

# Save hierarchy_df table
print("Saving hierarchy_df table...")
hierarchy_df.write.save_as_table("hierarchy_df", mode="overwrite")

# Create fenic_summary - aggregate module summaries
print("Creating module summaries...")

# Filter to public modules only
public_modules = api_df.filter(
    (fc.col("type") == "module") & 
    (fc.col("is_public") == True) & 
    (~fc.col("name").starts_with("_"))
)

# Create module summaries based on docstrings
module_summaries = public_modules.select(
    fc.col("name").alias("module"),
    fc.coalesce(fc.col("docstring"), fc.lit("No description available")).alias("summary")
)

# Create a project summary by aggregating module information
print("Creating project summary...")
project_summary_df = module_summaries.agg(
    fc.semantic.reduce(
        "Create a comprehensive summary of the Fenic project based on these module descriptions: "
        "Module: {module}, Description: {summary}. "
        "The summary should explain what Fenic is, its main features, and key capabilities.",
        model_alias="flash"
    ).alias("project_summary")
)

# Save fenic_summary table
print("Saving fenic_summary table...")
project_summary_df.write.save_as_table("fenic_summary", mode="overwrite")

print("\nSuccessfully created all required tables:")
print("- api_df: Contains all API elements with metadata")
print("- hierarchy_df: Contains hierarchy information with depth and path parts")
print("- fenic_summary: Contains project overview")

# Verify tables were created
print("\nVerifying tables...")
for table_name in ["api_df", "hierarchy_df", "fenic_summary"]:
    if session.catalog.does_table_exist(table_name):
        count = session.table(table_name).count()
        print(f"✓ {table_name}: {count} rows")
    else:
        print(f"✗ {table_name}: NOT FOUND")