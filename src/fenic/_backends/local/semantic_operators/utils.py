import logging
from enum import Enum
from textwrap import dedent
from typing import (
    List,
)

import polars as pl
from pydantic import BaseModel, create_model

logger = logging.getLogger(__name__)

SIMPLE_INSTRUCTION_SYSTEM_PROMPT = dedent("""\
    Follow the user's instruction exactly and generate only the requested output.

    Requirements:
    1. Follow the instruction exactly as written
    2. Output only what is requested - no explanations, no prefixes, no metadata
    3. Be concise and direct
    4. Do not add formatting or structure unless explicitly requested""")

# Shared schema explanation template for all structured operations
SCHEMA_EXPLANATION_INSTRUCTION_FRAGMENT = (
    "How to read the output schema:\n"
    "- Nested fields are expressed using dot notation (e.g., 'organization.name' means 'name' is a subfield of 'organization')\n"
    "- Lists are denoted using 'list of [type]' (e.g., 'employees' is a list of str)\n"
    "- For lists: 'fieldname[item].subfield' means each item in the list has that subfield\n"
    "- Type annotations are shown in parentheses (e.g., string, integer, boolean, date)\n"
    "- Fields marked (optional) can be omitted if not applicable"
)

def create_classification_pydantic_model(allowed_values: List[str]) -> type[BaseModel]:
    """Creates a Pydantic model from a list of allowed string values using a dynamic Enum.

    Args:
        allowed_values (List[str]): The list of allowed string values.

    Returns:
        Type[BaseModel]: A Pydantic model class with a field for the Enum values.
    """
    enum_name = "LabelEnum"
    enum_members = {value.upper(): value for value in allowed_values}
    enum_cls = Enum(enum_name, enum_members)

    return create_model(
        "EnumModel",
        output=(enum_cls, ...),
    )

def filter_invalid_embeddings_expr(embedding_column: str) -> pl.Expr:
    """Filter out rows with invalid embeddings.

    Args:
        embedding_column (pl.Expr): The column containing the embeddings.

    Returns:
        pl.Expr: A filter expression that removes rows with invalid embeddings.
    """
    return (
        pl.col(embedding_column).is_not_null()  # 1. Array itself is not null
        & ~pl.col(embedding_column).arr.contains(None)  # 2. No null elements
        & ~pl.col(embedding_column).arr.contains(float('nan'))  # 3. No NaN elements
    )
