import re
from enum import Enum
from typing import Any, Dict, Type

import polars as pl
from pydantic import BaseModel, create_model


def convert_row_to_instruction_context(row: Dict[str, Any]) -> str:
    """Format a row as text, returning None if any value is None."""
    return "\n".join(f"[{col.upper()}]: «{row[col]}»" for col in row.keys())


def uppercase_instruction_placeholder(instruction: str) -> str:
    """Convert placeholders in the format {column_name} to [COLUMN_NAME].

    Args:
        instruction (str): The instruction string with placeholders.

    Returns:
        str: The instruction with uppercase placeholders.
    """
    return re.sub(
        r"\{(\w+)\}", lambda match: f"[{match.group(1).upper()}]", instruction
    )


def stringify_enum_type(enum_type: Type[Enum]) -> str:
    """Convert enum values to a comma-separated string.

    Args:
        enum_categories (Type[Enum]): The enum class.

    Returns:
        str: Comma-separated string of enum values.
    """
    return ", ".join(f"{label.value}" for label in enum_type)


def create_classification_pydantic_model(enum_cls: Type[Enum]) -> Type[BaseModel]:
    """Creates a Pydantic model from an Enum class.

    Args:
        enum_cls (Type[Enum]): The Enum class to convert.

    Returns:
        Type[BaseModel]: A Pydantic model class with a field for the Enum values.
    """
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
