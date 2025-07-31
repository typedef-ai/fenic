import logging
import re
from enum import Enum
from typing import (
    Annotated,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Type,
    Union,
    get_args,
    get_origin,
)

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


# =============================================================================
# Schema-related utilities for structured semantic operations
# =============================================================================

# Shared schema explanation template for all structured operations
SCHEMA_EXPLANATION_INSTRUCTION_FRAGMENT = (
    "How to read the field schema:\n"
    "- Nested fields are expressed using dot notation (e.g., 'organization.name' means 'name' is a subfield of 'organization')\n"
    "- Lists are denoted using 'list of [type]' (e.g., 'employees' is a list of [string])\n"
    "- Type annotations are shown in parentheses (e.g., string, integer, boolean, date)\n"
)


def convert_pydantic_model_to_key_descriptions(schema: Type[BaseModel]) -> str:
    """Extract keys, types, and descriptions from a Pydantic model, including nested models and lists.

    This function is used by structured semantic operations (extract, map, etc.) to convert
    Pydantic schema models into human-readable field descriptions for LLM prompts.

    Args:
        schema (Type[BaseModel]): The Pydantic model class.

    Returns:
        str: Formatted string of model keys and descriptions.
    """
    result = []

    def get_type_name(annotation) -> str:
        origin = get_origin(annotation)
        args = get_args(annotation)

        if origin is Annotated:
            annotation = args[0]
            origin = get_origin(annotation)
            args = get_args(annotation)

        if origin is Union:
            non_none = [arg for arg in args if arg is not type(None)]
            type_str = "/".join(get_type_name(t) for t in non_none)
            if len(non_none) < len(args):
                return f"{type_str} (optional)"
            return type_str

        if origin in (list, List):
            return f"list of {get_type_name(args[0])}" if args else "list"

        if origin is Literal:
            return " or ".join(repr(a) for a in args)

        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return "object"

        return getattr(annotation, "__name__", str(annotation))

    def recurse(schema: Type[BaseModel], prefix: str = ""):
        for field_name, field_info in schema.model_fields.items():
            full_field_name = f"{prefix}.{field_name}" if prefix else field_name
            annotation = field_info.annotation
            description = field_info.description or ""

            # Unwrap Annotated
            if get_origin(annotation) is Annotated:
                annotation = get_args(annotation)[0]

            origin = get_origin(annotation)
            args = get_args(annotation)
            is_optional = False

            # Handle Optional[T]
            if origin is Union and any(a is type(None) for a in args):
                is_optional = True
                # Unwrap Optional[T] to T
                non_none_args = [a for a in args if a is not type(None)]
                if len(non_none_args) == 1:
                    annotation = non_none_args[0]
                    origin = get_origin(annotation)
                    args = get_args(annotation)

            type_str = get_type_name(annotation)
            if is_optional:
                type_str += " (optional)"

            # Handle nested BaseModel
            if isinstance(annotation, type) and issubclass(annotation, BaseModel):
                result.append(f"{full_field_name} ({type_str}): {description}")
                recurse(annotation, full_field_name)
                continue

            # Handle list of BaseModels
            if origin in (list, List) and get_args(annotation):
                elem_type = get_args(annotation)[0]
                if isinstance(elem_type, type) and issubclass(elem_type, BaseModel):
                    result.append(f"{full_field_name} (list of objects): {description}")
                    recurse(elem_type, f"{full_field_name}[item]")
                    continue

            # Leaf field
            result.append(f"{full_field_name} ({type_str}): {description}")

    recurse(schema)
    return "\n".join(result)


def validate_structured_response(
    json_resp: Optional[str],
    model_class: Type[BaseModel],
    operator_name: str
) -> Optional[Dict[str, Any]]:
    """Validate and parse a structured JSON response from an LLM.

    This function provides standardized validation for structured outputs across
    semantic operations that use Pydantic schemas.

    Args:
        json_resp: The JSON response string from the LLM (can be None)
        model_class: The Pydantic model class to validate against
        operator_name: Name of the operation (for logging purposes)

    Returns:
        Validated dictionary representation of the model, or None if validation fails
    """
    logger = logging.getLogger(__name__)

    if json_resp is None:
        return None

    try:
        validated_model = model_class.model_validate_json(json_resp)
        return validated_model.model_dump(mode="json")
    except Exception as e:
        logger.warning(
            f"invalid model output: {json_resp} for {operator_name}: {e}",
            exc_info=True,
        )
        return None