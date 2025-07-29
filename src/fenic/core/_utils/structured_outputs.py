import typing
from typing import List, Union, get_args, get_origin

from pydantic import BaseModel


class OutputFormatValidationError(Exception):
    """Error raised when a semantic operation schema is invalid."""


def validate_output_format(
    model: type[BaseModel],
) -> None:
    """Check a Pydantic model type to ensure it is valid schema for semantic operations.

    This function validates schemas used by semantic operations like extract, map, etc.
    to ensure they have proper field descriptions and supported types.

    Args:
        model: The Pydantic model class to validate

    Raises:
        SemanticSchemaValidationError: If the schema is invalid
    """
    # Check the field structure and the types for the pydantic model
    if len(model.__pydantic_fields__.items()) == 0:
        raise OutputFormatValidationError(
            "Output schema cannot be empty. "
            "Please specify at least one output field."
        )

    for field_name, field_info in model.__pydantic_fields__.items():
        if field_info.description is None:
            raise OutputFormatValidationError(
                f"Extract schema field {field_name} has no description. Please specify a description for each field."
            )

        _validate_semantic_field_type(field_info.annotation, field_name)


def _validate_semantic_field_type(annotation, field_name: str) -> None:
    """Recursively validate field types for semantic operations."""
    # Handle basic types
    if annotation in (bool, int, float, str):
        return

    # Handle Pydantic models
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        validate_output_format(annotation)
        return

    # Handle generic types (List, Optional, Union, etc.)
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is list or origin is List:
        if not args:
            raise TypeError(f"List type in field {field_name} must specify element type")
        element_type = args[0]
        _validate_semantic_field_type(element_type, field_name)
        return

    elif origin is Union:
        # Only support Optional (Union[T, None]), reject other unions
        if len(args) == 2 and type(None) in args:
            # This is Optional[T] - validate the non-None type
            non_none_type = next(arg for arg in args if arg is not type(None))
            _validate_semantic_field_type(non_none_type, field_name)
            return
        else:
            # This is a Union with multiple non-None types - not supported
            raise OutputFormatValidationError(
                f"Union types are not supported in field {field_name}. Only Optional[T] is allowed."
            )

    elif origin is typing.Literal:
        # Literal types are allowed
        return

    # If we get here, it's an unsupported type
    raise OutputFormatValidationError(
        f"Unsupported data type in semantic schema field '{field_name}': {annotation}. "
        "Supported types are: str, int, float, bool, List[T], Optional[T], Literal, and nested Pydantic models."
    )
