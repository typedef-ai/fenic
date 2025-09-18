
from __future__ import annotations

import logging
import typing
from typing import (
    Annotated,
    List,
    Literal,
    Type,
    Union,
    get_args,
    get_origin,
)

from pydantic import BaseModel

logger = logging.getLogger(__name__)

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

def check_if_model_uses_unserializable_features(model: type[BaseModel]) -> None:
    """Validate a Pydantic model to determine if information will be lost when serializing.

    We cannot serialize the arbitrary python code in pydantic field/model validators, as well as
    the use of `default_factory` with an arbitrary callable. If the model contains usages of these,
    we can still serialize them, but these validators will not be included in the serialized form.
    """
    decorators = getattr(model, "__pydantic_decorators__", None)
    if decorators:
        has_field_validators = bool(getattr(decorators, "field_validators", None))
        has_model_validators = bool(getattr(decorators, "model_validators", None))
        if has_field_validators or has_model_validators:
            logger.warning("Field/Model validators are not supported for cloud execution and will be ignored.")

    fields = getattr(model, "model_fields", {})
    fields_with_default_factory = []
    fields_with_callable_default = []
    for f in fields.values():
        # Block callables in defaults
        if callable(getattr(f, "default_factory", None)):
            fields_with_default_factory.append(f)
        if callable(getattr(f, "default", None)):
            fields_with_callable_default.append(f)

    if fields_with_default_factory:
        logger.warning(f"Usage of `default_factory` in fields: {fields_with_default_factory} is not supported for "
                       f"cloud execution and will be ignored.")
    if fields_with_callable_default:
        logger.warning(f"Usage of `default` with a `Callable` or nested pydantic model in fields: "
                       f"{fields_with_callable_default} is not supported for cloud execution and will be ignored.")