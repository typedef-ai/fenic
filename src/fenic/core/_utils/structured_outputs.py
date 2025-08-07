import json
import logging
import typing
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

from jsonschema import ValidationError, validate
from pydantic import BaseModel

from fenic.core._logical_plan.resolved_types import ResolvedResponseFormat
from fenic.core.types.datatypes import (
    ArrayType,
    BooleanType,
    DoubleType,
    FloatType,
    IntegerType,
    StringType,
    StructType,
)


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


def convert_resolved_response_format_to_key_descriptions(resolved_format: ResolvedResponseFormat) -> str:
    """Extract keys, types, and descriptions from a ResolvedResponseFormat.

    This function is used by structured semantic operations (extract, map, etc.) to convert
    ResolvedResponseFormat into human-readable field descriptions for LLM prompts.

    Args:
        resolved_format: The ResolvedResponseFormat object.

    Returns:
        str: Formatted string of model keys and descriptions.
    """
    result = []

    def recurse_struct_type(struct_type, prefix: str = ""):
        for field in struct_type.struct_fields:
            full_field_name = f"{prefix}.{field.name}" if prefix else field.name
            # Locate corresponding schema node for this field
            schema_node = _get_schema_node_for_path(resolved_format.schema, full_field_name)
            # Description sourced from the JSON schema
            description = ""
            if isinstance(schema_node, dict):
                description = schema_node.get("description", "")

            # Get type name from the data type and schema node (to support Literal/enum)
            type_str = _get_data_type_name(field.data_type, schema_node)

            # Handle nested structs
            if isinstance(field.data_type, StructType):
                result.append(f"{full_field_name} (object): {description}")
                recurse_struct_type(field.data_type, full_field_name)
                continue

            # Handle arrays of structs
            if isinstance(field.data_type, ArrayType) and isinstance(field.data_type.element_type, StructType):
                result.append(f"{full_field_name} (list of objects): {description}")
                recurse_struct_type(field.data_type.element_type, f"{full_field_name}[item]")
                continue

            # Leaf field
            result.append(f"{full_field_name} ({type_str}): {description}")

    def _get_data_type_name(data_type, schema_node: Optional[Dict[str, Any]]) -> str:
        # If schema indicates enum/const, prefer that for Literal-like display
        literal_display = _literal_from_schema(schema_node)
        if literal_display is not None and not isinstance(data_type, (StructType, ArrayType)):
            return literal_display

        if isinstance(data_type, StructType):
            return "object"
        if isinstance(data_type, ArrayType):
            # For arrays, try to derive element type from items schema
            items_node = None
            if isinstance(schema_node, dict):
                items_node = schema_node.get("items") if isinstance(schema_node.get("items"), dict) else None
                if isinstance(items_node, dict):
                    items_node = _resolve_ref_if_needed(resolved_format.schema, items_node)
            element_type_name = _get_data_type_name(data_type.element_type, items_node)
            return f"list of {element_type_name}"

        # Map fenic primitive/logical singletons to Python type names used in the Pydantic path
        if data_type == StringType:
            return "str"
        if data_type == IntegerType:
            return "int"
        if data_type in (FloatType, DoubleType):
            return "float"
        if data_type == BooleanType:
            return "bool"

        # Fallback to string form
        return str(data_type)

    def _get_schema_node_for_path(schema: Dict[str, Any], path: str) -> Optional[Dict[str, Any]]:
        """Traverse a JSON schema to the node for a dot path with optional [item] segments."""
        def resolve_ref(node: Dict[str, Any]) -> Dict[str, Any]:
            return _resolve_ref_if_needed(schema, node)

        node = schema
        # Root may have $ref
        node = resolve_ref(node)
        # Ensure we start at properties
        props = node.get("properties")
        if not isinstance(props, dict):
            return None
        current = node
        segments = path.split(".") if path else []
        for seg in segments:
            is_item = seg.endswith("[item]")
            key = seg[:-6] if is_item else seg
            current = resolve_ref(current)
            if not isinstance(current, dict):
                return None
            properties = current.get("properties")
            if not isinstance(properties, dict) or key not in properties:
                return None
            current = resolve_ref(properties[key])
            if is_item:
                items = current.get("items") if isinstance(current, dict) else None
                if isinstance(items, dict):
                    current = resolve_ref(items)
                else:
                    return None
        return current if isinstance(current, dict) else None

    def _resolve_ref_if_needed(root_schema: Dict[str, Any], node: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve $ref, preserving metadata (e.g., description) on the referring node.

        If a node has $ref and also includes keys like description, merge them over the target.
        """
        if not isinstance(node, dict):
            return node
        ref = node.get("$ref")
        if not isinstance(ref, str) or not ref.startswith("#/"):
            return node
        target: Any = root_schema
        for part in ref[2:].split("/"):
            if isinstance(target, dict):
                target = target.get(part)
            else:
                return node
        if not isinstance(target, dict):
            return node
        # Merge, preferring properties on the referring node (excluding $ref itself)
        merged = dict(target)
        for k, v in node.items():
            if k == "$ref":
                continue
            merged[k] = v
        return merged

    def _literal_from_schema(node: Optional[Dict[str, Any]]) -> Optional[str]:
        if not isinstance(node, dict):
            return None
        node = _resolve_ref_if_needed(resolved_format.schema, node)
        if "enum" in node and isinstance(node["enum"], list):
            return " or ".join(repr(v) for v in node["enum"])
        if "const" in node:
            return repr(node["const"])
        return None

    if resolved_format.struct_type is None:
        return ""
    recurse_struct_type(resolved_format.struct_type)
    return "\n".join(result)


def validate_parsed_json_with_resolved_format(
    parsed_json: dict[str, Any],
    resolved_format: ResolvedResponseFormat,
) -> dict[str, Any]:
    """Validate and parse a structured JSON response from an LLM."""
    # Validate against the JSON schema from ResolvedResponseFormat
    validate(instance=parsed_json, schema=resolved_format.schema)
    return parsed_json


def validate_structured_response_with_resolved_format(
    json_resp: Optional[Union[str, dict[str, Any]]],
    resolved_format: ResolvedResponseFormat,
    operator_name: str
) -> Optional[Dict[str, Any]]:
    """Validate and parse a structured JSON response using ResolvedResponseFormat.

    This function provides standardized validation for structured outputs across
    semantic operations that use ResolvedResponseFormat.

    Args:
        json_resp: The JSON response string from the LLM (can be None)
        resolved_format: The ResolvedResponseFormat object
        operator_name: Name of the operation (for logging purposes)

    Returns:
        Validated dictionary representation of the model, or None if validation fails
    """
    logger = logging.getLogger(__name__)

    if json_resp is None:
        return None
    if isinstance(json_resp, str):
        json_resp = json.loads(json_resp)

    try:
        return validate_parsed_json_with_resolved_format(json_resp, resolved_format)
    except json.JSONDecodeError as e:
        logger.warning(
            f"Invalid JSON in model output: {json_resp} for {operator_name}: {e}",
            exc_info=True,
        )
        return None
    except ValidationError as e:
        logger.warning(
            f"JSON schema validation failed for {operator_name}: {e.message} at {e.path}",
            exc_info=True,
        )
        return None
    except Exception as e:
        logger.warning(
            f"Unexpected error validating model output: {json_resp} for {operator_name}: {e}",
            exc_info=True,
        )
        return None
