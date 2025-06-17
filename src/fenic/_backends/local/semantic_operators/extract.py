import logging
from typing import (
    Annotated,
    Any,
    Dict,
    List,
    Optional,
    Type,
    Union,
    get_args,
    get_origin,
)

import polars as pl
from pydantic import BaseModel

from fenic._backends.local.semantic_operators.base import (
    BaseSingleColumnInputOperator,
    CompletionOnlyRequestSender,
)
from fenic._inference.language_model import InferenceConfiguration, LanguageModel

logger = logging.getLogger(__name__)


class Extract(BaseSingleColumnInputOperator[str, Dict[str, Any]]):
    SYSTEM_PROMPT = (
        "You are an expert at structured data extraction. "
        "Your task is to extract relevant information from a given document. Your output must be a structured JSON object. "
        "Expected JSON keys and descriptions:\n"
        "{keys}"
        "Notes on the structure:\n"
        "- Field names with parent.child notation indicate nested objects\n"
        "- [item] notation indicates items within a list\n"
        "- Type information is provided in parentheses\n"
    )

    def __init__(
        self,
        input: pl.Series,
        schema: type[BaseModel],
        model: LanguageModel,
        max_output_tokens: int,
        temperature: float,
    ):
        self.output_model = schema
        super().__init__(
            input,
            CompletionOnlyRequestSender(
                operator_name="semantic.extract",
                inference_config=InferenceConfiguration(
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                    response_format=self.output_model,
                ),
                model=model,
            ),
            None,
        )

    def build_system_message(self) -> str:
        return self.SYSTEM_PROMPT.format(
            keys=convert_pydantic_model_to_key_descriptions(self.output_model)
        )

    def postprocess(
        self, responses: List[Optional[str]]
    ) -> List[Optional[Dict[str, Any]]]:
        validated_data = []
        for json_resp in responses:
            if json_resp is None:
                validated_data.append(None)
                continue
            try:
                validated_model = self.output_model.model_validate_json(json_resp)
                validated_data.append(validated_model.model_dump(mode="json"))
            except Exception as e:
                logger.warning(
                    f"invalid model output: {json_resp} for semantic.extract: {e}",
                    exc_info=True,
                )
                validated_data.append(None)
        return validated_data


def convert_pydantic_model_to_key_descriptions(schema: Type[BaseModel]) -> str:
    """Extract keys, types, and descriptions from a Pydantic model, including nested models and lists.
    Designed for LLM-structured data extraction.

    Args:
        schema (Type[BaseModel]): The Pydantic model class.

    Returns:
        str: Formatted string of model keys and descriptions.
    """
    result = []

    def get_type_name(annotation) -> str:
        """Get a human-readable type name for an annotation."""
        origin = get_origin(annotation)
        args = get_args(annotation)

        # Handle Annotated
        if origin is Annotated:
            annotation = args[0]
            origin = get_origin(annotation)
            args = get_args(annotation)

        # Handle Optional and Union
        if origin is Union:
            non_none = [arg for arg in args if arg is not type(None)]
            type_str = "/".join(get_type_name(t) for t in non_none)
            if len(non_none) < len(args):  # there's a NoneType → optional
                return f"{type_str} (optional)"
            return type_str

        # Handle List[...] types
        if origin is list:
            if not args:
                return "list"
            element_type = args[0]
            return f"list of {get_type_name(element_type)}"

        # Handle Dict, Tuple, etc.
        if origin in (dict, tuple):
            return origin.__name__

        # Handle nested BaseModel
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return "object"

        return getattr(annotation, "__name__", str(annotation))

    def recurse(schema: Type[BaseModel], prefix: str = ""):
        """Recursively extract field information from a model."""
        for field_name, field_info in schema.model_fields.items():
            full_field_name = f"{prefix}.{field_name}" if prefix else field_name
            annotation = field_info.annotation
            description = getattr(field_info, "description", "") or ""

            type_str = get_type_name(annotation)

            # Nested model
            if isinstance(annotation, type) and issubclass(annotation, BaseModel):
                result.append(f"{full_field_name} ({type_str}): {description}")
                recurse(annotation, full_field_name)
                continue

            # List of nested models
            origin = get_origin(annotation)
            args = get_args(annotation)
            if origin is list and args:
                element_type = args[0]
                if isinstance(element_type, type) and issubclass(
                    element_type, BaseModel
                ):
                    result.append(f"{full_field_name} (list of objects): {description}")
                    recurse(element_type, f"{full_field_name}[item]")
                    continue

            # Default: primitive or simple container
            result.append(f"{full_field_name} ({type_str}): {description}")

    recurse(schema)
    return "\n".join(result)
