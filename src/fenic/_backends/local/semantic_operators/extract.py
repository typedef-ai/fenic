import logging
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
        "Your task is to extract relevant information from a given document using only the information explicitly stated in the text. "
        "You must adhere strictly to the provided field definitions. Do not infer or generate information that is not directly supported by the document.\n\n"

        "The field schema below defines the structure of the information you are expected to extract.\n\n"

        "How to read the field schema:\n"
        "- Nested fields are expressed using dot notation (e.g., 'organization.name' means 'name' is a subfield of 'organization')\n"
        "- Lists are denoted using 'list of [type]' (e.g., 'employees' is a list of [string])\n"
        "- Type annotations are shown in parentheses (e.g., string, integer, boolean, date)\n\n"

        "Extraction Guidelines:\n"
        "1. Extract only what is explicitly present or clearly supported in the document—do not guess or extrapolate.\n"
        "2. For list fields, extract all items that match the field description.\n"
        "3. If a field is not found in the document, return null for single values and [] for lists.\n"
        "4. Ensure all field names in your structured output exactly match the field schema.\n"
        "5. Be thorough and precise—capture all relevant content without changing or omitting meaning.\n\n"

        "Field Schema:\n"
        "{keys}"
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
