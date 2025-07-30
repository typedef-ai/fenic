from typing import Any, Dict, List, Optional, Union

import jinja2
import polars as pl
from pydantic import BaseModel

from fenic._backends.local.semantic_operators.base import (
    BaseMultiColumnInputOperator,
    CompletionOnlyRequestSender,
)
from fenic._backends.local.semantic_operators.utils import (
    SCHEMA_EXPLANATION_INSTRUCTION_FRAGMENT,
    convert_pydantic_model_to_key_descriptions,
    validate_structured_response,
)
from fenic._inference.language_model import InferenceConfiguration, LanguageModel
from fenic.core.types import (
    MapExample,
    MapExampleCollection,
)


class Map(BaseMultiColumnInputOperator[str, str]):
    STRING_OUTPUT_SYSTEM_PROMPT = (
        "Follow the user's instruction exactly and generate only the requested output.\n\n"
        "Requirements:\n"
        "1. Follow the instruction exactly as written\n"
        "2. Output only what is requested - no explanations, no prefixes, no metadata\n"
        "3. Be concise and direct\n"
        "4. Do not add formatting or structure unless explicitly requested"
    )

    RESPONSE_FORMAT_SYSTEM_PROMPT = jinja2.Template(
        "Follow the user's instruction exactly and generate a structured output according to the field schema.\n\n"
        "Field Schema:\n"
        "{{ schema_details }}\n\n"
        "{{ schema_explanation }}\n\n"
        "Requirements:\n"
        "1. Follow the instruction exactly as written\n"
        "2. Generate output that matches the provided schema exactly\n"
        "3. Include all required fields - no extra fields, no missing fields\n"
        "5. Each field's content must match its description precisely"
    ),

    def __init__(
        self,
        input: pl.Series,
        jinja_template: str,
        model: LanguageModel,
        max_tokens: int,
        temperature: float,
        response_format: Optional[type[BaseModel]] = None,
        examples: Optional[MapExampleCollection] = None,
    ):
        super().__init__(
            input,
            request_sender=CompletionOnlyRequestSender(
                model=model,
                operator_name="semantic.map",
                inference_config=InferenceConfiguration(
                    max_output_tokens=max_tokens,
                    response_format=response_format,
                    temperature=temperature,
                ),
            ),
            jinja_template=jinja2.Template(jinja_template),
            examples=examples,
        )
        self.response_format = response_format

    def build_system_message(self) -> str:
        is_structured_response = self.response_format is not None
        if is_structured_response:
            schema_details = convert_pydantic_model_to_key_descriptions(self.response_format)
            return self.RESPONSE_FORMAT_SYSTEM_PROMPT.render(
                schema_explanation=SCHEMA_EXPLANATION_INSTRUCTION_FRAGMENT,
                schema_details=schema_details
            )
        else:
            return self.STRING_OUTPUT_SYSTEM_PROMPT

    def postprocess(
        self, responses: List[Optional[str]]
    ) -> Union[List[Optional[Dict[str, Any]]], List[Optional[str]]]:
        if self.response_format is None:
            return responses
        return [
            validate_structured_response(
                json_resp, self.response_format, "semantic.map"
            )
            for json_resp in responses
        ]

    def convert_example_to_assistant_message(self, example: MapExample) -> str:
        """Convert a MapExample to an assistant message string.

        If the example output is a BaseModel instance, serialize it to JSON.
        Otherwise, return the string output directly.
        """
        if isinstance(example.output, BaseModel):
            return example.output.model_dump_json()
        return example.output
