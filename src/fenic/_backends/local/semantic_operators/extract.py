import logging
from typing import Any, Dict, List, Optional

import jinja2
import polars as pl
from pydantic import BaseModel

from fenic._backends.local.semantic_operators.base import (
    BaseSingleColumnInputOperator,
    CompletionOnlyRequestSender,
)
from fenic._backends.local.semantic_operators.utils import (
    SCHEMA_EXPLANATION_INSTRUCTION_FRAGMENT,
    convert_pydantic_model_to_key_descriptions,
    validate_structured_response,
)
from fenic._inference.language_model import InferenceConfiguration, LanguageModel

logger = logging.getLogger(__name__)


class Extract(BaseSingleColumnInputOperator[str, Dict[str, Any]]):
    EXTRACT_SYSTEM_PROMPT = jinja2.Template(
        "Extract information from the document according to the field schema.\n\n"
        "Field Schema:\n"
        "{{ schema_details }}\n\n"
        "{{ schema_explanation }}\n\n"
        "Requirements:\n"
        "1. Extract only information explicitly stated in the document\n"
        "2. Do not infer, guess, or generate information not present\n"
        "3. Include all required fields - no extra fields, no missing fields\n"
        "4. For list fields, extract all items that match the field description\n"
        "5. Be thorough and precise - capture all relevant content without changing meaning"
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
        schema_details = convert_pydantic_model_to_key_descriptions(self.output_model)
        return self.EXTRACT_SYSTEM_PROMPT.render(
            schema_explanation=SCHEMA_EXPLANATION_INSTRUCTION_FRAGMENT,
            schema_details=schema_details
        )

    def postprocess(
        self, responses: List[Optional[str]]
    ) -> List[Optional[Dict[str, Any]]]:
        return [
            validate_structured_response(
                json_resp, self.output_model, "semantic.extract"
            )
            for json_resp in responses
        ]
