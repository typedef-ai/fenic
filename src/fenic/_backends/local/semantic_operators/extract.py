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
from fenic.core._logical_plan.resolved_types import ResolvedModelAlias

logger = logging.getLogger(__name__)


class Extract(BaseSingleColumnInputOperator[str, Dict[str, Any]]):
    EXTRACT_PROMPT_TEMPLATE = (
        "You are an expert at structured data extraction. "
        "Your task is to extract relevant information from a given document using only the information explicitly stated in the text. "
        "You must adhere strictly to the provided field definitions. Do not infer or generate information that is not directly supported by the document.\n\n"
        "Extraction Guidelines:\n"
        "1. Extract only what is explicitly present or clearly supported in the document—do not guess or extrapolate.\n"
        "2. For list fields, extract all items that match the field description.\n"
        "3. If a field is not found in the document, return null for single values and [] for lists.\n"
        "4. Ensure all field names in your structured output exactly match the field schema.\n"
        "5. Be thorough and precise—capture all relevant content without changing or omitting meaning.\n\n"
        "{{ schema_explanation }}\n"
        "Field Schema:\n"
        "{{ schema_details }}"
    )

    # Pre-compiled template as class variable
    _TEMPLATE = jinja2.Template(EXTRACT_PROMPT_TEMPLATE)

    def __init__(
        self,
        input: pl.Series,
        schema: type[BaseModel],
        model: LanguageModel,
        max_output_tokens: int,
        temperature: float,
        model_alias: Optional[ResolvedModelAlias] = None,
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
                    model_profile=model_alias.profile if model_alias else None,
                ),
                model=model,
            ),
            None,
        )

    def build_system_message(self) -> str:
        schema_details = convert_pydantic_model_to_key_descriptions(self.output_model)
        return self._TEMPLATE.render(
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
