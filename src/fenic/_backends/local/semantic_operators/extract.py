import logging
from textwrap import dedent
from typing import Any, Dict, List, Optional

import jinja2
import polars as pl

from fenic._backends.local.semantic_operators.base import (
    BaseSingleColumnInputOperator,
    CompletionOnlyRequestSender,
)
from fenic._backends.local.semantic_operators.utils import (
    SCHEMA_EXPLANATION_INSTRUCTION_FRAGMENT,
)
from fenic._inference.language_model import InferenceConfiguration, LanguageModel
from fenic.core._logical_plan.resolved_types import (
    ResolvedModelAlias,
    ResolvedResponseFormat,
)
from fenic.core._utils.structured_outputs import (
    convert_resolved_response_format_to_key_descriptions,
    validate_structured_response_with_resolved_format,
)

logger = logging.getLogger(__name__)


class Extract(BaseSingleColumnInputOperator[str, Dict[str, Any]]):
    EXTRACT_SYSTEM_PROMPT = jinja2.Template(
        dedent("""\
        Extract information from the document according to the output schema.

        Output Schema:
        {{ schema_definition }}

        {{ schema_explanation }}

        Requirements:
        1. Extract only information explicitly stated in the document
        2. Do not infer, guess, or generate information not present
        3. Include all required fields - no extra fields, no missing fields
        4. For list fields, extract all items that match the field description
        5. Be thorough and precise - capture all relevant content without changing meaning
        """).strip()
    )

    def __init__(
        self,
        input: pl.Series,
        response_format: ResolvedResponseFormat,
        model: LanguageModel,
        max_output_tokens: int,
        temperature: float,
        model_alias: Optional[ResolvedModelAlias] = None,
    ):
        self.resolved_format = response_format
        super().__init__(
            input,
            CompletionOnlyRequestSender(
                operator_name="semantic.extract",
                inference_config=InferenceConfiguration(
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                    response_format=response_format,
                    model_profile=model_alias.profile if model_alias else None,
                ),
                model=model,
            ),
            None,
        )

    def build_system_message(self) -> str:
        schema_definition = convert_resolved_response_format_to_key_descriptions(self.resolved_format)
        return self.EXTRACT_SYSTEM_PROMPT.render(
            schema_explanation=SCHEMA_EXPLANATION_INSTRUCTION_FRAGMENT,
            schema_definition=schema_definition
        )

    def postprocess(
        self, responses: List[Optional[str]]
    ) -> List[Optional[Dict[str, Any]]]:
        return [
            validate_structured_response_with_resolved_format(
                json_resp, self.resolved_format, "semantic.extract"
            )
            for json_resp in responses
        ]
