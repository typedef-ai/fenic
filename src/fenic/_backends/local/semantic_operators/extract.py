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
from fenic.core._utils.schema import convert_custom_dtype_to_polars
from fenic.core.error import InternalError

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
        request_timeout: Optional[float] = None,
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
                    request_timeout=request_timeout,
                ),
                model=model,
            ),
            None,
            convert_custom_dtype_to_polars(self.resolved_format.struct_type)
        )

    def build_system_message(self) -> str:
        if not self.resolved_format.prompt_schema_definition:
            raise InternalError("Missing prompt_schema_definition for structured response format in semantic.extract")
        return self.EXTRACT_SYSTEM_PROMPT.render(
            schema_explanation=SCHEMA_EXPLANATION_INSTRUCTION_FRAGMENT,
            schema_definition=self.resolved_format.prompt_schema_definition
        )

    def postprocess(
        self, responses: List[Optional[str]]
    ) -> List[Optional[Dict[str, Any]]]:
        return [
            self.resolved_format.parse_structured_response(
                json_resp, "semantic.extract"
            )
            for json_resp in responses
        ]
