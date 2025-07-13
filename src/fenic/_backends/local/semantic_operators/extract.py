import logging
from typing import Any, Dict, List, Optional

import polars as pl
from pydantic import BaseModel

from fenic._backends.local.semantic_operators.base import (
    BaseSingleColumnInputOperator,
    CompletionOnlyRequestSender,
)
from fenic._backends.local.semantic_operators.utils import (
    SchemaOperationType,
    build_schema_prompt_section,
    validate_structured_response,
)
from fenic._inference.language_model import InferenceConfiguration, LanguageModel

logger = logging.getLogger(__name__)


class Extract(BaseSingleColumnInputOperator[str, Dict[str, Any]]):
    SYSTEM_PROMPT_PREFIX = (
        "You are an expert at structured data extraction. "
        "Your task is to extract relevant information from a given document using only the information explicitly stated in the text. "
        "You must adhere strictly to the provided field definitions. Do not infer or generate information that is not directly supported by the document.\n\n"
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
        return (
            self.SYSTEM_PROMPT_PREFIX +
            build_schema_prompt_section(self.output_model, SchemaOperationType.EXTRACT)
        )

    def postprocess(
        self, responses: List[Optional[str]]
    ) -> List[Optional[Dict[str, Any]]]:
        return [
            validate_structured_response(json_resp, self.output_model, "semantic.extract")
            for json_resp in responses
        ]
