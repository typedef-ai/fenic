import json
import logging
from typing import List, Optional

import polars as pl

from fenic._backends.local.semantic_operators.base import (
    BaseMultiColumnInputOperator,
    CompletionOnlyRequestSender,
)
from fenic._backends.local.semantic_operators.types import (
    SimpleBooleanOutputModelResponse,
)
from fenic._constants import MAX_TOKENS_DETERMINISTIC_OUTPUT_SIZE
from fenic._inference.language_model import InferenceConfiguration, LanguageModel
from fenic.core.types import PredicateExample, PredicateExampleCollection
import jinja2

logger = logging.getLogger(__name__)

class Predicate(BaseMultiColumnInputOperator[str, bool]):
    SYSTEM_PROMPT = (
        "Evaluate the user's question or claim and respond with either True or False.\n\n"
        "Requirements:\n"
        "1. Output ONLY True or False - nothing else\n"
        "2. If the answer is unclear or ambiguous, output False\n"
        "3. Evaluate based solely on the information provided"
    )

    def __init__(
        self,
        input: pl.Series,
        jinja_template: str,
        model: LanguageModel,
        temperature: float,
        examples: Optional[PredicateExampleCollection] = None,
    ):
        super().__init__(
            input,
            CompletionOnlyRequestSender(
                operator_name="semantic.predicate",
                inference_config=InferenceConfiguration(
                  max_output_tokens=MAX_TOKENS_DETERMINISTIC_OUTPUT_SIZE,
                  response_format=SimpleBooleanOutputModelResponse,
                  temperature=temperature,
                ),
                model=model,
            ),
            jinja_template=jinja2.Template(jinja_template),
            examples=examples,
        )

    def build_system_message(self) -> str:
        return self.SYSTEM_PROMPT

    def postprocess(self, responses: List[Optional[str]]) -> List[Optional[bool]]:
        predictions = []
        for response in responses:
            if not response:
                predictions.append(None)
            else:
                try:
                    data = json.loads(response)["output"]
                    predictions.append(data)
                except Exception as e:
                    logger.warning(
                        f"Invalid model output: {response} for semantic.predicate: {e}",
                    )
                    predictions.append(None)
        return predictions

    def convert_example_to_assistant_message(self, example: PredicateExample) -> str:
        return SimpleBooleanOutputModelResponse(output=example.output).model_dump_json()
