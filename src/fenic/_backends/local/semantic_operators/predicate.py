import json
import logging
from textwrap import dedent
from typing import List, Optional

import jinja2
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
from fenic.core._logical_plan.resolved_types import (
    ResolvedModelAlias,
    ResolvedResponseFormat,
)
from fenic.core.types import PredicateExample, PredicateExampleCollection

logger = logging.getLogger(__name__)

PREDICATE_FORMAT = ResolvedResponseFormat.from_pydantic_model(
    SimpleBooleanOutputModelResponse,
    generate_struct_type=False,
)


class Predicate(BaseMultiColumnInputOperator[str, bool]):
    SYSTEM_PROMPT = dedent("""\
    Evaluate the user's question or claim and respond with either true or false.

    Requirements:
    1. Output ONLY true or false - nothing else
    2. For yes/no questions, respond with "true" for yes and "false" for no
    3. If the answer is unclear or ambiguous, output false""")

    def __init__(
        self,
        input: pl.Series,
        jinja_template: str,
        model: LanguageModel,
        temperature: float,
        examples: Optional[PredicateExampleCollection] = None,
        model_alias: Optional[ResolvedModelAlias] = None,
        request_timeout: Optional[float] = None,
    ):
        super().__init__(
            input,
            CompletionOnlyRequestSender(
                operator_name="semantic.predicate",
                inference_config=InferenceConfiguration(
                  max_output_tokens=MAX_TOKENS_DETERMINISTIC_OUTPUT_SIZE,
                  response_format=PREDICATE_FORMAT,
                  temperature=temperature,
                  model_profile=model_alias.profile if model_alias else None,
                  request_timeout=request_timeout,
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
