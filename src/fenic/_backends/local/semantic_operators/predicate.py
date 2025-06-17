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
from fenic._backends.local.semantic_operators.utils import (
    convert_row_to_instruction_context,
    uppercase_instruction_placeholder,
)
from fenic._constants import MAX_TOKENS_DETERMINISTIC_OUTPUT_SIZE
from fenic._inference.language_model import InferenceConfiguration, LanguageModel
from fenic.core.types import PredicateExample, PredicateExampleCollection

logger = logging.getLogger(__name__)




class Predicate(BaseMultiColumnInputOperator[str, bool]):
    SYSTEM_PROMPT = (
        "You are an AI assistant designed evaluate boolean claims. "
        "Your task is to determine whether a claim is supported by the provided context fields. "
        "Each input message will have two sections:\n"
        "1. A claim labeled with the prefix: ###Claim\n"
        "2. One or more context fields labeled with the prefix: ###Context\n"
        "Claims reference context fields using square brackets [LIKE_THIS]. "
        "Each context field will be labeled with its name in square brackets, matching the references in the claim. "
        "A claim is True if and only if it is supported by information in the context. "
        "A claim is False if it contradicts the context OR if the context doesn't provide enough information to verify it. "
        "Respond with either True or False."
        "Your response should evaluate the claim based on the context fields provided without using any external information."
    )

    def __init__(
        self,
        input: pl.DataFrame,
        user_instruction: str,
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
            examples,
        )
        self.user_instruction = uppercase_instruction_placeholder(user_instruction)

    def build_system_message(self) -> str:
        return self.SYSTEM_PROMPT

    def build_user_message(self, context_fields: dict[str, str]) -> str:
        prompt = (
            "### Claim\n"
            f"{self.user_instruction}\n\n"
            "### Context\n"
            f"{convert_row_to_instruction_context(context_fields)}"
        )
        return prompt

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
