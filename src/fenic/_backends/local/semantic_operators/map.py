from typing import List, Optional

import polars as pl
from pydantic import BaseModel

from fenic._backends.local.semantic_operators.base import (
    BaseMultiColumnInputOperator,
    CompletionOnlyRequestSender,
)
from fenic._backends.local.semantic_operators.utils import (
    convert_row_to_instruction_context,
    uppercase_instruction_placeholder,
)
from fenic._inference.language_model import InferenceConfiguration, LanguageModel
from fenic.core._utils.misc import parse_instruction
from fenic.core.types import (
    MapExampleCollection,
)


class Map(BaseMultiColumnInputOperator[str, str]):
    SYSTEM_PROMPT = (
        "You are an AI assistant designed to follow instructions. "
        "Your task is to generate responses based on instructions that reference one or more context fields. "
        "Each input message will have two sections:\n"
        "1. An instruction labeled with the prefix: ###Instruction\n"
        "2. One or more context fields labeled with the prefix: ###Context\n"
        "The instruction will reference the context fields using square brackets [LIKE_THIS]. "
        "Each context field will be labeled with its name in square brackets, matching the references in the instruction. "
        "Your response should fulfill the instruction by appropriately integrating each of the referenced context fields without using any external information. "
        "Your response should not include unnecessary preamble or explanation. " # anthropic LOVES a friendly preamble.
    )

    def __init__(
        self,
        input: pl.DataFrame,
        user_instruction: str,
        model: LanguageModel,
        max_tokens: int,
        temperature: float,
        model_alias: Optional[str] = None,
        response_format: Optional[type[BaseModel]] = None,
        examples: Optional[MapExampleCollection] = None,
    ):
        super().__init__(
            input,
            CompletionOnlyRequestSender(
                model=model,
                operator_name="semantic.map",
                inference_config=InferenceConfiguration(
                    max_output_tokens=max_tokens,
                    response_format=response_format,
                    temperature=temperature,
                ),
            ),
            examples,
        )
        self.referenced_cols = parse_instruction(user_instruction)
        self.user_instruction = uppercase_instruction_placeholder(user_instruction)

    def build_system_message(self) -> str:
        return self.SYSTEM_PROMPT

    def build_user_message(self, input: dict[str, str]) -> str:
        prompt = (
            "### Instruction\n"
            f"{self.user_instruction}\n\n"
            "### Context\n"
            f"{convert_row_to_instruction_context(input)}"
        )

        return prompt

    def postprocess(self, responses: List[Optional[str]]) -> List[Optional[str]]:
        return responses
