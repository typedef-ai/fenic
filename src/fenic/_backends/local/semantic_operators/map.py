from typing import List, Optional

import polars as pl
from pydantic import BaseModel

from fenic._backends.local.semantic_operators.base import (
    BaseSingleColumnInputOperator,
    CompletionOnlyRequestSender,
)
from fenic._inference.language_model import InferenceConfiguration, LanguageModel
from fenic.core.types import (
    MapExampleCollection,
)


class Map(BaseSingleColumnInputOperator[str, str]):
    SYSTEM_PROMPT = (
        "You are a helpful assistant that generates responses based on the instructions. "
    )

    def __init__(
        self,
        input: pl.Series,
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

    def build_system_message(self) -> str:
        return self.SYSTEM_PROMPT

    def postprocess(self, responses: List[Optional[str]]) -> List[Optional[str]]:
        return responses
