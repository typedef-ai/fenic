import logging
from typing import List, Optional, Union

import polars as pl

from fenic._backends.local.semantic_operators.base import (
    BaseSingleColumnInputOperator,
    CompletionOnlyRequestSender,
)
from fenic._inference.language_model import InferenceConfiguration, LanguageModel
from fenic.core._logical_plan.resolved_types import ResolvedModelAlias
from fenic.core.types import (
    KeyPoints,
    Paragraph,
)

logger = logging.getLogger(__name__)

class Summarize(BaseSingleColumnInputOperator[str, str]):
    SYSTEM_PROMPT = (
        '''You are tasked with summarizing the provided text.

            Core Requirements:
            - Use ONLY information explicitly stated in the source text
            - Maintain the original meaning, tone, and intent without distortion
            - Preserve the relative importance of ideas as presented in the original
            - Include context necessary for understanding key points
            - Omit redundant information and minor details

            Quality Criteria:
            - Completeness: Capture all major themes and conclusions
            - Accuracy: Ensure factual correctness and proper attribution
            - Coherence: Create logical flow between summarized points
            - Objectivity: Avoid inserting personal interpretation or bias

            Format: {format_description}

            Your summary should enable readers to understand the essential content without needing to read the original text, while clearly indicating this is a condensed version. '''
    )

    def __init__(
        self,
        input: pl.Series,
        format: Union[KeyPoints, Paragraph],
        temperature: float,
        model: LanguageModel,
        model_alias: Optional[ResolvedModelAlias] = None,
        request_timeout: Optional[float] = None,
    ):
        self.format = format

        super().__init__(
            input,
            CompletionOnlyRequestSender(
                operator_name="semantic.summarize",
                inference_config=InferenceConfiguration(
                    max_output_tokens=self.get_max_tokens(),
                    temperature=temperature,
                    model_profile=model_alias.profile if model_alias else None,
                    request_timeout=request_timeout,
                ),
                model=model,
            ),
            None,
        )

    def postprocess(self, responses) -> List[Optional[str]]:
        return responses

    def build_system_message(self) -> str:
        return self.SYSTEM_PROMPT.format(format_description=str(self.format))

    def get_max_tokens(self) -> int:
        return self.format.max_tokens()