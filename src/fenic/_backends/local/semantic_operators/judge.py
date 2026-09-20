"""Native physical judge operator, sharing fenic's registered inference client."""

import json
from typing import Optional, Sequence

import polars as pl

from fenic._inference.language_model import LanguageModel
from fenic.core._logical_plan.resolved_types import ResolvedModelAlias
from fenic.core._utils.schema import convert_custom_dtype_to_polars
from fenic.core.types.judge import JudgeQuestion, judge_schema


class Judge:
    """Execute one batch of states without constructing an SDK client or UDF."""

    def __init__(
        self,
        input: pl.Series,
        questions: Sequence[JudgeQuestion],
        model: LanguageModel,
        model_alias: Optional[ResolvedModelAlias] = None,
        request_timeout: Optional[float] = None,
    ):
        self.input = input
        self.questions = tuple(questions)
        self.model = model
        self.model_alias = model_alias
        self.request_timeout = request_timeout

    def execute(self) -> pl.Series:
        responses = self.model.get_judgments(
            self.input.to_list(),
            self.questions,
            model_profile=self.model_alias.profile if self.model_alias else None,
            request_timeout=self.request_timeout,
        )
        return pl.Series(
            self.input.name,
            [
                json.loads(response.completion) if response is not None else None
                for response in responses
            ],
            dtype=convert_custom_dtype_to_polars(judge_schema(self.questions)),
        )
