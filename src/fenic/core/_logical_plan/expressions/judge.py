"""Native logical expression for a shared-state set of typed judgments."""

from __future__ import annotations

import hashlib
import math
from typing import Optional, Sequence

from fenic.core._interfaces.session_state import BaseSessionState
from fenic.core._logical_plan.expressions.base import LogicalExpr, SemanticExpr
from fenic.core._logical_plan.resolved_types import ResolvedModelAlias
from fenic.core.error import TypeMismatchError, ValidationError
from fenic.core.types.datatypes import StringType
from fenic.core.types.judge import (
    JudgeQuestion,
    judge_schema,
    questions_json,
    validate_questions,
)
from fenic.core.types.schema import ColumnField


class SemanticJudgeExpr(SemanticExpr):
    """One state expression, many questions, a probability-bearing struct."""

    function_name = "semantic.judge"

    def __init__(
        self,
        state: LogicalExpr,
        questions: Sequence[JudgeQuestion],
        model_alias: Optional[ResolvedModelAlias] = None,
        request_timeout: Optional[float] = None,
    ):
        if request_timeout is not None and (
            isinstance(request_timeout, bool)
            or not math.isfinite(request_timeout)
            or request_timeout <= 0
        ):
            raise ValueError("Judge request_timeout must be positive")
        self.state = state
        self.questions = validate_questions(questions)
        self.model_alias = model_alias
        self.request_timeout = request_timeout
        self.return_type = judge_schema(self.questions)

    def children(self) -> list[LogicalExpr]:
        return [self.state]

    def _validate_completion_parameters(self, session_state: BaseSessionState) -> None:
        from fenic.core._logical_plan.utils import fetch_model_and_completion_parameters

        _, _, parameters = fetch_model_and_completion_parameters(
            self.model_alias, session_state.session_config
        )
        if not parameters.supports_judge:
            raise ValidationError(
                "semantic.judge requires a provider supporting typed judgments"
            )

    def to_column_field(self, plan, session_state: BaseSessionState) -> ColumnField:
        self._validate_completion_parameters(session_state)
        if self.state.to_column_field(plan, session_state).data_type != StringType:
            raise TypeMismatchError.from_message(
                "semantic.judge state must be a string column"
            )
        return ColumnField(name=str(self), data_type=self.return_type)

    def __str__(self) -> str:
        digest = hashlib.sha256(questions_json(self.questions).encode()).hexdigest()[
            :12
        ]
        return f"semantic.judge_{digest}({self.state})"

    def _eq_specific(self, other: SemanticJudgeExpr) -> bool:
        return (
            self.questions == other.questions
            and self.model_alias == other.model_alias
            and self.request_timeout == other.request_timeout
        )
