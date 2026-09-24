"""Provider-independent, immutable questions for semantic.judge."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Sequence

from fenic.core.types.datatypes import FloatType, StringType, StructField, StructType


def _slug(value: str) -> str:
    return re.sub(r"\W", "_", value, flags=re.ASCII).strip("_").lower()


@dataclass(frozen=True)
class JudgeQuestion:
    """A named closed-set question with immutable criteria and deterministic fields."""

    name: str
    kind: str
    instructions: str
    options: tuple[tuple[str, Optional[str]], ...] = ()
    levels: tuple[str, ...] = ()
    criteria: tuple[tuple[str, str], ...] = ()
    premise: Optional[str] = None

    def __post_init__(self) -> None:
        """Reject malformed questions before they reach the planner or provider."""
        if not isinstance(self.name, str) or not re.fullmatch(
            r"[A-Za-z_]\w*", self.name
        ):
            raise ValueError("Judge question names must be identifiers")
        if not isinstance(self.instructions, str) or not self.instructions.strip():
            raise ValueError("Judge instructions must be a nonempty string")
        if self.kind not in {"noul", "choice", "score"}:
            raise ValueError("Unknown judge question kind")
        if not all(
            isinstance(value, tuple)
            for value in (self.options, self.levels, self.criteria)
        ):
            raise ValueError(
                "Judge criteria must be immutable tuples; use the question factories"
            )
        if any(
            not isinstance(pair, tuple) or len(pair) != 2
            for pair in (*self.options, *self.criteria)
        ):
            raise ValueError(
                "Judge option and criterion entries must be immutable pairs"
            )
        if self.kind == "choice":
            if not 2 <= len(self.options) <= 255 or self.levels or self.criteria:
                raise ValueError(
                    "Choice requires 2..255 options and no score/noul criteria"
                )
            names = [name for name, _ in self.options]
            if any(not isinstance(name, str) or not _slug(name) for name in names):
                raise ValueError("Choice options need nonempty string names")
            if len(set(names)) != len(names) or len(
                {_slug(name) for name in names}
            ) != len(names):
                raise ValueError("Choice option names collide")
            if any(
                description is not None and not isinstance(description, str)
                for _, description in self.options
            ):
                raise ValueError("Choice option descriptions must be strings or None")
        elif self.kind == "score":
            if not 2 <= len(self.levels) <= 10 or self.options or self.criteria:
                raise ValueError(
                    "Score requires 2..10 levels and no other criteria"
                )
            if any(
                not isinstance(level, str) or not level.strip() for level in self.levels
            ):
                raise ValueError("Score levels must be nonempty strings")
        elif (
            self.options
            or self.levels
            or any(
                key not in {"true", "false"} or not isinstance(value, str)
                for key, value in self.criteria
            )
            or len(dict(self.criteria)) != len(self.criteria)
        ):
            raise ValueError("Noul criteria must be unique true/false descriptions")
        if self.premise is not None and (
            not isinstance(self.premise, str) or self.premise == self.name
        ):
            raise ValueError("A premise must name a different Noul question")

    @classmethod
    def noul(
        cls,
        *,
        name: str,
        instructions: str,
        criteria: Optional[Mapping[str, str]] = None,
        premise: Optional[str] = None,
    ) -> JudgeQuestion:
        """Ask a yes/no question, retaining its affirmative probability."""
        return cls(
            name,
            "noul",
            instructions,
            criteria=tuple((criteria or {}).items()),
            premise=premise,
        )

    @classmethod
    def choice(
        cls,
        *,
        name: str,
        instructions: str,
        options: Mapping[str, Optional[str]],
        premise: Optional[str] = None,
    ) -> JudgeQuestion:
        """Select from described options, retaining their distribution."""
        return cls(
            name,
            "choice",
            instructions,
            options=tuple(options.items()),
            premise=premise,
        )

    @classmethod
    def score(
        cls,
        *,
        name: str,
        instructions: str,
        levels: Sequence[str],
        premise: Optional[str] = None,
    ) -> JudgeQuestion:
        """Rate on ordered levels, retaining the value and distribution."""
        return cls(name, "score", instructions, levels=tuple(levels), premise=premise)

    def body(self) -> dict[str, Any]:
        """Return a fresh SDK-compatible question body, without its transport name."""
        body: dict[str, Any] = {"type": self.kind, "instructions": self.instructions}
        if self.kind == "choice":
            body["criteria"] = dict(self.options)
        elif self.kind == "score":
            body["criteria"] = list(self.levels)
        elif self.criteria:
            body["criteria"] = dict(self.criteria)
        return body

    def fields(self) -> list[StructField]:
        """Return the ordered output fields contributed by this question."""
        if self.kind == "noul":
            fields = [StructField(f"{self.name}_p", FloatType)]
        else:
            fields = [
                StructField(
                    self.name, StringType if self.kind == "choice" else FloatType
                ),
                StructField(f"{self.name}_confidence", FloatType),
            ]
            suffixes = (
                [_slug(name) for name, _ in self.options]
                if self.kind == "choice"
                else [str(index) for index in range(len(self.levels))]
            )
            fields.extend(
                StructField(f"{self.name}_p_{suffix}", FloatType) for suffix in suffixes
            )
        if self.premise:
            fields.append(StructField(f"{self.name}_premise_p", FloatType))
        return fields

    def to_dict(self) -> dict[str, Any]:
        """Serialize ordered criteria without SDK types."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> JudgeQuestion:
        """Reconstruct the immutable question from its JSON representation."""
        data = dict(value)
        for key in ("options", "criteria"):
            data[key] = tuple(tuple(pair) for pair in data.get(key, ()))
        data["levels"] = tuple(data.get("levels", ()))
        return cls(**data)


def validate_questions(questions: Sequence[JudgeQuestion]) -> tuple[JudgeQuestion, ...]:
    """Reject ambiguous schemas and invalid or cyclic speculative premises."""
    items = tuple(questions)
    if not items or any(not isinstance(question, JudgeQuestion) for question in items):
        raise ValueError("judge requires a nonempty sequence of JudgeQuestion values")
    by_name = {question.name: question for question in items}
    fields = [field.name for question in items for field in question.fields()]
    if len(by_name) != len(items) or len(set(fields)) != len(fields):
        raise ValueError("Judge question names or generated output fields collide")
    for question in items:
        seen = {question.name}
        premise = question.premise
        while premise:
            if (
                premise not in by_name
                or by_name[premise].kind != "noul"
                or premise in seen
            ):
                raise ValueError(
                    "Judge premises must form an acyclic graph of Noul questions"
                )
            seen.add(premise)
            premise = by_name[premise].premise
    return items


def judge_schema(questions: Sequence[JudgeQuestion]) -> StructType:
    """Build a deterministic, flattened struct schema."""
    return StructType(
        [
            field
            for question in validate_questions(questions)
            for field in question.fields()
        ]
    )


def questions_json(questions: Sequence[JudgeQuestion]) -> str:
    """Canonical ordered payload for equality, caches, and protobuf transport."""
    return json.dumps(
        [question.to_dict() for question in questions],
        sort_keys=True,
        separators=(",", ":"),
    )


def _probability(value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(
            "Judge probability/confidence must be finite and between zero and one"
        )
    return float(value)


def flatten_answers(
    questions: Sequence[JudgeQuestion], answers: dict[str, Any]
) -> dict[str, Any]:
    """Validate answer vectors.

    Score consistency uses an empirical compatibility policy derived from
    observed two-decimal score/probability outputs in evidence revision
    ef27f01917a16113533463fa8797ab0b71f87ba6. It is not a guaranteed service
    contract.
    """
    output: dict[str, Any] = {}
    for question in questions:
        answer = answers.get(question.name)
        if not isinstance(answer, dict) or answer.get("type") != question.kind:
            raise ValueError("Missing or wrong-kind judge answer")
        if question.kind == "noul":
            output[f"{question.name}_p"] = _probability(answer.get("noul"))
            continue
        keys = (
            [name for name, _ in question.options]
            if question.kind == "choice"
            else [str(index) for index in range(len(question.levels))]
        )
        raw = answer.get("probabilities")
        if not isinstance(raw, dict):
            raise ValueError("Missing judge distribution")
        raw = {str(key): value for key, value in raw.items()}
        if set(raw) != set(keys):
            raise ValueError(
                "Judge distribution does not match the requested answer set"
            )
        probabilities = [_probability(raw[key]) for key in keys]
        if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-6):
            raise ValueError("Judge distribution must sum to one")
        output[f"{question.name}_confidence"] = _probability(answer.get("confidence"))
        if question.kind == "choice":
            selected = answer.get("choice")
            if selected not in keys or raw[selected] < max(probabilities) - 1e-6:
                raise ValueError("Judge choice is not a highest-probability option")
            output[question.name] = selected
        else:
            score = answer.get("score")
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(score)
                or not 0 <= score <= len(keys) - 1
            ):
                raise ValueError("Judge score is outside its level range")
            expected_score = sum(
                index * probability for index, probability in enumerate(probabilities)
            )
            score_tolerance = 0.005 * (1 + len(keys) * (len(keys) - 1) / 2) + 1e-6
            if not math.isclose(
                float(score), expected_score, rel_tol=0, abs_tol=score_tolerance
            ):
                raise ValueError(
                    "Judge score does not match its probability-weighted expectation"
                )
            output[question.name] = float(score)
        for key, probability in zip(keys, probabilities, strict=True):
            suffix = _slug(key) if question.kind == "choice" else key
            output[f"{question.name}_p_{suffix}"] = probability
    for question in questions:
        if question.premise:
            output[f"{question.name}_premise_p"] = output[f"{question.premise}_p"]
    return {
        field.name: output[field.name]
        for field in judge_schema(questions).struct_fields
    }
