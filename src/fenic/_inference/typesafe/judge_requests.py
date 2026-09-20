"""Estimated request packing; vendor token counts remain authoritative."""

import json
from typing import Callable, Sequence

from fenic.core.types.judge import JudgeQuestion, validate_questions

# https://docs.typesafe.ai/models.md, read 2026-09-19.
STATE_AND_QUESTION_LIMIT = 32_000
REQUEST_LIMIT = 64_000


def partition_questions(
    state: str,
    questions: Sequence[JudgeQuestion],
    count_tokens: Callable[[str], int],
) -> list[tuple[JudgeQuestion, ...]]:
    """Pack intact states and premise-connected question groups without truncation."""
    questions = validate_questions(questions)
    state_tokens = count_tokens(state)
    costs = {
        q.name: count_tokens(json.dumps(q.body(), sort_keys=True)) for q in questions
    }
    if any(state_tokens + cost > STATE_AND_QUESTION_LIMIT for cost in costs.values()):
        raise ValueError(
            "Estimated state plus longest question exceeds the provider envelope"
        )
    # Connected components keep speculative answers alongside every premise they need.
    groups: list[set[str]] = []
    for question in questions:
        names = {question.name}
        if question.premise:
            names.add(question.premise)
        overlaps = [group for group in groups if group & names]
        for group in overlaps:
            names.update(group)
            groups.remove(group)
        groups.append(names)
    groups.sort(
        key=lambda group: min(i for i, q in enumerate(questions) if q.name in group)
    )
    packed: list[tuple[JudgeQuestion, ...]] = []
    current: set[str] = set()
    used = state_tokens
    for group in groups:
        cost = sum(costs[name] for name in group)
        if state_tokens + cost > REQUEST_LIMIT:
            raise ValueError(
                "An indivisible judge question group exceeds the provider envelope"
            )
        if current and used + cost > REQUEST_LIMIT:
            packed.append(tuple(q for q in questions if q.name in current))
            current, used = set(), state_tokens
        current.update(group)
        used += cost
    if current:
        packed.append(tuple(q for q in questions if q.name in current))
    return packed
