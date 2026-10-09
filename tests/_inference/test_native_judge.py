"""Native judge contracts with a real SDK response model and fake transport."""

import asyncio
import hashlib
import json
import socket
import time
from dataclasses import replace
from types import SimpleNamespace
from typing import get_args
from unittest.mock import AsyncMock, Mock

import polars as pl
import pytest
from pydantic import ValidationError as PydanticValidationError

import fenic as fc
from fenic._backends.local.model_registry import SessionModelRegistry
from fenic._backends.local.semantic_operators.decision import DecisionRequestSender
from fenic._inference.cache.key_builder import compute_request_fingerprint
from fenic._inference.cache.sqlite_cache import SQLiteLLMCache
from fenic._inference.model_client import FatalException, TransientException
from fenic._inference.rate_limit_strategy import InputTokenRateLimitStrategy
from fenic._inference.types import (
    FenicCompletionsRequest,
    FenicCompletionsResponse,
    LMRequestMessages,
)
from fenic._inference.typesafe.judge_requests import partition_questions
from fenic._inference.typesafe.typesafe_provider import TypeSafeModelProvider
from fenic.core._inference.model_catalog import (
    ModelProvider,
    TypeSafeLanguageModelName,
    model_catalog,
)
from fenic.core._logical_plan.expressions.judge import SemanticJudgeExpr
from fenic.core._logical_plan.resolved_types import ResolvedModelAlias
from fenic.core._resolved_session_config import (
    ResolvedLanguageModelConfig,
    ResolvedSemanticConfig,
    ResolvedTypeSafeModelConfig,
)
from fenic.core._serde.proto.expression_serde import (
    deserialize_logical_expr,
    serialize_logical_expr,
)
from fenic.core._serde.proto.serde_context import SerdeContext
from fenic.core.error import ExecutionError, ValidationError
from fenic.core.types.judge import (
    flatten_answers,
    judge_schema,
    questions_json,
    validate_questions,
)


@pytest.fixture(autouse=True)
def clear_typesafe_endpoint(monkeypatch):
    monkeypatch.delenv("TYPESAFE_BASE_URL", raising=False)


def questions():
    return (
        fc.JudgeQuestion.noul(name="ok", instructions="Is the state acceptable?"),
        fc.JudgeQuestion.choice(
            name="kind",
            instructions="What kind?",
            options={"a": "First", "b": "Second"},
        ),
        fc.JudgeQuestion.score(
            name="severity",
            instructions="How severe?",
            levels=["low", "high"],
            premise="ok",
        ),
    )


def answer(body, state="text"):
    if body["type"] == "noul":
        return {"type": "noul", "noul": 0.9 if state != "bad" else 0.1}
    if body["type"] == "choice":
        keys = list(body["criteria"])
        return {
            "type": "choice",
            "choice": keys[0],
            "confidence": 1.0,
            "probabilities": {key: float(index == 0) for index, key in enumerate(keys)},
        }
    return {
        "type": "score",
        "score": 0.75,
        "confidence": 0.5,
        "probabilities": {0: 0.25, 1: 0.75},
        "legend": {index: level for index, level in enumerate(body["criteria"])},
    }


@pytest.fixture
def native_session(tmp_path, monkeypatch):
    response_type = pytest.importorskip("typesafe_sdk").SystemOneResponse

    async def no_validation(_providers):
        return None

    monkeypatch.setattr(
        "fenic._backends.local.model_registry._validate_provider_api_keys",
        no_validation,
    )
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    calls = []

    async def evaluate(state, bodies, **_kwargs):
        calls.append((state, bodies))
        return response_type.model_validate(
            {
                "answers": {name: answer(body, state) for name, body in bodies.items()},
                "usage": {"input_tokens": 100, "output_tokens": 20},
                "model": "jev-1.13.0",
            }
        )

    sdk = SimpleNamespace(system_one=evaluate, aclose=AsyncMock())
    monkeypatch.setattr(
        TypeSafeModelProvider, "create_aio_client", lambda self: sdk
    )
    session = fc.Session.get_or_create(
        fc.SessionConfig(
            app_name="native_judge",
            db_path=tmp_path,
            semantic=fc.SemanticConfig(
                language_models={
                    "judge": fc.TypeSafeLanguageModel(
                        model_name="jev-1.13.0", rpm=1200, tpm=1_000_000
                    )
                },
                llm_response_cache=fc.LLMResponseCacheConfig(),
            ),
        )
    )

    def refuse_network(*_args, **_kwargs):
        raise AssertionError("offline judge tests must not open a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse_network)
    yield session, calls, sdk
    session.stop(skip_usage_summary=True)


def test_public_expression_and_native_execution(native_session):
    session, calls, _ = native_session
    client = session._session_state.get_language_model(
        ResolvedModelAlias("judge", None)
    ).client
    assert isinstance(client.rate_limit_strategy, InputTokenRateLimitStrategy)
    expr = fc.semantic.judge(
        state="text", questions=list(questions()), model_alias="judge"
    )
    assert isinstance(expr._logical_expr, SemanticJudgeExpr)
    source = session.create_dataframe(pl.DataFrame({"text": ["x", "x", "", None]}))
    execution = source.with_column("j", expr).unnest("j").collect()
    result = execution.data
    assert len(calls) == 2
    assert all(len(bodies) == len(questions()) for _, bodies in calls)
    assert result["ok_p"].to_list() == pytest.approx([0.9, 0.9, 0.9, None], nan_ok=True)
    assert result["severity"].to_list()[:3] == pytest.approx([0.75] * 3)
    assert result["severity_premise_p"].to_list()[:3] == pytest.approx([0.9] * 3)
    assert execution.metrics.total_lm_metrics.num_output_tokens == 40
    source.with_column("j", expr).to_polars()
    assert len(calls) == 2


def test_schema_roundtrip_and_equality():
    original = fc.semantic.judge(
        state="text", questions=list(questions()), request_timeout=17
    )._logical_expr
    context = SerdeContext()
    restored = deserialize_logical_expr(
        serialize_logical_expr(original, context), context
    )
    assert original == restored
    assert (
        original
        != fc.semantic.judge(
            state="text", questions=list(questions()), request_timeout=18
        )._logical_expr
    )
    assert (
        original
        != fc.semantic.judge(
            state="other", questions=list(questions()), request_timeout=17
        )._logical_expr
    )
    assert original.return_type == judge_schema(questions())


@pytest.mark.parametrize(
    ("level_count", "valid"),
    [(1, False), (2, True), (10, True), (11, False)],
)
def test_score_level_range_is_enforced_before_inference(level_count, valid):
    levels = [f"level-{index}" for index in range(level_count)]

    if not valid:
        with pytest.raises(ValueError, match="2..10"):
            fc.JudgeQuestion.score(
                name="score", instructions="How strong?", levels=levels
            )
        with pytest.raises(ValueError, match="2..10"):
            fc.JudgeQuestion(
                name="score",
                kind="score",
                instructions="How strong?",
                levels=tuple(levels),
            )
        with pytest.raises(ValueError, match="2..10"):
            fc.JudgeQuestion.from_dict(
                {
                    "name": "score",
                    "kind": "score",
                    "instructions": "How strong?",
                    "levels": levels,
                }
            )
        return

    question = fc.JudgeQuestion.score(
        name="score", instructions="How strong?", levels=levels
    )
    direct = fc.JudgeQuestion(
        name="score",
        kind="score",
        instructions="How strong?",
        levels=tuple(levels),
    )
    assert question == direct
    assert fc.JudgeQuestion.from_dict(question.to_dict()) == question


def test_cache_fingerprint_keeps_questions_order_and_endpoint():
    request = FenicCompletionsRequest(
        LMRequestMessages("", [], "state"),
        None,
        None,
        None,
        None,
        judge_questions=questions(),
    )
    first = compute_request_fingerprint(request, "judge")
    changed = replace(
        request,
        judge_questions=(replace(questions()[0], instructions="Another question"),),
    )
    assert first != compute_request_fingerprint(changed, "judge")
    assert first != compute_request_fingerprint(
        replace(request, judge_questions=tuple(reversed(questions()))), "judge"
    )
    assert first != compute_request_fingerprint(
        request, "judge", base_url="https://example.invalid"
    )
    assert first != compute_request_fingerprint(
        replace(request, judge_questions=None), "judge"
    )


def test_judge_decoder_upgrade_misses_persistent_v1_cache(native_session, tmp_path):
    session, calls, _ = native_session
    client = session._session_state.get_language_model(
        ResolvedModelAlias("judge", None)
    ).client
    question = replace(questions()[2], premise=None)
    messages = LMRequestMessages("", [], "x")
    legacy_payload = {
        "model": client.model,
        "base_url": client.model_provider_class._base_url,
        "messages": messages.encode().hex(),
        "max_tokens": None,
        "temperature": None,
        "model_profile": None,
        "profile_hash": None,
        "top_logprobs": None,
        "request_kind": "judge-v1",
        "judge_questions": questions_json([question]),
    }
    legacy_key = hashlib.sha256(
        json.dumps(legacy_payload, sort_keys=True).encode()
    ).hexdigest()
    legacy_response = FenicCompletionsResponse(
        json.dumps(
            {
                "severity": 0.1,
                "severity_confidence": 0.8,
                "severity_p_0": 0.9,
                "severity_p_1": 0.1,
            }
        ),
        None,
    )
    cache_path = str(tmp_path / "decoder-upgrade.db")
    writer = SQLiteLLMCache(db_path=cache_path, namespace="upgrade")
    assert writer.set(legacy_key, legacy_response, client.model)
    writer.close()
    reader = SQLiteLLMCache(db_path=cache_path, namespace="upgrade")
    original_cache = client.cache
    client.cache = reader
    try:
        frame = session.create_dataframe({"text": ["x"]}).with_column(
            "j", fc.semantic.judge(state="text", questions=[question])
        )
        row = frame.collect().data["j"].to_list()[0]
        assert len(calls) == 1, "the legacy decoded entry bypassed the new decoder"
        assert row["severity"] == pytest.approx(0.75)
        assert frame.collect().data["j"].to_list()[0] == row
        assert len(calls) == 1, "the fresh namespace should still cache valid answers"
        assert reader.get(legacy_key).completion == legacy_response.completion
    finally:
        client.cache = original_cache
        reader.close()


def test_structured_judge_state_cache_identity_is_canonical_and_distinct_from_text():
    question = (questions()[0],)
    state = {
        "input": "unicode ✓",
        "examples": [{"input": "first", "response": ["yes", None]}],
    }
    serialized_state = json.dumps(
        state, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    structured = FenicCompletionsRequest(
        LMRequestMessages("", [], serialized_state),
        None,
        None,
        None,
        None,
        judge_questions=question,
        judge_state=state,
    )
    reordered = replace(
        structured,
        judge_state={
            "examples": [{"response": ["yes", None], "input": "first"}],
            "input": "unicode ✓",
        },
    )
    literal_json_text = replace(
        structured,
        messages=LMRequestMessages("", [], serialized_state),
        judge_state=None,
    )

    key = compute_request_fingerprint(structured, "judge")
    assert key == compute_request_fingerprint(reordered, "judge")
    assert key != compute_request_fingerprint(literal_json_text, "judge")


def test_structured_judge_state_reaches_sdk_transport_as_object(
    native_session, monkeypatch
):
    import httpx2
    from typesafe_sdk import AsyncTypeSafeClient

    session, _, _ = native_session
    model = session._session_state.get_language_model(
        ResolvedModelAlias("judge", None)
    )
    client = model.client
    prepared_bodies = []
    counted_values = []

    async def transport(request):
        body = json.loads(request.content)
        prepared_bodies.append(body)
        return httpx2.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "answers": {
                    name: {"type": "noul", "noul": 0.9}
                    for name in body["questions"]
                },
                "usage": {"input_tokens": 100, "output_tokens": 20},
            },
            request=request,
        )

    def count_tokens(value):
        assert isinstance(value, str)
        counted_values.append(value)
        return len(value)

    monkeypatch.setattr(
        client,
        "_client",
        AsyncTypeSafeClient(
            api_key="test-key",
            base_url="https://typesafe.invalid",
            transport=httpx2.MockTransport(transport),
        ),
    )
    monkeypatch.setattr(client.token_counter, "count_tokens", count_tokens)
    state = {
        "input": "hello ✓",
        "examples": [
            {"input": "nested", "response": {"label": "yes", "scores": [1, None]}}
        ],
    }
    structured_response = model.get_judgments([state], (questions()[0],))[0]
    decision_sender = DecisionRequestSender(
        SimpleNamespace(
            model=model,
            inference_config=SimpleNamespace(
                temperature=0, model_profile=None, request_timeout=None
            ),
        ),
        "Is this acceptable?",
    )
    outputs = decision_sender.send_requests(
        [
            LMRequestMessages(
                "",
                [SimpleNamespace(user="nested", assistant='{"label":"yes"}')],
                "hello ✓",
            )
        ]
    )
    plain_text_response = model.get_judgments(
        ['{"this is":"plain JSON-looking text"}'], (questions()[0],)
    )[0]

    assert structured_response.completion == '{"ok_p": 0.9}'
    assert outputs == ['{"output": true}']
    assert plain_text_response.completion == '{"ok_p": 0.9}'
    assert prepared_bodies[0]["state"] == state
    assert prepared_bodies[1]["state"] == {
        "input": "hello ✓",
        "examples": [{"input": "nested", "response": '{"label":"yes"}'}],
    }
    assert prepared_bodies[2]["state"] == '{"this is":"plain JSON-looking text"}'
    assert all(isinstance(value, str) for value in counted_values)
    canonical_state = json.dumps(
        state, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    assert canonical_state in counted_values


@pytest.mark.parametrize(
    "probabilities",
    [
        {"a": 0, "b": 0},
        {"a": float("nan"), "b": 1},
        {"a": 1},
        {"a": 1.1, "b": -0.1},
    ],
)
def test_invalid_vectors_are_refused(probabilities):
    with pytest.raises(ValueError):
        flatten_answers(
            [questions()[1]],
            {
                "kind": {
                    "type": "choice",
                    "choice": "a",
                    "confidence": 0.5,
                    "probabilities": probabilities,
                }
            },
        )


@pytest.mark.parametrize("option_count", [12, 255])
@pytest.mark.parametrize("mass", [0.99, 1.01])
def test_rounded_choice_vectors_are_normalized_with_bounded_mass(option_count, mass):
    options = {f"opt{index}": None for index in range(option_count)}
    question = fc.JudgeQuestion.choice(
        name="kind", instructions="Which category?", options=options
    )
    values = [0.9, 0.06, 0.01 if mass < 1 else 0.03, 0.01, 0.01]
    probabilities = {
        key: values[index] if index < len(values) else 0.0
        for index, key in enumerate(options)
    }
    result = flatten_answers(
        [question],
        {
            "kind": {
                "type": "choice",
                "choice": "opt0",
                "confidence": 0.8,
                "probabilities": probabilities,
            }
        },
    )
    assert result["kind"] == "opt0"
    assert sum(result[f"kind_p_{key}"] for key in options) == pytest.approx(1)
    for key, probability in probabilities.items():
        assert result[f"kind_p_{key}"] == pytest.approx(probability / mass)


@pytest.mark.parametrize(
    "malformed",
    [
        "negative",
        "nan",
        "inf",
        "above_one",
        "bool",
        "string",
        "zero",
        "low_mass",
        "high_mass",
        "missing",
        "extra",
        "wrong_argmax",
    ],
)
@pytest.mark.parametrize("option_count", [12, 255])
def test_rounded_choice_policy_still_rejects_malformed_vectors(malformed, option_count):
    options = {f"opt{index}": None for index in range(option_count)}
    question = fc.JudgeQuestion.choice(
        name="kind", instructions="Which category?", options=options
    )
    probabilities = dict.fromkeys(options, 0.0)
    probabilities["opt0"] = 0.9
    probabilities["opt1"] = 0.1
    invalid_values = {
        "negative": -0.01,
        "nan": float("nan"),
        "inf": float("inf"),
        "above_one": 1.01,
        "bool": True,
        "string": "0",
    }
    if malformed in invalid_values:
        probabilities["opt2"] = invalid_values[malformed]
    elif malformed == "zero":
        probabilities = dict.fromkeys(options, 0.0)
    elif malformed in {"low_mass", "high_mass"}:
        probabilities["opt0"] = 0.87 if malformed == "low_mass" else 0.93
    elif malformed == "missing":
        probabilities.pop("opt2")
    elif malformed == "extra":
        probabilities["extra"] = 0.0
    with pytest.raises(ValueError):
        flatten_answers(
            [question],
            {
                "kind": {
                    "type": "choice",
                    "choice": "opt1" if malformed == "wrong_argmax" else "opt0",
                    "confidence": 0.8,
                    "probabilities": probabilities,
                }
            },
        )


@pytest.mark.parametrize("mass", [0.99, 1.01, 0.97, 1.03])
def test_rounded_choice_cache_accepts_only_validated_mass(native_session, mass):
    session, calls, sdk = native_session
    original = sdk.system_one

    async def rounded(state, bodies, **kwargs):
        result = await original(state, bodies, **kwargs)
        answer = result.answers["kind"].model_copy(
            update={"probabilities": {"a": 0.9, "b": mass - 0.9}}
        )
        return result.model_copy(update={"answers": {"kind": answer}})

    sdk.system_one = rounded
    frame = session.create_dataframe({"text": ["x"]}).with_column(
        "j", fc.semantic.judge(state="text", questions=[questions()[1]])
    )
    valid = abs(mass - 1) <= 0.010001
    for _ in range(2):
        row = frame.collect().data["j"].to_list()[0]
        if valid:
            assert row["kind"] == "a"
            assert row["kind_p_a"] == pytest.approx(0.9 / mass)
        else:
            assert row is None
    assert len(calls) == (1 if valid else 2)


@pytest.mark.parametrize(
    ("score", "probabilities", "expected"),
    [
        (
            0.75,
            [
                0.4300000071525574,
                0.3799999952316284,
                0.17000000178813934,
                0.019999999552965164,
            ],
            0.7799999974668026,
        ),
        (
            2.559999942779541,
            [
                0.0,
                0.009999999776482582,
                0.38999998569488525,
                0.6000000238418579,
            ],
            2.590000042691827,
        ),
    ],
)
def test_score_accepts_archived_two_decimal_compatibility_values(
    score, probabilities, expected
):
    question = fc.JudgeQuestion.score(
        name="severity",
        instructions="How severe?",
        levels=["low", "medium", "high", "critical"],
    )
    answers = {
        "severity": {
            "type": "score",
            "score": score,
            "confidence": 0.5,
            "probabilities": dict(enumerate(probabilities)),
            "legend": dict(enumerate(question.levels)),
        }
    }

    assert sum(index * value for index, value in enumerate(probabilities)) == expected
    assert flatten_answers([question], answers)["severity"] == score


@pytest.mark.parametrize(("level_count", "offset", "raises"), [(2, -1e-7, False), (2, 1e-7, True), (10, -1e-7, False), (10, 1e-7, True)])
def test_score_tolerance_scales_with_level_count(level_count, offset, raises):
    score_tolerance = 0.005 * (1 + level_count * (level_count - 1) / 2) + 1e-6
    question = fc.JudgeQuestion.score(
        name="severity",
        instructions="How severe?",
        levels=[f"level-{index}" for index in range(level_count)],
    )
    answers = {
        "severity": {
            "type": "score",
            "score": score_tolerance + offset,
            "confidence": 0.5,
            "probabilities": {
                index: float(index == 0) for index in range(level_count)
            },
            "legend": dict(enumerate(question.levels)),
        }
    }
    if raises:
        with pytest.raises(ValueError, match="expectation"):
            flatten_answers([question], answers)
    else:
        assert flatten_answers([question], answers)["severity"] == pytest.approx(
            score_tolerance + offset
        )


def test_score_rejects_material_contradiction():
    question = fc.JudgeQuestion.score(
        name="severity",
        instructions="How severe?",
        levels=["low", "high"],
    )
    answers = {
        "severity": {
            "type": "score",
            "score": 1,
            "confidence": 0.5,
            "probabilities": {"0": 1, "1": 0},
            "legend": {"0": "low", "1": "high"},
        }
    }

    with pytest.raises(ValueError, match="expectation"):
        flatten_answers([question], answers)


@pytest.mark.parametrize(
    "legend",
    [
        None,
        {},
        {"0": "high", "1": "low"},
        {"0": "low"},
        {"0": "low", "1": "high", "2": "extra"},
        {"0": "low", "1": "other"},
    ],
)
def test_score_rejects_legend_that_does_not_match_requested_levels(legend):
    question = fc.JudgeQuestion.score(
        name="severity", instructions="How severe?", levels=["low", "high"]
    )
    with pytest.raises(ValueError, match="legend"):
        flatten_answers(
            [question],
            {
                "severity": {
                    "type": "score",
                    "score": 0,
                    "confidence": 1,
                    "probabilities": {"0": 1, "1": 0},
                    "legend": legend,
                }
            },
        )


@pytest.mark.parametrize("legend", [{0: "low", 1: "high"}, {"0": "low", "1": "high"}])
def test_score_accepts_legend_with_integer_or_json_string_indexes(legend):
    question = fc.JudgeQuestion.score(
        name="severity", instructions="How severe?", levels=["low", "high"]
    )
    result = flatten_answers(
        [question],
        {
            "severity": {
                "type": "score",
                "score": 0,
                "confidence": 1,
                "probabilities": {"0": 1, "1": 0},
                "legend": legend,
            }
        },
    )
    assert result["severity_p_0"] == 1
    assert result["severity_p_1"] == 0


def test_reversed_score_legend_is_not_cached_and_billed_usage_is_retained(
    native_session,
):
    session, calls, sdk = native_session
    response_type = pytest.importorskip("typesafe_sdk").SystemOneResponse
    original = sdk.system_one

    async def reversed_legend(state, bodies, **kwargs):
        result = await original(state, bodies, **kwargs)
        payload = result.model_dump()
        payload["answers"]["severity"].update(
            score=0,
            confidence=1,
            probabilities={0: 1, 1: 0},
            legend={0: "high", 1: "low"},
        )
        return response_type.model_validate(payload)

    sdk.system_one = reversed_legend
    frame = session.create_dataframe({"text": ["x"]}).with_column(
        "j", fc.semantic.judge(state="text", questions=list(questions()))
    )
    for _ in range(2):
        result = frame.collect()
        assert result.data["j"].to_list() == [None]
        assert result.metrics.total_lm_metrics.num_uncached_input_tokens == 100
        assert result.metrics.total_lm_metrics.num_output_tokens == 20
    assert len(calls) == 2


def test_inconsistent_score_is_not_cached_and_billed_usage_is_retained(
    native_session,
):
    session, calls, sdk = native_session
    original = sdk.system_one

    async def inconsistent_score(state, bodies, **kwargs):
        result = await original(state, bodies, **kwargs)
        answers = {
            name: answer.model_copy(
                update={"score": 0.25}
            )
            if answer.type == "score"
            else answer
            for name, answer in result.answers.items()
        }
        return result.model_copy(update={"answers": answers})

    sdk.system_one = inconsistent_score
    frame = session.create_dataframe({"text": ["x"]}).with_column(
        "j", fc.semantic.judge(state="text", questions=list(questions()))
    )
    for _ in range(2):
        result = frame.collect()
        assert result.data["j"].to_list() == [None]
        assert result.metrics.total_lm_metrics.num_uncached_input_tokens == 100
        assert result.metrics.total_lm_metrics.num_output_tokens == 20
    assert len(calls) == 2


def test_field_collision_and_missing_premise_are_refused():
    with pytest.raises(ValueError, match="collide"):
        validate_questions(
            [
                questions()[0],
                fc.JudgeQuestion.choice(
                    name="ok_p", instructions="Which?", options={"a": None, "b": None}
                ),
            ]
        )
    with pytest.raises(ValueError, match="acyclic"):
        validate_questions([questions()[2]])
    with pytest.raises(ValueError, match="collide"):
        fc.JudgeQuestion.choice(
            name="x", instructions="Which?", options={"A-B": None, "A B": None}
        )


def test_partition_preserves_state_bound_and_premises():
    with pytest.raises(ValueError, match="longest"):
        partition_questions("x" * 32000, questions(), len)
    many = [
        fc.JudgeQuestion.noul(name=f"q{i}", instructions="x" * 20000) for i in range(4)
    ]
    packed = partition_questions("state", many, len)
    assert len(packed) == 2
    assert [q for group in packed for q in group] == many
    assert sum(len(group) for group in partition_questions("", questions(), len)) == 3


def test_failed_vector_is_not_cached_and_usage_is_retained(native_session):
    session, calls, sdk = native_session
    original = sdk.system_one

    async def malformed(state, bodies, **kwargs):
        result = await original(state, bodies, **kwargs)
        return result.model_copy(update={"answers": {}})

    sdk.system_one = malformed
    frame = session.create_dataframe({"text": ["x"]}).with_column(
        "j",
        fc.semantic.judge(state="text", questions=list(questions())),
    )
    for _ in range(2):
        result = frame.collect()
        assert result.data["j"].to_list() == [None]
        assert result.metrics.total_lm_metrics.num_uncached_input_tokens == 100
        assert result.metrics.total_lm_metrics.num_output_tokens == 20
        assert result.metrics.total_lm_metrics.cost > 0
    assert len(calls) == 2


@pytest.mark.parametrize(
    ("input_tokens", "output_tokens", "complete_usage"),
    [
        (100, 20, True),
        (None, 20, False),
        (100, None, False),
        (None, None, False),
    ],
)
def test_valid_answers_with_incomplete_usage_are_cached_without_aggregate_totals(
    native_session, monkeypatch, caplog, input_tokens, output_tokens, complete_usage
):
    session, calls, sdk = native_session
    client = session._session_state.get_language_model(
        ResolvedModelAlias("judge", None)
    ).client
    reconcile = Mock(wraps=client._reconcile_completion)
    settle = Mock(wraps=client.rate_limit_strategy.settle)
    observe = Mock(wraps=client._output_estimator.observe)
    monkeypatch.setattr(client, "_reconcile_completion", reconcile)
    monkeypatch.setattr(client.rate_limit_strategy, "settle", settle)
    monkeypatch.setattr(client._output_estimator, "observe", observe)
    original = sdk.system_one

    async def response_with_usage(state, bodies, **kwargs):
        result = await original(state, bodies, **kwargs)
        return SimpleNamespace(
            answers=result.answers,
            usage=SimpleNamespace(
                input_tokens=input_tokens, output_tokens=output_tokens
            ),
        )

    sdk.system_one = response_with_usage
    frame = session.create_dataframe({"text": ["row-content-sentinel"]}).with_column(
        "j", fc.semantic.judge(state="text", questions=list(questions()))
    )
    first = frame.collect()
    second = frame.collect()

    assert first.data["j"].to_list()[0] is not None
    assert second.data["j"].to_list()[0] is not None
    assert len(calls) == 1
    assert first.metrics.total_lm_metrics.num_requests == 1
    assert reconcile.call_count == int(complete_usage)
    if complete_usage:
        assert first.metrics.total_lm_metrics.num_uncached_input_tokens == 100
        assert first.metrics.total_lm_metrics.num_output_tokens == 20
        assert first.metrics.total_lm_metrics.cost == pytest.approx(
            model_catalog.calculate_completion_model_cost(
                model_provider=ModelProvider.TYPESAFE,
                model_name="jev-1.13.0",
                uncached_input_tokens=100,
                cached_input_tokens_read=0,
                output_tokens=20,
            )
        )
        assert settle.call_count == observe.call_count == 1
    else:
        assert first.metrics.total_lm_metrics.num_uncached_input_tokens == 0
        assert first.metrics.total_lm_metrics.num_output_tokens == 0
        assert first.metrics.total_lm_metrics.cost == 0
        assert settle.call_count == observe.call_count == 0
        assert "incomplete usage" in caplog.text
        assert "row-content-sentinel" not in caplog.text


@pytest.mark.parametrize(
    ("input_tokens", "output_tokens"),
    [
        (-1, 20),
        (100, -1),
        (True, 20),
        (100, True),
        (1.5, 20),
        (100, 1.5),
        ("100", 20),
        (100, "20"),
    ],
)
def test_invalid_reported_usage_is_not_cached(
    native_session, input_tokens, output_tokens
):
    session, calls, sdk = native_session
    original = sdk.system_one

    async def response_with_invalid_usage(state, bodies, **kwargs):
        result = await original(state, bodies, **kwargs)
        return SimpleNamespace(
            answers=result.answers,
            usage=SimpleNamespace(
                input_tokens=input_tokens, output_tokens=output_tokens
            ),
        )

    sdk.system_one = response_with_invalid_usage
    frame = session.create_dataframe({"text": ["x"]}).with_column(
        "j", fc.semantic.judge(state="text", questions=list(questions()))
    )
    for _ in range(2):
        result = frame.collect()
        assert result.data["j"].to_list() == [None]
        assert result.metrics.total_lm_metrics.num_requests == 0
    assert len(calls) == 2


def test_bad_state_type(native_session):
    session, calls, _ = native_session
    with pytest.raises(Exception, match="string"):
        session.create_dataframe({"text": [7]}).with_column(
            "j",
            fc.semantic.judge(state="text", questions=list(questions())),
        )
    assert not calls


def test_size_rejections_warn_once_and_skip_only_rejected_rows(
    native_session, monkeypatch, caplog
):
    session, calls, _ = native_session
    client = session._session_state.get_language_model(
        ResolvedModelAlias("judge", None)
    ).client
    monkeypatch.setattr(client.token_counter, "count_tokens", len)
    secret = "valid-state-sentinel"
    frame = session.create_dataframe(
        {"text": [secret, "x" * 32000, None, "y" * 32000]}
    ).with_column(
        "j",
        fc.semantic.judge(state="text", questions=[questions()[0]]),
    )
    result = frame.to_polars()["j"].to_list()
    assert result[0] is not None
    assert result[1:] == [None, None, None]
    assert len(calls) == 1
    warnings = [
        record.message
        for record in caplog.records
        if "size or packing validation rejected" in record.message
    ]
    assert warnings == [
        "Typed judgment size or packing validation rejected 2 input row(s); "
        "returning null judgments for those rows."
    ]
    assert secret not in caplog.text


def test_oversized_question_group_warns_per_input_owner(
    native_session, monkeypatch, caplog
):
    session, calls, _ = native_session
    client = session._session_state.get_language_model(
        ResolvedModelAlias("judge", None)
    ).client
    monkeypatch.setattr(client.token_counter, "count_tokens", len)
    oversized = fc.JudgeQuestion.noul(
        name="too_large", instructions="x" * 32000
    )
    result = (
        session.create_dataframe({"text": ["first", "second"]})
        .with_column("j", fc.semantic.judge(state="text", questions=[oversized]))
        .to_polars()
    )
    assert result["j"].to_list() == [None, None]
    assert not calls
    assert caplog.text.count("size or packing validation rejected") == 1
    assert "2 input row(s)" in caplog.text


def test_valid_judgments_do_not_emit_size_rejection_warning(native_session, caplog):
    session, calls, _ = native_session
    result = (
        session.create_dataframe({"text": ["x"]})
        .with_column("j", fc.semantic.judge(state="text", questions=[questions()[0]]))
        .to_polars()
    )
    assert result["j"].to_list()[0] is not None
    assert len(calls) == 1
    assert "size or packing validation rejected" not in caplog.text


def test_settlement_once_per_request(native_session, monkeypatch):
    session, calls, _ = native_session
    client = session._session_state.get_language_model(
        ResolvedModelAlias("judge", None)
    ).client
    settlement = Mock(wraps=client._reconcile_completion)
    monkeypatch.setattr(client, "_reconcile_completion", settlement)
    frame = session.create_dataframe({"text": ["x", "x"]}).with_column(
        "j",
        fc.semantic.judge(state="text", questions=list(questions())),
    )
    result = frame.collect()
    assert len(calls) == settlement.call_count == 1
    assert result.metrics.total_lm_metrics.num_requests == 1
    assert result.metrics.total_lm_metrics.num_uncached_input_tokens == 100
    frame.collect()
    assert settlement.call_count == 1


def test_cancellation_is_not_swallowed(native_session):
    session, _, sdk = native_session

    async def cancelled(*_args, **_kwargs):
        raise asyncio.CancelledError()

    sdk.system_one = cancelled
    client = session._session_state.get_language_model(
        ResolvedModelAlias("judge", None)
    ).client
    request = FenicCompletionsRequest(
        LMRequestMessages("", [], "x"),
        None,
        None,
        None,
        None,
        judge_questions=questions(),
    )
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(client.make_single_request(request))


def test_untyped_request_is_refused_without_transport(native_session):
    session, calls, _ = native_session
    client = session._session_state.get_language_model(
        ResolvedModelAlias("judge", None)
    ).client
    request = FenicCompletionsRequest(
        LMRequestMessages("", [], "x"), None, None, None, None
    )
    assert isinstance(asyncio.run(client.make_single_request(request)), FatalException)
    assert not calls


@pytest.mark.parametrize(
    ("constructor_name", "factory_name"),
    [
        ("TypeSafeClient", "create_client"),
        ("AsyncTypeSafeClient", "create_aio_client"),
    ],
)
def test_sdk_retries_are_disabled(monkeypatch, constructor_name, factory_name):
    import typesafe_sdk

    from fenic._constants import MAX_MODEL_CLIENT_TIMEOUT

    constructor = Mock()
    monkeypatch.setattr(typesafe_sdk, constructor_name, constructor)
    getattr(TypeSafeModelProvider(), factory_name)()
    assert constructor.call_args.kwargs["retry"].max_retries == 0
    assert constructor.call_args.kwargs["timeout"] == MAX_MODEL_CLIENT_TIMEOUT


def test_typesafe_provider_normalizes_effective_endpoint(monkeypatch):
    import typesafe_sdk
    from typesafe_sdk.constants import BASE_URL_ENV, DEFAULT_BASE_URL

    sync_constructor = Mock()
    async_constructor = Mock()
    monkeypatch.setattr(typesafe_sdk, "TypeSafeClient", sync_constructor)
    monkeypatch.setattr(typesafe_sdk, "AsyncTypeSafeClient", async_constructor)
    monkeypatch.setenv(BASE_URL_ENV, "https://endpoint.example/")
    custom = TypeSafeModelProvider()
    monkeypatch.setenv(BASE_URL_ENV, "https://changed.example/")
    custom.create_client()
    custom.create_aio_client()
    explicit = TypeSafeModelProvider("https://explicit.example/")
    monkeypatch.delenv(BASE_URL_ENV, raising=False)
    default = TypeSafeModelProvider()
    monkeypatch.setenv(BASE_URL_ENV, "   ")
    blank = TypeSafeModelProvider()

    assert custom._base_url == "https://endpoint.example"
    assert explicit._base_url == "https://explicit.example"
    assert default._base_url == DEFAULT_BASE_URL.rstrip("/")
    assert blank._base_url == DEFAULT_BASE_URL.rstrip("/")
    assert sync_constructor.call_args.kwargs["base_url"] == custom._base_url
    assert async_constructor.call_args.kwargs["base_url"] == custom._base_url


def test_effective_endpoint_separates_typesafe_cache_identity(monkeypatch):
    from typesafe_sdk.constants import BASE_URL_ENV

    request = FenicCompletionsRequest(
        LMRequestMessages("", [], "state"),
        None,
        None,
        None,
        None,
        judge_questions=questions(),
    )
    monkeypatch.setenv(BASE_URL_ENV, "https://one.example/")
    first = TypeSafeModelProvider()
    monkeypatch.setenv(BASE_URL_ENV, "https://two.example/")
    second = TypeSafeModelProvider()
    equivalent = TypeSafeModelProvider("https://one.example/")

    first_key = compute_request_fingerprint(
        request, "jev-1.13.0", base_url=first._base_url
    )
    assert first_key != compute_request_fingerprint(
        request, "jev-1.13.0", base_url=second._base_url
    )
    assert first_key == compute_request_fingerprint(
        request, "jev-1.13.0", base_url=equivalent._base_url
    )


@pytest.mark.parametrize(
    ("base_url", "env_url", "should_validate"),
    [
        (None, None, True),
        (None, "https://api.typesafe.ai/", True),
        (None, "https://endpoint.example/", False),
        ("https://api.typesafe.ai/", "https://endpoint.example/", True),
        ("https://explicit.example/", "https://endpoint.example/", False),
    ],
)
def test_registry_validates_only_typesafe_default_endpoint(
    monkeypatch, base_url, env_url, should_validate
):
    from typesafe_sdk.constants import BASE_URL_ENV

    if env_url is None:
        monkeypatch.delenv(BASE_URL_ENV, raising=False)
    else:
        monkeypatch.setenv(BASE_URL_ENV, env_url)
    validated = []

    async def validate(providers):
        validated.append(providers)

    def initialize(_self, model_config, *_args):
        provider = TypeSafeModelProvider(model_config.base_url)
        return SimpleNamespace(
            client=SimpleNamespace(model_provider_class=provider)
        )

    monkeypatch.setattr(
        "fenic._backends.local.model_registry._validate_provider_api_keys",
        validate,
    )
    monkeypatch.setattr(
        SessionModelRegistry, "_initialize_language_model", initialize
    )
    SessionModelRegistry(
        ResolvedSemanticConfig(
            language_models=ResolvedLanguageModelConfig(
                model_configs={
                    "judge": ResolvedTypeSafeModelConfig(
                        model_name="jev-1.13.0",
                        rpm=1,
                        tpm=1,
                        base_url=base_url,
                    )
                },
                default_model="judge",
            )
        )
    )

    assert bool(validated) is should_validate


@pytest.mark.parametrize("list_error", [None, RuntimeError("offline validation")])
def test_validation_uses_short_lived_client_and_always_closes(monkeypatch, list_error):
    import typesafe_sdk

    client = SimpleNamespace(
        models=SimpleNamespace(list=AsyncMock(side_effect=list_error)),
        aclose=AsyncMock(),
    )
    constructor = Mock(return_value=client)
    provider = TypeSafeModelProvider()
    monkeypatch.setattr(typesafe_sdk, "AsyncTypeSafeClient", constructor)

    if list_error is None:
        asyncio.run(provider.validate_api_key())
    else:
        with pytest.raises(RuntimeError, match="offline validation"):
            asyncio.run(provider.validate_api_key())

    assert constructor.call_args.kwargs["timeout"] == 10
    assert constructor.call_args.kwargs["retry"].max_retries == 0
    client.aclose.assert_awaited_once()


def test_scheduler_keeps_long_requests_and_cancels_short_deadlines(native_session):
    session, calls, sdk = native_session
    original = sdk.system_one
    cancellations = 0

    async def provider_response(state, bodies, **kwargs):
        nonlocal cancellations
        if state == "long":
            return await original(state, bodies, **kwargs)
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellations += 1
            raise

    sdk.system_one = provider_response
    long_result = (
        session.create_dataframe({"text": ["long"]})
        .with_column(
            "j",
            fc.semantic.judge(
                state="text", questions=list(questions()), request_timeout=11
            ),
        )
        .collect()
    )
    assert long_result.data["j"].to_list()[0] is not None
    assert len(calls) == 1

    short_frame = session.create_dataframe({"text": ["short"]}).with_column(
        "j",
        fc.semantic.judge(
            state="text", questions=list(questions()), request_timeout=0.01
        ),
    )
    with pytest.raises(ExecutionError, match="maximum number of retries"):
        short_frame.collect()
    assert cancellations == 3


def test_unsupported_typesafe_profile_fails_before_provider_call(native_session):
    session, calls, _ = native_session
    with pytest.raises((ExecutionError, ValidationError), match="profile"):
        session.create_dataframe({"text": ["x"]}).with_column(
            "j",
            fc.semantic.judge(
                state="text",
                questions=list(questions()),
                model_alias=fc.ModelAlias(name="judge", profile="unsupported"),
            ),
        )
    assert not calls


def test_typesafe_model_literal_matches_catalog_aliases():
    catalog_names = set(
        model_catalog.provider_model_collections[
            ModelProvider.TYPESAFE
        ].completion_models
    )
    assert set(get_args(TypeSafeLanguageModelName)) == catalog_names
    assert fc.TypeSafeLanguageModel(
        model_name="jev-latest", rpm=1, tpm=1
    ).model_name == "jev-latest"
    with pytest.raises(PydanticValidationError):
        fc.TypeSafeLanguageModel(model_name="unknown", rpm=1, tpm=1)



def test_connection_failure_uses_scheduler_retry(native_session):
    from typesafe_sdk import TypeSafeAPIConnectionError

    session, calls, sdk = native_session
    original = sdk.system_one
    attempts = []

    async def transient(state, bodies, **kwargs):
        attempts.append(state)
        if len(attempts) == 1:
            raise TypeSafeAPIConnectionError("offline transport failure")
        return await original(state, bodies, **kwargs)

    sdk.system_one = transient
    result = (
        session.create_dataframe({"text": ["x"]})
        .with_column(
            "j",
            fc.semantic.judge(state="text", questions=[questions()[0]]),
        )
        .collect()
    )
    assert len(attempts) == 2
    assert len(calls) == 1
    assert result.metrics.total_lm_metrics.num_uncached_input_tokens == 100


def test_fatal_provider_error(native_session, caplog):
    from typesafe_sdk import TypeSafeAuthenticationError, TypeSafeError

    from fenic.core.error import ExecutionError

    assert issubclass(TypeSafeAuthenticationError, TypeSafeError)
    session, calls, sdk = native_session
    attempts = []
    private_body = "synthetic response body must remain private"

    async def fatal(*_args, **_kwargs):
        attempts.append(1)
        raise TypeSafeError(private_body)

    sdk.system_one = fatal
    frame = session.create_dataframe({"text": ["x"]}).with_column(
        "j", fc.semantic.judge(state="text", questions=[questions()[0]])
    )
    with pytest.raises(ExecutionError, match="TypeSafeError") as raised:
        frame.collect()
    assert len(attempts) == 1
    assert not calls
    assert private_body not in str(raised.value)
    assert private_body not in caplog.text


@pytest.mark.parametrize(
    ("error_type", "status"),
    [
        pytest.param("TypeSafeBadRequestError", 400),
        pytest.param("TypeSafeUnprocessableEntityError", 422),
        pytest.param("TypeSafeAPIResponseValidationError", 200),
    ],
)
def test_request_validation_error_returns_null_without_cache_or_usage(
    native_session, caplog, error_type, status
):
    import httpx2
    import typesafe_sdk

    session, calls, sdk = native_session
    original = sdk.system_one
    rejected_attempts = []
    private_body = "request-rejection-secret"
    error_class = getattr(typesafe_sdk, error_type)

    async def reject_one(state, bodies, **kwargs):
        if state == "reject":
            rejected_attempts.append(state)
            if error_type == "TypeSafeAPIResponseValidationError":
                raise error_class(
                    status, {"detail": private_body}, httpx2.Headers(), "answers"
                )
            raise error_class(
                status, {"detail": private_body}, httpx2.Headers(), private_body
            )
        return await original(state, bodies, **kwargs)

    sdk.system_one = reject_one
    frame = session.create_dataframe({"text": ["valid", "reject"]}).with_column(
        "j", fc.semantic.judge(state="text", questions=[questions()[0]])
    )
    first = frame.collect()
    assert first.data["j"].to_list()[0] is not None
    assert first.data["j"].to_list()[1] is None
    assert first.metrics.total_lm_metrics.num_requests == 1
    second = frame.collect()
    assert second.data["j"].to_list()[0] is not None
    assert second.data["j"].to_list()[1] is None
    assert second.metrics.total_lm_metrics.num_requests == 0
    assert len(calls) == 1
    assert rejected_attempts == ["reject", "reject"]
    assert private_body not in caplog.text


@pytest.mark.parametrize(
    ("error_type", "status"),
    [
        pytest.param("TypeSafeAuthenticationError", 401),
        pytest.param("TypeSafePermissionDeniedError", 403),
    ],
)
def test_access_errors_remain_fatal_and_sanitized(
    native_session, caplog, error_type, status
):
    import httpx2
    import typesafe_sdk

    session, calls, sdk = native_session
    private_body = "access-error-secret"
    error_class = getattr(typesafe_sdk, error_type)

    async def fail_access(*_args, **_kwargs):
        raise error_class(
            status, {"detail": private_body}, httpx2.Headers(), private_body
        )

    sdk.system_one = fail_access
    frame = session.create_dataframe({"text": ["x"]}).with_column(
        "j", fc.semantic.judge(state="text", questions=[questions()[0]])
    )
    with pytest.raises(ExecutionError, match=error_type) as raised:
        frame.collect()
    assert not calls
    assert private_body not in str(raised.value)
    assert private_body not in caplog.text


def test_typesafe_retry_deadline_preserves_body_sanitization(native_session, caplog):
    import httpx2
    from typesafe_sdk import TypeSafeRateLimitError

    session, _, sdk = native_session
    client = session._session_state.get_language_model(
        ResolvedModelAlias("judge", None)
    ).client
    private_body = "synthetic-private-retry-body"

    async def rate_limit(*_args, **_kwargs):
        raise TypeSafeRateLimitError(
            429, {"detail": private_body}, httpx2.Headers({"retry-after-ms": "2000"})
        )

    sdk.system_one = rate_limit
    request = FenicCompletionsRequest(
        LMRequestMessages("", [], "x"),
        None,
        None,
        None,
        None,
        judge_questions=(questions()[0],),
    )
    started = time.monotonic()
    response = asyncio.run_coroutine_threadsafe(
        client.make_single_request(request), client._event_loop
    ).result(1)
    assert isinstance(response, TransientException)
    assert response.retry_not_before >= started + 2
    assert str(response.exception) == "TypeSafeRateLimitError"
    assert private_body not in repr(response)
    assert private_body not in caplog.text


def test_choice_mass_compatibility_does_not_normalize_score_vectors():
    question = replace(questions()[2], premise=None)
    raw = answer(question.body())
    raw["probabilities"] = {0: 0.24, 1: 0.75}
    with pytest.raises(ValueError, match="sum"):
        flatten_answers([question], {question.name: raw})


@pytest.mark.parametrize("timeout", [0, -1, 601, float("inf"), float("nan"), True])
def test_invalid_timeout(timeout):
    with pytest.raises((ValueError, ValidationError)):
        fc.semantic.judge(
            state="text", questions=list(questions()), request_timeout=timeout
        )


def test_timeout_maximum():
    from fenic._constants import MAX_MODEL_CLIENT_TIMEOUT

    expr = fc.semantic.judge(
        state="text",
        questions=list(questions()),
        request_timeout=MAX_MODEL_CLIENT_TIMEOUT,
    )
    assert expr._logical_expr.request_timeout == MAX_MODEL_CLIENT_TIMEOUT
    with pytest.raises(ValidationError, match="max timeout"):
        fc.semantic.judge(
            state="text",
            questions=list(questions()),
            request_timeout=MAX_MODEL_CLIENT_TIMEOUT + 0.5,
        )


def test_invalid_model_alias():
    from pydantic import ValidationError as ArgumentValidationError

    with pytest.raises(ArgumentValidationError, match="model_alias"):
        fc.semantic.judge(
            state="text",
            questions=list(questions()),
            model_alias=7,
        )
