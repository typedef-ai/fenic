"""Native judge contracts with a real SDK response model and fake transport."""

import asyncio
import socket
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import polars as pl
import pytest

import fenic as fc
from fenic._inference.cache.key_builder import compute_request_fingerprint
from fenic._inference.model_client import FatalException
from fenic._inference.rate_limit_strategy import InputTokenRateLimitStrategy
from fenic._inference.types import FenicCompletionsRequest, LMRequestMessages
from fenic._inference.typesafe.judge_requests import partition_questions
from fenic._inference.typesafe.typesafe_provider import TypeSafeModelProvider
from fenic.core._logical_plan.expressions.judge import SemanticJudgeExpr
from fenic.core._logical_plan.resolved_types import ResolvedModelAlias
from fenic.core._serde.proto.expression_serde import (
    deserialize_logical_expr,
    serialize_logical_expr,
)
from fenic.core._serde.proto.serde_context import SerdeContext
from fenic.core.error import ValidationError
from fenic.core.types.judge import flatten_answers, judge_schema, validate_questions


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
    monkeypatch.setattr(TypeSafeModelProvider, "create_aio_client", lambda self: sdk)
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
    assert len(calls) == 2


def test_bad_state_type(native_session):
    session, calls, _ = native_session
    with pytest.raises(Exception, match="string"):
        session.create_dataframe({"text": [7]}).with_column(
            "j",
            fc.semantic.judge(state="text", questions=list(questions())),
        )
    assert not calls


def test_envelope_failure_has_no_call(native_session, monkeypatch):
    session, calls, _ = native_session
    client = session._session_state.get_language_model(
        ResolvedModelAlias("judge", None)
    ).client
    monkeypatch.setattr(client.token_counter, "count_tokens", len)
    frame = session.create_dataframe({"text": ["x" * 32000]}).with_column(
        "j",
        fc.semantic.judge(state="text", questions=[questions()[0]]),
    )
    assert frame.to_polars()["j"].to_list() == [None]
    assert not calls


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

    constructor = Mock()
    monkeypatch.setattr(typesafe_sdk, constructor_name, constructor)
    getattr(TypeSafeModelProvider(), factory_name)()
    assert constructor.call_args.kwargs["retry"].max_retries == 0


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
