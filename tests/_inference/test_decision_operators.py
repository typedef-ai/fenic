"""Offline coverage for explicit closed-set decision-provider routing."""

import json
import socket
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import polars as pl
import pytest
from pydantic import BaseModel, Field

import fenic as fc
from fenic._backends.local.semantic_operators.analyze_sentiment import (
    EXAMPLES,
    AnalyzeSentiment,
)
from fenic._backends.local.semantic_operators.base import CompletionOnlyRequestSender
from fenic._backends.local.semantic_operators.classify import Classify
from fenic._backends.local.semantic_operators.decision import DecisionRequestSender
from fenic._backends.local.semantic_operators.predicate import Predicate
from fenic._inference.language_model import InferenceConfiguration
from fenic._inference.types import FenicCompletionsResponse, LMRequestMessages
from fenic._inference.typesafe.typesafe_provider import TypeSafeModelProvider
from fenic.core._inference.model_catalog import ModelProvider
from fenic.core._logical_plan.plans.join import SemanticJoin
from fenic.core._logical_plan.resolved_types import (
    ResolvedClassDefinition,
    ResolvedModelAlias,
)
from fenic.core._serde.proto.expression_serde import (
    deserialize_logical_expr,
    serialize_logical_expr,
)
from fenic.core._serde.proto.proto_serde import ProtoSerde
from fenic.core._serde.proto.serde_context import SerdeContext
from fenic.core.error import ConfigurationError, ValidationError


@pytest.fixture
def decision_session(tmp_path, monkeypatch):
    sdk_module = pytest.importorskip("typesafe_sdk")
    monkeypatch.setattr(
        "fenic._backends.local.model_registry._validate_provider_api_keys", AsyncMock()
    )
    calls = []

    async def evaluate(state, bodies, **_kwargs):
        assert isinstance(state, dict)
        calls.append((state, bodies))
        answers = {}
        for name, body in bodies.items():
            if body["type"] == "noul":
                text = state["input"]
                probability = 0.5 if "tie" in text else 0.1 if "bad" in text else 0.9
                answers[name] = {"type": "noul", "noul": probability}
            else:
                keys = list(body["criteria"])
                selected = keys[1] if "bad" in state["input"] else keys[0]
                answers[name] = {
                    "type": "choice",
                    "choice": selected,
                    "confidence": 1.0,
                    "probabilities": {key: float(key == selected) for key in keys},
                }
        return sdk_module.SystemOneResponse.model_validate(
            {
                "model": "jev-1.13.0",
                "answers": answers,
                "usage": {"input_tokens": 100, "output_tokens": 20},
            }
        )

    sdk = SimpleNamespace(system_one=evaluate, aclose=AsyncMock())
    monkeypatch.setattr(TypeSafeModelProvider, "create_aio_client", lambda self: sdk)
    session = fc.Session.get_or_create(
        fc.SessionConfig(
            app_name="decision_operators",
            db_path=tmp_path,
            semantic=fc.SemanticConfig(
                language_models={
                    "decisions": fc.TypeSafeLanguageModel(
                        model_name="jev-1.13.0", rpm=1200, tpm=1_000_000
                    )
                },
                llm_response_cache=fc.LLMResponseCacheConfig(),
            ),
        )
    )

    def refuse_network(*_args, **_kwargs):
        raise AssertionError("offline decision tests must not open a connection")

    monkeypatch.setattr(socket.socket, "connect", refuse_network)
    model = session._session_state.get_language_model(
        ResolvedModelAlias("decisions", None)
    )
    monkeypatch.setattr(
        model,
        "get_completions",
        Mock(side_effect=AssertionError("closed decisions must not use completions")),
    )
    yield session, model, calls, sdk
    session.stop(skip_usage_summary=True)


def test_predicate_filter_and_cache(decision_session, monkeypatch):
    session, model, calls, _ = decision_session
    submit = Mock(wraps=model.client.make_batch_requests)
    monkeypatch.setattr(model.client, "make_batch_requests", submit)
    examples = fc.PredicateExampleCollection(
        examples=[fc.PredicateExample(input={"text": "example"}, output=True)]
    )
    prompt = "Read all lines:\n{{ text }}\nIs this acceptable?"
    predicate = fc.semantic.predicate(
        prompt,
        text=fc.col("text"),
        examples=examples,
        model_alias="decisions",
        request_timeout=7,
    )
    frame = session.create_dataframe(
        {"text": ["good\nkeep this line", "bad", "tie", None]}
    )
    result = frame.with_column("ok", predicate).collect()
    assert result.data["ok"].to_list() == [True, False, False, None]
    assert result.data["ok"].dtype == pl.Boolean
    assert len(calls) == 3
    assert result.metrics.total_lm_metrics.num_uncached_input_tokens == 300
    assert submit.call_args.kwargs["request_timeout"] == 7
    assert (
        calls[0][0]["input"]
        == "Read all lines:\ngood\nkeep this line\nIs this acceptable?"
    )
    assert calls[0][0]["examples"] == [
        {
            "input": "Read all lines:\nexample\nIs this acceptable?",
            "response": '{"output":true}',
        }
    ]
    question = calls[0][1]["decision"]
    assert question["type"] == "noul"
    assert question["instructions"].startswith(Predicate.SYSTEM_PROMPT)
    assert frame.filter(predicate).to_polars()["text"].to_list() == [
        "good\nkeep this line"
    ]
    assert len(calls) == 3


def test_classification_preserves_labels(decision_session):
    session, _, calls, _ = decision_session
    classes = [
        fc.ClassDefinition(label="A-B", description="First\nverbatim description"),
        fc.ClassDefinition(label="A B", description="Second description"),
        fc.ClassDefinition(label="其他", description="Unicode label"),
    ]
    examples = fc.ClassifyExampleCollection(
        examples=[fc.ClassifyExample(input="example", output="A B")]
    )
    result = (
        session.create_dataframe({"text": ["good", "bad", "", None]})
        .select(fc.semantic.classify("text", classes, examples=examples).alias("label"))
        .to_polars()
    )
    assert result["label"].to_list() == ["A-B", "A B", None, None]
    assert result["label"].dtype == pl.String
    assert len(calls) == 2
    body = calls[0][1]["decision"]
    assert body["type"] == "choice"
    assert [json.loads(value) for value in body["criteria"].values()] == [
        {"label": item.label, "description": item.description} for item in classes
    ]
    assert calls[0][0]["examples"] == [
        {"input": "example", "response": '{"output":"A B"}'}
    ]


def test_sentiment_preserves_examples(decision_session):
    session, _, calls, _ = decision_session
    result = (
        session.create_dataframe({"text": ["good", "bad", None]})
        .select(fc.semantic.analyze_sentiment("text").alias("sentiment"))
        .to_polars()
    )
    assert result["sentiment"].to_list() == ["positive", "negative", None]
    assert len(calls[0][0]["examples"]) == len(EXAMPLES.examples)
    assert calls[0][0]["examples"][0]["input"] == EXAMPLES.examples[0].input
    body = calls[0][1]["decision"]
    assert body["instructions"].startswith(AnalyzeSentiment.SYSTEM_PROMPT)
    assert [json.loads(value)["label"] for value in body["criteria"].values()] == [
        "positive",
        "negative",
        "neutral",
    ]


def test_join_predicate_and_timeout(decision_session, monkeypatch):
    session, model, calls, _ = decision_session
    submit = Mock(wraps=model.client.make_batch_requests)
    monkeypatch.setattr(model.client, "make_batch_requests", submit)
    left = session.create_dataframe({"left": ["good", "bad", None]})
    right = session.create_dataframe({"right": ["good", None]})
    examples = fc.JoinExampleCollection(
        examples=[
            fc.JoinExample(left_on="sample left", right_on="sample right", output=True)
        ]
    )
    result = left.semantic.join(
        right,
        "Compare {{ left_on }} with {{ right_on }}",
        left_on=fc.col("left"),
        right_on=fc.col("right"),
        examples=examples,
        model_alias="decisions",
        request_timeout=9,
    ).to_polars()
    assert result.to_dicts() == [{"left": "good", "right": "good"}]
    assert len(calls) == 2
    assert submit.call_args.kwargs["request_timeout"] == 9
    assert (
        calls[0][0]["examples"][0]["input"] == "Compare sample left with sample right"
    )


def test_filter_size_rejections_warn_once_and_exclude_only_rejected_rows(
    decision_session, monkeypatch, caplog
):
    session, model, calls, _ = decision_session

    def count_tokens(value):
        if "filter-secret" in str(value):
            return 32_000
        return len(value)

    monkeypatch.setattr(model.client.token_counter, "count_tokens", count_tokens)
    predicate = fc.semantic.predicate(
        "Check {{ text }}", text=fc.col("text"), model_alias="decisions"
    )
    result = (
        session.create_dataframe({"text": ["good", "filter-secret", None]})
        .filter(predicate)
        .to_polars()
    )

    assert result["text"].to_list() == ["good"]
    assert len(calls) == 1
    warnings = [
        record.message
        for record in caplog.records
        if "size or packing validation rejected" in record.message
    ]
    assert warnings == [
        "Typed judgment size or packing validation rejected 1 input row(s); "
        "returning null judgments for those rows."
    ]
    assert "filter-secret" not in caplog.text


def test_join_size_rejections_warn_once_per_candidate_pair(
    decision_session, monkeypatch, caplog
):
    session, model, calls, _ = decision_session

    def count_tokens(value):
        if "join-secret" in str(value):
            return 32_000
        return len(value)

    monkeypatch.setattr(model.client.token_counter, "count_tokens", count_tokens)
    result = (
        session.create_dataframe({"left": ["good", "join-secret", None]})
        .semantic.join(
            session.create_dataframe({"right": ["good", "join-secret", None]}),
            "Compare {{ left_on }} with {{ right_on }}",
            left_on=fc.col("left"),
            right_on=fc.col("right"),
            model_alias="decisions",
        )
        .to_polars()
    )

    assert result.to_dicts() == [{"left": "good", "right": "good"}]
    assert len(calls) == 1
    warnings = [
        record.message
        for record in caplog.records
        if "size or packing validation rejected" in record.message
    ]
    assert warnings == [
        "Typed judgment size or packing validation rejected 3 input row(s); "
        "returning null judgments for those rows."
    ]
    assert "join-secret" not in caplog.text


def test_join_timeout_survives_proto_roundtrip(decision_session, monkeypatch):
    session, model, calls, _ = decision_session
    submit = Mock(wraps=model.client.make_batch_requests)
    monkeypatch.setattr(model.client, "make_batch_requests", submit)
    left = session.create_dataframe({"left": ["good"]})
    right = session.create_dataframe({"right": ["good"]})
    join = left.semantic.join(
        right,
        "Compare {{ left_on }} with {{ right_on }}",
        left_on=fc.col("left"),
        right_on=fc.col("right"),
        model_alias="decisions",
        request_timeout=9,
    )

    result = fc.DataFrame._from_logical_plan(
        ProtoSerde.deserialize(ProtoSerde.serialize(join._logical_plan)),
        session._session_state,
    ).to_polars()

    assert result.to_dicts() == [{"left": "good", "right": "good"}]
    assert len(calls) == 1
    assert submit.call_args.kwargs["request_timeout"] == 9


@pytest.mark.parametrize("operation", ["predicate", "classify", "sentiment"])
def test_null_batches_and_serde(decision_session, operation):
    session, _, calls, _ = decision_session
    expr = {
        "predicate": lambda: fc.semantic.predicate(
            "Check {{ text }}", text=fc.col("text")
        ),
        "classify": lambda: fc.semantic.classify("text", ["yes", "no"]),
        "sentiment": lambda: fc.semantic.analyze_sentiment("text"),
    }[operation]()
    context = SerdeContext()
    restored = deserialize_logical_expr(
        serialize_logical_expr(expr._logical_expr, context), context
    )
    assert restored == expr._logical_expr
    frame = session.create_dataframe(
        pl.DataFrame({"text": pl.Series([None, None], dtype=pl.String)})
    )
    result = frame.select(
        fc.Column._from_logical_expr(restored).alias("out")
    ).to_polars()
    assert result["out"].to_list() == [None, None]
    assert result["out"].dtype == (
        pl.Boolean if operation == "predicate" else pl.String
    )
    assert not calls


def test_invalid_answers_are_not_cached(decision_session):
    session, _, calls, sdk = decision_session
    original = sdk.system_one

    async def invalid(*args, **kwargs):
        return (await original(*args, **kwargs)).model_copy(update={"answers": {}})

    sdk.system_one = invalid
    frame = session.create_dataframe({"text": ["good"]}).select(
        fc.semantic.classify("text", ["yes", "no"]).alias("label")
    )
    for _ in range(2):
        result = frame.collect()
        assert result.data["label"].to_list() == [None]
        assert result.metrics.total_lm_metrics.num_uncached_input_tokens == 100
    assert len(calls) == 2


def test_options_and_examples_change_cache(decision_session):
    session, _, calls, _ = decision_session
    frame = session.create_dataframe({"text": ["good"]})
    first = [
        fc.ClassDefinition(label="yes", description="First description"),
        fc.ClassDefinition(label="no", description="Second description"),
    ]
    frame.select(fc.semantic.classify("text", first)).collect()
    changed = [
        fc.ClassDefinition(label="yes", description="Changed description"),
        first[1],
    ]
    frame.select(fc.semantic.classify("text", changed)).collect()
    examples = fc.ClassifyExampleCollection(
        examples=[fc.ClassifyExample(input="extra guidance", output="no")]
    )
    frame.select(fc.semantic.classify("text", changed, examples=examples)).collect()
    assert len(calls) == 3
    assert calls[0][1] != calls[1][1]
    assert calls[1][0] != calls[2][0]


def test_non_strict_predicate(decision_session):
    session, _, calls, _ = decision_session
    frame = session.create_dataframe(
        pl.DataFrame({"text": pl.Series([None], dtype=pl.String)})
    )
    result = frame.select(
        fc.semantic.predicate(
            "Check {{ text }}",
            text=fc.col("text"),
            strict=False,
        ).alias("ok")
    ).to_polars()
    assert result["ok"].to_list() == [True]
    assert len(calls) == 1
    assert calls[0][0]["input"].startswith("Check ")


@pytest.mark.parametrize("operation", ["map", "extract", "summarize", "reduce"])
def test_open_ended_is_refused(decision_session, operation, monkeypatch):
    from fenic._inference.language_model import LanguageModel

    _, model, calls, _ = decision_session
    monkeypatch.setattr(
        model, "get_completions", LanguageModel.get_completions.__get__(model)
    )
    with pytest.raises(ConfigurationError, match="unsupported"):
        model.get_completions(
            [LMRequestMessages("instructions", [], "text")],
            max_tokens=128,
            operation_name=f"semantic.{operation}",
        )
    assert not calls


@pytest.mark.parametrize("operation", ["map", "extract", "summarize", "reduce"])
@pytest.mark.parametrize("explicit_alias", [False, True], ids=["default", "explicit"])
def test_open_ended_plans_are_refused(
    decision_session, monkeypatch, operation, explicit_alias
):
    class Output(BaseModel):
        value: str = Field(description="An open-ended value")

    session, model, calls, _ = decision_session
    get_judgments = Mock(side_effect=AssertionError("planning must not infer"))
    monkeypatch.setattr(model, "get_judgments", get_judgments)
    get_model = Mock(side_effect=AssertionError("planning must not resolve a client"))
    monkeypatch.setattr(session._session_state, "get_language_model", get_model)
    frame = session.create_dataframe({"text": ["text"]})
    model_arg = {"model_alias": "decisions"} if explicit_alias else {}
    if operation == "map":
        expr = fc.semantic.map(
            "Describe {{ text }}", text=fc.col("text"), **model_arg
        )
    elif operation == "extract":
        expr = fc.semantic.extract("text", Output, **model_arg)
    elif operation == "summarize":
        expr = fc.semantic.summarize("text", **model_arg)
    else:
        expr = fc.semantic.reduce(
            "Summarize these notes", "text", **model_arg
        )

    with pytest.raises(ValidationError, match=rf"semantic\.{operation}"):
        if operation == "reduce":
            frame.group_by("text").agg(expr)
        else:
            frame.select(expr)
    get_model.assert_not_called()
    get_judgments.assert_not_called()
    model.get_completions.assert_not_called()
    assert not calls


def _build_closed_set_plan(frame, operation, **options):
    if operation == "predicate":
        return frame.select(
            fc.semantic.predicate(
                "Check {{ text }}", text=fc.col("text"), **options
            )
        )
    if operation == "filter":
        return frame.filter(
            fc.semantic.predicate(
                "Check {{ text }}", text=fc.col("text"), **options
            )
        )
    if operation == "classify":
        return frame.select(fc.semantic.classify("text", ["yes", "no"], **options))
    if operation == "sentiment":
        return frame.select(fc.semantic.analyze_sentiment("text", **options))
    if operation == "join":
        right = frame.select(fc.col("text").alias("right_text"))
        return frame.semantic.join(
            right,
            "Compare {{ left_on }} and {{ right_on }}",
            left_on=fc.col("text"),
            right_on=fc.col("right_text"),
            **options,
        )
    raise AssertionError(f"Unknown operation: {operation}")


@pytest.mark.parametrize("operation", ["predicate", "filter", "classify", "sentiment"])
@pytest.mark.parametrize("explicit_alias", [False, True], ids=["default", "explicit"])
def test_typesafe_temperature_rejected_during_planning(
    decision_session, monkeypatch, operation, explicit_alias
):
    session, model, calls, _ = decision_session
    get_model = Mock(side_effect=AssertionError("planning must not resolve a client"))
    monkeypatch.setattr(session._session_state, "get_language_model", get_model)
    get_judgments = Mock(side_effect=AssertionError("planning must not infer"))
    monkeypatch.setattr(model, "get_judgments", get_judgments)
    frame = session.create_dataframe({"text": ["text"]})
    options = {"temperature": 0.2}
    if explicit_alias:
        options["model_alias"] = "decisions"
    with pytest.raises(ValidationError, match="temperature=0"):
        _build_closed_set_plan(frame, operation, **options)
    get_model.assert_not_called()
    get_judgments.assert_not_called()
    model.get_completions.assert_not_called()
    assert not calls


@pytest.mark.parametrize("explicit_alias", [False, True], ids=["default", "explicit"])
def test_typesafe_join_temperature_rejected_during_planning(
    decision_session, monkeypatch, explicit_alias
):
    # The public join has no temperature argument, but restored logical joins do.
    session, model, calls, _ = decision_session
    get_model = Mock(side_effect=AssertionError("planning must not resolve a client"))
    monkeypatch.setattr(session._session_state, "get_language_model", get_model)
    left = session.create_dataframe({"text": ["text"]})
    right = session.create_dataframe({"right_text": ["text"]})
    with pytest.raises(ValidationError, match="temperature=0"):
        SemanticJoin.from_session_state(
            left=left._logical_plan,
            right=right._logical_plan,
            left_on=fc.col("text")._logical_expr,
            right_on=fc.col("right_text")._logical_expr,
            jinja_template="Compare {{ left_on }} and {{ right_on }}",
            strict=True,
            temperature=0.2,
            model_alias=ResolvedModelAlias("decisions") if explicit_alias else None,
            session_state=session._session_state,
        )
    get_model.assert_not_called()
    model.get_completions.assert_not_called()
    assert not calls


@pytest.mark.parametrize("operation", ["predicate", "filter", "classify", "sentiment", "join"])
def test_typesafe_profile_rejected_during_planning(
    decision_session, monkeypatch, operation
):
    session, model, calls, _ = decision_session
    get_model = Mock(side_effect=AssertionError("planning must not resolve a client"))
    monkeypatch.setattr(session._session_state, "get_language_model", get_model)
    get_judgments = Mock(side_effect=AssertionError("planning must not infer"))
    monkeypatch.setattr(model, "get_judgments", get_judgments)
    frame = session.create_dataframe({"text": ["text"]})
    with pytest.raises(ValidationError, match="model profiles"):
        _build_closed_set_plan(
            frame, operation, model_alias=fc.ModelAlias(name="decisions", profile="unknown")
        )
    get_model.assert_not_called()
    get_judgments.assert_not_called()
    model.get_completions.assert_not_called()
    assert not calls


@pytest.mark.parametrize("operation", ["predicate", "filter", "classify", "sentiment", "join"])
def test_typesafe_valid_controls_build_schema(decision_session, monkeypatch, operation):
    session, model, calls, _ = decision_session
    get_model = Mock(side_effect=AssertionError("planning must not resolve a client"))
    monkeypatch.setattr(session._session_state, "get_language_model", get_model)
    get_judgments = Mock(side_effect=AssertionError("planning must not infer"))
    monkeypatch.setattr(model, "get_judgments", get_judgments)
    frame = session.create_dataframe({"text": ["text"]})
    options = {"model_alias": "decisions"}
    if operation != "join":
        options["temperature"] = 0
    result = _build_closed_set_plan(frame, operation, **options)
    assert result.schema.column_fields
    get_model.assert_not_called()
    get_judgments.assert_not_called()
    model.get_completions.assert_not_called()
    assert not calls


def test_typesafe_none_temperature_is_valid_for_logical_join(decision_session):
    session, _, calls, _ = decision_session
    left = session.create_dataframe({"text": ["text"]})
    right = session.create_dataframe({"right_text": ["text"]})
    join = SemanticJoin.from_session_state(
        left=left._logical_plan,
        right=right._logical_plan,
        left_on=fc.col("text")._logical_expr,
        right_on=fc.col("right_text")._logical_expr,
        jinja_template="Compare {{ left_on }} and {{ right_on }}",
        strict=True,
        temperature=None,
        session_state=session._session_state,
    )
    assert join.schema().column_fields
    assert not calls


def test_runtime_decision_controls_remain_guarded(decision_session):
    _, model, calls, _ = decision_session
    for temperature, profile, message in [
        (0.2, None, "temperature=0"),
        (0, "unknown", "model profiles"),
    ]:
        sender = CompletionOnlyRequestSender(
            model,
            "semantic.predicate",
            InferenceConfiguration(
                max_output_tokens=None,
                temperature=temperature,
                model_profile=profile,
            ),
        )
        with pytest.raises(ConfigurationError, match=message):
            DecisionRequestSender(sender, "Check this")
    assert not calls


def test_non_typesafe_plans_keep_temperature_and_profiles(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "offline-test-key")
    sdk = SimpleNamespace(
        system_one=AsyncMock(side_effect=AssertionError("planning must not infer")),
        aclose=AsyncMock(),
    )
    monkeypatch.setattr(TypeSafeModelProvider, "create_aio_client", lambda self: sdk)
    monkeypatch.setattr(
        "fenic._backends.local.model_registry._validate_provider_api_keys", AsyncMock()
    )

    def refuse_network(*_args, **_kwargs):
        raise AssertionError("planning must not open a connection")

    monkeypatch.setattr(socket.socket, "connect", refuse_network)
    session = fc.Session.get_or_create(
        fc.SessionConfig(
            app_name="mixed_decision_planning",
            db_path=tmp_path,
            semantic=fc.SemanticConfig(
                language_models={
                    "decisions": fc.TypeSafeLanguageModel(
                        model_name="jev-1.13.0", rpm=1200, tpm=1_000_000
                    ),
                    "completion": fc.AnthropicLanguageModel(
                        model_name="claude-haiku-4-5",
                        rpm=1200,
                        input_tpm=1_000_000,
                        output_tpm=1_000_000,
                        profiles={"neutral": fc.AnthropicLanguageModel.Profile()},
                        default_profile="neutral",
                    ),
                },
                default_language_model="decisions",
            ),
        )
    )
    try:
        get_model = Mock(side_effect=AssertionError("planning must not resolve a client"))
        monkeypatch.setattr(session._session_state, "get_language_model", get_model)
        frame = session.create_dataframe({"text": ["text"]})
        other_provider = fc.ModelAlias(name="completion", profile="neutral")

        class Output(BaseModel):
            value: str = Field(description="An open-ended value")

        # Even with a completion-capable model in the same session, the default
        # decision model cannot build an open-ended operation.
        with pytest.raises(ValidationError, match=r"semantic\.map"):
            frame.select(fc.semantic.map("Describe {{ text }}", text=fc.col("text")))

        for plan in [
            frame.select(
                fc.semantic.map(
                    "Describe {{ text }}", text=fc.col("text"),
                    model_alias=other_provider, temperature=0.4,
                )
            ),
            frame.select(
                fc.semantic.extract(
                    "text", Output, model_alias=other_provider, temperature=0.4
                )
            ),
            frame.select(
                fc.semantic.summarize(
                    "text", model_alias=other_provider, temperature=0.4
                )
            ),
            frame.group_by("text").agg(
                fc.semantic.reduce(
                    "Summarize these notes", "text",
                    model_alias=other_provider, temperature=0.4,
                )
            ),
            *(
                _build_closed_set_plan(
                    frame, operation,
                    model_alias=other_provider,
                    **({"temperature": 0.4} if operation != "join" else {}),
                )
                for operation in ("predicate", "filter", "classify", "sentiment", "join")
            ),
        ]:
            assert plan.schema.column_fields
        get_model.assert_not_called()
        sdk.system_one.assert_not_called()
    finally:
        session.stop(skip_usage_summary=True)


def test_class_count_still_rejected_by_runtime_guard(decision_session):
    session, _, calls, _ = decision_session
    frame = session.create_dataframe({"text": ["text"]})
    # Class-count validation remains in the physical sender, not in TD-5624.
    from fenic.core.error import ExecutionError

    with pytest.raises(ExecutionError, match="2..255") as raised:
        frame.select(fc.semantic.classify("text", [str(i) for i in range(256)])).to_polars()
    assert isinstance(raised.value.__cause__, ConfigurationError)
    assert not calls


@pytest.mark.parametrize("operation", ["predicate", "classify", "sentiment"])
def test_other_provider_path_is_unchanged(operation):
    model = SimpleNamespace(
        provider=ModelProvider.OPENAI,
        get_completions=Mock(
            return_value=[FenicCompletionsResponse('{"output":true}', None)]
        ),
        get_judgments=Mock(side_effect=AssertionError("must not use judgments")),
    )
    kwargs = {
        "input": pl.Series(["text"]),
        "model": model,
        "temperature": 0.4,
        "request_timeout": 5,
    }
    if operation == "predicate":
        operator = Predicate(jinja_template="Check {{ text }}", **kwargs)
    elif operation == "classify":
        operator = Classify(
            classes=[
                ResolvedClassDefinition("yes", "affirmative"),
                ResolvedClassDefinition("no", "negative"),
            ],
            **kwargs,
        )
    else:
        operator = AnalyzeSentiment(**kwargs)
    assert isinstance(operator.request_sender, CompletionOnlyRequestSender)
    messages = operator.build_request_messages_batch()
    operator.request_sender.send_requests(messages)
    call = model.get_completions.call_args.kwargs
    assert call["messages"] == messages
    assert call["temperature"] == 0.4
    assert call["request_timeout"] == 5
    assert (
        call["operation_name"]
        == f"semantic.{operation if operation != 'sentiment' else 'analyze_sentiment'}"
    )
    model.get_judgments.assert_not_called()


def test_other_provider_join():
    from fenic._backends.local.semantic_operators.join import Join
    from fenic._constants import LEFT_ON_KEY, RIGHT_ON_KEY

    model = SimpleNamespace(
        provider=ModelProvider.OPENAI,
        get_completions=Mock(
            return_value=[
                FenicCompletionsResponse('{"output":true}', None),
                FenicCompletionsResponse('{"output":false}', None),
            ]
        ),
        get_judgments=Mock(side_effect=AssertionError("must not use judgments")),
    )
    result = Join(
        pl.DataFrame({LEFT_ON_KEY: ["first", "second"]}),
        pl.DataFrame({RIGHT_ON_KEY: ["right"]}),
        "Compare {{ left_on }} with {{ right_on }}",
        strict=True,
        model=model,
        temperature=0.4,
        request_timeout=9,
    ).execute()
    assert result.to_dicts() == [{LEFT_ON_KEY: "first", RIGHT_ON_KEY: "right"}]
    call = model.get_completions.call_args.kwargs
    assert call["request_timeout"] is None  # Preserve the existing completion route.
    assert call["temperature"] == 0.4
    model.get_judgments.assert_not_called()
