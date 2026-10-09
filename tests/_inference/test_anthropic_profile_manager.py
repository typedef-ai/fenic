import asyncio
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, Field

pytest.importorskip("anthropic")

import anthropic

from fenic._inference.anthropic.anthropic_batch_chat_completions_client import (
    AnthropicBatchCompletionsClient,
)
from fenic._inference.anthropic.anthropic_profile_manager import (
    AnthropicCompletionsProfileManager,
)
from fenic._inference.model_client import FatalException
from fenic._inference.types import FenicCompletionsRequest, LMRequestMessages
from fenic.core._inference.model_catalog import ModelProvider, model_catalog
from fenic.core._inference.output_token_limits import (
    ANTHROPIC_ADAPTIVE_THINKING_EFFORT_RATIOS,
)
from fenic.core._logical_plan.resolved_types import ResolvedResponseFormat
from fenic.core._resolved_session_config import ResolvedAnthropicModelProfile
from fenic.core.error import ValidationError


class _StructuredResult(BaseModel):
    answer: str


class _NestedScore(BaseModel):
    score: int = Field(ge=1, le=5)


class _NestedStructuredResult(BaseModel):
    label: str
    detail: _NestedScore


def _make_anthropic_client(
    params, profiles, default_profile_name, model_name="claude-opus-4-8"
):
    client = AnthropicBatchCompletionsClient.__new__(AnthropicBatchCompletionsClient)
    client.model_provider = ModelProvider.ANTHROPIC
    client.model = model_name
    client._model_parameters = params
    client._profile_manager = AnthropicCompletionsProfileManager(
        model_parameters=params,
        profile_configurations=profiles,
        default_profile_name=default_profile_name,
    )
    client._output_formatter_tool_name = "output_formatter"
    client._output_formatter_tool_description = "Format structured output."
    return client


def _make_request(max_completion_tokens, structured_output=None):
    return FenicCompletionsRequest(
        messages=LMRequestMessages(system="", examples=[], user="hello"),
        max_completion_tokens=max_completion_tokens,
        top_logprobs=None,
        structured_output=structured_output,
        temperature=None,
    )


def _capture_structured_output_payload(client, request, monkeypatch):
    captured_payload = {}

    async def capture(payload):
        captured_payload.update(payload)
        return None, None

    monkeypatch.setattr(client, "_handle_structured_output_streaming_response", capture)
    asyncio.run(client.make_single_request(request))
    return captured_payload


@pytest.mark.parametrize(
    "model_name", ["claude-opus-4-8", "claude-opus-5", "claude-haiku-5-5"]
)
def test_adaptive_effort_profile_uses_output_config(model_name):
    params = model_catalog.get_completion_model_parameters(
        ModelProvider.ANTHROPIC, model_name
    )
    profile = AnthropicCompletionsProfileManager(
        model_parameters=params,
        profile_configurations={
            "deep": ResolvedAnthropicModelProfile(effort="xhigh"),
        },
        default_profile_name="deep",
    ).get_profile_by_name(None)

    assert profile.thinking_enabled
    assert profile.thinking_token_budget == (
        ANTHROPIC_ADAPTIVE_THINKING_EFFORT_RATIOS["xhigh"]
        * params.max_output_tokens
    )
    assert profile.uses_adaptive_thinking
    assert profile.effort == "xhigh"
    assert profile.thinking_config["type"] == "adaptive"
    assert profile.output_config == {"effort": "xhigh"}


def test_fable_default_profile_uses_required_adaptive_thinking():
    params = model_catalog.get_completion_model_parameters(
        ModelProvider.ANTHROPIC, "claude-fable-5"
    )
    profile = AnthropicCompletionsProfileManager(
        model_parameters=params,
    ).get_profile_by_name(None)

    assert profile.thinking_enabled
    assert profile.uses_adaptive_thinking
    assert profile.thinking_config["type"] == "adaptive"


def test_fable_empty_named_profile_uses_required_adaptive_thinking():
    params = model_catalog.get_completion_model_parameters(
        ModelProvider.ANTHROPIC, "claude-fable-5"
    )
    profile = AnthropicCompletionsProfileManager(
        model_parameters=params,
        profile_configurations={
            "default": ResolvedAnthropicModelProfile(),
        },
        default_profile_name="default",
    ).get_profile_by_name(None)

    assert profile.thinking_enabled
    assert profile.uses_adaptive_thinking
    assert profile.thinking_config["type"] == "adaptive"


def test_fable_effort_profile_reserves_adaptive_thinking_budget():
    params = model_catalog.get_completion_model_parameters(
        ModelProvider.ANTHROPIC, "claude-fable-5"
    )
    client = _make_anthropic_client(
        params,
        profiles={
            "deep": ResolvedAnthropicModelProfile(effort="xhigh"),
        },
        default_profile_name="deep",
    )
    request = _make_request(max_completion_tokens=10_000)

    profile = client._profile_manager.get_profile_by_name(None)
    assert profile.thinking_token_budget == 121_600
    assert client._get_max_output_token_request_limit(request) == 128_000


def test_manual_thinking_budget_profile_still_uses_budget_tokens():
    params = model_catalog.get_completion_model_parameters(
        ModelProvider.ANTHROPIC, "claude-opus-4-5"
    )
    profile = AnthropicCompletionsProfileManager(
        model_parameters=params,
        profile_configurations={
            "budget": ResolvedAnthropicModelProfile(
                thinking_token_budget=2048, effort="high"
            ),
        },
        default_profile_name="budget",
    ).get_profile_by_name(None)

    assert profile.thinking_enabled
    assert profile.thinking_token_budget == 2048
    assert not profile.uses_adaptive_thinking
    assert profile.effort == "high"
    assert profile.thinking_config["type"] == "enabled"
    assert profile.thinking_config["budget_tokens"] == 2048
    assert profile.output_config == {"effort": "high"}


def _assert_structured_output_format(payload, response_format):
    """Adaptive requests use output_config.format, not a formatter tool."""
    assert "tools" not in payload
    assert "tool_choice" not in payload
    output_format = payload["output_config"]["format"]
    assert output_format["type"] == "json_schema"
    assert output_format["schema"] == anthropic.transform_schema(
        response_format.json_schema
    )


@pytest.mark.parametrize(
    "model_name, effort",
    [
        ("claude-opus-5", "high"),
        ("claude-sonnet-5-5", "low"),
        ("claude-haiku-5-5", "max"),
    ],
)
def test_adaptive_thinking_structured_output_uses_output_format(
    model_name, effort, monkeypatch
):
    """With tool_choice auto the model can answer in text, so adaptive profiles use structured outputs."""
    params = model_catalog.get_completion_model_parameters(
        ModelProvider.ANTHROPIC, model_name
    )
    client = _make_anthropic_client(
        params,
        profiles={
            "deep": ResolvedAnthropicModelProfile(effort=effort),
        },
        default_profile_name="deep",
        model_name=model_name,
    )
    response_format = ResolvedResponseFormat.from_pydantic_model(
        _StructuredResult, generate_struct_type=False
    )
    request = _make_request(
        max_completion_tokens=512, structured_output=response_format
    )

    payload = _capture_structured_output_payload(client, request, monkeypatch)

    assert payload["thinking"] == {"type": "adaptive"}
    assert payload["output_config"]["effort"] == effort
    _assert_structured_output_format(payload, response_format)
    assert client._profile_manager.get_profile_by_name(None).output_config == {
        "effort": effort
    }


def test_sonnet_55_default_profile_never_forces_formatter_tool(monkeypatch):
    """Sonnet 5.5 rejects disabled thinking and forced tool choice."""
    params = model_catalog.get_completion_model_parameters(
        ModelProvider.ANTHROPIC, "claude-sonnet-5-5"
    )
    client = _make_anthropic_client(
        params,
        profiles={},
        default_profile_name=None,
        model_name="claude-sonnet-5-5",
    )
    response_format = ResolvedResponseFormat.from_pydantic_model(
        _StructuredResult, generate_struct_type=False
    )
    request = _make_request(
        max_completion_tokens=512, structured_output=response_format
    )
    request.temperature = 0.0

    payload = _capture_structured_output_payload(client, request, monkeypatch)

    assert payload["thinking"] == {"type": "adaptive"}
    assert "effort" not in payload["output_config"]
    _assert_structured_output_format(payload, response_format)
    assert "temperature" not in payload


def test_haiku_55_default_profile_disables_thinking_and_forces_formatter_tool(
    monkeypatch,
):
    """Haiku 5.5 accepts disabled thinking and forced tool choice at its default effort."""
    params = model_catalog.get_completion_model_parameters(
        ModelProvider.ANTHROPIC, "claude-haiku-5-5"
    )
    client = _make_anthropic_client(
        params,
        profiles={},
        default_profile_name=None,
        model_name="claude-haiku-5-5",
    )
    request = _make_request(
        max_completion_tokens=512,
        structured_output=ResolvedResponseFormat.from_pydantic_model(
            _StructuredResult, generate_struct_type=False
        ),
    )
    request.temperature = 0.0

    payload = _capture_structured_output_payload(client, request, monkeypatch)

    assert payload["thinking"] == {"type": "disabled"}
    assert "output_config" not in payload
    assert payload["tool_choice"] == {
        "name": "output_formatter",
        "type": "tool",
    }
    assert "temperature" not in payload


def test_structured_output_format_closes_objects_and_drops_unsupported_constraints(
    monkeypatch,
):
    """Structured outputs need additionalProperties false on every object and no numeric bounds."""
    params = model_catalog.get_completion_model_parameters(
        ModelProvider.ANTHROPIC, "claude-sonnet-5-5"
    )
    client = _make_anthropic_client(
        params,
        profiles={},
        default_profile_name=None,
        model_name="claude-sonnet-5-5",
    )
    response_format = ResolvedResponseFormat.from_pydantic_model(
        _NestedStructuredResult, generate_struct_type=False
    )
    request = _make_request(
        max_completion_tokens=512, structured_output=response_format
    )

    payload = _capture_structured_output_payload(client, request, monkeypatch)
    schema = payload["output_config"]["format"]["schema"]

    assert schema["additionalProperties"] is False
    nested = schema["$defs"]["_NestedScore"]
    assert nested["additionalProperties"] is False
    assert "minimum" not in nested["properties"]["score"]
    assert "maximum" not in nested["properties"]["score"]
    assert "minimum" in response_format.json_schema["$defs"]["_NestedScore"]["properties"]["score"]


def test_forced_formatter_tool_keeps_original_schema(monkeypatch):
    params = model_catalog.get_completion_model_parameters(
        ModelProvider.ANTHROPIC, "claude-haiku-5-5"
    )
    client = _make_anthropic_client(
        params,
        profiles={},
        default_profile_name=None,
        model_name="claude-haiku-5-5",
    )
    response_format = ResolvedResponseFormat.from_pydantic_model(
        _NestedStructuredResult, generate_struct_type=False
    )
    request = _make_request(
        max_completion_tokens=512, structured_output=response_format
    )

    payload = _capture_structured_output_payload(client, request, monkeypatch)

    assert "strict" not in payload["tools"][0]
    assert payload["tools"][0]["input_schema"] == response_format.json_schema


def _stream_events(*deltas):
    events = [
        SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(**delta))
        for delta in deltas
    ]
    events.append(
        SimpleNamespace(type="message_stop", message=SimpleNamespace(usage="usage"))
    )
    return events


def _client_streaming(events):
    class _Stream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def __aiter__(self):
            async def iterate():
                for event in events:
                    yield event

            return iterate()

    client = AnthropicBatchCompletionsClient.__new__(AnthropicBatchCompletionsClient)
    client._client = SimpleNamespace(
        messages=SimpleNamespace(stream=lambda **payload: _Stream())
    )
    return client


_MIXED_STREAM = _stream_events(
    {"type": "thinking_delta", "thinking": "reasoning"},
    {"type": "signature_delta", "signature": "sig"},
    {"type": "text_delta", "text": '{"answer": '},
    {"type": "input_json_delta", "partial_json": '{"answer": "tool"}'},
    {"type": "text_delta", "text": '"text"}'},
)


def test_structured_stream_reads_text_when_request_uses_output_format():
    client = _client_streaming(_MIXED_STREAM)

    content, usage = asyncio.run(
        client._handle_structured_output_streaming_response({"output_config": {}})
    )

    assert content == '{"answer": "text"}'
    assert usage == "usage"


def test_structured_stream_reads_tool_json_when_request_uses_formatter_tool():
    client = _client_streaming(_MIXED_STREAM)

    content, _ = asyncio.run(
        client._handle_structured_output_streaming_response({"tools": [{}]})
    )

    assert content == '{"answer": "tool"}'


def _record_count_tokens_calls(client):
    calls = []

    def count_tokens(**payload):
        calls.append(payload)
        return SimpleNamespace(input_tokens=len(calls))

    client._sync_client = SimpleNamespace(
        messages=SimpleNamespace(count_tokens=count_tokens)
    )
    return calls


_EMPTY_NAMED_PROFILE = {"default": ResolvedAnthropicModelProfile()}
_EFFORT_PROFILE = {"default": ResolvedAnthropicModelProfile(effort="low")}


@pytest.mark.parametrize(
    "model_name, profiles, expect_output_format",
    [
        ("claude-haiku-5-5", {}, False),
        ("claude-haiku-5-5", _EMPTY_NAMED_PROFILE, False),
        ("claude-haiku-5-5", _EFFORT_PROFILE, True),
        ("claude-opus-5", {}, False),
        ("claude-opus-5", _EFFORT_PROFILE, True),
        ("claude-sonnet-5-5", {}, True),
        ("claude-sonnet-5-5", _EMPTY_NAMED_PROFILE, True),
        ("claude-sonnet-5-5", _EFFORT_PROFILE, True),
        ("claude-opus-5-5", {}, True),
        ("claude-fable-5-1", {}, True),
        ("claude-haiku-4-5", {}, False),
    ],
)
def test_schema_token_estimate_counts_the_format_the_request_sends(
    model_name, profiles, expect_output_format, monkeypatch
):
    """The estimate follows the effective profile, not the model, so input TPM matches the request."""
    params = model_catalog.get_completion_model_parameters(
        ModelProvider.ANTHROPIC, model_name
    )
    client = _make_anthropic_client(
        params,
        profiles=profiles,
        default_profile_name="default" if profiles else None,
        model_name=model_name,
    )
    response_format = ResolvedResponseFormat.from_pydantic_model(
        _StructuredResult, generate_struct_type=False
    )
    request = _make_request(
        max_completion_tokens=512, structured_output=response_format
    )
    calls = _record_count_tokens_calls(client)

    client._count_auxiliary_input_tokens(request)
    request_payload = _capture_structured_output_payload(client, request, monkeypatch)
    (estimate_payload,) = calls

    if expect_output_format:
        _assert_structured_output_format(estimate_payload, response_format)
        assert (
            estimate_payload["output_config"]["format"]
            == request_payload["output_config"]["format"]
        )
        assert "tools" not in request_payload
    else:
        assert "output_config" not in estimate_payload
        assert estimate_payload["tools"][0]["input_schema"] == response_format.json_schema
        assert estimate_payload["tools"][0]["name"] == request_payload["tools"][0]["name"]
        assert estimate_payload["tool_choice"] == request_payload["tool_choice"]
        assert "format" not in request_payload.get("output_config", {})


def test_schema_token_estimates_are_cached_per_format():
    params = model_catalog.get_completion_model_parameters(
        ModelProvider.ANTHROPIC, "claude-haiku-5-5"
    )
    client = _make_anthropic_client(
        params, profiles={}, default_profile_name=None, model_name="claude-haiku-5-5"
    )
    response_format = ResolvedResponseFormat.from_pydantic_model(
        _StructuredResult, generate_struct_type=False
    )
    calls = _record_count_tokens_calls(client)

    tool_tokens = client.estimate_response_format_tokens(response_format, False)
    format_tokens = client.estimate_response_format_tokens(response_format, True)

    assert client.estimate_response_format_tokens(response_format, False) == tool_tokens
    assert client.estimate_response_format_tokens(response_format, True) == format_tokens
    assert len(calls) == 2
    assert "tools" in calls[0] and "output_config" in calls[1]


def test_manual_thinking_structured_output_does_not_force_formatter_tool(monkeypatch):
    params = model_catalog.get_completion_model_parameters(
        ModelProvider.ANTHROPIC, "claude-opus-4-5"
    )
    client = _make_anthropic_client(
        params,
        profiles={
            "budget": ResolvedAnthropicModelProfile(thinking_token_budget=2048),
        },
        default_profile_name="budget",
        model_name="claude-opus-4-5",
    )
    request = _make_request(
        max_completion_tokens=512,
        structured_output=ResolvedResponseFormat.from_pydantic_model(
            _StructuredResult, generate_struct_type=False
        ),
    )

    payload = _capture_structured_output_payload(client, request, monkeypatch)

    assert payload["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert "tools" in payload
    assert "tool_choice" not in payload


def test_adaptive_effort_profile_caps_request_limit_at_model_window():
    params = model_catalog.get_completion_model_parameters(
        ModelProvider.ANTHROPIC, "claude-opus-4-8"
    )
    client = _make_anthropic_client(
        params,
        profiles={
            "deep": ResolvedAnthropicModelProfile(effort="xhigh"),
        },
        default_profile_name="deep"
    )
    # Adaptive thinking shares the output window: the request limit is capped
    # at the model maximum instead of failing when visible + budget overflows.
    request = _make_request(max_completion_tokens=10_000)
    assert client._get_max_output_token_request_limit(request) == 128_000


def test_adaptive_effort_max_profile_is_usable():
    params = model_catalog.get_completion_model_parameters(
        ModelProvider.ANTHROPIC, "claude-opus-4-8"
    )
    client = _make_anthropic_client(
        params,
        profiles={
            "deepest": ResolvedAnthropicModelProfile(effort="max"),
        },
        default_profile_name="deepest"
    )
    request = _make_request(max_completion_tokens=512)
    assert client._get_max_output_token_request_limit(request) == 128_000


def test_adaptive_effort_profile_rejects_visible_tokens_above_model_limit():
    params = model_catalog.get_completion_model_parameters(
        ModelProvider.ANTHROPIC, "claude-opus-4-8"
    )
    client = _make_anthropic_client(
        params,
        profiles={
            "deep": ResolvedAnthropicModelProfile(effort="xhigh"),
        },
        default_profile_name="deep"
    )
    request = _make_request(max_completion_tokens=200_000)

    with pytest.raises(ValidationError, match="less than or equal to 128000"):
        client._get_max_output_token_request_limit(request)


def test_make_single_request_returns_fatal_exception_for_invalid_token_budget():
    params = model_catalog.get_completion_model_parameters(
        ModelProvider.ANTHROPIC, "claude-opus-4-8"
    )
    client = _make_anthropic_client(
        params,
        profiles={
            "deep": ResolvedAnthropicModelProfile(effort="xhigh"),
        },
        default_profile_name="deep"
    )
    request = _make_request(max_completion_tokens=200_000)

    result = asyncio.run(client.make_single_request(request))
    assert isinstance(result, FatalException)
    assert isinstance(result.exception, ValidationError)


def test_adaptive_effort_profile_uses_request_sized_rate_limit_estimate():
    params = model_catalog.get_completion_model_parameters(
        ModelProvider.ANTHROPIC, "claude-opus-4-8"
    )
    client = _make_anthropic_client(
        params,
        profiles={
            "deep": ResolvedAnthropicModelProfile(effort="xhigh"),
        },
        default_profile_name="deep"
    )
    request = _make_request(max_completion_tokens=512)

    profile = client._profile_manager.get_profile_by_name(None)
    assert profile.thinking_token_budget == 121_600
    assert client._get_max_output_token_request_limit(request) == 122_112
    assert client._estimate_output_tokens(request) == 999


def test_effort_only_profile_on_non_adaptive_model_does_not_enable_thinking():
    params = model_catalog.get_completion_model_parameters(
        ModelProvider.ANTHROPIC, "claude-opus-4-5"
    )
    profile = AnthropicCompletionsProfileManager(
        model_parameters=params,
        profile_configurations={
            "effort_only": ResolvedAnthropicModelProfile(effort="high"),
        },
        default_profile_name="effort_only",
    ).get_profile_by_name(None)

    assert not profile.thinking_enabled
    assert profile.thinking_token_budget == 0
    assert not profile.uses_adaptive_thinking
    assert profile.effort == "high"
    assert profile.thinking_config["type"] == "disabled"
    assert profile.output_config == {"effort": "high"}


def test_manual_thinking_budget_profile_still_rejects_reasoning_overflow():
    params = model_catalog.get_completion_model_parameters(
        ModelProvider.ANTHROPIC, "claude-opus-4-5"
    )
    client = _make_anthropic_client(
        params,
        profiles={
            "budget": ResolvedAnthropicModelProfile(thinking_token_budget=60_000),
        },
        default_profile_name="budget"
    )
    request = _make_request(max_completion_tokens=10_000)

    with pytest.raises(ValidationError, match="plus estimated reasoning tokens"):
        client._get_max_output_token_request_limit(request)


def test_manual_thinking_budget_profile_uses_budget_for_rate_limit_estimate():
    params = model_catalog.get_completion_model_parameters(
        ModelProvider.ANTHROPIC, "claude-opus-4-5"
    )
    client = _make_anthropic_client(
        params,
        profiles={
            "budget": ResolvedAnthropicModelProfile(thinking_token_budget=2048),
        },
        default_profile_name="budget"
    )
    request = _make_request(max_completion_tokens=512)

    assert client._estimate_output_tokens(request) == 2560
