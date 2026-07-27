import asyncio

import pytest
from pydantic import BaseModel

pytest.importorskip("anthropic")

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


@pytest.mark.parametrize("model_name", ["claude-opus-4-8", "claude-opus-5"])
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


def test_adaptive_thinking_structured_output_forces_formatter_tool(monkeypatch):
    params = model_catalog.get_completion_model_parameters(
        ModelProvider.ANTHROPIC, "claude-opus-5"
    )
    client = _make_anthropic_client(
        params,
        profiles={
            "deep": ResolvedAnthropicModelProfile(effort="high"),
        },
        default_profile_name="deep",
        model_name="claude-opus-5",
    )
    request = _make_request(
        max_completion_tokens=512,
        structured_output=ResolvedResponseFormat.from_pydantic_model(
            _StructuredResult, generate_struct_type=False
        ),
    )

    payload = _capture_structured_output_payload(client, request, monkeypatch)

    assert payload["thinking"] == {"type": "adaptive"}
    assert payload["output_config"] == {"effort": "high"}
    assert payload["tool_choice"] == {
        "name": "output_formatter",
        "type": "tool",
    }


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
