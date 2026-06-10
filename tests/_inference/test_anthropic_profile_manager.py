import pytest

pytest.importorskip("anthropic")

from fenic._inference.anthropic.anthropic_batch_chat_completions_client import (
    AnthropicBatchCompletionsClient,
)
from fenic._inference.anthropic.anthropic_profile_manager import (
    AnthropicCompletionsProfileManager,
)
from fenic._inference.types import FenicCompletionsRequest, LMRequestMessages
from fenic.core._inference.model_catalog import ModelProvider, model_catalog
from fenic.core._inference.output_token_limits import (
    ANTHROPIC_ADAPTIVE_THINKING_EFFORT_RATIOS,
)
from fenic.core._resolved_session_config import ResolvedAnthropicModelProfile
from fenic.core.error import ValidationError


def _make_anthropic_client(params, profiles, default_profile_name):
    client = AnthropicBatchCompletionsClient.__new__(AnthropicBatchCompletionsClient)
    client.model_provider = ModelProvider.ANTHROPIC
    client.model = "claude-opus-4-8"
    client._model_parameters = params
    client._profile_manager = AnthropicCompletionsProfileManager(
        model_parameters=params,
        profile_configurations=profiles,
        default_profile_name=default_profile_name,
    )
    return client


def _make_request(max_completion_tokens):
    return FenicCompletionsRequest(
        messages=LMRequestMessages(system="", examples=[], user="hello"),
        max_completion_tokens=max_completion_tokens,
        top_logprobs=None,
        structured_output=None,
        temperature=None,
    )


def test_adaptive_effort_profile_uses_output_config():
    params = model_catalog.get_completion_model_parameters(
        ModelProvider.ANTHROPIC, "claude-opus-4-8"
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


def test_adaptive_effort_profile_rejects_max_tokens_above_model_limit():
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
    request = _make_request(max_completion_tokens=10_000)

    with pytest.raises(ValidationError, match="plus estimated reasoning tokens"):
        client._get_max_output_token_request_limit(request)


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
