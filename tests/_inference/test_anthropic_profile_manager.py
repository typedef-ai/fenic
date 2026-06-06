import pytest

pytest.importorskip("anthropic")

from fenic._inference.anthropic.anthropic_profile_manager import (
    AnthropicCompletionsProfileManager,
)
from fenic.core._inference.model_catalog import ModelProvider, model_catalog
from fenic.core._resolved_session_config import ResolvedAnthropicModelProfile


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
    assert profile.thinking_token_budget == 0
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
    assert profile.thinking_config["type"] == "enabled"
    assert profile.thinking_config["budget_tokens"] == 2048
    assert profile.output_config == {"effort": "high"}
