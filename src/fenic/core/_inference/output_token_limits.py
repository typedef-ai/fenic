"""Helpers for validating visible completion and reasoning token budgets."""

import math
from typing import Final, Optional, Union

from fenic.core._inference.model_catalog import CompletionModelParameters, ModelProvider
from fenic.core._resolved_session_config import (
    ResolvedAnthropicModelConfig,
    ResolvedAnthropicModelProfile,
    ResolvedGoogleModelConfig,
    ResolvedGoogleModelProfile,
    ResolvedModelConfig,
    ResolvedOpenAIModelConfig,
    ResolvedOpenAIModelProfile,
    ResolvedOpenRouterModelConfig,
    ResolvedOpenRouterModelProfile,
)
from fenic.core.error import ValidationError

ResolvedModelProfile = Union[
    ResolvedAnthropicModelProfile,
    ResolvedGoogleModelProfile,
    ResolvedOpenAIModelProfile,
    ResolvedOpenRouterModelProfile,
]

OPENAI_REASONING_TOKEN_ESTIMATES: Final[dict[str, int]] = {
    "none": 0,
    "minimal": 2048,
    "low": 4096,
    "medium": 8192,
    "high": 16384,
    "xhigh": 32768,
    "max": 65536,
}

OPENROUTER_REASONING_EFFORT_RATIOS: Final[dict[str, float]] = {
    "none": 0.0,
    "minimal": 0.10,
    "low": 0.20,
    "medium": 0.50,
    "high": 0.80,
    "xhigh": 0.95,
    "max": 0.95,
}

GOOGLE_THINKING_LEVEL_TOKEN_ESTIMATES: Final[dict[str, int]] = {
    "minimal": 2048,
    "low": 8192,
    "medium": 16384,
    "high": 32768,
}

# Gemini thinking budgets are suggestions, not hard limits. The provider-side
# request limit applies this margin on top of the expected thinking budget so
# overruns do not eat into the visible completion budget. The plan-time
# estimate must apply the same margin so plan validation is exactly as strict
# as the runtime request construction.
GOOGLE_REASONING_SAFETY_MARGIN: Final[float] = 1.5

ANTHROPIC_ADAPTIVE_THINKING_EFFORT_RATIOS: Final[dict[str, float]] = {
    "low": 0.20,
    "medium": 0.50,
    "high": 0.80,
    "xhigh": 0.95,
    "max": 1.00,
}


def validate_effective_output_token_limit(
    *,
    model_provider: ModelProvider,
    model_name: str,
    model_max_output_tokens: int,
    requested_completion_tokens: Optional[int],
    estimated_reasoning_tokens: int = 0,
    reasoning_shares_output_window: bool = False,
) -> Optional[int]:
    """Return the provider request limit after validating reasoning headroom.

    `requested_completion_tokens` is the visible completion budget. Reasoning
    models often require a larger provider-side output limit to preserve that
    visible budget. Fail when the combined visible + estimated reasoning budget
    exceeds the model cap instead of silently shrinking visible output.

    When `reasoning_shares_output_window` is True (Anthropic adaptive
    thinking), the reasoning budget is not a fixed reservation: the provider
    decides how much to think within `max_tokens`. In that mode only the
    visible budget is validated against the model cap, and the returned
    request limit is capped at the model maximum instead of failing, since
    high effort levels intentionally allow thinking up to the full window.
    """
    if requested_completion_tokens is None:
        return None
    if requested_completion_tokens > model_max_output_tokens:
        raise ValidationError(
            f"[{model_provider.value}:{model_name}] max_output_tokens must be a positive integer less than or equal to {model_max_output_tokens}"
        )
    effective_output_tokens = requested_completion_tokens + estimated_reasoning_tokens
    if reasoning_shares_output_window:
        return min(effective_output_tokens, model_max_output_tokens)
    if effective_output_tokens > model_max_output_tokens:
        max_visible_tokens = max(model_max_output_tokens - estimated_reasoning_tokens, 0)
        raise ValidationError(
            f"[{model_provider.value}:{model_name}] max_output_tokens={requested_completion_tokens} "
            f"plus estimated reasoning tokens={estimated_reasoning_tokens} requires "
            f"{effective_output_tokens} total output tokens, exceeding the model limit "
            f"of {model_max_output_tokens}. Lower max_output_tokens to at most "
            f"{max_visible_tokens}, lower reasoning effort, or use a model with a larger output limit."
        )
    return effective_output_tokens


def resolve_openai_reasoning_effort(
    completion_parameters: CompletionModelParameters,
    reasoning_effort: Optional[str],
) -> str:
    """Resolve the effective OpenAI reasoning effort for a profile.

    Reasoning effort behavior varies by model:
    - o-series/gpt-5 models: do not support disabling reasoning, default to
      the lowest supported effort (minimal or low)
    - gpt-5.1+ models: support 'none' to disable reasoning, default to 'none'
    - models with a catalog-level default use that default
    """
    if reasoning_effort:
        return reasoning_effort
    if completion_parameters.default_reasoning_effort:
        return completion_parameters.default_reasoning_effort
    if completion_parameters.supports_disabled_reasoning:
        return "none"
    if completion_parameters.supports_minimal_reasoning:
        return "minimal"
    return "low"


def estimate_reasoning_tokens_for_resolved_profile(
    *,
    model_config: ResolvedModelConfig,
    completion_parameters: CompletionModelParameters,
    profile_name: Optional[str],
) -> int:
    """Estimate provider-side reasoning tokens for a resolved model profile."""
    profile = _resolve_profile(model_config, profile_name)

    if isinstance(model_config, ResolvedOpenAIModelConfig):
        return _estimate_openai_reasoning_tokens(completion_parameters, profile)
    if isinstance(model_config, ResolvedGoogleModelConfig):
        return _estimate_google_reasoning_tokens(completion_parameters, profile)
    if isinstance(model_config, ResolvedAnthropicModelConfig):
        return _estimate_anthropic_reasoning_tokens(completion_parameters, profile)
    if isinstance(model_config, ResolvedOpenRouterModelConfig):
        return _estimate_openrouter_reasoning_tokens(completion_parameters, profile)
    return 0


def reasoning_shares_output_window_for_resolved_profile(
    *,
    model_config: ResolvedModelConfig,
    completion_parameters: CompletionModelParameters,
    profile_name: Optional[str],
) -> bool:
    """Whether the resolved profile uses a reasoning budget that shares the output window.

    True only for Anthropic adaptive-thinking effort profiles, where the
    thinking budget is an adaptive maximum inside `max_tokens` rather than a
    fixed reservation on top of the visible completion budget.
    """
    if not isinstance(model_config, ResolvedAnthropicModelConfig):
        return False
    if not completion_parameters.uses_adaptive_thinking:
        return False
    profile = _resolve_profile(model_config, profile_name)
    return isinstance(profile, ResolvedAnthropicModelProfile) and bool(profile.effort)


def _resolve_profile(
    model_config: ResolvedModelConfig, profile_name: Optional[str]
) -> Optional[ResolvedModelProfile]:
    selected_profile_name = profile_name or getattr(model_config, "default_profile", None)
    profiles = getattr(model_config, "profiles", None) or {}
    return profiles.get(selected_profile_name) if selected_profile_name else None


def _estimate_openai_reasoning_tokens(
    completion_parameters: CompletionModelParameters,
    profile: Optional[ResolvedOpenAIModelProfile],
) -> int:
    if not completion_parameters.supports_reasoning:
        return 0
    reasoning_effort = resolve_openai_reasoning_effort(
        completion_parameters, profile.reasoning_effort if profile else None
    )
    return OPENAI_REASONING_TOKEN_ESTIMATES[reasoning_effort]


def _estimate_google_reasoning_tokens(
    completion_parameters: CompletionModelParameters,
    profile: Optional[ResolvedGoogleModelProfile],
) -> int:
    if not completion_parameters.supports_reasoning:
        return 0
    if completion_parameters.supported_thinking_levels:
        thinking_level = (profile.thinking_level if profile else None) or "low"
        return _with_google_safety_margin(
            GOOGLE_THINKING_LEVEL_TOKEN_ESTIMATES[thinking_level]
        )
    if profile is None:
        if completion_parameters.supports_disabled_reasoning:
            return 0
        return _with_google_safety_margin(1024)
    thinking_token_budget = profile.thinking_token_budget
    if thinking_token_budget is None or thinking_token_budget == 0:
        return 0
    if thinking_token_budget > 0:
        return _with_google_safety_margin(thinking_token_budget)
    return _with_google_safety_margin(16384)


def _with_google_safety_margin(expected_thinking_tokens: int) -> int:
    return int(GOOGLE_REASONING_SAFETY_MARGIN * expected_thinking_tokens)


def _estimate_anthropic_reasoning_tokens(
    completion_parameters: CompletionModelParameters,
    profile: Optional[ResolvedAnthropicModelProfile],
) -> int:
    if profile is None:
        return 0
    if profile.thinking_token_budget and completion_parameters.supports_reasoning:
        return profile.thinking_token_budget
    if profile.effort and completion_parameters.uses_adaptive_thinking:
        return math.ceil(
            ANTHROPIC_ADAPTIVE_THINKING_EFFORT_RATIOS[profile.effort]
            * completion_parameters.max_output_tokens
        )
    return 0


def _estimate_openrouter_reasoning_tokens(
    completion_parameters: CompletionModelParameters,
    profile: Optional[ResolvedOpenRouterModelProfile],
) -> int:
    if not completion_parameters.supports_reasoning:
        return 0
    if profile is not None and profile.reasoning_max_tokens:
        return profile.reasoning_max_tokens
    reasoning_effort = (profile.reasoning_effort if profile else None) or "low"
    return math.ceil(
        OPENROUTER_REASONING_EFFORT_RATIOS[reasoning_effort]
        * completion_parameters.max_output_tokens
    )
