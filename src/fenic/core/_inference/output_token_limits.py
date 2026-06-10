"""Helpers for validating visible completion and reasoning token budgets."""

import math
from typing import Final, Optional

from fenic.core._inference.model_catalog import CompletionModelParameters, ModelProvider
from fenic.core._resolved_session_config import (
    ResolvedAnthropicModelConfig,
    ResolvedGoogleModelConfig,
    ResolvedModelConfig,
    ResolvedOpenAIModelConfig,
    ResolvedOpenRouterModelConfig,
)
from fenic.core.error import ValidationError

OPENAI_REASONING_TOKEN_ESTIMATES: Final[dict[str, int]] = {
    "none": 0,
    "minimal": 2048,
    "low": 4096,
    "medium": 8192,
    "high": 16384,
    "xhigh": 32768,
}

OPENROUTER_REASONING_EFFORT_RATIOS: Final[dict[str, float]] = {
    "none": 0.0,
    "minimal": 0.10,
    "low": 0.20,
    "medium": 0.50,
    "high": 0.80,
    "xhigh": 0.95,
}

GOOGLE_THINKING_LEVEL_TOKEN_ESTIMATES: Final[dict[str, int]] = {
    "minimal": 2048,
    "low": 8192,
    "medium": 16384,
    "high": 32768,
}

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
) -> Optional[int]:
    """Return the provider request limit after validating reasoning headroom.

    `requested_completion_tokens` is the visible completion budget. Reasoning
    models often require a larger provider-side output limit to preserve that
    visible budget. Fail when the combined visible + estimated reasoning budget
    exceeds the model cap instead of silently shrinking visible output.
    """
    if requested_completion_tokens is None:
        return None
    if requested_completion_tokens > model_max_output_tokens:
        raise ValidationError(
            f"[{model_provider.value}:{model_name}] max_output_tokens must be a positive integer less than or equal to {model_max_output_tokens}"
        )
    effective_output_tokens = requested_completion_tokens + estimated_reasoning_tokens
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


def estimate_reasoning_tokens_for_resolved_profile(
    *,
    model_config: ResolvedModelConfig,
    completion_parameters: CompletionModelParameters,
    profile_name: Optional[str],
) -> int:
    """Estimate provider-side reasoning tokens for a resolved model profile."""
    selected_profile_name = profile_name or getattr(model_config, "default_profile", None)
    profiles = getattr(model_config, "profiles", None) or {}
    profile = profiles.get(selected_profile_name) if selected_profile_name else None

    if isinstance(model_config, ResolvedOpenAIModelConfig):
        return _estimate_openai_reasoning_tokens(completion_parameters, profile)
    if isinstance(model_config, ResolvedGoogleModelConfig):
        return _estimate_google_reasoning_tokens(completion_parameters, profile)
    if isinstance(model_config, ResolvedAnthropicModelConfig):
        return _estimate_anthropic_reasoning_tokens(completion_parameters, profile)
    if isinstance(model_config, ResolvedOpenRouterModelConfig):
        return _estimate_openrouter_reasoning_tokens(completion_parameters, profile)
    return 0


def _estimate_openai_reasoning_tokens(completion_parameters, profile) -> int:
    if not completion_parameters.supports_reasoning:
        return 0
    reasoning_effort = getattr(profile, "reasoning_effort", None)
    if not reasoning_effort:
        if completion_parameters.default_reasoning_effort:
            reasoning_effort = completion_parameters.default_reasoning_effort
        elif completion_parameters.supports_disabled_reasoning:
            reasoning_effort = "none"
        elif completion_parameters.supports_minimal_reasoning:
            reasoning_effort = "minimal"
        else:
            reasoning_effort = "low"
    return OPENAI_REASONING_TOKEN_ESTIMATES[reasoning_effort]


def _estimate_google_reasoning_tokens(completion_parameters, profile) -> int:
    if not completion_parameters.supports_reasoning:
        return 0
    if completion_parameters.supported_thinking_levels:
        thinking_level = getattr(profile, "thinking_level", None) or "low"
        return GOOGLE_THINKING_LEVEL_TOKEN_ESTIMATES[thinking_level]
    if profile is None:
        if completion_parameters.supports_disabled_reasoning:
            return 0
        return 1024
    thinking_token_budget = getattr(profile, "thinking_token_budget", None)
    if thinking_token_budget is None or thinking_token_budget == 0:
        return 0
    if thinking_token_budget > 0:
        return thinking_token_budget
    return 16384


def _estimate_anthropic_reasoning_tokens(completion_parameters, profile) -> int:
    if profile is None:
        return 0
    thinking_token_budget = getattr(profile, "thinking_token_budget", None)
    if thinking_token_budget and completion_parameters.supports_reasoning:
        return thinking_token_budget
    effort = getattr(profile, "effort", None)
    if effort and completion_parameters.uses_adaptive_thinking:
        return math.ceil(
            ANTHROPIC_ADAPTIVE_THINKING_EFFORT_RATIOS[effort]
            * completion_parameters.max_output_tokens
        )
    return 0


def _estimate_openrouter_reasoning_tokens(completion_parameters, profile) -> int:
    if not completion_parameters.supports_reasoning:
        return 0
    reasoning_max_tokens = getattr(profile, "reasoning_max_tokens", None)
    if reasoning_max_tokens:
        return reasoning_max_tokens
    reasoning_effort = getattr(profile, "reasoning_effort", None) or "low"
    return math.ceil(
        OPENROUTER_REASONING_EFFORT_RATIOS[reasoning_effort]
        * completion_parameters.max_output_tokens
    )
