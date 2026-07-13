import asyncio
from types import SimpleNamespace

import pytest

from fenic._inference.common_openai.openai_chat_completions_core import (
    OpenAIChatCompletionsCore,
)
from fenic._inference.common_openai.openai_profile_manager import (
    OpenAICompletionProfileConfiguration,
)
from fenic._inference.types import FenicCompletionsRequest, LMRequestMessages
from fenic.core._inference.model_catalog import (
    CompletionModelParameters,
    ModelProvider,
)
from fenic.core._inference.output_token_limits import (
    GOOGLE_REASONING_SAFETY_MARGIN,
    estimate_reasoning_tokens_for_resolved_profile,
    validate_effective_output_token_limit,
)
from fenic.core._logical_plan.resolved_types import ResolvedModelAlias
from fenic.core._logical_plan.utils import validate_completion_parameters
from fenic.core._resolved_session_config import (
    ResolvedAnthropicModelConfig,
    ResolvedAnthropicModelProfile,
    ResolvedGoogleModelConfig,
    ResolvedGoogleModelProfile,
    ResolvedLanguageModelConfig,
    ResolvedOpenAIModelConfig,
    ResolvedOpenAIModelProfile,
    ResolvedOpenRouterModelConfig,
    ResolvedOpenRouterModelProfile,
    ResolvedSemanticConfig,
    ResolvedSessionConfig,
)
from fenic.core.error import ValidationError


def test_effective_output_token_limit_rejects_reasoning_overflow():
    with pytest.raises(ValidationError, match="Lower max_output_tokens to at most 60"):
        validate_effective_output_token_limit(
            model_provider=ModelProvider.OPENAI,
            model_name="test-model",
            model_max_output_tokens=100,
            requested_completion_tokens=70,
            estimated_reasoning_tokens=40,
        )


def test_openai_core_output_limit_uses_internal_model_identity():
    core = OpenAIChatCompletionsCore(
        model="gpt-4.1-nano",
        model_provider=ModelProvider.OPENAI,
        token_counter=None,
        client=None,
    )
    request = FenicCompletionsRequest(
        messages=LMRequestMessages(system="", examples=[], user="hello"),
        max_completion_tokens=512,
        top_logprobs=None,
        structured_output=None,
        temperature=None,
    )

    assert (
        core.get_max_output_token_request_limit(
            request,
            OpenAICompletionProfileConfiguration(
                expected_additional_reasoning_tokens=0
            ),
        )
        == 512
    )


class FakeOpenAICompletions:
    def __init__(self):
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="ok", refusal=None),
                    finish_reason="stop",
                    logprobs=None,
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=1,
                prompt_tokens_details=None,
                completion_tokens=1,
                completion_tokens_details=None,
            ),
        )


def _make_openai_core_with_fake_completions():
    fake_completions = FakeOpenAICompletions()
    return (
        OpenAIChatCompletionsCore(
            model="gpt-4.1-nano",
            model_provider=ModelProvider.OPENAI,
            token_counter=None,
            client=SimpleNamespace(
                chat=SimpleNamespace(completions=fake_completions),
                beta=None,
            ),
        ),
        fake_completions,
    )


def test_openai_core_omits_zero_temperature():
    core, fake_completions = _make_openai_core_with_fake_completions()
    request = FenicCompletionsRequest(
        messages=LMRequestMessages(system="", examples=[], user="hello"),
        max_completion_tokens=512,
        top_logprobs=None,
        structured_output=None,
        temperature=0,
    )

    asyncio.run(
        core.make_single_request(request, OpenAICompletionProfileConfiguration())
    )

    assert "temperature" not in fake_completions.kwargs


def test_openai_core_sends_nonzero_temperature():
    core, fake_completions = _make_openai_core_with_fake_completions()
    request = FenicCompletionsRequest(
        messages=LMRequestMessages(system="", examples=[], user="hello"),
        max_completion_tokens=512,
        top_logprobs=None,
        structured_output=None,
        temperature=0.2,
    )

    asyncio.run(
        core.make_single_request(request, OpenAICompletionProfileConfiguration())
    )

    assert fake_completions.kwargs["temperature"] == 0.2


def test_openrouter_effort_estimate_uses_model_output_ratio():
    params = CompletionModelParameters(
        input_token_cost=0,
        output_token_cost=0,
        context_window_length=1000,
        max_output_tokens=1000,
        supports_reasoning=True,
        supports_disabled_reasoning=False,
    )
    model_config = ResolvedOpenRouterModelConfig(
        model_name="openai/gpt-test",
        profiles={"deep": ResolvedOpenRouterModelProfile(reasoning_effort="xhigh")},
    )

    assert (
        estimate_reasoning_tokens_for_resolved_profile(
            model_config=model_config,
            completion_parameters=params,
            profile_name="deep",
        )
        == 950
    )


def test_openrouter_max_effort_uses_maximum_output_ratio():
    params = CompletionModelParameters(
        input_token_cost=0,
        output_token_cost=0,
        context_window_length=1000,
        max_output_tokens=1000,
        supports_reasoning=True,
        supports_disabled_reasoning=False,
    )
    model_config = ResolvedOpenRouterModelConfig(
        model_name="openai/gpt-5.6-sol",
        profiles={"max": ResolvedOpenRouterModelProfile(reasoning_effort="max")},
    )

    assert (
        estimate_reasoning_tokens_for_resolved_profile(
            model_config=model_config,
            completion_parameters=params,
            profile_name="max",
        )
        == 950
    )


def test_google_explicit_empty_budget_profile_estimates_no_reasoning_tokens():
    params = CompletionModelParameters(
        input_token_cost=0,
        output_token_cost=0,
        context_window_length=1000,
        max_output_tokens=1000,
        supports_reasoning=True,
        supports_disabled_reasoning=False,
    )
    model_config = ResolvedGoogleModelConfig(
        model_name="gemini-test",
        model_provider=ModelProvider.GOOGLE_DEVELOPER,
        rpm=100,
        tpm=1000,
        profiles={"off": ResolvedGoogleModelProfile(thinking_token_budget=None)},
    )

    assert (
        estimate_reasoning_tokens_for_resolved_profile(
            model_config=model_config,
            completion_parameters=params,
            profile_name="off",
        )
        == 0
    )
    assert (
        estimate_reasoning_tokens_for_resolved_profile(
            model_config=model_config,
            completion_parameters=params,
            profile_name=None,
        )
        == 1536  # 1024 expected thinking tokens with the 1.5x safety margin
    )


@pytest.mark.parametrize(
    ("model_config", "max_tokens"),
    [
        (
            ResolvedOpenAIModelConfig(
                model_name="gpt-5.5",
                rpm=100,
                tpm=1000,
                profiles={"deep": ResolvedOpenAIModelProfile(reasoning_effort="xhigh")},
            ),
            100_000,
        ),
        (
            ResolvedAnthropicModelConfig(
                model_name="claude-opus-4-5",
                rpm=100,
                input_tpm=1000,
                output_tpm=1000,
                profiles={"deep": ResolvedAnthropicModelProfile(thinking_token_budget=60_000)},
            ),
            10_000,
        ),
        (
            ResolvedGoogleModelConfig(
                model_name="gemini-3.1-pro-preview",
                model_provider=ModelProvider.GOOGLE_DEVELOPER,
                rpm=100,
                tpm=1000,
                profiles={"deep": ResolvedGoogleModelProfile(thinking_level="high")},
            ),
            50_000,
        ),
    ],
)
def test_completion_parameter_validation_rejects_reasoning_overflow(
    model_config, max_tokens
):
    session_config = ResolvedSessionConfig(
        app_name="test",
        db_path=None,
        semantic=ResolvedSemanticConfig(
            language_models=ResolvedLanguageModelConfig(
                model_configs={"model": model_config},
                default_model="model",
            )
        ),
    )

    with pytest.raises(ValidationError, match="plus estimated reasoning tokens"):
        validate_completion_parameters(
            ResolvedModelAlias(name="model", profile="deep"),
            session_config,
            temperature=0,
            max_tokens=max_tokens,
        )


def test_openai_default_profile_estimate_disables_reasoning_when_supported():
    params = CompletionModelParameters(
        input_token_cost=0,
        output_token_cost=0,
        context_window_length=1000,
        max_output_tokens=1000,
        supports_reasoning=True,
        supports_disabled_reasoning=True,
    )
    model_config = ResolvedOpenAIModelConfig(
        model_name="gpt-test",
        rpm=100,
        tpm=1000,
    )

    assert (
        estimate_reasoning_tokens_for_resolved_profile(
            model_config=model_config,
            completion_parameters=params,
            profile_name=None,
        )
        == 0
    )


def test_anthropic_estimate_without_profile_reserves_no_reasoning_tokens():
    params = CompletionModelParameters(
        input_token_cost=0,
        output_token_cost=0,
        context_window_length=1000,
        max_output_tokens=1000,
        supports_reasoning=True,
    )
    model_config = ResolvedAnthropicModelConfig(
        model_name="claude-test",
        rpm=100,
        input_tpm=1000,
        output_tpm=1000,
    )

    assert (
        estimate_reasoning_tokens_for_resolved_profile(
            model_config=model_config,
            completion_parameters=params,
            profile_name=None,
        )
        == 0
    )


def test_openrouter_effort_estimate_rejects_reasoning_overflow():
    params = CompletionModelParameters(
        input_token_cost=0,
        output_token_cost=0,
        context_window_length=100_000,
        max_output_tokens=100_000,
        supports_reasoning=True,
        supports_disabled_reasoning=False,
    )
    model_config = ResolvedOpenRouterModelConfig(
        model_name="openai/gpt-test",
        profiles={"deep": ResolvedOpenRouterModelProfile(reasoning_effort="high")},
    )
    estimated_reasoning_tokens = estimate_reasoning_tokens_for_resolved_profile(
        model_config=model_config,
        completion_parameters=params,
        profile_name="deep",
    )

    with pytest.raises(ValidationError, match="plus estimated reasoning tokens"):
        validate_effective_output_token_limit(
            model_provider=ModelProvider.OPENROUTER,
            model_name=model_config.model_name,
            model_max_output_tokens=params.max_output_tokens,
            requested_completion_tokens=60_000,
            estimated_reasoning_tokens=estimated_reasoning_tokens,
        )


def test_adaptive_anthropic_profile_passes_completion_parameter_validation():
    session_config = ResolvedSessionConfig(
        app_name="test",
        db_path=None,
        semantic=ResolvedSemanticConfig(
            language_models=ResolvedLanguageModelConfig(
                model_configs={
                    "model": ResolvedAnthropicModelConfig(
                        model_name="claude-opus-4-8",
                        rpm=100,
                        input_tpm=1000,
                        output_tpm=1000,
                        profiles={"deepest": ResolvedAnthropicModelProfile(effort="max")},
                    )
                },
                default_model="model",
            )
        ),
    )

    # Adaptive thinking shares the output window, so even effort="max" admits
    # any visible budget within the model limit.
    validate_completion_parameters(
        ResolvedModelAlias(name="model", profile="deepest"),
        session_config,
        temperature=0,
        max_tokens=100_000,
    )


def test_google_plan_estimate_matches_runtime_safety_margin():
    params = CompletionModelParameters(
        input_token_cost=0,
        output_token_cost=0,
        context_window_length=100_000,
        max_output_tokens=100_000,
        supports_reasoning=True,
        supports_disabled_reasoning=False,
    )
    model_config = ResolvedGoogleModelConfig(
        model_name="gemini-test",
        model_provider=ModelProvider.GOOGLE_DEVELOPER,
        rpm=100,
        tpm=1000,
        profiles={"budget": ResolvedGoogleModelProfile(thinking_token_budget=20_000)},
    )

    # Plan-layer estimate must equal the runtime request construction estimate
    # (expected thinking budget scaled by the shared safety margin).
    assert (
        estimate_reasoning_tokens_for_resolved_profile(
            model_config=model_config,
            completion_parameters=params,
            profile_name="budget",
        )
        == int(GOOGLE_REASONING_SAFETY_MARGIN * 20_000)
    )


def test_openai_core_sends_verbosity_when_configured():
    core, fake_completions = _make_openai_core_with_fake_completions()
    request = FenicCompletionsRequest(
        messages=LMRequestMessages(system="", examples=[], user="hello"),
        max_completion_tokens=512,
        top_logprobs=None,
        structured_output=None,
        temperature=None,
    )

    asyncio.run(
        core.make_single_request(
            request, OpenAICompletionProfileConfiguration(verbosity="high")
        )
    )

    assert fake_completions.kwargs["verbosity"] == "high"


def test_openai_core_omits_verbosity_when_unset():
    core, fake_completions = _make_openai_core_with_fake_completions()
    request = FenicCompletionsRequest(
        messages=LMRequestMessages(system="", examples=[], user="hello"),
        max_completion_tokens=512,
        top_logprobs=None,
        structured_output=None,
        temperature=None,
    )

    asyncio.run(
        core.make_single_request(request, OpenAICompletionProfileConfiguration())
    )

    assert "verbosity" not in fake_completions.kwargs
