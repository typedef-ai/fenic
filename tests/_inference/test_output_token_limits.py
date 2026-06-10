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
        == 1024
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
                model_name="claude-opus-4-8",
                rpm=100,
                input_tpm=1000,
                output_tpm=1000,
                profiles={"deep": ResolvedAnthropicModelProfile(effort="xhigh")},
            ),
            100_000,
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
