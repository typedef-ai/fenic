import asyncio
from types import SimpleNamespace

import pytest

from fenic._inference.common_openai.openai_chat_completions_core import (
    OpenAIChatCompletionsCore,
)
from fenic._inference.common_openai.openai_profile_manager import (
    OpenAICompletionsProfileManager,
)
from fenic._inference.types import FenicCompletionsRequest, LMRequestMessages
from fenic.core._inference.model_catalog import ModelProvider, model_catalog
from tests._inference.test_output_token_limits import FakeOpenAICompletions


def test_luna_catalog_matches_documented_pricing_and_limits():
    params = model_catalog.get_completion_model_parameters(
        ModelProvider.OPENAI, "gpt-6-luna"
    )
    assert params.input_token_cost == pytest.approx(0.10 / 1_000_000)
    assert params.cached_input_token_read_cost == pytest.approx(0.01 / 1_000_000)
    assert params.cached_input_token_write_cost == pytest.approx(0.125 / 1_000_000)
    assert params.output_token_cost == pytest.approx(0.50 / 1_000_000)
    assert params.context_window_length == 1_050_000
    assert params.max_output_tokens == 128_000
    assert params.supports_reasoning
    assert params.supports_disabled_reasoning
    assert params.supports_custom_temperature
    assert not params.supports_minimal_reasoning
    tier = params.tiered_input_token_costs[272_000]
    assert tier.input_token_cost == pytest.approx(0.20 / 1_000_000)
    assert tier.output_token_cost == pytest.approx(0.75 / 1_000_000)


@pytest.mark.parametrize(
    ("fixture_name", "alias"),
    [
        ("local_session_config", "test_model"),
        ("examples_session_config", "default"),
        ("multi_model_local_session_config", "model_1"),
    ],
)
@pytest.mark.parametrize("temperature", [0, 0.2])
def test_default_luna_profile_disables_reasoning_and_accepts_temperature(
    request, fixture_name, alias, temperature, caplog
):
    config = request.getfixturevalue(fixture_name)
    model = config.semantic.language_models[alias]
    assert model.model_name == "gpt-6-luna"
    assert model.profiles[model.default_profile].reasoning_effort == "none"
    assert model.profiles[model.default_profile].verbosity is None
    resolved = config._to_resolved_config().semantic.language_models.model_configs[
        alias
    ]
    params = model_catalog.get_completion_model_parameters(
        ModelProvider.OPENAI, resolved.model_name
    )
    manager = OpenAICompletionsProfileManager(
        params,
        profile_configurations=resolved.profiles,
        default_profile_name=resolved.default_profile,
    )
    profile = manager.get_profile_by_name(None)
    assert profile.reasoning_effort == "none"
    assert profile.expected_additional_reasoning_tokens == 0
    completions = FakeOpenAICompletions()
    core = OpenAIChatCompletionsCore(
        model=resolved.model_name,
        model_provider=ModelProvider.OPENAI,
        token_counter=None,
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=completions), beta=None
        ),
    )
    inference_request = FenicCompletionsRequest(
        messages=LMRequestMessages(system="", examples=[], user="hello"),
        max_completion_tokens=32,
        top_logprobs=None,
        structured_output=None,
        temperature=temperature,
    )
    asyncio.run(core.make_single_request(inference_request, profile))

    assert completions.kwargs["reasoning_effort"] == "none"
    if temperature:
        assert completions.kwargs["temperature"] == temperature
    assert "Ignoring temperature parameter." not in caplog.text
