import asyncio
from types import SimpleNamespace

import pytest

from fenic._inference.common_openai.openai_chat_completions_core import (
    OpenAIChatCompletionsCore,
)
from fenic._inference.common_openai.openai_profile_manager import (
    OpenAICompletionProfileConfiguration,
)
from fenic._inference.language_model import LanguageModel
from fenic._inference.types import FenicCompletionsRequest, LMRequestMessages
from fenic.core._inference.model_catalog import ModelProvider
from tests._inference.test_output_token_limits import FakeOpenAICompletions


def _make_core(model):
    completions = FakeOpenAICompletions()
    core = OpenAIChatCompletionsCore(
        model=model,
        model_provider=ModelProvider.OPENAI,
        token_counter=None,
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=completions), beta=None
        ),
    )
    return core, completions


@pytest.mark.parametrize(
    ("model", "effort"),
    [("gpt-4.1-nano", None), ("gpt-6-sol", "none")],
)
@pytest.mark.parametrize("temperature", [0, 0.0, 0.2, None])
def test_openai_core_preserves_supported_temperature(model, effort, temperature):
    core, completions = _make_core(model)
    request = FenicCompletionsRequest(
        messages=LMRequestMessages(system="", examples=[], user="hello"),
        max_completion_tokens=512,
        top_logprobs=None,
        structured_output=None,
        temperature=temperature,
    )

    asyncio.run(
        core.make_single_request(
            request, OpenAICompletionProfileConfiguration(reasoning_effort=effort)
        )
    )

    if temperature is None:
        assert "temperature" not in completions.kwargs
    else:
        assert completions.kwargs["temperature"] == temperature


def test_openai_core_still_ignores_temperature_with_reasoning(caplog):
    core, completions = _make_core("gpt-5-nano")
    request = FenicCompletionsRequest(
        messages=LMRequestMessages(system="", examples=[], user="hello"),
        max_completion_tokens=512,
        top_logprobs=None,
        structured_output=None,
        temperature=0.2,
    )

    asyncio.run(
        core.make_single_request(
            request,
            OpenAICompletionProfileConfiguration(reasoning_effort="minimal"),
        )
    )

    assert "temperature" not in completions.kwargs
    assert "Ignoring temperature parameter." in caplog.text


@pytest.mark.parametrize("effort", [None, "medium"])
@pytest.mark.parametrize("temperature", [0, 0.0, None])
def test_openai_core_omits_zero_with_reasoning_without_warning(
    effort, temperature, caplog
):
    core, completions = _make_core("gpt-6-sol")
    request = FenicCompletionsRequest(
        messages=LMRequestMessages(system="", examples=[], user="hello"),
        max_completion_tokens=512,
        top_logprobs=None,
        structured_output=None,
        temperature=temperature,
    )

    asyncio.run(
        core.make_single_request(
            request, OpenAICompletionProfileConfiguration(reasoning_effort=effort)
        )
    )

    assert "temperature" not in completions.kwargs
    assert "Ignoring temperature parameter." not in caplog.text


def test_gpt_5_nano_language_model_still_omits_temperature():
    core, completions = _make_core("gpt-5-nano")
    client = SimpleNamespace(
        model_provider=ModelProvider.OPENAI,
        model="gpt-5-nano",
        context_tokens_per_minute=100_000,
        make_batch_requests=lambda requests, **kwargs: requests,
    )
    model = LanguageModel(client)
    requests = model.get_completions(
        [LMRequestMessages(system="", examples=[], user="hello")],
        max_tokens=512,
        temperature=0,
    )
    assert requests[0].temperature is None

    asyncio.run(
        core.make_single_request(
            requests[0],
            OpenAICompletionProfileConfiguration(reasoning_effort="minimal"),
        )
    )

    assert completions.kwargs["reasoning_effort"] == "minimal"
    assert "temperature" not in completions.kwargs
