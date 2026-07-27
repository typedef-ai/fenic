import asyncio
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from fenic._inference.model_client import FatalException, TransientException
from fenic._inference.openrouter.openrouter_batch_chat_completions_client import (
    ANTHROPIC_STRUCTURED_OUTPUTS_BETA_HEADER,
    OpenRouterBatchChatCompletionsClient,
)
from fenic._inference.openrouter.openrouter_profile_manager import (
    OpenRouterCompletionsProfileManager,
)
from fenic._inference.openrouter.openrouter_provider import OpenRouterModelProvider
from fenic._inference.types import FenicCompletionsRequest, LMRequestMessages
from fenic.core._inference.model_catalog import (
    CompletionModelParameters,
    ModelProvider,
    model_catalog,
)
from fenic.core._logical_plan.resolved_types import ResolvedResponseFormat
from fenic.core._resolved_session_config import ResolvedOpenRouterModelProfile
from fenic.core.metrics import LMMetrics


class _Answer(BaseModel):
    answer: bool


class _FakeCompletions:
    def __init__(self, response):
        self.response = response
        self.create_calls = []
        self.parse_calls = []

    async def create(self, **kwargs):
        self.create_calls.append(kwargs)
        return self.response

    async def parse(self, **kwargs):
        self.parse_calls.append(kwargs)
        return self.response


def _tool_call(
    arguments: str = '{"answer":true}',
    *,
    name: str = "output_formatter",
):
    return SimpleNamespace(
        id="call-1",
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _custom_tool_call():
    return SimpleNamespace(
        id="call-1",
        type="custom",
        custom=SimpleNamespace(name="output_formatter", input="{}"),
    )


def _response(*, content=None, tool_calls=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content,
                    refusal=None,
                    tool_calls=tool_calls,
                ),
                finish_reason="tool_calls" if tool_calls else "stop",
                logprobs=None,
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=2,
            prompt_tokens_details=None,
            completion_tokens=1,
            completion_tokens_details=None,
            model_extra={"cost": 0.0},
        ),
    )


def _request(*, profile=None):
    return FenicCompletionsRequest(
        messages=LMRequestMessages(system="system", examples=[], user="question"),
        max_completion_tokens=64,
        top_logprobs=None,
        structured_output=ResolvedResponseFormat.from_pydantic_model(_Answer),
        temperature=None,
        model_profile=profile,
    )


def _client(
    supported_parameters,
    response,
    *,
    model="test/model",
    prefer_tools=False,
    reasoning_effort=None,
    reasoning_max_tokens=None,
    uses_adaptive_thinking=False,
    max_output_tokens=256,
):
    parameters = CompletionModelParameters(
        input_token_cost=0.0,
        output_token_cost=0.0,
        context_window_length=8_192,
        max_output_tokens=max_output_tokens,
        supports_reasoning="reasoning" in supported_parameters,
        uses_adaptive_thinking=uses_adaptive_thinking,
        supported_parameters=set(supported_parameters),
    )
    profiles = None
    default_profile_name = None
    if (
        prefer_tools
        or reasoning_effort is not None
        or reasoning_max_tokens is not None
    ):
        profiles = {
            "tools": ResolvedOpenRouterModelProfile(
                structured_output_strategy=(
                    "prefer_tools" if prefer_tools else None
                ),
                reasoning_effort=reasoning_effort,
                reasoning_max_tokens=reasoning_max_tokens,
            )
        }
        default_profile_name = "tools"

    completions = _FakeCompletions(response)
    client = OpenRouterBatchChatCompletionsClient.__new__(
        OpenRouterBatchChatCompletionsClient
    )
    client.model = model
    client.model_provider = ModelProvider.OPENROUTER
    client.model_provider_class = SimpleNamespace(_base_url=None)
    client._model_parameters = parameters
    client._profile_manager = OpenRouterCompletionsProfileManager(
        model_parameters=parameters,
        profile_configurations=profiles,
        default_profile_name=default_profile_name,
    )
    client._aio_client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    client._metrics = LMMetrics()
    return client, completions


def test_native_response_format_is_the_default_when_both_strategies_are_supported():
    client, completions = _client(
        {"structured_outputs", "tools", "tool_choice"},
        _response(content='{"answer":true}'),
    )

    result = asyncio.run(client.make_single_request(_request()))

    assert result.completion == '{"answer":true}'
    assert not completions.create_calls
    assert len(completions.parse_calls) == 1
    payload = completions.parse_calls[0]
    assert payload["response_format"] is _Answer
    assert "tools" not in payload
    assert "tool_choice" not in payload


@pytest.mark.parametrize(
    ("supported_parameters", "prefer_tools"),
    [
        ({"tools", "tool_choice"}, False),
        ({"structured_outputs", "tools", "tool_choice"}, True),
    ],
    ids=["tools-only-model", "prefer-tools-profile"],
)
def test_tool_strategy_forces_the_named_formatter(
    supported_parameters, prefer_tools
):
    client, completions = _client(
        supported_parameters,
        _response(tool_calls=[_tool_call()]),
        prefer_tools=prefer_tools,
    )

    result = asyncio.run(
        client.make_single_request(
            _request(profile="tools" if prefer_tools else None)
        )
    )

    assert result.completion == '{"answer":true}'
    assert not completions.parse_calls
    assert len(completions.create_calls) == 1
    payload = completions.create_calls[0]
    assert payload["tool_choice"] == {
        "type": "function",
        "function": {"name": "output_formatter"},
    }
    assert "parallel_tool_calls" not in payload
    assert "extra_headers" not in payload
    assert payload["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "output_formatter",
                "description": (
                    "Format the output of the model to correspond strictly to "
                    "the provided schema."
                ),
                "parameters": _Answer.model_json_schema()
                | {"additionalProperties": False},
                "strict": True,
            },
        }
    ]


@pytest.mark.parametrize(
    "tool_calls",
    [
        None,
        [],
        [_tool_call(name="some_other_tool")],
        [_tool_call(), _tool_call()],
        [_custom_tool_call()],
        [_tool_call(arguments="not json")],
        [_tool_call(arguments='{"answer":"true"}')],
    ],
    ids=[
        "missing-tool-calls",
        "empty-tool-calls",
        "wrong-tool",
        "multiple-tool-calls",
        "custom-tool-call",
        "malformed-json",
        "schema-invalid-arguments",
    ],
)
def test_malformed_formatter_response_is_transient(tool_calls):
    client, completions = _client(
        {"tools", "tool_choice"},
        _response(content="unstructured fallback", tool_calls=tool_calls),
    )

    result = asyncio.run(client.make_single_request(_request()))

    assert isinstance(result, TransientException)
    assert len(completions.create_calls) == 1
    assert client.get_metrics().num_requests == 1


def test_tool_only_model_without_tool_choice_is_rejected_before_dispatch():
    client, completions = _client(
        {"tools"},
        _response(tool_calls=[_tool_call()]),
    )

    result = asyncio.run(client.make_single_request(_request()))

    assert isinstance(result, FatalException)
    assert "tool_choice" in str(result.exception)
    assert not completions.create_calls
    assert not completions.parse_calls


def test_prefer_tools_falls_back_to_native_when_tool_choice_is_unavailable():
    client, completions = _client(
        {"structured_outputs", "tools"},
        _response(content='{"answer":true}'),
        prefer_tools=True,
    )

    result = asyncio.run(
        client.make_single_request(_request(profile="tools"))
    )

    assert result.completion == '{"answer":true}'
    assert not completions.create_calls
    assert len(completions.parse_calls) == 1


def test_anthropic_tool_strategy_enables_strict_tool_beta():
    client, completions = _client(
        {"structured_outputs", "tools", "tool_choice", "reasoning"},
        _response(tool_calls=[_tool_call()]),
        model="anthropic/claude-opus-5",
        prefer_tools=True,
        uses_adaptive_thinking=True,
    )

    result = asyncio.run(
        client.make_single_request(_request(profile="tools"))
    )

    assert result.completion == '{"answer":true}'
    assert completions.create_calls[0]["extra_headers"] == {
        "x-anthropic-beta": ANTHROPIC_STRUCTURED_OUTPUTS_BETA_HEADER
    }


def test_anthropic_fallback_model_enables_strict_tool_beta():
    adaptive_fallback = "anthropic/test-adaptive-strict-fallback"
    model_catalog.add_model(
        ModelProvider.OPENROUTER,
        adaptive_fallback,
        CompletionModelParameters(
            input_token_cost=0.0,
            output_token_cost=0.0,
            context_window_length=8_192,
            max_output_tokens=256,
            supports_reasoning=True,
            uses_adaptive_thinking=True,
            supported_parameters={
                "structured_outputs",
                "tools",
                "tool_choice",
                "reasoning",
            },
        ),
    )
    parameters = {"structured_outputs", "tools", "tool_choice"}
    client, completions = _client(
        parameters,
        _response(tool_calls=[_tool_call()]),
        prefer_tools=True,
    )
    client._profile_manager = OpenRouterCompletionsProfileManager(
        model_parameters=client._model_parameters,
        profile_configurations={
            "fallback": ResolvedOpenRouterModelProfile(
                models=[adaptive_fallback],
                structured_output_strategy="prefer_tools",
            )
        },
        default_profile_name="fallback",
    )

    result = asyncio.run(client.make_single_request(_request(profile="fallback")))

    assert result.completion == '{"answer":true}'
    assert completions.create_calls[0]["extra_headers"] == {
        "x-anthropic-beta": ANTHROPIC_STRUCTURED_OUTPUTS_BETA_HEADER
    }


def test_unsupported_anthropic_fallback_does_not_enable_strict_tool_beta():
    old_fallback = "anthropic/test-no-strict-tool-fallback"
    model_catalog.add_model(
        ModelProvider.OPENROUTER,
        old_fallback,
        CompletionModelParameters(
            input_token_cost=0.0,
            output_token_cost=0.0,
            context_window_length=8_192,
            max_output_tokens=256,
            supported_parameters={"tools", "tool_choice"},
        ),
    )
    client, completions = _client(
        {"structured_outputs", "tools", "tool_choice"},
        _response(tool_calls=[_tool_call()]),
    )
    client._profile_manager = OpenRouterCompletionsProfileManager(
        model_parameters=client._model_parameters,
        profile_configurations={
            "fallback": ResolvedOpenRouterModelProfile(
                models=[old_fallback],
                structured_output_strategy="prefer_tools",
            )
        },
        default_profile_name="fallback",
    )

    result = asyncio.run(client.make_single_request(_request(profile="fallback")))

    assert result.completion == '{"answer":true}'
    assert "extra_headers" not in completions.create_calls[0]


def test_anthropic_manual_thinking_uses_native_structured_output_when_available():
    client, completions = _client(
        {"structured_outputs", "tools", "tool_choice", "reasoning"},
        _response(content='{"answer":true}'),
        model="anthropic/claude-sonnet-4",
        prefer_tools=True,
    )

    result = asyncio.run(
        client.make_single_request(_request(profile="tools"))
    )

    assert result.completion == '{"answer":true}'
    assert not completions.create_calls
    assert len(completions.parse_calls) == 1


def test_anthropic_manual_thinking_without_native_output_is_rejected():
    client, completions = _client(
        {"tools", "tool_choice", "reasoning"},
        _response(tool_calls=[_tool_call()]),
        model="anthropic/claude-sonnet-4",
    )

    result = asyncio.run(client.make_single_request(_request()))

    assert isinstance(result, FatalException)
    assert "manual thinking" in str(result.exception)
    assert not completions.create_calls
    assert not completions.parse_calls


def test_anthropic_adaptive_thinking_can_force_formatter():
    client, completions = _client(
        {"tools", "tool_choice", "reasoning"},
        _response(tool_calls=[_tool_call()]),
        model="anthropic/claude-opus-5",
        uses_adaptive_thinking=True,
    )

    result = asyncio.run(client.make_single_request(_request()))

    assert result.completion == '{"answer":true}'
    assert len(completions.create_calls) == 1


def test_anthropic_reasoning_budget_uses_native_output_on_adaptive_model():
    client, completions = _client(
        {"structured_outputs", "tools", "tool_choice", "reasoning"},
        _response(content='{"answer":true}'),
        model="anthropic/claude-opus-4.6",
        prefer_tools=True,
        reasoning_max_tokens=1_024,
        uses_adaptive_thinking=True,
        max_output_tokens=4_096,
    )

    result = asyncio.run(
        client.make_single_request(_request(profile="tools"))
    )

    assert result.completion == '{"answer":true}'
    assert not completions.create_calls
    assert len(completions.parse_calls) == 1


def test_anthropic_manual_thinking_fallback_prevents_forced_formatter():
    old_fallback = "anthropic/test-manual-thinking-fallback"
    model_catalog.add_model(
        ModelProvider.OPENROUTER,
        old_fallback,
        CompletionModelParameters(
            input_token_cost=0.0,
            output_token_cost=0.0,
            context_window_length=8_192,
            max_output_tokens=256,
            supports_reasoning=True,
            uses_adaptive_thinking=False,
            supported_parameters={"tools", "tool_choice", "reasoning"},
        ),
    )
    client, completions = _client(
        {"tools", "tool_choice", "reasoning"},
        _response(tool_calls=[_tool_call()]),
        model="anthropic/test-adaptive-primary",
        uses_adaptive_thinking=True,
    )
    client._profile_manager = OpenRouterCompletionsProfileManager(
        model_parameters=client._model_parameters,
        profile_configurations={
            "fallback": ResolvedOpenRouterModelProfile(models=[old_fallback])
        },
        default_profile_name="fallback",
    )

    result = asyncio.run(client.make_single_request(_request(profile="fallback")))

    assert isinstance(result, FatalException)
    assert "manual thinking" in str(result.exception)
    assert not completions.create_calls


def test_disabling_reasoning_allows_forced_formatter_on_older_anthropic_model():
    client, completions = _client(
        {"tools", "tool_choice", "reasoning"},
        _response(tool_calls=[_tool_call()]),
        model="anthropic/claude-sonnet-4",
        reasoning_effort="none",
    )

    result = asyncio.run(
        client.make_single_request(_request(profile="tools"))
    )

    assert result.completion == '{"answer":true}'
    assert len(completions.create_calls) == 1
    assert "extra_headers" not in completions.create_calls[0]


@pytest.mark.parametrize(
    (
        "model_id",
        "reasoning",
        "uses_adaptive_thinking",
        "requires_adaptive_thinking",
    ),
    [
        (
            "anthropic/test-model",
            {
                "supported_efforts": ["high", "medium", "low"],
                "mandatory": True,
            },
            True,
            True,
        ),
        (
            "~anthropic/test-model",
            {"supported_efforts": ["high", "medium", "low"]},
            True,
            False,
        ),
        ("anthropic/test-model", {"mandatory": False}, False, False),
        ("anthropic/test-model", "malformed", False, False),
    ],
)
def test_openrouter_translates_anthropic_thinking_mode(
    model_id,
    reasoning,
    uses_adaptive_thinking,
    requires_adaptive_thinking,
):
    parameters = OpenRouterModelProvider()._translate_model(
        {
            "id": model_id,
            "pricing": {"prompt": "0", "completion": "0"},
            "context_length": 1_024,
            "top_provider": {"max_completion_tokens": 256},
            "supported_parameters": ["reasoning"],
            "reasoning": reasoning,
        }
    )

    assert parameters.uses_adaptive_thinking is uses_adaptive_thinking
    assert parameters.requires_adaptive_thinking is requires_adaptive_thinking
