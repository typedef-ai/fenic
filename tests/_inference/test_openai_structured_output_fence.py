"""Fence fallback for structured output in the OpenAI core.

A response wrapped in one matched backtick fence is accepted after strict
validation fails, and a throttled warning tells the user their model did not
honor response_format. Anything other than one matched fence fails as before.
"""

import asyncio
import json
import logging

import httpx
import pytest
from openai import AsyncOpenAI
from pydantic import BaseModel, field_validator
from pydantic import ValidationError as PydanticValidationError

from fenic._inference.common_openai.openai_chat_completions_core import (
    OpenAIChatCompletionsCore,
)
from fenic._inference.model_client import FatalException
from fenic.core._inference.model_catalog import ModelProvider
from fenic.core._logical_plan.resolved_types import ResolvedResponseFormat
from tests._inference.test_openai_structured_output_diagnostics import (
    FENCE,
    NL,
    OBJECT_FORMAT,
    SvcModel,
    _core,
    _request,
    _run,
)

LOGGER = "fenic._inference.common_openai.openai_chat_completions_core"
BODY = '{"service": "api", "port": 8080}'


def _fence_warnings(caplog):
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING
        and "response_format" in record.getMessage()
    ]


@pytest.mark.parametrize(
    "content",
    [
        FENCE + "json" + NL + BODY + NL + FENCE,
        FENCE + NL + BODY + NL + FENCE,
        FENCE + "JSON title=x" + NL + BODY + NL + FENCE,
        FENCE + "`" + NL + BODY + NL + FENCE + "`",
        NL + "  " + FENCE + "json" + NL + BODY + NL + FENCE + NL + "  ",
        FENCE + "json\r\n" + BODY + "\r\n" + FENCE + "\r\n",
        FENCE + "json" + NL + NL + "  " + BODY + NL + NL + "  " + FENCE,
    ],
    ids=[
        "info-string",
        "no-info-string",
        "ignored-info-string",
        "four-backticks",
        "surrounding-whitespace",
        "crlf",
        "padded-body",
    ],
)
def test_matched_fence_is_stripped(content):
    result = _run(_core(content=content), _request(OBJECT_FORMAT, "semantic.extract"))

    assert not isinstance(result, FatalException)
    assert SvcModel.model_validate_json(result.completion).port == 8080
    assert FENCE not in result.completion


@pytest.mark.parametrize(
    "content",
    [
        FENCE + "json" + NL + BODY,
        BODY + NL + FENCE,
        "Here it is" + NL + FENCE + "json" + NL + BODY + NL + FENCE,
        FENCE + "json" + NL + BODY + NL + FENCE + NL + "Hope that helps",
        "~~~json" + NL + BODY + NL + "~~~",
        FENCE + NL + FENCE + "json" + NL + BODY + NL + FENCE + NL + FENCE,
        FENCE + "`" + NL + BODY + NL + FENCE,
        FENCE + NL + BODY + NL + FENCE + "`",
        FENCE + BODY + FENCE,
        FENCE + "json" + NL + '{"service": "api"}' + NL + FENCE,
        FENCE + "json" + NL + "not json" + NL + FENCE,
    ],
    ids=[
        "no-close",
        "no-open",
        "prose-before",
        "prose-after",
        "tilde-fence",
        "two-layers",
        "longer-open",
        "longer-close",
        "single-line",
        "inner-fails-schema",
        "inner-not-json",
    ],
)
def test_anything_else_fails_as_before(content, caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        result = _run(
            _core(content=content), _request(OBJECT_FORMAT, "semantic.extract")
        )

    assert isinstance(result, FatalException)
    assert isinstance(result.exception.__cause__, PydanticValidationError)
    assert "could not be validated" in str(result.exception)
    assert _fence_warnings(caplog) == []


def test_conforming_response_does_not_warn(caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        result = _run(_core(content=BODY), _request(OBJECT_FORMAT, "semantic.extract"))

    assert result.completion == BODY
    assert _fence_warnings(caplog) == []


def test_custom_validator_failure_is_not_unwrapped():
    class Rejecting(BaseModel):
        value: str

        @field_validator("value")
        @classmethod
        def reject(cls, value):
            raise ValueError("no")

    fenced_value = FENCE + NL + '{"value": "x"}' + NL + FENCE
    response_format = ResolvedResponseFormat(
        pydantic_model=Rejecting, json_schema={}, prompt_schema_definition=""
    )
    content = json.dumps({"value": fenced_value})

    result = _run(_core(content=content), _request(response_format, "semantic.extract"))

    assert isinstance(result, FatalException)


FENCED = FENCE + "json" + NL + BODY + NL + FENCE


def test_warning_names_model_and_operator_and_is_throttled(caplog):
    core = _core(content=FENCED)
    request = _request(OBJECT_FORMAT, "semantic.extract")

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        for _ in range(50):
            assert not isinstance(_run(core, request), FatalException)

    warnings = _fence_warnings(caplog)
    assert len(warnings) == 1
    assert "gpt-4.1-nano" in warnings[0]
    assert "semantic.extract" in warnings[0]


def test_each_operator_warns_once(caplog):
    core = _core(content=FENCED)

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        for operator in ("semantic.extract", "semantic.map", "semantic.extract"):
            _run(core, _request(OBJECT_FORMAT, operator))

    assert len(_fence_warnings(caplog)) == 2


def test_reduce_groups_count_as_one_operator(caplog):
    core = _core(content=FENCED)

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        for group in range(20):
            _run(core, _request(OBJECT_FORMAT, f"semantic.reduce(group={group})"))

    assert len(_fence_warnings(caplog)) == 1


def test_warning_is_rearmed_when_metrics_reset(caplog):
    core = _core(content=FENCED)
    request = _request(OBJECT_FORMAT, "semantic.extract")

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        _run(core, request)
        core.reset_metrics()
        _run(core, request)

    assert len(_fence_warnings(caplog)) == 2


def test_recovered_response_is_counted_as_a_request():
    core = _core(content=FENCED)

    _run(core, _request(OBJECT_FORMAT, "semantic.extract"))

    assert core.get_metrics().num_requests == 1


def test_real_sdk_error_is_recovered_without_a_second_request():
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(
            200,
            json={
                "id": "c",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-4.1-nano",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": FENCED},
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
            client = AsyncOpenAI(api_key="test", http_client=http, max_retries=0)
            core = OpenAIChatCompletionsCore(
                model="gpt-4.1-nano",
                model_provider=ModelProvider.OPENAI,
                token_counter=None,
                client=client,
            )
            return await core.make_single_request(
                _request(OBJECT_FORMAT, "semantic.extract"), None
            )

    result = asyncio.run(run())

    assert not isinstance(result, FatalException)
    assert json.loads(result.completion) == json.loads(BODY)
    assert len(calls) == 1
