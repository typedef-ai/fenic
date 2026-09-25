"""Diagnosable failures for structured-output validation in the OpenAI core.

When the OpenAI SDK's .parse() rejects a model's response against the requested
schema, the resulting pydantic error is opaque by itself. The core should re-raise it
as a fatal error that names the operator, the expected schema, and a preview of what
the model actually returned.
"""

import asyncio
import enum
from types import SimpleNamespace

from pydantic import BaseModel, create_model
from pydantic import ValidationError as PydanticValidationError

from fenic._inference.common_openai.openai_chat_completions_core import (
    OpenAIChatCompletionsCore,
)
from fenic._inference.model_client import FatalException
from fenic._inference.types import (
    FenicCompletionsRequest,
    LMRequestMessages,
)
from fenic.core._inference.model_catalog import ModelProvider
from fenic.core._logical_plan.resolved_types import ResolvedResponseFormat

LabelEnum = enum.Enum("LabelEnum", {"TOOLCHAIN": "toolchain", "NETWORK": "network"})
EnumModel = create_model("EnumModel", output=(LabelEnum, ...))
ENUM_FORMAT = ResolvedResponseFormat(
    pydantic_model=EnumModel, json_schema={}, prompt_schema_definition=""
)


class SvcModel(BaseModel):
    service: str
    port: int


OBJECT_FORMAT = ResolvedResponseFormat(
    pydantic_model=SvcModel, json_schema={}, prompt_schema_definition=""
)

FENCE = chr(96) * 3
NL = chr(10)


def _validation_error(raw_input: str) -> PydanticValidationError:
    try:
        EnumModel.model_validate_json(raw_input)
    except PydanticValidationError as e:
        return e
    raise AssertionError("expected a validation error")


class FakeParseCompletions:
    def __init__(self, error=None):
        self.error = error

    async def parse(self, **kwargs):
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"service": "api", "port": 1}', refusal=None
                    ),
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


def _core(parse_error=None):
    return OpenAIChatCompletionsCore(
        model="gpt-4.1-nano",
        model_provider=ModelProvider.OPENAI,
        token_counter=None,
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=None),
            beta=SimpleNamespace(
                chat=SimpleNamespace(
                    completions=FakeParseCompletions(error=parse_error)
                )
            ),
        ),
    )


def _request(response_format, operation_name):
    return FenicCompletionsRequest(
        messages=LMRequestMessages(system="", examples=[], user="hello"),
        max_completion_tokens=512,
        top_logprobs=None,
        structured_output=response_format,
        temperature=None,
        operation_name=operation_name,
    )


def _run(core, request):
    return asyncio.run(core.make_single_request(request, None))


def test_bare_enum_value_produces_diagnosable_fatal():
    core = _core(parse_error=_validation_error("toolchain"))
    request = _request(ENUM_FORMAT, "semantic.classify")

    result = _run(core, request)

    assert isinstance(result, FatalException)
    message = str(result.exception)
    assert "semantic.classify" in message
    assert "EnumModel" in message
    assert "'toolchain'" in message
    assert "does not honour response_format" in message


def test_fenced_json_produces_diagnosable_fatal():
    fenced = FENCE + "json" + NL + '{"service": "api"}' + NL + FENCE
    core = _core(parse_error=_validation_error(fenced))
    request = _request(OBJECT_FORMAT, "semantic.extract")

    result = _run(core, request)

    assert isinstance(result, FatalException)
    message = str(result.exception)
    assert "semantic.extract" in message
    assert "SvcModel" in message
    assert "does not honour response_format" in message


def test_missing_operation_name_falls_back_without_crashing():
    core = _core(parse_error=_validation_error("toolchain"))
    request = _request(ENUM_FORMAT, operation_name=None)

    result = _run(core, request)

    assert isinstance(result, FatalException)
    assert "unknown operator" in str(result.exception)


def test_conforming_response_still_succeeds():
    core = _core(parse_error=None)
    request = _request(ENUM_FORMAT, "semantic.classify")

    result = _run(core, request)

    assert not isinstance(result, FatalException)
    assert result.completion == '{"service": "api", "port": 1}'
