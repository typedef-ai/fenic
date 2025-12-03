"""Unit tests for request fingerprint generation."""

from fenic._inference.cache.key_builder import compute_request_fingerprint
from fenic._inference.types import (
    FenicCompletionsRequest,
    FewShotExample,
    LMRequestFile,
    LMRequestMessages,
)
from fenic.core._logical_plan.resolved_types import ResolvedResponseFormat
from tests.conftest import _save_pdf_file


def _build_request(
    *,
    messages: LMRequestMessages,
    max_tokens: int = 100,
    temperature: float = 0.7,
    top_logprobs: int | None = None,
    structured_output: ResolvedResponseFormat | None = None,
    model_profile: str | None = None,
) -> FenicCompletionsRequest:
    return FenicCompletionsRequest(
        messages=messages,
        max_completion_tokens=max_tokens,
        top_logprobs=top_logprobs,
        structured_output=structured_output,
        temperature=temperature,
        model_profile=model_profile,
    )


def _fingerprint(
    request: FenicCompletionsRequest,
    model: str = "gpt-4o-mini",
    profile_hash: str | None = None,
) -> str:
    return compute_request_fingerprint(request, model, profile_hash=profile_hash)


def test_request_fingerprint_is_deterministic():
    messages = LMRequestMessages(system="You are helpful", examples=[], user="Hello")
    request1 = _build_request(messages=messages)
    request2 = _build_request(messages=messages)

    assert _fingerprint(request1) == _fingerprint(request2)


def test_request_fingerprint_changes_with_model():
    messages = LMRequestMessages(system="You are helpful", examples=[], user="Hello")
    request = _build_request(messages=messages)

    assert _fingerprint(request) != _fingerprint(request, model="gpt-4o")


def test_request_fingerprint_changes_with_messages():
    messages1 = LMRequestMessages(system="You are helpful", examples=[], user="Hello")
    messages2 = LMRequestMessages(system="You are helpful", examples=[], user="Goodbye")

    assert _fingerprint(_build_request(messages=messages1)) != _fingerprint(
        _build_request(messages=messages2)
    )


def test_request_fingerprint_changes_with_temperature():
    messages = LMRequestMessages(system="You are helpful", examples=[], user="Hello")

    assert _fingerprint(_build_request(messages=messages, temperature=0.7)) != _fingerprint(
        _build_request(messages=messages, temperature=0.9)
    )


def test_request_fingerprint_changes_with_max_tokens():
    messages = LMRequestMessages(system="You are helpful", examples=[], user="Hello")

    assert _fingerprint(_build_request(messages=messages, max_tokens=100)) != _fingerprint(
        _build_request(messages=messages, max_tokens=200)
    )


def test_request_fingerprint_changes_with_profiles():
    messages = LMRequestMessages(system="You are helpful", examples=[], user="Hello")

    assert _fingerprint(
        _build_request(messages=messages, model_profile="fast")
    ) != _fingerprint(_build_request(messages=messages, model_profile="thorough"))


def test_request_fingerprint_accounts_for_examples():
    messages1 = LMRequestMessages(
        system="You are helpful",
        examples=[FewShotExample(user="Hi", assistant="Hello")],
        user="Hello",
    )
    messages2 = LMRequestMessages(
        system="You are helpful",
        examples=[FewShotExample(user="Hi", assistant="Greetings")],
        user="Hello",
    )

    assert _fingerprint(_build_request(messages=messages1)) != _fingerprint(
        _build_request(messages=messages2)
    )


def test_request_fingerprint_includes_structured_output():
    from pydantic import BaseModel

    class Model1(BaseModel):
        name: str

    class Model2(BaseModel):
        age: int

    messages = LMRequestMessages(system="You are helpful", examples=[], user="Hello")
    format1 = ResolvedResponseFormat(
        pydantic_model=Model1,
        json_schema=Model1.model_json_schema(),
        prompt_schema_definition="",
    )
    format2 = ResolvedResponseFormat(
        pydantic_model=Model2,
        json_schema=Model2.model_json_schema(),
        prompt_schema_definition="",
    )

    assert _fingerprint(
        _build_request(messages=messages, structured_output=format1)
    ) != _fingerprint(_build_request(messages=messages, structured_output=format2))


def test_request_fingerprint_accounts_for_pdf_files(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    _save_pdf_file(pdf_path, page_count=2, text_content="PDF content")

    messages_one = LMRequestMessages(
        system="You are helpful",
        examples=[],
        user=None,
        user_file=LMRequestFile(path=str(pdf_path), page_range=(0, 1)),
    )
    messages_two = LMRequestMessages(
        system="You are helpful",
        examples=[],
        user=None,
        user_file=LMRequestFile(path=str(pdf_path), page_range=(1, 2)),
    )

    assert _fingerprint(_build_request(messages=messages_one)) != _fingerprint(
        _build_request(messages=messages_two)
    )


def test_request_fingerprint_is_valid_sha256():
    messages = LMRequestMessages(system="You are helpful", examples=[], user="Hello")
    key = _fingerprint(_build_request(messages=messages))

    assert len(key) == 64
    assert all(c in "0123456789abcdef" for c in key)


def test_request_fingerprint_top_logprobs_included():
    messages = LMRequestMessages(system="You are helpful", examples=[], user="Hello")

    assert _fingerprint(
        _build_request(messages=messages, top_logprobs=None)
    ) != _fingerprint(_build_request(messages=messages, top_logprobs=5))


def test_request_fingerprint_respects_profile_hash_even_without_profile_name():
    messages = LMRequestMessages(system="You are helpful", examples=[], user="Hello")
    request = _build_request(messages=messages)

    assert _fingerprint(request, profile_hash="hash1") != _fingerprint(
        request, profile_hash="hash2"
    )

