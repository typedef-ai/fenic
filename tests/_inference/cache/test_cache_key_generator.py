"""Unit tests for CacheKeyGenerator."""

from fenic._inference.cache.key_generator import CacheKeyGenerator
from fenic._inference.types import (
    FenicCompletionsRequest,
    FewShotExample,
    LMRequestMessages,
)
from fenic.core._logical_plan.resolved_types import ResolvedResponseFormat


class TestCacheKeyGenerator:
    """Test suite for CacheKeyGenerator."""

    def test_compute_key_deterministic(self):
        """Test that identical requests generate identical keys."""
        messages = LMRequestMessages(
            system="You are helpful",
            examples=[],
            user="Hello, world!",
        )

        request1 = FenicCompletionsRequest(
            messages=messages,
            max_completion_tokens=100,
            top_logprobs=None,
            structured_output=None,
            temperature=0.7,
        )

        request2 = FenicCompletionsRequest(
            messages=messages,
            max_completion_tokens=100,
            top_logprobs=None,
            structured_output=None,
            temperature=0.7,
        )

        key1 = CacheKeyGenerator.compute_key(request1, "gpt-4o-mini")
        key2 = CacheKeyGenerator.compute_key(request2, "gpt-4o-mini")

        assert key1 == key2

    def test_compute_key_different_models(self):
        """Test that different models generate different keys."""
        messages = LMRequestMessages(
            system="You are helpful",
            examples=[],
            user="Hello, world!",
        )

        request = FenicCompletionsRequest(
            messages=messages,
            max_completion_tokens=100,
            top_logprobs=None,
            structured_output=None,
            temperature=0.7,
        )

        key1 = CacheKeyGenerator.compute_key(request, "gpt-4o-mini")
        key2 = CacheKeyGenerator.compute_key(request, "gpt-4o")

        assert key1 != key2

    def test_compute_key_different_messages(self):
        """Test that different messages generate different keys."""
        messages1 = LMRequestMessages(
            system="You are helpful",
            examples=[],
            user="Hello, world!",
        )

        messages2 = LMRequestMessages(
            system="You are helpful",
            examples=[],
            user="Goodbye, world!",
        )

        request1 = FenicCompletionsRequest(
            messages=messages1,
            max_completion_tokens=100,
            top_logprobs=None,
            structured_output=None,
            temperature=0.7,
        )

        request2 = FenicCompletionsRequest(
            messages=messages2,
            max_completion_tokens=100,
            top_logprobs=None,
            structured_output=None,
            temperature=0.7,
        )

        key1 = CacheKeyGenerator.compute_key(request1, "gpt-4o-mini")
        key2 = CacheKeyGenerator.compute_key(request2, "gpt-4o-mini")

        assert key1 != key2

    def test_compute_key_different_temperature(self):
        """Test that different temperatures generate different keys."""
        messages = LMRequestMessages(
            system="You are helpful",
            examples=[],
            user="Hello, world!",
        )

        request1 = FenicCompletionsRequest(
            messages=messages,
            max_completion_tokens=100,
            top_logprobs=None,
            structured_output=None,
            temperature=0.7,
        )

        request2 = FenicCompletionsRequest(
            messages=messages,
            max_completion_tokens=100,
            top_logprobs=None,
            structured_output=None,
            temperature=0.9,
        )

        key1 = CacheKeyGenerator.compute_key(request1, "gpt-4o-mini")
        key2 = CacheKeyGenerator.compute_key(request2, "gpt-4o-mini")

        assert key1 != key2

    def test_compute_key_different_max_tokens(self):
        """Test that different max_completion_tokens generate different keys."""
        messages = LMRequestMessages(
            system="You are helpful",
            examples=[],
            user="Hello, world!",
        )

        request1 = FenicCompletionsRequest(
            messages=messages,
            max_completion_tokens=100,
            top_logprobs=None,
            structured_output=None,
            temperature=0.7,
        )

        request2 = FenicCompletionsRequest(
            messages=messages,
            max_completion_tokens=200,
            top_logprobs=None,
            structured_output=None,
            temperature=0.7,
        )

        key1 = CacheKeyGenerator.compute_key(request1, "gpt-4o-mini")
        key2 = CacheKeyGenerator.compute_key(request2, "gpt-4o-mini")

        assert key1 != key2

    def test_compute_key_different_profile(self):
        """Test that different model profiles generate different keys."""
        messages = LMRequestMessages(
            system="You are helpful",
            examples=[],
            user="Hello, world!",
        )

        request1 = FenicCompletionsRequest(
            messages=messages,
            max_completion_tokens=100,
            top_logprobs=None,
            structured_output=None,
            temperature=0.7,
            model_profile="fast",
        )

        request2 = FenicCompletionsRequest(
            messages=messages,
            max_completion_tokens=100,
            top_logprobs=None,
            structured_output=None,
            temperature=0.7,
            model_profile="thorough",
        )

        key1 = CacheKeyGenerator.compute_key(request1, "gpt-4o-mini")
        key2 = CacheKeyGenerator.compute_key(request2, "gpt-4o-mini")

        assert key1 != key2

    def test_compute_key_with_examples(self):
        """Test that examples are included in key generation."""
        messages1 = LMRequestMessages(
            system="You are helpful",
            examples=[
                FewShotExample(user="Hi", assistant="Hello"),
            ],
            user="Hello, world!",
        )

        messages2 = LMRequestMessages(
            system="You are helpful",
            examples=[
                FewShotExample(user="Hi", assistant="Greetings"),
            ],
            user="Hello, world!",
        )

        request1 = FenicCompletionsRequest(
            messages=messages1,
            max_completion_tokens=100,
            top_logprobs=None,
            structured_output=None,
            temperature=0.7,
        )

        request2 = FenicCompletionsRequest(
            messages=messages2,
            max_completion_tokens=100,
            top_logprobs=None,
            structured_output=None,
            temperature=0.7,
        )

        key1 = CacheKeyGenerator.compute_key(request1, "gpt-4o-mini")
        key2 = CacheKeyGenerator.compute_key(request2, "gpt-4o-mini")

        # Different examples should produce different keys
        assert key1 != key2

    def test_compute_key_with_structured_output(self):
        """Test that structured output schema is included in key generation."""
        from pydantic import BaseModel

        # Create simple Pydantic models for testing
        class Model1(BaseModel):
            name: str

        class Model2(BaseModel):
            age: int

        messages = LMRequestMessages(
            system="You are helpful",
            examples=[],
            user="Hello, world!",
        )

        # Create ResolvedResponseFormat instances
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

        request1 = FenicCompletionsRequest(
            messages=messages,
            max_completion_tokens=100,
            top_logprobs=None,
            structured_output=format1,
            temperature=0.7,
        )

        request2 = FenicCompletionsRequest(
            messages=messages,
            max_completion_tokens=100,
            top_logprobs=None,
            structured_output=format2,
            temperature=0.7,
        )

        key1 = CacheKeyGenerator.compute_key(request1, "gpt-4o-mini")
        key2 = CacheKeyGenerator.compute_key(request2, "gpt-4o-mini")

        # Different schemas should produce different keys
        assert key1 != key2

    def test_compute_key_format(self):
        """Test that generated keys are valid SHA-256 hashes."""
        messages = LMRequestMessages(
            system="You are helpful",
            examples=[],
            user="Hello, world!",
        )

        request = FenicCompletionsRequest(
            messages=messages,
            max_completion_tokens=100,
            top_logprobs=None,
            structured_output=None,
            temperature=0.7,
        )

        key = CacheKeyGenerator.compute_key(request, "gpt-4o-mini")

        # SHA-256 produces 64 hex characters
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)

    def test_compute_key_with_top_logprobs(self):
        """Test that top_logprobs affects key generation."""
        messages = LMRequestMessages(
            system="You are helpful",
            examples=[],
            user="Hello, world!",
        )

        request1 = FenicCompletionsRequest(
            messages=messages,
            max_completion_tokens=100,
            top_logprobs=None,
            structured_output=None,
            temperature=0.7,
        )

        request2 = FenicCompletionsRequest(
            messages=messages,
            max_completion_tokens=100,
            top_logprobs=5,
            structured_output=None,
            temperature=0.7,
        )

        key1 = CacheKeyGenerator.compute_key(request1, "gpt-4o-mini")
        key2 = CacheKeyGenerator.compute_key(request2, "gpt-4o-mini")

        # Different top_logprobs should produce different keys
        assert key1 != key2
