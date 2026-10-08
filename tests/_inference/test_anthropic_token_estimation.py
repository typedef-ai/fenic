"""Local input-token estimation for the Anthropic completions client.

Anthropic has no local tokenizer, so the client scales a tiktoken cl100k_base
count by a per-model ratio from the catalog. Claude Opus 4.7 introduced a
tokenizer that emits ~30% more tokens than the legacy one, so each model must
estimate with the ratio for its tokenizer generation.
"""

import math
from typing import get_args

import pytest

pytest.importorskip("anthropic")

import tiktoken  # noqa: E402

from fenic._inference.anthropic.anthropic_batch_chat_completions_client import (  # noqa: E402
    CL100K_TO_ANTHROPIC_TOKENIZER_RATIOS,
    AnthropicBatchCompletionsClient,
)
from fenic._inference.rate_limit_strategy import (  # noqa: E402
    SeparatedTokenRateLimitStrategy,
)
from fenic._inference.types import (  # noqa: E402
    FenicCompletionsRequest,
    LMRequestMessages,
)
from fenic.core._inference.model_catalog import (  # noqa: E402
    AnthropicLanguageModelName,
    CompletionModelParameters,
    ModelProvider,
    model_catalog,
)

TEXT = "Four score and seven years ago our fathers brought forth, upon this continent, a new nation."


def _client(monkeypatch, model: str) -> AnthropicBatchCompletionsClient:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    return AnthropicBatchCompletionsClient(
        model=model,
        rate_limit_strategy=SeparatedTokenRateLimitStrategy(
            rpm=100, input_tpm=10_000, output_tpm=10_000
        ),
    )


@pytest.mark.parametrize(
    "model,ratio",
    [
        ("claude-haiku-4-5", 1.05),
        ("claude-haiku-4-5-20251001", 1.05),
        ("claude-haiku-5-5", 1.40),
        ("claude-opus-5-5", 1.40),
    ],
)
def test_count_tokens_uses_the_model_tokenizer_ratio(monkeypatch, model, ratio):
    client = _client(monkeypatch, model)
    cl100k_tokens = len(tiktoken.get_encoding("cl100k_base").encode(TEXT))

    assert client.count_tokens(TEXT) == math.ceil(cl100k_tokens * ratio)


def test_new_tokenizer_models_reserve_more_input_tokens(monkeypatch):
    request = FenicCompletionsRequest(
        messages=LMRequestMessages(system=TEXT, examples=[], user=TEXT),
        max_completion_tokens=128,
        top_logprobs=None,
        structured_output=None,
        temperature=0.0,
    )

    legacy = _client(monkeypatch, "claude-haiku-4-5").estimate_tokens_for_request(request)
    new = _client(monkeypatch, "claude-haiku-5-5").estimate_tokens_for_request(request)

    assert new.input_tokens / legacy.input_tokens == pytest.approx(
        CL100K_TO_ANTHROPIC_TOKENIZER_RATIOS["claude-opus-4-7"]
        / CL100K_TO_ANTHROPIC_TOKENIZER_RATIOS["claude-3"],
        rel=0.02,
    )


def test_tokenizer_defaults_to_the_legacy_claude_tokenizer():
    params = CompletionModelParameters(
        input_token_cost=0, output_token_cost=0, context_window_length=1, max_output_tokens=1
    )

    assert params.anthropic_tokenizer == "claude-3"


def test_opus_4_7_and_later_models_use_the_new_tokenizer():
    # Measured with tools/predictive_token_accuracy.py; everything else
    # (Claude 3 through Opus/Sonnet 4.6 and Haiku 4.5) uses the legacy default.
    new_tokenizer_models = {
        model
        for model in get_args(AnthropicLanguageModelName)
        if model_catalog.get_completion_model_parameters(
            ModelProvider.ANTHROPIC, model
        ).anthropic_tokenizer
        == "claude-opus-4-7"
    }

    assert new_tokenizer_models == {
        "claude-opus-4-7",
        "claude-opus-4-8",
        "claude-opus-5",
        "claude-opus-5-5",
        "claude-sonnet-5",
        "claude-sonnet-5-5",
        "claude-fable-5",
        "claude-fable-5-1",
        "claude-haiku-5-5",
    }
