from fenic._inference.openai.openai_batch_chat_completions_client import (
    OpenAIBatchChatCompletionsClient,
)
from fenic._inference.rate_limit_strategy import UnifiedTokenRateLimitStrategy, TokenEstimate
from fenic._inference.types import FenicCompletionsRequest, LMRequestMessages, ResponseUsage
from fenic.core._resolved_session_config import ResolvedAdaptiveTokenEstimationConfig


def _req():
    return FenicCompletionsRequest(
        messages=LMRequestMessages(system="s", examples=[], user="u"),
        max_completion_tokens=512,
        top_logprobs=None,
        structured_output=None,
        temperature=0.0,
    )


def test_disabled_matches_static_ceiling_after_observations():
    client = OpenAIBatchChatCompletionsClient(
        model="gpt-4o-mini",
        rate_limit_strategy=UnifiedTokenRateLimitStrategy(rpm=1000, tpm=1_000_000),
        adaptive_estimation=ResolvedAdaptiveTokenEstimationConfig(enabled=False),
    )
    try:
        req = _req()
        baseline = client.estimate_tokens_for_request(req).output_tokens
        for _ in range(40):
            client._reconcile_completion(
                req,
                TokenEstimate(input_tokens=10, output_tokens=baseline),
                ResponseUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            )
        # disabled -> estimate never moves from the static ceiling
        assert client.estimate_tokens_for_request(req).output_tokens == baseline == 512
    finally:
        client.shutdown()
