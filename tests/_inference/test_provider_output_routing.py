from fenic._inference.openai.openai_batch_chat_completions_client import (
    OpenAIBatchChatCompletionsClient,
)
from fenic._inference.rate_limit_strategy import (
    TokenEstimate,
    UnifiedTokenRateLimitStrategy,
)
from fenic._inference.types import (
    FenicCompletionsRequest,
    LMRequestMessages,
    ResponseUsage,
)
from fenic.core._resolved_session_config import ResolvedAdaptiveTokenEstimationConfig


def _req(max_tokens=512):
    return FenicCompletionsRequest(
        messages=LMRequestMessages(system="s", examples=[], user="u"),
        max_completion_tokens=max_tokens,
        top_logprobs=None,
        structured_output=None,
        temperature=0.0,
    )


def _openai_client(margin=1.0):
    return OpenAIBatchChatCompletionsClient(
        model="gpt-4o-mini",
        rate_limit_strategy=UnifiedTokenRateLimitStrategy(rpm=1000, tpm=1_000_000),
        adaptive_estimation=ResolvedAdaptiveTokenEstimationConfig(enabled=True, safety_margin=margin),
    )


def test_openai_output_estimate_drops_after_learning():
    client = _openai_client(margin=1.0)
    try:
        req = _req(512)
        ceiling = client.estimate_tokens_for_request(req).output_tokens  # cold = static ceiling
        for _ in range(40):
            client._reconcile_completion(
                req,
                TokenEstimate(input_tokens=10, output_tokens=ceiling),
                ResponseUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
            )
        learned = client.estimate_tokens_for_request(req).output_tokens
        assert learned < ceiling
        assert learned == 20  # p95 of constant 20 * margin 1.0
        # the API cap is unchanged (still the generous ceiling)
        assert client._get_max_output_token_request_limit(req) == 512
    finally:
        client.shutdown()
