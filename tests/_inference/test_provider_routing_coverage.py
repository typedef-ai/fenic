"""Targeted coverage tests for adaptive-token-estimation across providers.

All tests are HERMETIC: no real network calls happen. Provider clients that
require an API-key environment variable are guarded by ``monkeypatch.setenv``
with a dummy value. The Anthropic SDK and Google genai SDK do NOT validate
keys at construction time, so a dummy string is sufficient to build the
client without raising.

Test inventory
--------------
1. test_thinking_tokens_feed_settlement_and_estimator
   Guards the ``usage.completion_tokens + usage.thinking_tokens`` path in
   ``ModelClient._reconcile_completion``. Uses the synthetic _FakeCompletionsClient
   from test_model_client_estimation.py (replicated inline to stay self-contained).

2. test_anthropic_routing_learns_and_caps_unchanged (enabled path)
3. test_anthropic_routing_disabled_stays_at_ceiling (disabled path)
   Both guard AnthropicBatchCompletionsClient adaptive routing.

4. test_gemini_routing_learns
   Guards GeminiNativeChatCompletionsClient adaptive routing.

5. test_openrouter_estimator_learns_even_though_strategy_ignores_tokens
   Guards OpenRouterBatchChatCompletionsClient: estimator still learns even
   though AdaptiveBackoffRateLimitStrategy.settle() is a no-op.
"""

import time

import pytest

from fenic._inference.model_client import ModelClient
from fenic._inference.rate_limit_strategy import (
    AdaptiveBackoffRateLimitStrategy,
    SeparatedTokenRateLimitStrategy,
    TokenEstimate,
    UnifiedTokenRateLimitStrategy,
)
from fenic._inference.token_counter import TiktokenTokenCounter
from fenic._inference.types import (
    FenicCompletionsRequest,
    FenicCompletionsResponse,
    LMRequestMessages,
    ResponseUsage,
)
from fenic.core._inference.model_catalog import ModelProvider
from fenic.core._resolved_session_config import (
    ResolvedAdaptiveTokenEstimationConfig,
    ResolvedAnthropicModelProfile,
)
from fenic.core.metrics import LMMetrics

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _req(max_tokens: int = 512) -> FenicCompletionsRequest:
    return FenicCompletionsRequest(
        messages=LMRequestMessages(system="s", examples=[], user="u"),
        max_completion_tokens=max_tokens,
        top_logprobs=None,
        structured_output=None,
        temperature=0.0,
    )


# ---------------------------------------------------------------------------
# Minimal fake client (mirrors _FakeCompletionsClient from
# test_model_client_estimation.py but self-contained here)
# ---------------------------------------------------------------------------

class _StubProviderClass:
    _base_url = None


class _FakeCompletionsClient(
    ModelClient[FenicCompletionsRequest, FenicCompletionsResponse]
):
    def __init__(self, strategy, adaptive_estimation=None):
        super().__init__(
            model="gpt-4o-mini",
            model_provider=ModelProvider.OPENAI,
            model_provider_class=_StubProviderClass(),
            rate_limit_strategy=strategy,
            token_counter=TiktokenTokenCounter(model_name="gpt-4o-mini"),
            adaptive_estimation=adaptive_estimation,
        )
        self._metrics = LMMetrics()

    async def make_single_request(self, request):
        return None

    def estimate_tokens_for_request(self, request) -> TokenEstimate:
        return TokenEstimate(input_tokens=10, output_tokens=10)

    def get_metrics(self) -> LMMetrics:
        return self._metrics

    def reset_metrics(self):
        self._metrics = LMMetrics()

    def _get_max_output_token_request_limit(self, request):
        return request.max_completion_tokens


def _fake_client(enabled: bool = True, margin: float = 1.0) -> _FakeCompletionsClient:
    strategy = UnifiedTokenRateLimitStrategy(rpm=1000, tpm=1_000_000)
    cfg = ResolvedAdaptiveTokenEstimationConfig(enabled=enabled, safety_margin=margin)
    return _FakeCompletionsClient(strategy, adaptive_estimation=cfg)


# ---------------------------------------------------------------------------
# Test 1: thinking tokens feed the estimator as completion + thinking
# ---------------------------------------------------------------------------

def test_thinking_tokens_feed_settlement_and_estimator():
    """_reconcile_completion must observe completion+thinking, not just completion.

    Setup: reserve 8704 output tokens, then feed 40 observations with
    completion_tokens=40, thinking_tokens=60 (total actual output = 100).
    After warming, _adaptive_output_reservation should converge to ~100, not 40.

    Also verifies bucket settlement: after a single reserve+reconcile the
    bucket capacity must increase (i.e. the over-reservation was refunded).
    """
    client = _fake_client(enabled=True, margin=1.0)
    try:
        req = _req(max_tokens=512)
        reserved_output = 8704

        # --- estimator warms up ---
        for _ in range(40):
            client._reconcile_completion(
                req,
                TokenEstimate(input_tokens=10, output_tokens=reserved_output),
                ResponseUsage(
                    prompt_tokens=10,
                    completion_tokens=40,
                    total_tokens=110,
                    thinking_tokens=60,
                ),
            )

        # (a) The estimator observed 40+60=100 each time, so p95 = 100.
        learned = client._adaptive_output_reservation(
            req, static_ceiling=reserved_output, reasoning=False
        )
        assert learned == 100, (
            f"Expected adaptive reservation 100 (completion+thinking), got {learned}"
        )

        # (b) A fresh bucket consumption + reconcile should yield a higher capacity
        # after reconciliation (over-reservation refunded).
        strategy = UnifiedTokenRateLimitStrategy(rpm=1000, tpm=1_000_000)
        cfg = ResolvedAdaptiveTokenEstimationConfig(enabled=True, safety_margin=1.0)
        settle_client = _FakeCompletionsClient(strategy, adaptive_estimation=cfg)
        try:
            settle_req = _req(512)
            reserved = TokenEstimate(input_tokens=10, output_tokens=8704)
            strategy.check_and_consume_rate_limit(reserved)
            before = strategy.unified_tokens_bucket._get_available_capacity(time.time())
            settle_client._reconcile_completion(
                settle_req,
                reserved,
                ResponseUsage(
                    prompt_tokens=10,
                    completion_tokens=40,
                    total_tokens=110,
                    thinking_tokens=60,
                ),
            )
            after = strategy.unified_tokens_bucket._get_available_capacity(time.time())
            # 8704 tokens were reserved; only 10+100=110 were actually used →
            # bucket must have grown.
            assert after > before, (
                "Bucket capacity should increase after refund of over-reservation"
            )
        finally:
            settle_client.shutdown()
    finally:
        client.shutdown()


# ---------------------------------------------------------------------------
# Test 2 + 3: Anthropic client learns and API cap is unchanged
# ---------------------------------------------------------------------------

def test_anthropic_routing_learns_and_caps_unchanged(monkeypatch):
    """AnthropicBatchCompletionsClient: estimate drops after warming; API cap unchanged."""
    pytest.importorskip("anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    from fenic._inference.anthropic.anthropic_batch_chat_completions_client import (
        AnthropicBatchCompletionsClient,
    )

    thinking_budget = 1024
    profile = ResolvedAnthropicModelProfile(thinking_token_budget=thinking_budget)

    client = AnthropicBatchCompletionsClient(
        model="claude-sonnet-4-6",
        rate_limit_strategy=SeparatedTokenRateLimitStrategy(
            rpm=1000, input_tpm=1_000_000, output_tpm=500_000
        ),
        profiles={"default": profile},
        default_profile_name="default",
        adaptive_estimation=ResolvedAdaptiveTokenEstimationConfig(
            enabled=True, safety_margin=1.0
        ),
    )
    try:
        req = _req(max_tokens=512)

        # cold estimate equals static ceiling (max_completion_tokens + thinking_budget)
        cold = client.estimate_tokens_for_request(req).output_tokens
        static_ceiling = req.max_completion_tokens + thinking_budget
        assert cold == static_ceiling, (
            f"Cold estimate {cold} should equal static ceiling {static_ceiling}"
        )

        # Warm the estimator with small actuals (completion=20, thinking=0)
        for _ in range(40):
            client._reconcile_completion(
                req,
                TokenEstimate(input_tokens=10, output_tokens=cold),
                ResponseUsage(
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                    thinking_tokens=0,
                ),
            )

        # learned estimate is below the cold ceiling
        learned = client.estimate_tokens_for_request(req).output_tokens
        assert learned < cold, (
            f"Learned estimate {learned} should be < cold ceiling {cold}"
        )
        # For pure p99 with constant 20 actual tokens and margin=1.0 → 20
        assert learned == 20, f"Expected learned=20, got {learned}"

        # API cap (_get_max_output_token_request_limit) must remain unchanged
        api_cap = client._get_max_output_token_request_limit(req)
        assert api_cap == static_ceiling, (
            f"API cap should be {static_ceiling} (unchanged), got {api_cap}"
        )
    finally:
        client.shutdown()


def test_anthropic_routing_disabled_stays_at_ceiling(monkeypatch):
    """When adaptive_estimation.enabled=False, estimate stays at the static ceiling."""
    pytest.importorskip("anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    from fenic._inference.anthropic.anthropic_batch_chat_completions_client import (
        AnthropicBatchCompletionsClient,
    )

    thinking_budget = 512
    profile = ResolvedAnthropicModelProfile(thinking_token_budget=thinking_budget)

    client = AnthropicBatchCompletionsClient(
        model="claude-sonnet-4-6",
        rate_limit_strategy=SeparatedTokenRateLimitStrategy(
            rpm=1000, input_tpm=1_000_000, output_tpm=500_000
        ),
        profiles={"default": profile},
        default_profile_name="default",
        adaptive_estimation=ResolvedAdaptiveTokenEstimationConfig(
            enabled=False, safety_margin=1.0
        ),
    )
    try:
        req = _req(max_tokens=512)
        static_ceiling = req.max_completion_tokens + thinking_budget

        # Feed many observations with small actuals
        for _ in range(40):
            client._reconcile_completion(
                req,
                TokenEstimate(input_tokens=10, output_tokens=static_ceiling),
                ResponseUsage(
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                    thinking_tokens=0,
                ),
            )

        estimate = client.estimate_tokens_for_request(req).output_tokens
        assert estimate == static_ceiling, (
            f"With estimation disabled, estimate {estimate} should equal "
            f"static ceiling {static_ceiling}"
        )
    finally:
        client.shutdown()


# ---------------------------------------------------------------------------
# Test 4: Gemini client learns
# ---------------------------------------------------------------------------

def test_gemini_routing_learns(monkeypatch):
    """GeminiNativeChatCompletionsClient: estimate drops after warming; API cap unchanged."""
    pytest.importorskip("google.genai")
    # GoogleDeveloperModelProvider reads GEMINI_API_KEY if present; set a dummy
    # to be safe. The client itself does not validate the key at construction.
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    from fenic._inference.google.gemini_native_chat_completions_client import (
        GeminiNativeChatCompletionsClient,
    )

    client = GeminiNativeChatCompletionsClient(
        rate_limit_strategy=UnifiedTokenRateLimitStrategy(rpm=1000, tpm=1_000_000),
        model_provider=ModelProvider.GOOGLE_DEVELOPER,
        model="gemini-2.5-flash",
        adaptive_estimation=ResolvedAdaptiveTokenEstimationConfig(
            enabled=True, safety_margin=1.0
        ),
    )
    try:
        req = _req(max_tokens=512)

        # cold estimate
        cold = client.estimate_tokens_for_request(req).output_tokens
        # static ceiling for Gemini (no thinking budget in default profile) = max_completion_tokens
        assert cold == req.max_completion_tokens, (
            f"Cold Gemini estimate {cold} should equal max_completion_tokens={req.max_completion_tokens}"
        )

        # Warm the estimator with small actuals
        for _ in range(40):
            client._reconcile_completion(
                req,
                TokenEstimate(input_tokens=10, output_tokens=cold),
                ResponseUsage(
                    prompt_tokens=10,
                    completion_tokens=25,
                    total_tokens=35,
                    thinking_tokens=0,
                ),
            )

        learned = client.estimate_tokens_for_request(req).output_tokens
        assert learned < cold, (
            f"Gemini learned estimate {learned} should be < cold ceiling {cold}"
        )
        assert learned == 25, f"Expected learned=25, got {learned}"

        # API cap unchanged
        api_cap = client._get_max_output_token_request_limit(req)
        assert api_cap == req.max_completion_tokens, (
            f"Gemini API cap should be {req.max_completion_tokens} (unchanged), got {api_cap}"
        )
    finally:
        client.shutdown()


# ---------------------------------------------------------------------------
# Test 5: OpenRouter estimator learns despite AdaptiveBackoff no-op settle()
# ---------------------------------------------------------------------------

def test_openrouter_estimator_learns_even_though_strategy_ignores_tokens(monkeypatch):
    """OpenRouterBatchChatCompletionsClient: estimator learns; AdaptiveBackoff settle is no-op."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    # OpenRouterModelProvider uses the OpenAI SDK (AsyncOpenAI) as its transport.
    # The OpenAI SDK raises if OPENAI_API_KEY is absent, even though OpenRouter
    # passes its own auth via default_headers. Set a dummy key to satisfy it.
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    # OpenRouter's provider singleton loads models lazily from the network when
    # model_catalog.get_completion_model_parameters is called for an unknown model.
    # Pre-register a fake model so the constructor never touches the network.
    from fenic.core._inference.model_catalog import (
        CompletionModelParameters,
        model_catalog,
    )

    _FAKE_OR_MODEL = "openai/fake-model-for-test"
    model_catalog.add_model(
        ModelProvider.OPENROUTER,
        _FAKE_OR_MODEL,
        CompletionModelParameters(
            input_token_cost=0.0,
            output_token_cost=0.0,
            context_window_length=128_000,
            max_output_tokens=8192,
            supported_parameters=set(),
        ),
    )

    from fenic._inference.openrouter.openrouter_batch_chat_completions_client import (
        OpenRouterBatchChatCompletionsClient,
    )

    strategy = AdaptiveBackoffRateLimitStrategy(rpm=100)
    client = OpenRouterBatchChatCompletionsClient(
        model=_FAKE_OR_MODEL,
        rate_limit_strategy=strategy,
        adaptive_estimation=ResolvedAdaptiveTokenEstimationConfig(
            enabled=True, safety_margin=1.0
        ),
    )
    try:
        req = _req(max_tokens=512)

        cold = client.estimate_tokens_for_request(req).output_tokens
        assert cold == req.max_completion_tokens

        # Warm the estimator
        for _ in range(40):
            client._reconcile_completion(
                req,
                TokenEstimate(input_tokens=10, output_tokens=cold),
                ResponseUsage(
                    prompt_tokens=10,
                    completion_tokens=30,
                    total_tokens=40,
                    thinking_tokens=0,
                ),
            )

        learned = client.estimate_tokens_for_request(req).output_tokens
        assert learned < cold, (
            f"OpenRouter learned estimate {learned} should be < cold ceiling {cold}"
        )
        assert learned == 30, f"Expected learned=30, got {learned}"

        # AdaptiveBackoff.settle() is a base no-op — must not raise
        reserved = TokenEstimate(input_tokens=10, output_tokens=512)
        actual = TokenEstimate(input_tokens=10, output_tokens=30)
        strategy.settle(reserved, actual)  # should be a silent no-op
    finally:
        client.shutdown()
