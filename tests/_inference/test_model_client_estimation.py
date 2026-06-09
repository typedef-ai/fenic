from fenic._inference.model_client import ModelClient
from fenic._inference.rate_limit_strategy import (
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
from fenic.core._resolved_session_config import ResolvedAdaptiveTokenEstimationConfig
from fenic.core.metrics import LMMetrics


class _StubProviderClass:
    _base_url = None


class _FakeCompletionsClient(ModelClient[FenicCompletionsRequest, FenicCompletionsResponse]):
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


def _request(max_tokens=512):
    return FenicCompletionsRequest(
        messages=LMRequestMessages(system="s", examples=[], user="u"),
        max_completion_tokens=max_tokens,
        top_logprobs=None,
        structured_output=None,
        temperature=0.0,
    )


def _client(enabled=True, margin=1.0):
    strategy = UnifiedTokenRateLimitStrategy(rpm=1000, tpm=1_000_000)
    cfg = ResolvedAdaptiveTokenEstimationConfig(enabled=enabled, safety_margin=margin)
    return _FakeCompletionsClient(strategy, adaptive_estimation=cfg)


def test_estimator_key_uses_profile_and_max_tokens():
    client = _client()
    try:
        key = client._estimator_key(_request(512))
        assert key == (client.get_profile_hash_for_request(_request(512)), 512)
    finally:
        client.shutdown()


def test_adaptive_reservation_cold_then_warm():
    client = _client(enabled=True, margin=1.0)
    try:
        req = _request(512)
        assert client._adaptive_output_reservation(req, static_ceiling=8704, reasoning=False) == 8704
        for _ in range(40):
            client._reconcile_completion(
                req,
                TokenEstimate(input_tokens=10, output_tokens=8704),
                ResponseUsage(prompt_tokens=10, completion_tokens=100, total_tokens=110),
            )
        assert client._adaptive_output_reservation(req, static_ceiling=8704, reasoning=False) == 100
    finally:
        client.shutdown()


def test_reconcile_settles_bucket_and_records_metrics():
    import time
    client = _client()
    try:
        req = _request(512)
        reserved = TokenEstimate(input_tokens=10, output_tokens=8704)
        client.rate_limit_strategy.check_and_consume_rate_limit(reserved)
        before = client.rate_limit_strategy.unified_tokens_bucket._get_available_capacity(time.time())
        client._reconcile_completion(
            req,
            reserved,
            ResponseUsage(prompt_tokens=10, completion_tokens=50, total_tokens=60),
        )
        after = client.rate_limit_strategy.unified_tokens_bucket._get_available_capacity(time.time())
        assert after > before
        assert client.get_metrics().num_reserved_output_tokens == 8704
    finally:
        client.shutdown()


def test_reconcile_skips_when_nothing_reserved():
    client = _client()
    try:
        req = _request(512)
        client._reconcile_completion(
            req,
            TokenEstimate(input_tokens=0, output_tokens=0),
            ResponseUsage(prompt_tokens=10, completion_tokens=50, total_tokens=60),
        )
        assert client.get_metrics().num_reserved_output_tokens == 0
    finally:
        client.shutdown()


def test_disabled_reservation_returns_ceiling_even_after_observations():
    client = _client(enabled=False)
    try:
        req = _request(512)
        for _ in range(40):
            client._reconcile_completion(
                req,
                TokenEstimate(input_tokens=10, output_tokens=8704),
                ResponseUsage(prompt_tokens=10, completion_tokens=100, total_tokens=110),
            )
        assert client._adaptive_output_reservation(req, static_ceiling=8704, reasoning=False) == 8704
    finally:
        client.shutdown()


def test_settlement_runs_when_disabled():
    """Settlement (bucket reconciliation) is always-on even when estimation is disabled.

    enabled=False only disables adaptive *estimation* (reservations use the static
    ceiling). Settlement still refunds over-reservation because it is pure accounting
    that cannot increase 429 risk.
    """
    import time

    client = _client(enabled=False)
    try:
        req = _request(512)
        reserved = TokenEstimate(input_tokens=10, output_tokens=8704)
        client.rate_limit_strategy.check_and_consume_rate_limit(reserved)
        before = client.rate_limit_strategy.unified_tokens_bucket._get_available_capacity(time.time())
        client._reconcile_completion(
            req,
            reserved,
            ResponseUsage(prompt_tokens=10, completion_tokens=50, total_tokens=60),
        )
        after = client.rate_limit_strategy.unified_tokens_bucket._get_available_capacity(time.time())
        # Settlement must have refunded the over-reservation even though estimation is disabled.
        assert after > before
    finally:
        client.shutdown()
