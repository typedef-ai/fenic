from fenic._inference import model_client as model_client_module
from fenic._inference.model_client import ModelClient
from fenic._inference.rate_limit_strategy import (
    TokenEstimate,
    UnifiedTokenRateLimitStrategy,
)
from fenic._inference.request_lifecycle import (
    RequestLifecycleEvent,
    compute_idle_gap_metrics,
)
from fenic._inference.token_counter import TiktokenTokenCounter
from fenic._inference.types import (
    FenicCompletionsRequest,
    FenicCompletionsResponse,
    LMRequestMessages,
)
from fenic.core._inference.model_catalog import ModelProvider
from fenic.core.metrics import LMMetrics


class _StubProviderClass:
    _base_url = None


class _FakeCompletionsClient(ModelClient[FenicCompletionsRequest, FenicCompletionsResponse]):
    def __init__(self):
        super().__init__(
            model="fake-model",
            model_provider=ModelProvider.OPENAI,
            model_provider_class=_StubProviderClass(),
            rate_limit_strategy=UnifiedTokenRateLimitStrategy(rpm=1_000, tpm=1_000_000),
            token_counter=TiktokenTokenCounter(model_name="gpt-4o-mini"),
        )
        self._metrics = LMMetrics()

    async def make_single_request(self, request):
        return FenicCompletionsResponse(completion="fake", logprobs=None)

    def estimate_tokens_for_request(self, request) -> TokenEstimate:
        return TokenEstimate(input_tokens=1, output_tokens=1)

    def get_metrics(self) -> LMMetrics:
        return self._metrics

    def reset_metrics(self):
        self._metrics = LMMetrics()

    def _get_max_output_token_request_limit(self, request):
        return request.max_completion_tokens


def _request():
    return FenicCompletionsRequest(
        messages=LMRequestMessages(system="system", examples=[], user="user"),
        max_completion_tokens=16,
        top_logprobs=None,
        structured_output=None,
        temperature=0.0,
    )


def test_lifecycle_events_label_serial_requests_and_measure_idle_gap(monkeypatch):
    timestamps = iter((100, 110, 120, 140, 150, 160))
    monkeypatch.setattr(model_client_module.time, "monotonic_ns", lambda: next(timestamps))

    client = _FakeCompletionsClient()
    events = []
    client.set_request_lifecycle_collector(events.append, execution_id="p0-fake-execution")
    try:
        client.make_batch_requests([_request()], operation_name="semantic.map")
        client.make_batch_requests([_request()], operation_name="semantic.extract")
    finally:
        client.shutdown()

    assert [event.event for event in events] == [
        "queued",
        "dispatched",
        "settled",
        "queued",
        "dispatched",
        "settled",
    ]
    assert {event.execution_id for event in events} == {"p0-fake-execution"}
    assert {event.model for event in events} == {"fake-model"}
    assert [event.operation_name for event in events] == [
        "semantic.map",
        "semantic.map",
        "semantic.map",
        "semantic.extract",
        "semantic.extract",
        "semantic.extract",
    ]

    metrics = compute_idle_gap_metrics(events)
    assert metrics.idle_gap_count == 1
    assert metrics.total_idle_gap_ns == 30
    assert metrics.total_non_rate_limited_idle_gap_ns == 30
    assert metrics.p50_idle_gap_ns == 30
    assert metrics.p95_idle_gap_ns == 30
    assert metrics.total_queue_delay_ns == 20
    assert metrics.total_rate_limited_ns == 0
    assert metrics.idle_fraction == 0.6


def test_idle_metrics_exclude_rate_limited_wait_from_attribution():
    def event(name, timestamp):
        return RequestLifecycleEvent(
            event=name,
            timestamp_ns=timestamp,
            execution_id="p0-fake-execution",
            batch_id="batch-1" if timestamp < 140 else "batch-2",
            request_index=0,
            operation_name="semantic.map" if timestamp < 140 else "semantic.extract",
            model="fake-model",
            provider="openai",
        )

    metrics = compute_idle_gap_metrics(
        [
            event("queued", 100),
            event("dispatched", 110),
            event("settled", 120),
            event("queued", 140),
            event("rate_limited", 150),
            event("dispatched", 160),
            event("settled", 170),
        ]
    )

    assert metrics.total_idle_gap_ns == 40
    assert metrics.total_rate_limited_ns == 10
    assert metrics.total_non_rate_limited_idle_gap_ns == 30


def test_lifecycle_marks_rate_limited_wait_once(monkeypatch):
    timestamps = iter((100, 110, 120, 130))
    monkeypatch.setattr(model_client_module.time, "monotonic_ns", lambda: next(timestamps))

    client = _FakeCompletionsClient()
    checks = iter((False, True))
    monkeypatch.setattr(client, "_check_and_consume_rate_limit", lambda _: next(checks))
    events = []
    client.set_request_lifecycle_collector(events.append, execution_id="p0-rate-limit")
    try:
        client.make_batch_requests([_request()], operation_name="semantic.map")
    finally:
        client.shutdown()

    assert [event.event for event in events] == [
        "queued",
        "rate_limited",
        "dispatched",
        "settled",
    ]


def test_idle_metrics_do_not_count_retry_backoff_as_execution_idle():
    def event(name, timestamp):
        return RequestLifecycleEvent(
            event=name,
            timestamp_ns=timestamp,
            execution_id="p0-fake-execution",
            batch_id="batch-1",
            request_index=0,
            operation_name="semantic.map",
            model="fake-model",
            provider="openai",
        )

    metrics = compute_idle_gap_metrics(
        [
            event("queued", 100),
            event("dispatched", 110),
            event("retried", 120),
            event("dispatched", 140),
            event("settled", 150),
        ]
    )

    assert metrics.idle_gap_count == 0
    assert metrics.total_idle_gap_ns == 0
    assert metrics.total_queue_delay_ns == 10
