"""Request lifecycle events used by execution-performance instrumentation.

The collector is intentionally optional: normal execution does not retain events. A
benchmark can attach a collector to a model client, label it with an execution ID,
and use these events to distinguish provider idle time from time spent queued.
"""

from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from math import ceil
from typing import Literal, Optional

RequestLifecycleEventType = Literal[
    "queued",
    "rate_limited",
    "dispatched",
    "settled",
    "retried",
    "failed",
]
RequestLifecycleCollector = Callable[["RequestLifecycleEvent"], None]


@dataclass(frozen=True)
class RequestLifecycleEvent:
    """One physical provider-request lifecycle transition.

    ``batch_id`` identifies the model-client batch. ``execution_id`` is supplied by
    an instrumented execution, so callers can correlate multiple semantic operators.
    Timestamps use :func:`time.monotonic_ns` and are comparable only within one process.
    """

    event: RequestLifecycleEventType
    timestamp_ns: int
    execution_id: Optional[str]
    batch_id: str
    request_index: int
    operation_name: str
    model: str
    provider: str


@dataclass(frozen=True)
class IdleGapMetrics:
    """Provider idle and queue-delay totals reconstructed from lifecycle events."""

    idle_gap_count: int
    total_idle_gap_ns: int
    total_non_rate_limited_idle_gap_ns: int
    p50_idle_gap_ns: Optional[int]
    p95_idle_gap_ns: Optional[int]
    total_queue_delay_ns: int
    total_rate_limited_ns: int
    idle_fraction: float


@dataclass
class _LifecycleState:
    inflight_requests: set[tuple[str, int]]
    queued_at_ns: dict[tuple[str, int], int]
    rate_limited_at_ns: dict[tuple[str, int], int]
    first_dispatch_ns: Optional[int] = None
    last_settlement_ns: Optional[int] = None
    final_timestamp_ns: Optional[int] = None


def compute_idle_gap_metrics(events: Iterable[RequestLifecycleEvent]) -> IdleGapMetrics:
    """Summarize idle intervals for each execution/model/provider stream.

    An idle interval starts only after all in-flight requests in a stream have
    settled or failed, and ends at the next dispatch. Queue delay is reported as
    the time from ``queued`` to ``dispatched``. A ``rate_limited`` transition
    identifies the portion of an idle interval that must not be attributed to
    upstream execution; the gross interval and its non-rate-limited remainder
    are both returned. Retry backoff stays in-flight, so it is excluded from both
    idle and queue delay; the metrics do not partition elapsed time under retries.
    B0 and B1 comparisons must use this same treatment.
    """

    states: dict[tuple[Optional[str], str, str], _LifecycleState] = defaultdict(
        lambda: _LifecycleState(set(), {}, {})
    )
    idle_gaps_ns: list[int] = []
    non_rate_limited_idle_gaps_ns: list[int] = []
    queue_delays_ns: list[int] = []
    rate_limited_delays_ns: list[int] = []
    stream_durations_ns: list[int] = []

    ordered_events = sorted(events, key=lambda event: event.timestamp_ns)
    for event in ordered_events:
        stream = (event.execution_id, event.model, event.provider)
        state = states[stream]
        request = (event.batch_id, event.request_index)
        state.final_timestamp_ns = event.timestamp_ns

        if event.event == "queued":
            state.queued_at_ns[request] = event.timestamp_ns
            continue

        if event.event == "rate_limited":
            state.rate_limited_at_ns.setdefault(request, event.timestamp_ns)
            continue

        if event.event == "dispatched":
            queued_at_ns = state.queued_at_ns.pop(request, None)
            if queued_at_ns is not None:
                queue_delays_ns.append(event.timestamp_ns - queued_at_ns)

            rate_limited_at_ns = state.rate_limited_at_ns.pop(request, None)

            if not state.inflight_requests and state.last_settlement_ns is not None:
                idle_gap_ns = event.timestamp_ns - state.last_settlement_ns
                idle_gaps_ns.append(idle_gap_ns)
                if rate_limited_at_ns is None:
                    non_rate_limited_idle_gaps_ns.append(idle_gap_ns)
                else:
                    rate_limited_ns = event.timestamp_ns - max(
                        rate_limited_at_ns, state.last_settlement_ns
                    )
                    rate_limited_delays_ns.append(rate_limited_ns)
                    non_rate_limited_idle_gaps_ns.append(idle_gap_ns - rate_limited_ns)

            state.inflight_requests.add(request)
            if state.first_dispatch_ns is None:
                state.first_dispatch_ns = event.timestamp_ns
            continue

        if event.event in {"settled", "failed"}:
            state.inflight_requests.discard(request)
            if not state.inflight_requests:
                state.last_settlement_ns = event.timestamp_ns

    for state in states.values():
        if state.first_dispatch_ns is None or state.final_timestamp_ns is None:
            continue
        stream_durations_ns.append(state.final_timestamp_ns - state.first_dispatch_ns)

    total_idle_gap_ns = sum(idle_gaps_ns)
    total_duration_ns = sum(stream_durations_ns)
    sorted_idle_gaps_ns = sorted(idle_gaps_ns)

    def percentile(percentile: float) -> Optional[int]:
        if not sorted_idle_gaps_ns:
            return None
        return sorted_idle_gaps_ns[ceil(percentile * len(sorted_idle_gaps_ns)) - 1]

    return IdleGapMetrics(
        idle_gap_count=len(idle_gaps_ns),
        total_idle_gap_ns=total_idle_gap_ns,
        total_non_rate_limited_idle_gap_ns=sum(non_rate_limited_idle_gaps_ns),
        p50_idle_gap_ns=percentile(0.50),
        p95_idle_gap_ns=percentile(0.95),
        total_queue_delay_ns=sum(queue_delays_ns),
        total_rate_limited_ns=sum(rate_limited_delays_ns),
        idle_fraction=total_idle_gap_ns / total_duration_ns if total_duration_ns else 0.0,
    )
