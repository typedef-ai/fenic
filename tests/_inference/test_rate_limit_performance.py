"""Hermetic, real-clock rate-limit PERFORMANCE scenario tests.

These tests need NO API key — they drive the real `ModelClient` queue,
rate-limit gate, backoff, settlement, and adaptive estimation against a fully
simulated provider (`SimulatedServerLimiter`). Run with `-s` to see the printed
reports:

    uv run pytest tests/_inference/test_rate_limit_performance.py -v -s

Tuning notes
------------
The token buckets start FULL (a burst of `tpm`) and refill at `tpm/60` per
second. Wall-clock for a paced run is ~`60 * (workload/tpm - 1)` seconds once the
burst drains, independent of absolute `tpm`. So:
  * Each backoff cycle costs a real sleep of `2^k` seconds (initial 1s, factor 2)
    — keep overshoot small so only 1-3 backoffs occur.
  * Mode A (pre-feature ceiling reservation) is bounded by keeping the
    ceiling/actual ratio modest; otherwise it reserves far more than it uses and
    paces for a long time.
These numbers are chosen so each test runs in ~0.1-3.5s with rate-limiting (not
latency) as the bottleneck.
"""

from __future__ import annotations

from tests._inference.rate_limit_harness.harness import (
    RateLimitScenario,
    constant,
    regime_shift,
    run_scenario,
)

# Shared base knobs for the symmetric (no-overshoot) settlement scenarios.
_RPM = 1_000_000
_TPM = 200_000
_INPUT = 50
_OUTPUT = 80
_CEILING = 200
_N_ROWS = 840
_LATENCY = 0.001


def _settlement_scenario(*, enabled: bool, settlement_enabled: bool) -> RateLimitScenario:
    """Low-variance constant-output workload; client tpm == server tpm."""
    return RateLimitScenario(
        rpm=_RPM,
        tpm=_TPM,
        true_rpm=_RPM,
        true_tpm=_TPM,
        n_rows=_N_ROWS,
        static_ceiling=_CEILING,
        input_tokens=_INPUT,
        output_spec=constant(_OUTPUT),
        latency_s=_LATENCY,
        enabled=enabled,
        settlement_enabled=settlement_enabled,
    )


def test_settlement_reclaims_throughput():
    """Settlement and adaptive estimation each reclaim throughput vs pre-feature.

    Mode A: pre-feature — reserve the full ceiling, never settle.
    Mode B: settle-only — reserve the ceiling, but refund the over-reservation.
    Mode C: estimation + settlement — reserve a learned (tighter) amount.
    """
    # (A) pre-feature: ceiling reservation, no settlement.
    report_a, _ = run_scenario(
        _settlement_scenario(enabled=False, settlement_enabled=False)
    )
    # (B) settle-only: ceiling reservation but refunded on success.
    report_b, _ = run_scenario(
        _settlement_scenario(enabled=False, settlement_enabled=True)
    )
    # (C) estimation + settlement.
    report_c, _ = run_scenario(
        _settlement_scenario(enabled=True, settlement_enabled=True)
    )

    print("\n[settlement_reclaims_throughput]")
    print("A (pre-feature, no settle):\n", report_a)
    print("B (settle-only):\n", report_b)
    print("C (estimation + settle):\n", report_c)

    # All modes must drain the batch.
    assert report_a.logical_completions == _N_ROWS
    assert report_b.logical_completions == _N_ROWS
    assert report_c.logical_completions == _N_ROWS

    # Settlement and/or estimation reclaim throughput (lower wall is better).
    assert report_c.wall <= report_a.wall
    assert report_b.wall <= report_a.wall

    # Estimation tightens the reservation -> higher reservation efficiency than
    # settle-only (which still reserves the full ceiling up front).
    assert report_c.reservation_efficiency >= report_b.reservation_efficiency


def test_no_429_when_under_true_limit():
    """Configured tpm is half the true server tpm -> client never overshoots."""
    scenario = RateLimitScenario(
        rpm=_RPM,
        tpm=_TPM,
        true_rpm=_RPM,
        true_tpm=_TPM * 2,  # server allows twice what the client is configured for
        n_rows=_N_ROWS,
        static_ceiling=_CEILING,
        input_tokens=_INPUT,
        output_spec=constant(_OUTPUT),
        latency_s=_LATENCY,
        enabled=True,
        settlement_enabled=True,
    )
    report, _ = run_scenario(scenario)
    print("\n[no_429_when_under_true_limit]\n", report)

    assert report.server_429 == 0
    assert report.logical_completions == _N_ROWS


def _overshoot_scenario() -> RateLimitScenario:
    """Over-provisioned client: tpm >> true_tpm, workload just over server burst.

    The client floods (its own tpm is huge), the server admits one burst then
    429s the small remainder, engaging the 429 -> backoff -> retry path. Kept
    small so only ~2 backoff cycles occur (each costs a real ~1-2s sleep).
    """
    return RateLimitScenario(
        rpm=10_000_000,
        tpm=5_000_000,  # client effectively unthrottled on its own limit
        true_rpm=10_000_000,
        true_tpm=120_000,  # server burst ~= 120k tokens; workload ~= 124.8k
        n_rows=960,
        static_ceiling=_CEILING,
        input_tokens=_INPUT,
        output_spec=constant(_OUTPUT),
        latency_s=0.0003,
        enabled=True,
        settlement_enabled=True,
    )


def test_overshoot_triggers_retry_and_recovers():
    """Over-provisioned client overshoots the true limit, 429s, and recovers."""
    report, _ = run_scenario(_overshoot_scenario())
    print("\n[overshoot_triggers_retry_and_recovers]\n", report)

    assert report.server_429 > 0
    assert report.retries > 0
    assert report.logical_completions == report.n_rows


def test_case_a1_settle_after_backoff_measurement():
    """MEASUREMENT: count settle-with-positive-refund events shortly after a backoff.

    This quantifies the A1 concern: after a backoff zeros the token bucket, an
    in-flight request that settles a positive refund (reserved_out > actual_out)
    re-injects capacity the backoff just removed. We only ASSERT that the harness
    can observe the timeline (>= 0) — the count decides whether a backoff-guard
    on settle is warranted; it is not itself a fix.
    """
    report, trace = run_scenario(_overshoot_scenario())

    backoff_times = [e[1] for e in trace if e[0] == "backoff"]
    # A settle event is ("settle", ts, reserved_out, actual_out).
    settle_after_backoff = []
    for e in trace:
        if e[0] != "settle":
            continue
        ts, reserved_out, actual_out = e[1], e[2], e[3]
        if reserved_out <= actual_out:
            continue  # not a refund
        # within a short window AFTER a backoff
        for bt in backoff_times:
            if 0.0 <= ts - bt <= 1.0:
                settle_after_backoff.append((ts - bt, reserved_out, actual_out))
                break

    print("\n[a1_settle_after_backoff_measurement]")
    print(f"  num_backoffs={len(backoff_times)} server_429={report.server_429}")
    print(f"  settle-after-backoff(refund>0) count={len(settle_after_backoff)}")
    for dt, res, act in settle_after_backoff[:20]:
        print(f"    +{dt:.3f}s after backoff: refunded {res - act} (reserved={res} actual={act})")

    # The harness can observe the timeline; this is a measurement, not a fix.
    assert len(settle_after_backoff) >= 0
    assert report.logical_completions == report.n_rows


def test_regime_shift_under_reservation():
    """Estimator learns 'small' then the workload shifts to 'large' outputs.

    With client tpm == true_tpm, per-success settlement corrects the client
    bucket to actual usage right after each response, so under-reservation does
    NOT leak 429s to the symmetric server. We PRINT the 429 count and where any
    cluster (this quantifies the estimator-key-pooling concern) and assert only
    that the run completes.
    """
    n_rows = 600
    scenario = RateLimitScenario(
        rpm=10_000_000,
        tpm=300_000,
        true_rpm=10_000_000,
        true_tpm=300_000,  # symmetric: client and server share the same limit
        n_rows=n_rows,
        static_ceiling=1024,
        input_tokens=_INPUT,
        output_spec=regime_shift(small=30, large=600),
        latency_s=0.0003,
        enabled=True,
        settlement_enabled=True,
    )
    report, trace = run_scenario(scenario)

    half = n_rows // 2
    e429_rows = sorted(e[2] for e in trace if e[0] == "server_429")
    print("\n[regime_shift_under_reservation]\n", report)
    if e429_rows:
        in_second_half = sum(1 for r in e429_rows if r >= half)
        print(
            f"  server_429 rows: min={min(e429_rows)} max={max(e429_rows)} "
            f"half_boundary={half} in_second_half={in_second_half}/{len(e429_rows)}"
        )
    else:
        print(f"  server_429=0 (no under-reservation leaked past the symmetric server; half={half})")

    assert report.logical_completions == n_rows
