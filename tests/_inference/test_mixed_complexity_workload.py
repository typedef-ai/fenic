"""Provider-free proof tests for the multi-step workload harness."""

from __future__ import annotations

import os

import pytest

from fenic._inference.request_lifecycle import (
    RequestLifecycleEvent,
    compute_idle_gap_metrics,
)
from tests._inference.rate_limit_harness.mixed_workload import (
    WorkloadScenario,
    assert_arm_parity,
    default_step_specs,
    run_workload_arm,
)


def test_step_draws_are_stable_by_step_and_row():
    """Step-local seeded distributions must not depend on dispatch order."""
    scenario = WorkloadScenario(n_rows=8, base_seed=91)

    first = scenario.step_output_draws()
    second = WorkloadScenario(n_rows=8, base_seed=91).step_output_draws()

    assert first == second
    assert set(first) == {step.step_id for step in default_step_specs()}
    assert all(len(draws) == 8 for draws in first.values())


def test_workload_arms_preserve_outputs_tokens_and_logical_totals(local_session):
    """The three arms are interchangeable as semantic computations."""
    scenario = WorkloadScenario(n_rows=8, base_seed=92, true_tpm=150_000)

    reports = [
        run_workload_arm(local_session, scenario, arm)
        for arm in ("barriered", "unbarriered_unfused", "unbarriered_fused")
    ]

    assert_arm_parity(reports)
    assert reports[0].used_fusion is False
    assert reports[2].used_fusion is True
    assert reports[1].used_fusion is False
    assert all(
        report.logical_completion_total == report.expected_logical_completion_total
        for report in reports
    )
    assert all(
        report.lm_metrics["num_requests"] == report.expected_logical_completion_total
        for report in reports
    )
    assert all(report.lifecycle_event_count > 0 for report in reports)
    assert all(report.server_429 == 0 for report in reports)


def test_modest_server_mismatch_exercises_real_retry_path(local_session):
    """The simulator must naturally expose 429/retry without a provider."""
    # This seeded workload consumes 16,503 actual server tokens. A 16,000 TPM
    # server admits nearly all work, then naturally returns a small overshoot.
    scenario = WorkloadScenario(n_rows=8, base_seed=93, tpm=200_000, true_tpm=16_000)

    report = run_workload_arm(local_session, scenario, "unbarriered_fused")

    assert report.server_429 > 0
    assert report.retries > 0
    assert report.logical_completion_total == report.expected_logical_completion_total


def test_multi_step_idle_metrics_exclude_rate_limited_wait():
    """P0 keeps provider-rate-limit wait out of multi-step execution idle."""
    events = (
        RequestLifecycleEvent("queued", 100, "workload", "step-01", 0, "semantic.extract", "sim", "openai"),
        RequestLifecycleEvent("dispatched", 110, "workload", "step-01", 0, "semantic.extract", "sim", "openai"),
        RequestLifecycleEvent("settled", 120, "workload", "step-01", 0, "semantic.extract", "sim", "openai"),
        RequestLifecycleEvent("queued", 121, "workload", "step-11", 0, "semantic.map", "sim", "openai"),
        RequestLifecycleEvent("rate_limited", 125, "workload", "step-11", 0, "semantic.map", "sim", "openai"),
        RequestLifecycleEvent("dispatched", 150, "workload", "step-11", 0, "semantic.map", "sim", "openai"),
        RequestLifecycleEvent("settled", 160, "workload", "step-11", 0, "semantic.map", "sim", "openai"),
    )

    metrics = compute_idle_gap_metrics(events)

    assert metrics.total_idle_gap_ns == 30
    assert metrics.total_rate_limited_ns == 25
    assert metrics.total_non_rate_limited_idle_gap_ns == 5


@pytest.mark.skipif(
    os.environ.get("RUN_MIXED_WORKLOAD_PILOT") != "1",
    reason="explicit real-clock benchmark only",
)
def test_mixed_complexity_workload_pilot(local_session, tmp_path):
    """Explicit real-clock 24-row pilot: RUN_MIXED_WORKLOAD_PILOT=1 pytest ... -s."""
    from pathlib import Path

    from tests._inference.rate_limit_harness.mixed_workload import run_pilot

    output_dir = Path(os.environ.get("MIXED_WORKLOAD_OUTPUT_DIR", tmp_path))
    receipt = run_pilot(local_session, output_dir=output_dir)

    assert receipt["arm_parity"] is True
    assert receipt["reports"][2]["used_fusion"] is True


@pytest.mark.skipif(
    os.environ.get("RUN_MIXED_WORKLOAD_MATRIX") != "1",
    reason="explicit real-clock benchmark only",
)
def test_mixed_complexity_workload_matrix(local_session, tmp_path):
    """Explicit benchmark command: RUN_MIXED_WORKLOAD_MATRIX=1 pytest ... -s."""
    from pathlib import Path

    from tests._inference.rate_limit_harness.mixed_workload import run_matrix

    output_dir = Path(os.environ.get("MIXED_WORKLOAD_OUTPUT_DIR", tmp_path))
    receipts = run_matrix(local_session, output_dir=output_dir)

    assert len(receipts) in (10, 12)  # Third 192-row seed may be reduced by the 45m guard.
    assert all(receipt["arm_parity"] for receipt in receipts)


@pytest.mark.skipif(
    os.environ.get("RUN_MIXED_WORKLOAD_RESUME") != "1",
    reason="explicit bounded benchmark recovery only",
)
def test_resume_partial_mixed_complexity_workload_matrix(local_session):
    """Complete one documented missing simulated third-seed lane after a runner fix."""
    from pathlib import Path

    from tests._inference.rate_limit_harness.mixed_workload import (
        resume_partial_third_seed,
    )

    output_dir = Path(os.environ["MIXED_WORKLOAD_OUTPUT_DIR"])
    receipt = resume_partial_third_seed(local_session, output_dir)

    assert receipt["arm_parity"] is True
    assert receipt["reports"][2]["used_fusion"] is True
