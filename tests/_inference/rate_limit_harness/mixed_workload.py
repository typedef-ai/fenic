"""Hermetic, real-clock multi-step workload harness built on the rate-limit base.

The workload drives genuine Fenic semantic operators through the real
``ModelClient`` queue, rate-limit gate, retry/backoff, settlement, B0 stream
path, P0 lifecycle collector, and (for one explicitly labelled overlay) B1's
``MapExtract`` execution.  The simulated client is the only completion client.
"""

from __future__ import annotations

import json
import re
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Literal

from pydantic import BaseModel, Field

from fenic import StringType, col, lit, semantic, text
from fenic._backends.local.physical_plan.transform import FusedMapExtractExec
from fenic._backends.local.transpiler.plan_converter import PlanConverter
from fenic._inference.model_client import TransientException
from fenic._inference.request_lifecycle import compute_idle_gap_metrics
from fenic._inference.types import (
    FenicCompletionsRequest,
    FenicCompletionsResponse,
    ResponseUsage,
)
from fenic.core.metrics import LMMetrics
from tests._inference.rate_limit_harness.harness import (
    OutputSpec,
    RateLimitScenario,
    SimulatedCompletionsClient,
    constant,
    lognormal,
    regime_shift,
)

Arm = Literal["barriered", "unbarriered_unfused", "unbarriered_fused"]
_MARKER = re.compile(r"workload-step-([a-z0-9-]+) row=(\d+)")
HARNESS_VERSION = "mixed-workload-harness-v1"


class _StepOutput(BaseModel):
    """Small schema shared by every simulated semantic.extract workload step."""

    label: str = Field(description="Deterministic simulated workload result")


@dataclass(frozen=True)
class WorkloadStepSpec:
    """One pass-shaped simulated semantic operation."""

    step_id: str
    operation: Literal["map", "extract"]
    input_tokens: int
    max_output_tokens: int
    output_spec: OutputSpec


def default_step_specs() -> tuple[WorkloadStepSpec, ...]:
    """Return the typedef-shaped 12-step envelope plus the B1-only overlay."""
    return (
        WorkloadStepSpec("01", "extract", 48, 160, constant(72)),
        WorkloadStepSpec("02", "extract", 72, 256, lognormal(4.4, 0.30)),
        WorkloadStepSpec("03", "map", 56, 160, constant(56)),
        WorkloadStepSpec("04", "map", 80, 128, lognormal(3.6, 0.25)),
        WorkloadStepSpec("05", "map", 88, 192, regime_shift(40, 120)),
        WorkloadStepSpec("06", "map", 56, 224, lognormal(4.2, 0.35)),
        WorkloadStepSpec("07", "map", 56, 192, constant(80)),
        WorkloadStepSpec("08", "map", 88, 192, lognormal(4.0, 0.25)),
        WorkloadStepSpec("09", "map", 56, 256, regime_shift(48, 180)),
        WorkloadStepSpec("10", "map", 88, 288, lognormal(4.5, 0.35)),
        WorkloadStepSpec("10a", "map", 56, 224, constant(96)),
        WorkloadStepSpec("11", "map", 32, 768, lognormal(5.5, 0.45)),
        WorkloadStepSpec("overlay-map", "map", 56, 224, constant(64)),
        WorkloadStepSpec("overlay-extract", "extract", 48, 256, constant(96)),
    )


@dataclass
class WorkloadScenario:
    """Versioned, seeded input for one mixed-complexity workload run."""

    n_rows: int
    base_seed: int = 1234
    rpm: int = 1_000_000
    tpm: int = 150_000
    true_rpm: int = 1_000_000
    true_tpm: int = 150_000
    latency_s: float = 0.001
    safety_margin: float = 1.15
    version: str = "mixed-workload-v1"
    steps: tuple[WorkloadStepSpec, ...] = field(default_factory=default_step_specs)

    def step_output_draws(self) -> dict[str, list[int]]:
        """Precompute deterministic output draws keyed by step, not dispatch order."""
        return {
            step.step_id: step.output_spec(self.n_rows, self.base_seed + index)
            for index, step in enumerate(self.steps, start=1)
        }

    def receipt_config(self) -> dict[str, object]:
        """Return only JSON-stable scenario metadata for a receipt."""
        return {
            "version": self.version,
            "n_rows": self.n_rows,
            "base_seed": self.base_seed,
            "rpm": self.rpm,
            "tpm": self.tpm,
            "true_rpm": self.true_rpm,
            "true_tpm": self.true_tpm,
            "latency_s": self.latency_s,
            "safety_margin": self.safety_margin,
            "steps": [
                {
                    "step_id": step.step_id,
                    "operation": step.operation,
                    "input_tokens": step.input_tokens,
                    "max_output_tokens": step.max_output_tokens,
                    "seed": self.base_seed + index,
                }
                for index, step in enumerate(self.steps, start=1)
            ],
        }


class WorkloadSimulatedCompletionsClient(SimulatedCompletionsClient):
    """Step-aware completion simulator that retains the real ModelClient machinery."""

    def __init__(self, scenario: WorkloadScenario):
        self.workload_scenario = scenario
        self.steps = {step.step_id: step for step in scenario.steps}
        self.draws = scenario.step_output_draws()
        bootstrap = RateLimitScenario(
            rpm=scenario.rpm,
            tpm=scenario.tpm,
            true_rpm=scenario.true_rpm,
            true_tpm=scenario.true_tpm,
            n_rows=scenario.n_rows,
            static_ceiling=max(step.max_output_tokens for step in scenario.steps),
            input_tokens=max(step.input_tokens for step in scenario.steps),
            output_spec=constant(1),
            seed=scenario.base_seed,
            latency_s=scenario.latency_s,
            safety_margin=scenario.safety_margin,
            enabled=True,
            settlement_enabled=True,
        )
        super().__init__(bootstrap)

    @staticmethod
    def _step_and_row(request: FenicCompletionsRequest) -> tuple[str, int]:
        user = request.messages.user or ""
        match = _MARKER.search(user)
        if match is None:
            raise ValueError(f"workload marker missing from simulated request: {user!r}")
        return match.group(1), int(match.group(2))

    def estimate_tokens_for_request(self, request: FenicCompletionsRequest):
        """Use each step's input/max-output contract with the shared estimator."""
        step_id, row_id = self._step_and_row(request)
        step = self.steps[step_id]
        output = self._adaptive_output_reservation(
            request, static_ceiling=step.max_output_tokens, reasoning=False
        )
        self.trace.append(("estimate", time.time(), step_id, row_id, output))
        from fenic._inference.rate_limit_strategy import TokenEstimate

        return TokenEstimate(input_tokens=step.input_tokens, output_tokens=output)

    def _get_max_output_token_request_limit(self, request: FenicCompletionsRequest) -> int:
        step_id, _ = self._step_and_row(request)
        return self.steps[step_id].max_output_tokens

    async def make_single_request(self, request: FenicCompletionsRequest):
        """Serve deterministic schema-valid results or a real retryable 429."""
        step_id, row_id = self._step_and_row(request)
        step = self.steps[step_id]
        actual_out = self.draws[step_id][row_id]
        total = step.input_tokens + actual_out
        self.trace.append(("dispatch", time.time(), step_id, row_id))

        import asyncio

        await asyncio.sleep(self.workload_scenario.latency_s)
        if not self.server.try_consume(time.time(), total):
            self.trace.append(("server_429", time.time(), step_id, row_id))
            return TransientException(Exception("simulated 429"))

        self._metrics.num_output_tokens += actual_out
        self._metrics.num_uncached_input_tokens += step.input_tokens
        self._metrics.num_requests += 1
        self.trace.append(("success", time.time(), step_id, row_id, actual_out))
        if step_id == "overlay-map":
            completion = f"workload-step-overlay-extract row={row_id} mapped"
        elif request.structured_output is not None:
            completion = json.dumps({"label": f"{step_id}-row-{row_id}"})
        else:
            completion = f"workload-step-{step_id} row={row_id} mapped"
        return FenicCompletionsResponse(
            completion=completion,
            logprobs=None,
            usage=ResponseUsage(
                prompt_tokens=step.input_tokens,
                completion_tokens=actual_out,
                total_tokens=total,
                thinking_tokens=0,
            ),
        )


@dataclass(frozen=True)
class WorkloadReport:
    """Comparable outcome of one arm, including the load-bearing parity fields."""

    arm: Arm
    harness_version: str
    scenario: dict[str, object]
    raw_step_output_draws: dict[str, tuple[int, ...]]
    wall_s: float
    logical_rows_per_s: float
    row_ids: tuple[int, ...]
    final_structs: tuple[dict[str, object], ...]
    per_step_logical_totals: dict[str, int]
    per_step_actual_output_tokens: dict[str, int]
    logical_completion_total: int
    expected_logical_completion_total: int
    server_429: int
    retries: int
    backoffs: int
    total_attempts: int
    reserved_output_tokens: int
    actual_output_tokens: int
    reservation_efficiency: float
    achieved_output_tpm: float
    lifecycle_event_count: int
    idle_gap_count: int
    total_idle_gap_ns: int
    total_non_rate_limited_idle_gap_ns: int
    p50_idle_gap_ns: int | None
    p95_idle_gap_ns: int | None
    total_queue_delay_ns: int
    total_rate_limited_ns: int
    used_fusion: bool
    per_step_receipts: dict[str, dict[str, int | float]]
    lm_metrics: dict[str, int | float]
    lifecycle_events: tuple[dict[str, object], ...]
    trace: tuple[tuple, ...]

    def receipt(self) -> dict[str, object]:
        """Serialize the report without hiding raw parity or scenario fields."""
        return asdict(self)


@contextmanager
def _installed_simulator(session, scenario: WorkloadScenario) -> Iterator[WorkloadSimulatedCompletionsClient]:
    """Replace the fixture client before any semantic action, then restore it."""
    model = session._session_state.get_language_model()
    original_client = model.client
    simulated = WorkloadSimulatedCompletionsClient(scenario)
    model.client = simulated
    try:
        yield simulated
    finally:
        model.client = original_client
        simulated.shutdown()


@contextmanager
def _fusion_disabled(disabled: bool) -> Iterator[None]:
    """Test-only converter seam for the unbarriered-but-unfused comparator."""
    if not disabled:
        yield
        return
    original = PlanConverter._try_convert_fused_map_extract
    PlanConverter._try_convert_fused_map_extract = lambda self, logical, cache_keys: None
    try:
        yield
    finally:
        PlanConverter._try_convert_fused_map_extract = original


def _step_expr(step: WorkloadStepSpec):
    """Build one real semantic expression carrying a parseable simulator marker."""
    marker = lit(f"workload-step-{step.step_id}")
    if step.operation == "map":
        return semantic.map(
            "{{ marker }} row={{ record_id }} input={{ payload }}",
            marker=marker,
            record_id=col("record_id"),
            payload=col("payload"),
        )
    marked_input = text.concat(
        marker,
        lit(" row="),
        col("record_id").cast(StringType),
        lit(" input="),
        col("payload"),
    )
    return semantic.extract(marked_input, _StepOutput)


def _overlay(base, *, n_rows: int, barriered: bool, action_metrics: list[LMMetrics]):
    """Build the explicitly non-production B1-eligible map-to-extract overlay."""
    mapped = base.select(
        "record_id",
        "payload",
        semantic.map(
            "{{ marker }} row={{ record_id }} input={{ payload }}",
            marker=lit("workload-step-overlay-map"),
            record_id=col("record_id"),
            payload=col("payload"),
        ).alias("overlay_map"),
    )
    if barriered:
        mapped = mapped.cache()
        mapped_result = mapped.collect("polars")
        action_metrics.append(mapped_result.metrics.total_lm_metrics)
        assert mapped_result.data.height == n_rows

    extracted = mapped.select(
        "record_id",
        semantic.extract(col("overlay_map"), _StepOutput).alias("overlay_summary"),
    )
    if barriered:
        extracted = extracted.cache()
        extracted_result = extracted.collect("polars")
        action_metrics.append(extracted_result.metrics.total_lm_metrics)
        assert extracted_result.data.height == n_rows
    return extracted


def _run_steps(session, scenario: WorkloadScenario, arm: Arm, action_metrics: list[LMMetrics]):
    """Build the pass-shaped DataFrame and apply cache/count only to baseline steps."""
    base = session.create_dataframe(
        {
            "record_id": list(range(scenario.n_rows)),
            "payload": [f"source-row-{row_id}" for row_id in range(scenario.n_rows)],
        }
    )
    if arm == "barriered":
        current = base
        for step in scenario.steps[:12]:
            current = current.with_column(f"step_{step.step_id}", _step_expr(step))
            current = current.cache()
            current_result = current.collect("polars")
            action_metrics.append(current_result.metrics.total_lm_metrics)
            assert current_result.data.height == scenario.n_rows
    else:
        # A single projection keeps independent pass-shaped operations from being
        # recalculated by nested lazy projections. The barriered arm above
        # materializes after each step to model typedef's current executor.
        current = base.select(
            "record_id",
            "payload",
            *[
                _step_expr(step).alias(f"step_{step.step_id}")
                for step in scenario.steps[:12]
            ],
        )

    # The overlay is deliberately separate from the typedef-shaped twelve
    # passes: it is the B1-eligible map -> extract chain under comparison.
    overlay = _overlay(
        base,
        n_rows=scenario.n_rows,
        barriered=arm == "barriered",
        action_metrics=action_metrics,
    )
    return current.join(overlay, on="record_id").select("record_id", "overlay_summary")


def run_workload_arm(session, scenario: WorkloadScenario, arm: Arm) -> WorkloadReport:
    """Execute one arm with real Fenic operators and return a comparable receipt."""
    events = []
    action_metrics: list[LMMetrics] = []
    used_fusion = False
    started = time.monotonic()
    with _installed_simulator(session, scenario) as client:
        client.set_request_lifecycle_collector(events.append, execution_id=f"{scenario.version}-{arm}")
        result = _run_steps(session, scenario, arm, action_metrics)
        original_execute = FusedMapExtractExec.execute_node

        def observe_fusion(self, child_dfs):
            nonlocal used_fusion
            used_fusion = True
            return original_execute(self, child_dfs)

        FusedMapExtractExec.execute_node = observe_fusion
        try:
            with _fusion_disabled(arm == "unbarriered_unfused"):
                output_result = result.collect("polars")
                action_metrics.append(output_result.metrics.total_lm_metrics)
                output = output_result.data.sort("record_id")
        finally:
            FusedMapExtractExec.execute_node = original_execute
            client.set_request_lifecycle_collector(None)
        wall_s = time.monotonic() - started
        successes = [event for event in client.trace if event[0] == "success"]
        per_step_logical = {step.step_id: 0 for step in scenario.steps}
        per_step_tokens = {step.step_id: 0 for step in scenario.steps}
        for _, _, step_id, _, actual_out in successes:
            per_step_logical[step_id] += 1
            per_step_tokens[step_id] += actual_out
        metrics = sum(action_metrics, LMMetrics())
        idle = compute_idle_gap_metrics(events)
        attempts = sum(1 for event in client.trace if event[0] == "dispatch")
        server_429 = sum(1 for event in client.trace if event[0] == "server_429")
        backoffs = sum(1 for event in client.trace if event[0] == "backoff")
        per_step_receipts = {}
        for step in scenario.steps:
            step_events = [event for event in client.trace if len(event) > 2 and event[2] == step.step_id]
            step_dispatches = sum(event[0] == "dispatch" for event in step_events)
            step_429s = sum(event[0] == "server_429" for event in step_events)
            estimate_values = [event[4] for event in step_events if event[0] == "estimate"]
            success_times = [event[1] for event in step_events if event[0] == "success"]
            dispatch_times = [event[1] for event in step_events if event[0] == "dispatch"]
            per_step_receipts[step.step_id] = {
                "logical_completions": per_step_logical[step.step_id],
                "actual_output_tokens": per_step_tokens[step.step_id],
                "dispatches": step_dispatches,
                "server_429s": step_429s,
                "reservation_calls": len(estimate_values),
                "reserved_output_tokens": sum(estimate_values),
                "min_reserved_output_tokens": min(estimate_values, default=0),
                "max_reserved_output_tokens": max(estimate_values, default=0),
                "active_wall_s": (
                    max(success_times) - min(dispatch_times)
                    if success_times and dispatch_times
                    else 0.0
                ),
            }

    return WorkloadReport(
        arm=arm,
        harness_version=HARNESS_VERSION,
        scenario=scenario.receipt_config(),
        raw_step_output_draws={key: tuple(values) for key, values in client.draws.items()},
        wall_s=wall_s,
        logical_rows_per_s=scenario.n_rows / wall_s if wall_s else 0.0,
        row_ids=tuple(output["record_id"].to_list()),
        final_structs=tuple(output["overlay_summary"].to_list()),
        per_step_logical_totals=per_step_logical,
        per_step_actual_output_tokens=per_step_tokens,
        logical_completion_total=metrics.num_requests,
        expected_logical_completion_total=scenario.n_rows * len(scenario.steps),
        server_429=server_429,
        retries=attempts - metrics.num_requests,
        backoffs=backoffs,
        total_attempts=attempts,
        reserved_output_tokens=metrics.num_reserved_output_tokens,
        actual_output_tokens=metrics.num_output_tokens,
        reservation_efficiency=metrics.num_output_tokens / max(1, metrics.num_reserved_output_tokens),
        achieved_output_tpm=60 * metrics.num_output_tokens / wall_s if wall_s else 0.0,
        lifecycle_event_count=len(events),
        idle_gap_count=idle.idle_gap_count,
        total_idle_gap_ns=idle.total_idle_gap_ns,
        total_non_rate_limited_idle_gap_ns=idle.total_non_rate_limited_idle_gap_ns,
        p50_idle_gap_ns=idle.p50_idle_gap_ns,
        p95_idle_gap_ns=idle.p95_idle_gap_ns,
        total_queue_delay_ns=idle.total_queue_delay_ns,
        total_rate_limited_ns=idle.total_rate_limited_ns,
        used_fusion=used_fusion,
        per_step_receipts=per_step_receipts,
        lm_metrics=asdict(metrics),
        lifecycle_events=tuple(asdict(event) for event in events),
        trace=tuple(client.trace),
    )


def assert_arm_parity(reports: list[WorkloadReport]) -> None:
    """Raise loudly if any arm changes semantic outputs, totals, or token draws."""
    if len(reports) != 3:
        raise ValueError("arm parity requires exactly barriered, unfused, and fused reports")
    baseline = reports[0]
    for report in reports[1:]:
        differing_fields = [
            name
            for name, observed, expected in (
                ("row_ids", report.row_ids, baseline.row_ids),
                ("final_structs", report.final_structs, baseline.final_structs),
                (
                    "per_step_logical_totals",
                    report.per_step_logical_totals,
                    baseline.per_step_logical_totals,
                ),
                (
                    "per_step_actual_output_tokens",
                    report.per_step_actual_output_tokens,
                    baseline.per_step_actual_output_tokens,
                ),
            )
            if observed != expected
        ]
        if differing_fields:
            raise AssertionError(
                "workload arm parity failed: "
                f"baseline={baseline.arm}, differing={report.arm}, fields={differing_fields}"
            )


def run_matrix(session, output_dir: Path) -> list[dict[str, object]]:
    """Run the explicit full matrix and persist one versioned receipt per scenario."""
    output_dir.mkdir(parents=True, exist_ok=True)
    receipts = []
    for n_rows in (96, 192):
        for base_seed in (101, 202, 303):
            for lane, true_tpm in (("matching", 150_000), ("modest_overshoot", 135_000)):
                scenario = WorkloadScenario(
                    n_rows=n_rows,
                    base_seed=base_seed,
                    true_tpm=true_tpm,
                )
                reports = [
                    run_workload_arm(session, scenario, arm)
                    for arm in ("barriered", "unbarriered_unfused", "unbarriered_fused")
                ]
                assert_arm_parity(reports)
                receipt = {
                    "scenario": scenario.receipt_config(),
                    "lane": lane,
                    "arm_parity": True,
                    "reports": [report.receipt() for report in reports],
                }
                path = output_dir / f"{scenario.version}-{lane}-{n_rows}-seed{base_seed}.json"
                path.write_text(json.dumps(receipt, indent=2) + "\n")
                receipts.append(receipt)
    return receipts


def run_pilot(session, output_dir: Path) -> dict[str, object]:
    """Run and persist the required 24-row matching-server pilot before the matrix."""
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario = WorkloadScenario(n_rows=24, base_seed=101)
    reports = [
        run_workload_arm(session, scenario, arm)
        for arm in ("barriered", "unbarriered_unfused", "unbarriered_fused")
    ]
    assert_arm_parity(reports)
    receipt = {
        "harness_version": HARNESS_VERSION,
        "scenario": scenario.receipt_config(),
        "lane": "matching-pilot",
        "arm_parity": True,
        "reports": [report.receipt() for report in reports],
    }
    path = output_dir / f"{scenario.version}-matching-pilot-{scenario.n_rows}-seed{scenario.base_seed}.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
