#!/usr/bin/env python3
"""Run the Captain-authorized, bounded live-provider execution validation.

This is an internal evidence harness, not a product command. It deliberately
prints only JSON measurements and never reads or prints credential values.
"""

from __future__ import annotations

import dataclasses
import json
import resource
import sys
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import polars as pl
from pydantic import BaseModel, Field

from fenic import OpenAILanguageModel, SemanticConfig, Session, SessionConfig, col, semantic
from fenic._backends.local.physical_plan import FusedMapExtractExec, ProjectionExec
from fenic._backends.local.transpiler.plan_converter import PlanConverter
from fenic._inference.request_lifecycle import compute_idle_gap_metrics


MODEL = "gpt-4.1-nano"
RPM = 250
TPM = 50_000
FUSION_SIZES = (64, 160, 320)
PRIOR_RUN_COST_UPPER_USD = 0.041574
HARD_PROJECTED_TOTAL_USD = 40.0
EVIDENCE_ROOT = Path(__file__).resolve().parent / "amendment-a-evidence"


class Signal(BaseModel):
    category: str = Field(
        description=(
            "Return exactly one of ALPHA, BETA, GAMMA, or DELTA. "
            "Use uppercase with no whitespace, punctuation, explanation, or other value."
        )
    )


def peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def new_session(tmpdir: str, label: str) -> Session:
    return Session.get_or_create(
        SessionConfig(
            app_name=f"fenic-exec-engine-validation-{label}-{time.time_ns()}",
            db_path=Path(tmpdir),
            semantic=SemanticConfig(
                language_models={
                    "validation": OpenAILanguageModel(
                        model_name=MODEL,
                        rpm=RPM,
                        tpm=TPM,
                    )
                },
                default_language_model="validation",
            ),
        )
    )


def metric_dict(metrics: Any) -> dict[str, Any]:
    return {
        "uncached_input_tokens": metrics.num_uncached_input_tokens,
        "cached_input_tokens": metrics.num_cached_input_tokens,
        "output_tokens": metrics.num_output_tokens,
        "reserved_output_tokens": metrics.num_reserved_output_tokens,
        "requests": metrics.num_requests,
        "cost_usd": round(metrics.cost, 9),
    }


def lifecycle_dict(events: list[Any]) -> dict[str, Any]:
    metrics = compute_idle_gap_metrics(events)
    return {
        "event_counts": dict(sorted(Counter(event.event for event in events).items())),
        "operations": dict(sorted(Counter(event.operation_name for event in events).items())),
        "idle": dataclasses.asdict(metrics),
    }


def raw_lifecycle_events(events: list[Any]) -> list[dict[str, Any]]:
    return [dataclasses.asdict(event) for event in events]


def normalize_category(value: object) -> str | None:
    if value is None:
        return None
    return str(value).strip().upper()


def write_evidence(name: str, evidence: dict[str, Any]) -> str:
    """Durably write synthetic run evidence before parity can raise."""
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    destination = EVIDENCE_ROOT / f"{name}.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)
    return str(destination.relative_to(Path.cwd()))


def fusion_arm_estimate_usd(rows: int) -> float:
    # Two operations per row; use the observed 512-token reservation, not the
    # original 384-token planning estimate, for Amendment A's remaining reserve.
    return rows * 2 * ((1_200 * 0.100 + 512 * 0.400) / 1_000_000)


def join_estimate_usd() -> float:
    return 16 * 16 * ((1_000 * 0.100 + 128 * 0.400) / 1_000_000)


def enforce_projected_budget(completed: list[dict[str, Any]], remaining_usd: float) -> None:
    actual_this_round = sum(item["metrics"]["cost_usd"] for item in completed)
    projected_total = PRIOR_RUN_COST_UPPER_USD + actual_this_round + 10 * remaining_usd
    if projected_total > HARD_PROJECTED_TOTAL_USD:
        raise RuntimeError(
            "validation projected total exceeds the $40 halt line: "
            f"${projected_total:.6f}"
        )


def source_frame(session: Session, rows: int):
    categories = ("ALPHA", "BETA", "GAMMA", "DELTA")
    expected = {index: categories[index % len(categories)] for index in range(rows)}
    source = session.create_dataframe(
        pl.DataFrame(
            {
                "record_id": pl.Series(range(rows), dtype=pl.Int64),
                "description": pl.Series(
                    [
                        (
                            f"Synthetic operational note {index}. CATEGORY: "
                            f"{expected[index]}. This deliberately short benchmark record "
                            "contains no customer, production, or secret data."
                        )
                        for index in range(rows)
                    ],
                    dtype=pl.String,
                ),
            }
        )
    )
    return source, expected


def map_extract_query(session: Session, rows: int, arm: Literal["fused", "unfused"]):
    source, expected = source_frame(session, rows)
    mapped = source.select(
        col("record_id"),
        semantic.map(
            "Return exactly the uppercase CATEGORY token ALPHA, BETA, GAMMA, or DELTA "
            "explicitly stated in this record. Return no other text: {{description}}",
            description=col("description"),
        ).alias("normalized"),
    )
    extracted = semantic.extract(col("normalized"), Signal).alias("signal")
    if arm == "fused":
        query = mapped.select(col("record_id"), extracted)
    else:
        # Keeping the mapped value visible makes B1's strict matcher decline fusion.
        query = (
            mapped.select(col("record_id"), col("normalized"), extracted)
            .select(col("record_id"), col("signal"))
        )
    physical = PlanConverter(session._session_state).convert(query._logical_plan)
    if arm == "fused" and not isinstance(physical, FusedMapExtractExec):
        raise AssertionError(f"expected fused physical plan, got {type(physical).__name__}")
    if arm == "unfused" and not isinstance(physical, ProjectionExec):
        raise AssertionError(f"expected unfused projection physical plan, got {type(physical).__name__}")
    return query, expected, type(physical).__name__


def run_fusion_arm(rows: int, arm: Literal["fused", "unfused"]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmpdir:
        session = new_session(tmpdir, f"{arm}-{rows}")
        try:
            query, expected, physical_plan = map_extract_query(session, rows, arm)
            language_model = session._session_state.get_language_model()
            events: list[Any] = []
            event_lock = threading.Lock()

            def collect(event: Any) -> None:
                with event_lock:
                    events.append(event)

            language_model.client.set_request_lifecycle_collector(
                collect, execution_id=f"validation-{arm}-{rows}"
            )
            started = time.perf_counter()
            actual = query.to_polars()
            wall_ms = round((time.perf_counter() - started) * 1000, 3)
            observed_raw = {
                int(record_id): signal["category"]
                for record_id, signal in actual.select("record_id", "signal").iter_rows()
            }
            observed_normalized = {
                record_id: normalize_category(value)
                for record_id, value in observed_raw.items()
            }
            mismatches = {
                str(record_id): {
                    "expected": expected[record_id],
                    "actual": observed_raw.get(record_id),
                    "normalized_actual": observed_normalized.get(record_id),
                }
                for record_id in expected
                if observed_normalized.get(record_id) != expected[record_id]
            }
            metrics = metric_dict(language_model.get_metrics())
            lifecycle = lifecycle_dict(events)
            evidence_path = write_evidence(
                f"{arm}-{rows}",
                {
                    "kind": "fusion",
                    "arm": arm,
                    "rows": rows,
                    "physical_plan": physical_plan,
                    "wall_ms": wall_ms,
                    "result_rows": actual.height,
                    "expected": expected,
                    "observed": observed_raw,
                    "observed_normalized": observed_normalized,
                    "mismatches": mismatches,
                    "metrics": metrics,
                    "lifecycle": lifecycle,
                    "lifecycle_events": raw_lifecycle_events(events),
                },
            )
            if mismatches:
                raise AssertionError(
                    f"semantic divergence in {arm}/{rows}: {len(mismatches)} normalized mismatches; "
                    f"evidence: {evidence_path}"
                )
            return {
                "kind": "fusion",
                "arm": arm,
                "rows": rows,
                "physical_plan": physical_plan,
                "wall_ms": wall_ms,
                "result_rows": actual.height,
                "semantic_parity": "pass",
                "evidence_path": evidence_path,
                "metrics": metrics,
                "lifecycle": lifecycle,
            }
        finally:
            session.stop(skip_usage_summary=True)


def run_join() -> dict[str, Any]:
    categories = ("ALPHA", "BETA", "GAMMA", "DELTA")
    left_rows = 16
    right_rows = 16
    expected_pairs = {
        (left_id, right_id)
        for left_id in range(left_rows)
        for right_id in range(right_rows)
        if left_id % len(categories) == right_id % len(categories)
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        session = new_session(tmpdir, "join-16x16")
        try:
            left = session.create_dataframe(
                {
                    "left_id": list(range(left_rows)),
                    "left_category": [categories[index % len(categories)] for index in range(left_rows)],
                }
            )
            right = session.create_dataframe(
                {
                    "right_id": list(range(right_rows)),
                    "right_category": [categories[index % len(categories)] for index in range(right_rows)],
                }
            )
            query = left.semantic.join(
                right,
                (
                    "Return true exactly when the two literal category tokens are identical. "
                    "Do not infer semantic similarity. Left: {{left_on}}; right: {{right_on}}."
                ),
                left_on=col("left_category"),
                right_on=col("right_category"),
            )
            language_model = session._session_state.get_language_model()
            events: list[Any] = []
            event_lock = threading.Lock()

            def collect(event: Any) -> None:
                with event_lock:
                    events.append(event)

            language_model.client.set_request_lifecycle_collector(
                collect, execution_id="validation-join-16x16"
            )
            started = time.perf_counter()
            actual = query.to_polars()
            wall_ms = round((time.perf_counter() - started) * 1000, 3)
            observed_pairs = {
                (int(row[0]), int(row[1]))
                for row in actual.select("left_id", "right_id").iter_rows()
            }
            metrics = metric_dict(language_model.get_metrics())
            lifecycle = lifecycle_dict(events)
            evidence_path = write_evidence(
                "join-16x16",
                {
                    "kind": "bounded_join",
                    "shape": [left_rows, right_rows],
                    "expected_pairs": sorted(expected_pairs),
                    "observed_pairs": sorted(observed_pairs),
                    "metrics": metrics,
                    "lifecycle": lifecycle,
                    "lifecycle_events": raw_lifecycle_events(events),
                },
            )
            if observed_pairs != expected_pairs:
                raise AssertionError(
                    f"semantic divergence in join: expected {len(expected_pairs)} survivors, "
                    f"observed {len(observed_pairs)}; evidence: {evidence_path}"
                )
            return {
                "kind": "bounded_join",
                "shape": [left_rows, right_rows],
                "pair_count": left_rows * right_rows,
                "pair_block_cap": 1024,
                "wall_ms": wall_ms,
                "peak_rss_bytes": peak_rss_bytes(),
                "result_rows": actual.height,
                "semantic_parity": "pass",
                "evidence_path": evidence_path,
                "metrics": metrics,
                "lifecycle": lifecycle,
            }
        finally:
            session.stop(skip_usage_summary=True)


def main() -> None:
    results: list[dict[str, Any]] = []
    scheduled_arms: list[tuple[str, int | None, float]] = [
        (arm, rows, fusion_arm_estimate_usd(rows))
        for rows in FUSION_SIZES
        for arm in ("unfused", "fused")
    ]
    scheduled_arms.append(("join", None, join_estimate_usd()))
    for position, (arm, rows, _estimate) in enumerate(scheduled_arms):
        enforce_projected_budget(results, sum(item[2] for item in scheduled_arms[position:]))
        if arm == "join":
            results.append(run_join())
        else:
            assert rows is not None
            results.append(run_fusion_arm(rows, arm))
    total_cost = sum(item["metrics"]["cost_usd"] for item in results)
    print(
        json.dumps(
            {
                "model": MODEL,
                "governor": {"rpm": RPM, "tpm": TPM},
                "results": results,
                "total_actual_cost_usd": round(total_cost, 9),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
