#!/usr/bin/env python3
# ruff: noqa: D103
"""Execute one validated streaming benchmark cell in an isolated process."""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import tempfile
import threading
import time
import uuid
from collections import Counter
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, get_args

try:
    from .models import (
        Cell,
        as_jsonable,
        cell_estimated_cost,
        expand_cells,
        load_matrix,
        require_metrics,
        result_hash,
    )
except ImportError:
    from models import (  # type: ignore[no-redef]
        Cell,
        as_jsonable,
        cell_estimated_cost,
        expand_cells,
        load_matrix,
        require_metrics,
        result_hash,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-spec", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


def _metrics_dict(metrics: Any) -> dict[str, Any]:
    if is_dataclass(metrics):
        return asdict(metrics)
    if hasattr(metrics, "model_dump"):
        return metrics.model_dump()
    return {
        name: getattr(metrics, name)
        for name in dir(metrics)
        if not name.startswith("_")
        and isinstance(getattr(metrics, name), (int, float, str, bool, type(None)))
    }


def peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def validate_execution_spec(spec: dict[str, Any]) -> tuple[Any, Cell]:
    """Authenticate the parent plan and revalidate the bounded cell."""
    expected_token = os.environ.get("FENIC_BENCHMARK_RUN_TOKEN")
    if not expected_token or spec.get("run_token") != expected_token:
        raise RuntimeError("execution spec is not authorized by the benchmark runner")
    matrix = load_matrix(Path(spec["matrix_path"]))
    cell = Cell(**spec["cell"])
    run_state = json.loads(Path(spec["run_state_path"]).read_text())
    if run_state.get("run_id") != spec.get("run_id") or run_state.get(
        "plan_id"
    ) != spec.get("plan_id"):
        raise RuntimeError("execution spec does not match its durable run ledger")
    reservation = run_state.get("active_reservation") or {}
    if reservation.get("cell_id") != cell.id:
        raise RuntimeError("execution cell has no active durable reservation")
    candidates = expand_cells(matrix, checkout=cell.checkout)
    if cell not in candidates:
        raise RuntimeError("execution cell is not derived from the validated matrix")
    estimate = float(spec["estimated_cost_usd"])
    if abs(estimate - float(reservation.get("estimated_cost_usd", -1))) > 1e-12:
        raise RuntimeError("execution cost does not match its durable reservation")
    if abs(estimate - cell_estimated_cost(matrix, cell)) > 1e-12:
        raise RuntimeError("execution cost is not derived from the validated matrix")
    if estimate < 0 or estimate > float(spec["cell_cost_cap_usd"]):
        raise RuntimeError("execution cell exceeds its reserved cost cap")
    if cell.execution_mode == "provider" and not spec.get("approve_provider_spend"):
        raise RuntimeError("provider execution lacks explicit spend acknowledgement")
    return matrix, cell


def _lifecycle_collector(
    client: Any, execution_id: str
) -> tuple[list[Any], dict[str, bool]]:
    """Attach lifecycle collection and report each measurement independently."""
    try:
        from fenic._inference.request_lifecycle import (
            RequestLifecycleEventType,
            compute_idle_gap_metrics,
        )
    except ImportError:
        return [], {
            name: False
            for name in (
                "event_counts",
                "idle_gap",
                "max_queue_depth",
                "rate_limit_events",
            )
        }
    event_types = set(get_args(RequestLifecycleEventType))
    collector = getattr(client, "set_request_lifecycle_collector", None)
    events: list[Any] = []
    if not callable(collector):
        return events, {
            name: False
            for name in (
                "event_counts",
                "idle_gap",
                "max_queue_depth",
                "rate_limit_events",
            )
        }
    lock = threading.Lock()

    def collect(event: Any) -> None:
        with lock:
            events.append(event)

    collector(collect, execution_id=execution_id)
    availability = {
        "event_counts": bool(event_types),
        "idle_gap": callable(compute_idle_gap_metrics)
        and {"queued", "settled"}.issubset(event_types),
        "max_queue_depth": {"queued", "dispatched"}.issubset(event_types),
        "rate_limit_events": "rate_limited" in event_types,
    }
    return events, availability


def _lifecycle_summary(
    events: list[Any], availability: dict[str, bool]
) -> dict[str, Any]:
    counts = (
        Counter(event.event for event in events)
        if availability["event_counts"]
        else None
    )
    maximum = None
    if availability["max_queue_depth"]:
        depth = 0
        maximum = 0
        for event in sorted(events, key=lambda item: item.timestamp_ns):
            if event.event == "queued":
                depth += 1
                maximum = max(maximum, depth)
            elif event.event == "dispatched":
                depth = max(0, depth - 1)
    idle_gap = None
    if availability["idle_gap"]:
        from fenic._inference.request_lifecycle import compute_idle_gap_metrics

        idle_gap = asdict(compute_idle_gap_metrics(events))
    return {
        "availability": {
            name: {
                "available": available,
                **(
                    {}
                    if available
                    else {"reason": "measurement is not exposed by this checkout"}
                ),
            }
            for name, available in availability.items()
        },
        "event_counts": dict(sorted(counts.items())) if counts is not None else None,
        "max_queue_depth": maximum,
        "idle_gap": idle_gap,
        "rate_limit_events": counts.get("rate_limited", 0)
        if counts is not None and availability["rate_limit_events"]
        else None,
    }


def run_simulated_cell(cell: Cell) -> dict[str, Any]:
    """Exercise today's bounded join path with the real scheduler and no provider."""
    from benchmarks.semantic_join_stream_adapter import Workload, run_arm
    from fenic._inference.model_client import ModelClient

    workload = Workload(
        left_rows=cell.rows,
        right_rows=cell.right_rows,
        pair_block_size=max(1, min(256, cell.physical_requests)),
        block_token_budget=14_000,
        rpm=100,
        batch_size=cell.batch_size,
        repetitions=1,
        latency_seconds=0.001,
    )
    calls = {"list": 0, "iterator": 0}
    original_list = ModelClient.make_batch_requests
    original_iterator = ModelClient.iter_batch_requests

    def counted_list(client: Any, *args: Any, **kwargs: Any) -> Any:
        calls["list"] += 1
        return original_list(client, *args, **kwargs)

    def counted_iterator(client: Any, *args: Any, **kwargs: Any) -> Any:
        calls["iterator"] += 1
        return original_iterator(client, *args, **kwargs)

    ModelClient.make_batch_requests = counted_list
    ModelClient.iter_batch_requests = counted_iterator
    try:
        started = time.monotonic_ns()
        raw = run_arm(workload, cell.arm == "streaming", cell.repetition)
        wall_ms = (time.monotonic_ns() - started) / 1_000_000
    finally:
        ModelClient.make_batch_requests = original_list
        ModelClient.iter_batch_requests = original_iterator
    lifecycle_counts = raw["lifecycle_counts"]
    rate_available = "queued" in lifecycle_counts and "settled" in lifecycle_counts
    return {
        "wall_clock_ms": wall_ms,
        "result_hash": result_hash([cell.expected_result_count, raw["output_tokens"]]),
        "result_count": raw["result_rows"],
        "lm_metrics": {
            "cost": 0.0,
            "num_requests": raw["request_count"],
            "num_output_tokens": raw["output_tokens"],
        },
        "lifecycle": {
            "availability": {
                "event_counts": {"available": True},
                "idle_gap": {
                    "available": False,
                    "reason": "no idle-gap collector is exposed by the simulator receipt",
                },
                "max_queue_depth": {"available": True},
                "rate_limit_events": {"available": rate_available},
            },
            "event_counts": lifecycle_counts,
            "idle_gap": None,
            "max_queue_depth": raw["max_live_requests"],
            "rate_limit_events": raw["simulated_429"] if rate_available else None,
        },
        "path_evidence": {
            "streaming_enabled": cell.arm == "streaming",
            "list_calls": calls["list"],
            "iterator_calls": calls["iterator"],
            "max_live_requests": raw["max_live_requests"],
            "configured_watermark": workload.watermark,
            "dispatch_count": raw["dispatch_count"],
        },
        "provider_execution": False,
    }


def run_provider_cell(matrix: Any, cell: Cell, work_dir: Path) -> dict[str, Any]:
    """Run one semantic.join arm against the configured real model."""
    import polars as pl

    import fenic as fc
    from fenic._backends.local.semantic_operators.base import BaseOperator
    from fenic._backends.local.semantic_operators.join import Join
    from fenic._inference.language_model import LanguageModel

    previous_stream = Join.stream_requests
    previous_batch = BaseOperator.request_batch_size
    session = None
    client = None
    calls = {"list": 0, "iterator": 0}
    try:
        with tempfile.TemporaryDirectory(
            prefix="streaming-provider-", dir=work_dir
        ) as tmpdir:
            session = fc.Session.get_or_create(
                fc.SessionConfig(
                    app_name=f"streaming-{uuid.uuid4().hex}",
                    db_path=Path(tmpdir),
                    semantic=fc.SemanticConfig(
                        language_models={
                            matrix.model_alias: fc.OpenAILanguageModel(
                                model_name=matrix.model_name,
                                rpm=matrix.limits.client_rpm,
                                tpm=matrix.limits.client_tpm,
                            )
                        },
                        default_language_model=matrix.model_alias,
                    ),
                )
            )
            model = session._session_state.get_language_model()
            client = model.client
            original_list = client.make_batch_requests
            original_iterator = client.iter_batch_requests

            def counted_list(*args: Any, **kwargs: Any) -> Any:
                calls["list"] += 1
                return original_list(*args, **kwargs)

            def counted_iterator(*args: Any, **kwargs: Any) -> Any:
                calls["iterator"] += 1
                return original_iterator(*args, **kwargs)

            client.make_batch_requests = counted_list
            client.iter_batch_requests = counted_iterator
            events, availability = _lifecycle_collector(client, cell.id)
            left = pl.DataFrame({"left_on": [f"left-{i}" for i in range(cell.rows)]})
            right = pl.DataFrame(
                {"right_on": [f"right-{i}" for i in range(cell.right_rows)]}
            )
            Join.stream_requests = cell.arm == "streaming"
            BaseOperator.request_batch_size = cell.batch_size
            join = Join(
                left, right, cell.prompt_template, True, LanguageModel(client), 0
            )
            started = time.monotonic_ns()
            result = join.execute()
            wall_ms = (time.monotonic_ns() - started) / 1_000_000
            metrics = _metrics_dict(client.get_metrics())
            require_metrics(metrics)
            lifecycle = _lifecycle_summary(events, availability)
            return {
                "wall_clock_ms": wall_ms,
                "result_hash": result_hash(result.to_dicts()),
                "result_count": len(result),
                "lm_metrics": metrics,
                "lifecycle": lifecycle,
                "path_evidence": {
                    "streaming_enabled": cell.arm == "streaming",
                    "list_calls": calls["list"],
                    "iterator_calls": calls["iterator"],
                },
                "provider_execution": True,
                "fenic_source": str(Path(fc.__file__).resolve()),
            }
    finally:
        Join.stream_requests = previous_stream
        BaseOperator.request_batch_size = previous_batch
        if client is not None:
            client.set_request_lifecycle_collector(None)
        if session is not None:
            session.stop(skip_usage_summary=True)


def execute(spec: dict[str, Any]) -> dict[str, Any]:
    matrix, cell = validate_execution_spec(spec)
    work_dir = Path(spec["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    payload = (
        run_simulated_cell(cell)
        if cell.execution_mode == "simulated"
        else run_provider_cell(matrix, cell, work_dir)
    )
    payload.update(
        {
            "cell_id": cell.id,
            "checkout": cell.checkout,
            "scenario_id": cell.scenario_id,
            "arm": cell.arm,
            "rows": cell.rows,
            "right_rows": cell.right_rows,
            "unique_inputs": cell.unique_inputs,
            "batch_size": cell.batch_size,
            "repetition": cell.repetition,
            "expected_result_count": cell.expected_result_count,
            "physical_requests": cell.physical_requests,
            "peak_rss_bytes": peak_rss_bytes(),
        }
    )
    return payload


def main() -> None:
    args = parse_args()
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    failure: dict[str, Any] | None = None
    try:
        payload = execute(json.loads(args.execution_spec.read_text()))
        args.receipt.write_text(
            json.dumps(as_jsonable(payload), indent=2, sort_keys=True) + "\n"
        )
    except BaseException as exc:
        failure = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        args.receipt.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        raise


if __name__ == "__main__":
    main()
