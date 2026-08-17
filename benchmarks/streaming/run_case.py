#!/usr/bin/env python3
# ruff: noqa: D103
"""Execute one benchmark cell in an isolated process.

The import of fenic is intentionally inside ``run_provider_cell``. Planning,
schema validation, and all provider-disabled paths therefore remain safe to
run on a machine without credentials or provider extras.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

try:
    from .models import (
        Cell,
        deterministic_inputs,
        lifecycle_unavailable,
        require_metrics,
        result_hash,
    )
except ImportError:  # Executed directly rather than as a package.
    from models import (  # type: ignore[no-redef]
        Cell,
        deterministic_inputs,
        lifecycle_unavailable,
        require_metrics,
        result_hash,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-json", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--client-rpm", type=int, required=True)
    parser.add_argument("--client-tpm", type=int, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-provider",
        action="store_true",
        help="explicit internal provider-call acknowledgement",
    )
    return parser.parse_args()


def _cell(raw: dict[str, Any]) -> Cell:
    return Cell(**raw)


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


def _lifecycle_collector(model: Any, execution_id: str) -> tuple[list[Any] | None, Any]:
    """Attach the optional lifecycle collector without inventing measurements."""
    try:
        from fenic._inference.request_lifecycle import compute_idle_gap_metrics
    except ImportError:
        return None, None
    if not hasattr(model.client, "set_request_lifecycle_collector"):
        return None, None

    events: list[Any] = []
    lock = threading.Lock()

    def collect(event: Any) -> None:
        with lock:
            events.append(event)

    model.client.set_request_lifecycle_collector(collect, execution_id=execution_id)
    return events, (lock, compute_idle_gap_metrics)


def _lifecycle_summary(events: list[Any] | None, state: Any) -> dict[str, Any]:
    if events is None:
        return lifecycle_unavailable()
    lock, compute_idle_gap_metrics = state
    with lock:
        captured = list(events)
    counts: dict[str, int] = {}
    queue_depth = 0
    max_queue_depth = 0
    for event in sorted(captured, key=lambda item: item.timestamp_ns):
        counts[event.event] = counts.get(event.event, 0) + 1
        if event.event == "queued":
            queue_depth += 1
            max_queue_depth = max(max_queue_depth, queue_depth)
        elif event.event == "dispatched":
            queue_depth = max(0, queue_depth - 1)
    return {
        "availability": {
            "event_counts": {"available": True},
            "idle_gap": {"available": True},
            "max_queue_depth": {"available": True},
            "rate_limit_events": {"available": True},
        },
        "event_counts": counts,
        "max_queue_depth": max_queue_depth,
        "idle_gap": asdict(compute_idle_gap_metrics(captured)),
        "rate_limit_events": counts.get("rate_limited", 0),
    }


def run_provider_cell(cell: Cell, args: argparse.Namespace) -> dict[str, Any]:
    if cell.scenario_kind != "operator":
        raise RuntimeError(
            "chain scenarios are reserved and provider-disabled in this landing"
        )
    # Importing these modules is itself part of the provider-backed execution
    # path. No import occurs for plan, validation, or disabled cells.
    import fenic as fc
    from fenic._backends.local.semantic_operators.map import Map
    from fenic._backends.local.semantic_operators.predicate import Predicate

    operator_class = Map if cell.operation == "map" else Predicate
    old_stream_requests = operator_class.stream_requests
    old_request_batch_size = operator_class.request_batch_size
    operator_class.stream_requests = cell.arm == "streaming"
    operator_class.request_batch_size = cell.batch_size
    args.work_dir.mkdir(parents=True, exist_ok=True)
    started_ns = 0
    try:
        with tempfile.TemporaryDirectory(
            prefix=f"streaming-{cell.id}-", dir=args.work_dir
        ) as tmpdir:
            session = fc.Session.get_or_create(
                fc.SessionConfig(
                    app_name=f"streaming-{uuid.uuid4().hex}",
                    db_path=Path(tmpdir),
                    semantic=fc.SemanticConfig(
                        language_models={
                            "benchmark": fc.OpenAILanguageModel(
                                model_name=args.model_name,
                                rpm=args.client_rpm,
                                tpm=args.client_tpm,
                            )
                        },
                        default_language_model="benchmark",
                        llm_response_cache=fc.LLMResponseCacheConfig(
                            ttl="1h", namespace=f"streaming-{uuid.uuid4().hex}"
                        ),
                    ),
                )
            )
            try:
                model = session._session_state.get_language_model()
                events, lifecycle_state = _lifecycle_collector(model, cell.id)
                source = session.create_dataframe(
                    {
                        "item": deterministic_inputs(
                            cell.rows, cell.unique_inputs, cell.input_seed
                        )
                    }
                )
                if cell.operation == "map":
                    expression = fc.semantic.map(
                        cell.prompt_template,
                        item=fc.col("item"),
                        max_output_tokens=cell.max_output_tokens,
                    )
                else:
                    expression = fc.semantic.predicate(
                        cell.prompt_template,
                        item=fc.col("item"),
                    )
                query = source.select(expression.alias("result"))
                started_ns = time.monotonic_ns()
                result = query.collect()
                wall_ns = time.monotonic_ns() - started_ns
                lm = result.metrics.total_lm_metrics
                metrics = _metrics_dict(lm)
                require_metrics(metrics)
                if events is not None:
                    model.client.set_request_lifecycle_collector(None)
                lifecycle = _lifecycle_summary(events, lifecycle_state)
                try:
                    values = result.data.to_dicts()
                except AttributeError:
                    values = result.data
                return {
                    "cell_id": cell.id,
                    "tested_commit": cell.checkout,
                    "checkout": cell.checkout,
                    "scenario_id": cell.scenario_id,
                    "execution_shape": cell.execution_shape,
                    "operation": cell.operation,
                    "arm": cell.arm,
                    "input_seed": cell.input_seed,
                    "rows": cell.rows,
                    "unique_inputs": cell.unique_inputs,
                    "batch_size": cell.batch_size,
                    "repetition": cell.repetition,
                    "wall_clock_ms": wall_ns / 1_000_000,
                    "rows_per_second": cell.rows / (wall_ns / 1_000_000_000),
                    "peak_rss_bytes": peak_rss_bytes(),
                    "result_hash": result_hash(values),
                    "result_count": len(result.data),
                    "lm_metrics": metrics,
                    "lifecycle": lifecycle,
                    "rate_limit_events": lifecycle["rate_limit_events"],
                    "provider_execution": True,
                    "fenic_source": str(Path(fc.__file__).resolve()),
                }
            finally:
                if "events" in locals() and events is not None:
                    model.client.set_request_lifecycle_collector(None)
                session.stop(skip_usage_summary=True)
    finally:
        operator_class.stream_requests = old_stream_requests
        operator_class.request_batch_size = old_request_batch_size


def main() -> None:
    args = parse_args()
    raw = json.loads(args.cell_json.read_text())
    cell = _cell(raw)
    if not cell.provider_execution:
        raise SystemExit("provider execution is disabled for this matrix scenario")
    if not args.allow_provider:
        raise SystemExit("refusing to call the provider without --allow-provider")
    print(json.dumps(run_provider_cell(cell, args), sort_keys=True))


if __name__ == "__main__":
    main()
