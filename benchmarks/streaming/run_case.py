#!/usr/bin/env python3
# ruff: noqa: D103
"""Execute one provider-free streaming benchmark cell."""

from __future__ import annotations

import argparse
import json
import resource
import sys
from pathlib import Path
from typing import Any

try:
    from .models import Cell, as_jsonable
except ImportError:
    from models import Cell, as_jsonable  # type: ignore[no-redef]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", required=True, help="validated cell JSON")
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


def peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def execute(cell: Cell) -> dict[str, Any]:
    """Run one bounded semantic.join arm through the simulated client."""
    if cell.execution_mode != "simulated" or cell.operation != "join":
        raise RuntimeError(
            "only the provider-free semantic.join scenario is executable"
        )

    import fenic as fc
    from benchmarks.semantic_join_stream_adapter import (
        SIMULATED_JOIN_STEP,
        Workload,
        assert_workload_geometry,
        run_arm,
    )
    from fenic._inference.model_client import ModelClient

    if dict(cell.step) != SIMULATED_JOIN_STEP:
        raise AssertionError(
            "declared scenario step does not match the simulated join adapter: "
            f"declared={dict(cell.step)!r}, adapter={SIMULATED_JOIN_STEP!r}"
        )

    workload = Workload(
        left_rows=cell.rows,
        right_rows=cell.right_rows,
        pair_block_size=cell.pair_block_size,
        block_token_budget=cell.block_token_budget,
        rpm=cell.rpm,
        batch_size=cell.batch_size,
        repetitions=1,
        latency_seconds=cell.latency_seconds,
        input_seed=cell.input_seed,
    )
    geometry = assert_workload_geometry(workload)
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
        raw = run_arm(
            workload,
            cell.arm == "streaming",
            cell.input_seed + cell.repetition,
        )
        # ``run_arm`` measures only ``join.execute()``.  Keep receipt encoding,
        # lifecycle accounting, and monkeypatch cleanup out of the sample.
        wall_ms = float(raw["wall_seconds"]) * 1_000
    finally:
        ModelClient.make_batch_requests = original_list
        ModelClient.iter_batch_requests = original_iterator

    counts = raw["lifecycle_counts"]
    return {
        "cell_id": cell.id,
        "checkout": cell.checkout,
        "scenario_id": cell.scenario_id,
        "operation": cell.operation,
        "arm": cell.arm,
        "rows": cell.rows,
        "right_rows": cell.right_rows,
        "unique_inputs": cell.unique_inputs,
        "pair_block_size": cell.pair_block_size,
        "block_token_budget": cell.block_token_budget,
        "rpm": cell.rpm,
        "latency_seconds": cell.latency_seconds,
        "batch_size": cell.batch_size,
        "repetition": cell.repetition,
        "input_seed": cell.input_seed,
        "wall_clock_ms": wall_ms,
        "peak_rss_bytes": peak_rss_bytes(),
        "result_hash": raw["result_hash"],
        "result_count": raw["result_rows"],
        "expected_result_count": cell.expected_result_count,
        "physical_requests": cell.physical_requests,
        "request_metrics": {
            "num_requests": raw["request_count"],
            "num_output_tokens": raw["output_tokens"],
        },
        "path_evidence": {
            "streaming_enabled": cell.arm == "streaming",
            "list_calls": calls["list"],
            "iterator_calls": calls["iterator"],
            "outstanding_admission_high_water": raw["max_live_requests"],
            "configured_watermark": workload.watermark,
        },
        "geometry": geometry,
        "lifecycle": {
            "availability": {
                "event_counts": {"available": True},
                "idle_gap": {
                    "available": False,
                    "reason": "the simulator receipt does not expose idle-gap measurements",
                },
                "max_queue_depth": {
                    "available": False,
                    "reason": "the adapter measures outstanding admission, not queue depth",
                },
                "rate_limit_events": {"available": True},
            },
            "event_counts": counts,
            "idle_gap": None,
            "max_queue_depth": None,
            "rate_limit_events": raw["simulated_429"],
        },
        "fenic_source": str(Path(fc.__file__).resolve()),
    }


def main() -> None:
    args = parse_args()
    cell = Cell(**json.loads(args.cell))
    receipt = execute(cell)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(as_jsonable(receipt), indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
