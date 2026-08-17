#!/usr/bin/env python3
# ruff: noqa: D103
"""Plan and run an on-demand streaming performance comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import secrets
import socket
import subprocess  # nosec B404 - fixed argv only; shell execution is never enabled
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

try:
    from .models import (
        Cell,
        as_jsonable,
        assert_interleaved_same_run,
        cell_estimated_cost,
        classify_comparison,
        environment_metadata,
        expand_cells,
        interleave_cells,
        load_matrix,
        median,
        projected_cost,
        projected_requests,
        require_metrics,
    )
except ImportError:
    from models import (  # type: ignore[no-redef]
        Cell,
        as_jsonable,
        assert_interleaved_same_run,
        cell_estimated_cost,
        classify_comparison,
        environment_metadata,
        expand_cells,
        interleave_cells,
        load_matrix,
        median,
        projected_cost,
        projected_requests,
        require_metrics,
    )


def git_output(checkout: Path, *args: str) -> str:
    completed = subprocess.run(  # nosec B603 B607 - fixed git command with validated arguments
        ["git", "-C", str(checkout), *args], text=True, capture_output=True, check=True
    )
    return completed.stdout.strip()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def harness_sha256() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).parent
    for path in sorted([*root.glob("*.py"), root / "matrix.schema.json"]):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def checkout_state(checkout: Path, expected_ref: str) -> dict[str, Any]:
    checkout = checkout.resolve()
    if not (checkout / ".git").exists() and not (checkout / "HEAD").exists():
        raise ValueError(f"checkout is not a Git worktree: {checkout}")
    head = git_output(checkout, "rev-parse", "HEAD")
    expected = git_output(checkout, "rev-parse", "--verify", expected_ref)
    dirty = bool(git_output(checkout, "status", "--porcelain"))
    if head != expected:
        raise ValueError(
            f"checkout HEAD {head} does not match {expected_ref} ({expected})"
        )
    if dirty:
        raise ValueError(f"checkout is dirty: {checkout}")
    return {
        "path": str(checkout),
        "head": head,
        "expected_ref": expected_ref,
        "expected_commit": expected,
        "dirty": False,
    }


def _derived_cells(matrix: Any, checkouts: Iterable[str]) -> list[Cell]:
    cells = [
        cell for label in checkouts for cell in expand_cells(matrix, checkout=label)
    ]
    cells = interleave_cells(cells, matrix.interleaving_seed)
    assert_interleaved_same_run(cells)
    return cells


def plan_document(
    matrix_path: Path,
    checkout: Path,
    expected_ref: str,
    output: Path,
    baseline_checkout: Path | None = None,
    baseline_ref: str | None = None,
) -> dict[str, Any]:
    matrix = load_matrix(matrix_path)
    checkouts = {"candidate": checkout_state(checkout, expected_ref)}
    if baseline_checkout is not None:
        if baseline_ref is None:
            raise ValueError("--baseline-ref is required with --baseline-checkout")
        checkouts["baseline"] = checkout_state(baseline_checkout, baseline_ref)
    cells = _derived_cells(matrix, checkouts)
    return {
        "plan_id": secrets.token_hex(16),
        "schema_version": matrix.schema_version,
        "scenario_version": matrix.scenario_version,
        "matrix_path": str(matrix_path.resolve()),
        "matrix_sha256": file_sha256(matrix_path),
        "schema_sha256": file_sha256(Path(__file__).with_name("matrix.schema.json")),
        "harness_sha256": harness_sha256(),
        "created_at": datetime.now(UTC).isoformat(),
        "checkouts": checkouts,
        "environment": environment_metadata(),
        "model": {"alias": matrix.model_alias, "name": matrix.model_name},
        "limits": as_jsonable(matrix.limits),
        "pricing": as_jsonable(matrix.pricing),
        "projected_requests": projected_requests(cells, provider_only=True),
        "projected_cost_usd": projected_cost(matrix, cells),
        "interleaving_seed": matrix.interleaving_seed,
        "cells": [as_jsonable(cell) for cell in cells],
        "output": str(output.resolve()),
    }


def write_json(path: Path, document: Any) -> None:
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def verify_plan(plan: dict[str, Any], cap: float) -> tuple[Any, list[Cell]]:
    if plan["environment"]["host"] != socket.gethostname():
        raise RuntimeError("refusing run: plan was created on a different host")
    matrix_path = Path(plan["matrix_path"])
    if file_sha256(matrix_path) != plan["matrix_sha256"]:
        raise RuntimeError("refusing run: matrix changed after planning")
    if (
        file_sha256(Path(__file__).with_name("matrix.schema.json"))
        != plan["schema_sha256"]
    ):
        raise RuntimeError("refusing run: matrix schema changed after planning")
    if harness_sha256() != plan["harness_sha256"]:
        raise RuntimeError("refusing run: benchmark harness changed after planning")
    for label, planned in plan["checkouts"].items():
        state = checkout_state(Path(planned["path"]), planned["expected_ref"])
        if state["head"] != planned["head"]:
            raise RuntimeError(
                f"refusing run: {label} checkout HEAD changed after planning"
            )
    matrix = load_matrix(matrix_path)
    cells = _derived_cells(matrix, plan["checkouts"])
    if [as_jsonable(cell) for cell in cells] != plan["cells"]:
        raise RuntimeError("refusing run: planned cells do not match the matrix")
    if (
        as_jsonable(matrix.limits) != plan["limits"]
        or as_jsonable(matrix.pricing) != plan["pricing"]
    ):
        raise RuntimeError(
            "refusing run: plan limits or pricing do not match the matrix"
        )
    if projected_requests(cells, provider_only=True) != plan["projected_requests"]:
        raise RuntimeError(
            "refusing run: projected request count is not derived from the matrix"
        )
    expected_cost = projected_cost(matrix, cells)
    if abs(expected_cost - float(plan["projected_cost_usd"])) > 1e-12:
        raise RuntimeError(
            "refusing run: projected cost is not derived from the matrix"
        )
    effective_cap = min(float(matrix.limits.max_cost_usd), float(cap))
    if expected_cost > effective_cap:
        raise RuntimeError(
            f"projected cost ${expected_cost:.6f} exceeds cap ${effective_cap:.6f}"
        )
    return matrix, cells


def _metrics_cost(receipt: dict[str, Any]) -> float:
    metrics = receipt.get("lm_metrics")
    require_metrics(metrics)
    return float(metrics["cost"])


def within_run_cap(
    actual: float,
    unreconciled_reserved: float,
    current_estimate: float,
    remaining_estimate: float,
    cap: float,
) -> bool:
    """Keep every paid, reserved, and forward-looking dollar under one cap."""
    return actual + unreconciled_reserved + current_estimate + remaining_estimate <= cap


def _path_engaged(items: list[dict[str, Any]]) -> bool:
    standard = [item["path_evidence"] for item in items if item["arm"] == "standard"]
    streaming = [item["path_evidence"] for item in items if item["arm"] == "streaming"]
    if not standard or not streaming:
        return False
    if "list_calls" in standard[0]:
        return all(
            item["list_calls"] > 0 and item["iterator_calls"] == 0 for item in standard
        ) and all(
            item["iterator_calls"] > 0 and item["list_calls"] == 0 for item in streaming
        )
    return (
        all(item["streaming_enabled"] is False for item in standard)
        and all(item["streaming_enabled"] is True for item in streaming)
        and all(
            item["max_live_requests"] > item["configured_watermark"]
            for item in standard
        )
        and all(
            item["max_live_requests"] <= item["configured_watermark"]
            for item in streaming
        )
    )


def aggregate_receipts(receipts: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    receipts = list(receipts)
    if (
        len({item.get("run_id") for item in receipts}) != 1
        or len({item.get("plan_id") for item in receipts}) != 1
    ):
        raise ValueError("receipts must belong to one run and one plan")
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for item in receipts:
        key = (
            item["checkout"],
            item["tested_commit"],
            item["scenario_id"],
            item["rows"],
            item["right_rows"],
            item["unique_inputs"],
            item["batch_size"],
        )
        groups[key].append(item)
    summaries = []
    for key, items in sorted(groups.items(), key=repr):
        standard = [item for item in items if item["arm"] == "standard"]
        streaming = [item for item in items if item["arm"] == "streaming"]
        hashes = {item["result_hash"] for item in items}
        expected_counts = {int(item["expected_result_count"]) for item in items}
        result_counts = {int(item["result_count"]) for item in items}
        request_counts_ok = all(
            int(item["lm_metrics"]["num_requests"]) == int(item["physical_requests"])
            for item in items
        )
        lifecycle_checks = []
        rate_limits: list[int] = []
        for item in items:
            lifecycle = item["lifecycle"]
            availability = lifecycle["availability"]
            if availability["event_counts"]["available"]:
                counts = lifecycle["event_counts"]
                expected = int(item["physical_requests"])
                lifecycle_checks.append(
                    counts.get("queued") == expected
                    and counts.get("settled") == expected
                    and counts.get("failed", 0) == 0
                )
            if availability["rate_limit_events"]["available"]:
                rate_limits.append(int(lifecycle["rate_limit_events"]))
        lifecycle_ok = all(lifecycle_checks) if lifecycle_checks else None
        rate_limit_total = sum(rate_limits) if len(rate_limits) == len(items) else None
        path_engaged = _path_engaged(items)
        correctness_ok = (
            len(hashes) == 1
            and len(expected_counts) == 1
            and result_counts == expected_counts
            and request_counts_ok
            and lifecycle_ok is not False
        )
        verdict = (
            classify_comparison(
                [item["wall_clock_ms"] for item in streaming],
                [item["wall_clock_ms"] for item in standard],
                cache_heavy=bool(items[0]["unique_inputs"] < items[0]["rows"]),
                correctness_ok=correctness_ok,
                rate_limit_events=rate_limit_total,
                path_engaged=path_engaged,
            )
            if standard and streaming
            else "INCONCLUSIVE"
        )
        summaries.append(
            {
                "checkout": key[0],
                "tested_commit": key[1],
                "scenario_id": key[2],
                "rows": key[3],
                "right_rows": key[4],
                "unique_inputs": key[5],
                "batch_size": key[6],
                "repetitions": max(item["repetition"] for item in items),
                "standard_median_ms": median(
                    item["wall_clock_ms"] for item in standard
                ),
                "streaming_median_ms": median(
                    item["wall_clock_ms"] for item in streaming
                ),
                "correctness_ok": correctness_ok,
                "request_counts_ok": request_counts_ok,
                "lifecycle_counts_ok": lifecycle_ok,
                "rate_limit_events": rate_limit_total,
                "idle_gap_available": all(
                    item["lifecycle"]["availability"]["idle_gap"]["available"]
                    for item in items
                ),
                "path_engaged": path_engaged,
                "verdict": verdict,
            }
        )
    return summaries


def write_summary(
    output: Path, receipts: list[dict[str, Any]], projected: float, actual: float
) -> None:
    summaries = aggregate_receipts(receipts)
    note = "Idle-gap measurements are absent where the checkout exposes no collector; timing regimes are established only from independently available rate-limit events, request counts, and path evidence."
    write_json(
        output / "summary.json",
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "projected_cost_usd": projected,
            "actual_cost_usd": actual,
            "measurement_availability": note,
            "cells": summaries,
        },
    )
    fields = list(summaries[0]) if summaries else []
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    lines = [
        "# Streaming benchmark summary",
        "",
        f"Projected cost: `${projected:.6f}`",
        f"Actual cost: `${actual:.6f}`",
        "",
        note,
        "",
        "| Checkout | Scenario | Rows | Right rows | Batch | Verdict |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    lines.extend(
        f"| {row['checkout']} | {row['scenario_id']} | {row['rows']} | {row['right_rows']} | {row['batch_size']} | {row['verdict']} |"
        for row in summaries
    )
    (output / "summary.md").write_text("\n".join(lines) + "\n")


def write_manifest(output: Path) -> None:
    entries = [
        {"path": str(path.relative_to(output)), "sha256": file_sha256(path)}
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    ]
    write_json(output / "manifest.json", {"algorithm": "sha256", "files": entries})


def _create_run_state(
    output: Path, plan: dict[str, Any], cap: float
) -> tuple[Path, dict[str, Any]]:
    unexpected = [path for path in output.iterdir() if path.name != "plan.json"]
    if unexpected:
        raise RuntimeError("refusing run: output directory is not a fresh planned run")
    state_path = output / "run-state.json"
    state = {
        "run_id": secrets.token_hex(16),
        "plan_id": plan["plan_id"],
        "cap_usd": cap,
        "actual_cost_usd": 0.0,
        "unreconciled_reserved_usd": 0.0,
        "completed_cells": [],
        "active_reservation": None,
        "status": "running",
    }
    descriptor = os.open(state_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(state, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return state_path, state


def run_plan(plan_path: Path, approve_provider_spend: bool, cap: float) -> None:
    plan = json.loads(plan_path.read_text())
    matrix, cells = verify_plan(plan, cap)
    provider_cells = [cell for cell in cells if cell.execution_mode == "provider"]
    if provider_cells and not approve_provider_spend:
        raise SystemExit(
            "refusing provider-backed cells without --approve-provider-spend"
        )
    output = plan_path.parent
    effective_cap = min(float(cap), matrix.limits.max_cost_usd)
    state_path, state = _create_run_state(output, plan, effective_cap)
    cells_dir = output / "cells"
    logs_dir = output / "logs"
    specs_dir = output / "execution-specs"
    for directory in (cells_dir, logs_dir, specs_dir, output / "work"):
        directory.mkdir()
    run_token = secrets.token_hex(32)
    receipts = []
    for index, cell in enumerate(cells):
        estimate = cell_estimated_cost(matrix, cell)
        remaining = projected_cost(matrix, cells[index + 1 :])
        accounted = state["actual_cost_usd"] + state["unreconciled_reserved_usd"]
        if not within_run_cap(
            state["actual_cost_usd"],
            state["unreconciled_reserved_usd"],
            estimate,
            remaining,
            effective_cap,
        ):
            state["status"] = "stopped_before_cap"
            write_json(state_path, state)
            raise SystemExit(
                f"stopping before {cell.id}: actual, reserved, and remaining estimate exceed cap"
            )
        state["active_reservation"] = {
            "cell_id": cell.id,
            "estimated_cost_usd": estimate,
        }
        write_json(state_path, state)
        spec_path = specs_dir / f"{cell.id}.json"
        receipt_path = cells_dir / f"{cell.id}.json"
        write_json(
            spec_path,
            {
                "run_token": run_token,
                "run_id": state["run_id"],
                "plan_id": plan["plan_id"],
                "run_state_path": str(state_path),
                "matrix_path": plan["matrix_path"],
                "cell": as_jsonable(cell),
                "estimated_cost_usd": estimate,
                "cell_cost_cap_usd": effective_cap - accounted,
                "approve_provider_spend": approve_provider_spend,
                "work_dir": str(output / "work"),
            },
        )
        checkout_path = Path(plan["checkouts"][cell.checkout]["path"])
        command = [
            "uv",
            "run",
            "--project",
            str(checkout_path),
            "--no-sync",
            "python",
            str(Path(__file__).with_name("run_case.py")),
            "--execution-spec",
            str(spec_path),
            "--receipt",
            str(receipt_path),
        ]
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONPATH": os.pathsep.join(
                    (str(checkout_path), str(checkout_path / "src"))
                ),
                "PYTHONDONTWRITEBYTECODE": "1",
                "FENIC_BENCHMARK_RUN_TOKEN": run_token,
            }
        )
        try:
            completed = subprocess.run(  # nosec B603 - fixed argv; shell execution is never enabled
                command,
                cwd=checkout_path,
                env=environment,
                text=True,
                capture_output=True,
                timeout=matrix.limits.cell_timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            (logs_dir / f"{cell.id}.stdout.log").write_text(exc.stdout or "")
            (logs_dir / f"{cell.id}.stderr.log").write_text(exc.stderr or "")
            completed = None
        else:
            (logs_dir / f"{cell.id}.stdout.log").write_text(completed.stdout)
            (logs_dir / f"{cell.id}.stderr.log").write_text(completed.stderr)
        if completed is None or completed.returncode:
            if cell.execution_mode == "provider":
                state["unreconciled_reserved_usd"] += estimate
            state["active_reservation"] = None
            state["status"] = "failed"
            write_json(state_path, state)
            write_manifest(output)
            suffix = (
                "; its provider reservation remains accounted"
                if cell.execution_mode == "provider"
                else ""
            )
            raise SystemExit(f"cell {cell.id} failed{suffix}")
        receipt = json.loads(receipt_path.read_text())
        try:
            actual = (
                _metrics_cost(receipt) if cell.execution_mode == "provider" else 0.0
            )
            if (
                cell.execution_mode == "simulated"
                and float(receipt["lm_metrics"].get("cost", -1)) != 0
            ):
                raise RuntimeError(
                    "simulated cell unexpectedly reported provider spend"
                )
        except (KeyError, TypeError, ValueError, RuntimeError):
            if cell.execution_mode == "provider":
                state["unreconciled_reserved_usd"] += estimate
            state["active_reservation"] = None
            state["status"] = "failed_metrics"
            write_json(state_path, state)
            write_manifest(output)
            raise
        source = receipt.get("fenic_source")
        if source and not Path(source).resolve().is_relative_to(
            (checkout_path / "src").resolve()
        ):
            raise RuntimeError(
                f"cell {cell.id} imported fenic outside its declared checkout"
            )
        state["actual_cost_usd"] += actual
        state["active_reservation"] = None
        state["completed_cells"].append(cell.id)
        receipt.update(
            {
                "run_id": state["run_id"],
                "plan_id": plan["plan_id"],
                "tested_commit": plan["checkouts"][cell.checkout]["head"],
                "cumulative_actual_spend_usd": state["actual_cost_usd"],
                "matrix_sha256": plan["matrix_sha256"],
                "schema_sha256": plan["schema_sha256"],
                "harness_sha256": plan["harness_sha256"],
            }
        )
        write_json(receipt_path, receipt)
        write_json(state_path, state)
        receipts.append(receipt)
        print(
            f"[{index + 1}/{len(cells)}] {cell.id}: ${state['actual_cost_usd']:.6f} cumulative",
            flush=True,
        )
    state["status"] = "complete"
    write_json(state_path, state)
    write_summary(
        output, receipts, plan["projected_cost_usd"], state["actual_cost_usd"]
    )
    write_manifest(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan_parser = commands.add_parser("plan")
    plan_parser.add_argument("--matrix", type=Path, required=True)
    plan_parser.add_argument("--checkout", type=Path, required=True)
    plan_parser.add_argument("--expect-ref", required=True)
    plan_parser.add_argument("--baseline-checkout", type=Path)
    plan_parser.add_argument("--baseline-ref")
    plan_parser.add_argument("--output", type=Path, required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--plan", type=Path, required=True)
    run_parser.add_argument("--approve-provider-spend", action="store_true")
    run_parser.add_argument("--max-cost-usd", type=float, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        if args.output.exists() and any(args.output.iterdir()):
            raise SystemExit("output directory must be new or empty")
        args.output.mkdir(parents=True, exist_ok=True)
        plan = plan_document(
            args.matrix,
            args.checkout,
            args.expect_ref,
            args.output,
            args.baseline_checkout,
            args.baseline_ref,
        )
        if plan["projected_cost_usd"] > plan["limits"]["max_cost_usd"]:
            raise SystemExit("matrix projected cost exceeds matrix cap")
        write_json(args.output / "plan.json", plan)
        print(
            json.dumps(
                {
                    "plan": str((args.output / "plan.json").resolve()),
                    "cells": len(plan["cells"]),
                    "projected_cost_usd": plan["projected_cost_usd"],
                },
                sort_keys=True,
            )
        )
    else:
        run_plan(args.plan, args.approve_provider_spend, args.max_cost_usd)


if __name__ == "__main__":
    main()
