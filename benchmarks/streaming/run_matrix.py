#!/usr/bin/env python3
# ruff: noqa: D103
"""Plan and run a provider-free streaming performance comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import socket
import subprocess  # nosec B404 - fixed argv only; shell execution is never enabled
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

try:
    from .models import (
        Cell,
        as_jsonable,
        assert_interleaved_same_run,
        classify_comparison,
        environment_metadata,
        expand_cells,
        interleave_cells,
        load_matrix,
        median,
    )
except ImportError:
    from models import (  # type: ignore[no-redef]
        Cell,
        as_jsonable,
        assert_interleaved_same_run,
        classify_comparison,
        environment_metadata,
        expand_cells,
        interleave_cells,
        load_matrix,
        median,
    )


def git_output(checkout: Path, *args: str) -> str:
    completed = subprocess.run(  # nosec B603 B607 - fixed git command with validated arguments
        ["git", "-C", str(checkout), *args],
        text=True,
        capture_output=True,
        check=True,
    )
    return completed.stdout.strip()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def harness_sha256() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).parent
    paths = [
        *root.glob("*.py"),
        root / "matrix.schema.json",
        root.parent / "semantic_join_stream_adapter.py",
    ]
    for path in sorted(paths):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def checkout_state(checkout: Path, expected_ref: str) -> dict[str, Any]:
    checkout = checkout.resolve()
    if not (checkout / ".git").exists() and not (checkout / "HEAD").exists():
        raise ValueError(f"checkout is not a Git worktree: {checkout}")
    head = git_output(checkout, "rev-parse", "HEAD")
    expected = git_output(checkout, "rev-parse", "--verify", expected_ref)
    if head != expected:
        raise ValueError(
            f"checkout HEAD {head} does not match {expected_ref} ({expected})"
        )
    if git_output(checkout, "status", "--porcelain"):
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
        cell
        for label in sorted(checkouts)
        for cell in expand_cells(matrix, checkout=label)
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
        "plan_id": uuid.uuid4().hex,
        "schema_version": matrix.schema_version,
        "scenario_version": matrix.scenario_version,
        "matrix_path": str(matrix_path.resolve()),
        "matrix_sha256": file_sha256(matrix_path),
        "schema_sha256": file_sha256(Path(__file__).with_name("matrix.schema.json")),
        "harness_sha256": harness_sha256(),
        "created_at": datetime.now(UTC).isoformat(),
        "checkouts": checkouts,
        "environment": environment_metadata(),
        "limits": as_jsonable(matrix.limits),
        "interleaving_seed": matrix.interleaving_seed,
        "cells": [as_jsonable(cell) for cell in cells],
        "output": str(output.resolve()),
    }


def write_json(path: Path, document: Any) -> None:
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def verify_plan(plan: dict[str, Any]) -> tuple[Any, list[Cell]]:
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
    if as_jsonable(matrix.limits) != plan["limits"]:
        raise RuntimeError("refusing run: plan limits do not match the matrix")
    return matrix, cells


def _path_engaged(items: list[dict[str, Any]]) -> bool:
    standard = [item["path_evidence"] for item in items if item["arm"] == "standard"]
    streaming = [item["path_evidence"] for item in items if item["arm"] == "streaming"]
    return (
        bool(standard and streaming)
        and all(
            item["list_calls"] > 0 and item["iterator_calls"] == 0 for item in standard
        )
        and all(
            item["iterator_calls"] > 0 and item["list_calls"] == 0 for item in streaming
        )
    )


def _admission_bound(items: list[dict[str, Any]]) -> bool:
    """Require the standard arm to exceed W and streaming to stay at or below it."""
    standard = [item["path_evidence"] for item in items if item["arm"] == "standard"]
    streaming = [item["path_evidence"] for item in items if item["arm"] == "streaming"]
    if not standard or not streaming:
        return False
    return all(
        int(item["outstanding_admission_high_water"])
        > int(item["configured_watermark"])
        for item in standard
    ) and all(
        int(item["outstanding_admission_high_water"])
        <= int(item["configured_watermark"])
        for item in streaming
    )


def _event_counts_ok(item: dict[str, Any]) -> bool:
    lifecycle = item["lifecycle"]
    if not lifecycle["availability"]["event_counts"]["available"]:
        return False
    counts = lifecycle["event_counts"]
    expected = int(item["physical_requests"])
    return (
        counts.get("queued") == expected
        and counts.get("settled") == expected
        and counts.get("failed", 0) == 0
    )


def aggregate_receipts(receipts: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    receipts = list(receipts)
    run_ids = {item.get("run_id") for item in receipts}
    plan_ids = {item.get("plan_id") for item in receipts}
    if (
        not receipts
        or None in run_ids
        or None in plan_ids
        or len(run_ids) != 1
        or len(plan_ids) != 1
    ):
        raise ValueError("receipts must identify exactly one run and one plan")
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for item in receipts:
        key = (
            item["checkout"],
            item["tested_commit"],
            item["scenario_id"],
            item["rows"],
            item["right_rows"],
            item["unique_inputs"],
            item["pair_block_size"],
            item["block_token_budget"],
            item["rpm"],
            item["latency_seconds"],
            item["batch_size"],
            item["input_seed"],
        )
        groups[key].append(item)

    summaries = []
    for key, items in sorted(groups.items(), key=repr):
        standard = [item for item in items if item["arm"] == "standard"]
        streaming = [item for item in items if item["arm"] == "streaming"]
        expected_counts = {int(item["expected_result_count"]) for item in items}
        result_counts = {int(item["result_count"]) for item in items}
        request_counts_ok = all(
            int(item["request_metrics"]["num_requests"])
            == int(item["physical_requests"])
            for item in items
        )
        geometry_ok = all(
            item["geometry"]["window_binds"]
            and item["geometry"]["multiple_pair_blocks"]
            and item["geometry"]["token_budget_splits"]
            for item in items
        )
        rate_limits: list[int] = []
        for item in items:
            lifecycle = item["lifecycle"]
            if lifecycle["availability"]["rate_limit_events"]["available"]:
                rate_limits.append(int(lifecycle["rate_limit_events"]))
        lifecycle_ok = all(_event_counts_ok(item) for item in items)
        rate_limit_total = sum(rate_limits) if len(rate_limits) == len(items) else None
        path_engaged = _path_engaged(items)
        admission_bound_ok = _admission_bound(items)
        correctness_ok = (
            len({item["result_hash"] for item in items}) == 1
            and len(expected_counts) == 1
            and result_counts == expected_counts
            and request_counts_ok
            and geometry_ok
            and lifecycle_ok
            and path_engaged
            and admission_bound_ok
        )
        verdict = (
            classify_comparison(
                [item["wall_clock_ms"] for item in streaming],
                [item["wall_clock_ms"] for item in standard],
                cache_heavy=bool(
                    items[0]["operation"] != "join"
                    and items[0]["unique_inputs"] < items[0]["rows"]
                ),
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
                "pair_block_size": key[6],
                "block_token_budget": key[7],
                "rpm": key[8],
                "latency_seconds": key[9],
                "batch_size": key[10],
                "input_seed": key[11],
                "repetitions": max(item["repetition"] for item in items),
                "standard_median_ms": median(
                    item["wall_clock_ms"] for item in standard
                ),
                "streaming_median_ms": median(
                    item["wall_clock_ms"] for item in streaming
                ),
                "correctness_ok": correctness_ok,
                "request_counts_ok": request_counts_ok,
                "geometry_ok": geometry_ok,
                "lifecycle_counts_ok": lifecycle_ok,
                "rate_limit_events": rate_limit_total,
                "idle_gap_available": all(
                    item["lifecycle"]["availability"]["idle_gap"]["available"]
                    for item in items
                ),
                "path_engaged": path_engaged,
                "admission_bound_ok": admission_bound_ok,
                "verdict": verdict,
            }
        )
    return summaries


def write_summary(output: Path, receipts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = aggregate_receipts(receipts)
    note = "Idle-gap and queue-depth measurements remain absent because the adapter exposes neither measurement. Missing fields stay null and never become inferred zeros."
    write_json(
        output / "summary.json",
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "provider_calls": 0,
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
        "Provider calls: `0`",
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
    return summaries


def write_manifest(output: Path) -> None:
    entries = [
        {"path": str(path.relative_to(output)), "sha256": file_sha256(path)}
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    ]
    write_json(output / "manifest.json", {"algorithm": "sha256", "files": entries})


def start_run(output: Path, plan_id: str) -> str:
    if any(path.name != "plan.json" for path in output.iterdir()):
        raise RuntimeError("refusing run: output directory is not a fresh planned run")
    run_id = uuid.uuid4().hex
    marker = output / ".run-started.json"
    descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump({"run_id": run_id, "plan_id": plan_id}, stream, sort_keys=True)
        stream.write("\n")
    return run_id


def raise_for_failed_verdicts(summaries: Iterable[dict[str, Any]]) -> None:
    failures = [
        item
        for item in summaries
        if item["verdict"] not in {"PASS", "OBSERVATIONAL"}
    ]
    if failures:
        raise SystemExit(
            f"benchmark hard gate failed for {len(failures)} comparison(s)"
        )


def _output_for_plan(plan_path: Path, plan: dict[str, Any]) -> Path:
    output = plan_path.parent.resolve()
    planned_output = Path(plan["output"]).resolve()
    if output != planned_output:
        raise RuntimeError(
            "refusing run: plan must execute from its planned output directory"
        )
    return output


def _log_text(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value or ""


def _runner_root() -> Path:
    return Path(__file__).parents[2]


def _case_command(cell: Cell, receipt_path: Path) -> list[str]:
    return [
        "uv",
        "run",
        "--project",
        str(_runner_root()),
        "--no-sync",
        "python",
        str(Path(__file__).with_name("run_case.py")),
        "--cell",
        json.dumps(as_jsonable(cell), separators=(",", ":")),
        "--receipt",
        str(receipt_path),
    ]


def _case_environment(checkout_path: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join(
                (str(_runner_root()), str(checkout_path / "src"))
            ),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    return environment


def run_plan(plan_path: Path) -> None:
    plan = json.loads(plan_path.read_text())
    output = _output_for_plan(plan_path, plan)
    matrix, cells = verify_plan(plan)
    run_id = start_run(output, plan["plan_id"])
    cells_dir = output / "cells"
    logs_dir = output / "logs"
    for directory in (cells_dir, logs_dir):
        directory.mkdir()
    receipts = []
    for index, cell in enumerate(cells):
        receipt_path = cells_dir / f"{cell.id}.json"
        checkout_path = Path(plan["checkouts"][cell.checkout]["path"])
        command = _case_command(cell, receipt_path)
        environment = _case_environment(checkout_path)
        try:
            completed = subprocess.run(  # nosec B603 - fixed argv; shell execution is never enabled
                command,
                cwd=_runner_root(),
                env=environment,
                text=True,
                capture_output=True,
                timeout=matrix.limits.cell_timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            (logs_dir / f"{cell.id}.stdout.log").write_text(_log_text(exc.stdout))
            (logs_dir / f"{cell.id}.stderr.log").write_text(_log_text(exc.stderr))
            write_manifest(output)
            raise SystemExit(f"provider-free cell {cell.id} timed out") from exc
        (logs_dir / f"{cell.id}.stdout.log").write_text(_log_text(completed.stdout))
        (logs_dir / f"{cell.id}.stderr.log").write_text(_log_text(completed.stderr))
        if completed.returncode:
            write_manifest(output)
            raise SystemExit(f"provider-free cell {cell.id} failed; see logs")
        receipt = json.loads(receipt_path.read_text())
        source = Path(receipt["fenic_source"]).resolve()
        if not source.is_relative_to((checkout_path / "src").resolve()):
            raise RuntimeError(
                f"cell {cell.id} imported fenic outside its declared checkout"
            )
        receipt.update(
            {
                "run_id": run_id,
                "plan_id": plan["plan_id"],
                "tested_commit": plan["checkouts"][cell.checkout]["head"],
                "matrix_sha256": plan["matrix_sha256"],
                "schema_sha256": plan["schema_sha256"],
                "harness_sha256": plan["harness_sha256"],
            }
        )
        write_json(receipt_path, receipt)
        receipts.append(receipt)
        print(f"[{index + 1}/{len(cells)}] {cell.id}", flush=True)
    summaries = write_summary(output, receipts)
    write_manifest(output)
    raise_for_failed_verdicts(summaries)


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
        write_json(args.output / "plan.json", plan)
        print(
            json.dumps(
                {
                    "plan": str((args.output / "plan.json").resolve()),
                    "cells": len(plan["cells"]),
                },
                sort_keys=True,
            )
        )
    else:
        run_plan(args.plan)


if __name__ == "__main__":
    main()
