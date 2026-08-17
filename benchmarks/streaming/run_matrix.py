#!/usr/bin/env python3
# ruff: noqa: D103
"""Plan and run the on-demand streaming performance gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import socket
import subprocess
import tempfile
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
        cost_within_cap,
        environment_metadata,
        expand_cells,
        interleave_cells,
        load_matrix,
        projected_cost,
        projected_requests,
        stamp_receipt,
    )
except ImportError:  # Executed directly rather than as a package.
    from models import (  # type: ignore[no-redef]
        Cell,
        as_jsonable,
        assert_interleaved_same_run,
        classify_comparison,
        cost_within_cap,
        environment_metadata,
        expand_cells,
        interleave_cells,
        load_matrix,
        projected_cost,
        projected_requests,
        stamp_receipt,
    )


def git_output(checkout: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(checkout), *args], text=True, capture_output=True, check=True
    )
    return completed.stdout.strip()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def harness_sha256() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
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
        "dirty": dirty,
    }


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
    cells = expand_cells(matrix, checkout="candidate")
    if baseline_checkout is not None:
        if baseline_ref is None:
            raise ValueError("--baseline-ref is required with --baseline-checkout")
        checkouts["baseline"] = checkout_state(baseline_checkout, baseline_ref)
        cells.extend(expand_cells(matrix, checkout="baseline"))
    cells = interleave_cells(cells, matrix.interleaving_seed)
    assert_interleaved_same_run(cells)
    return {
        "schema_version": matrix.schema_version,
        "scenario_version": matrix.scenario_version,
        "matrix_sha256": file_sha256(matrix_path),
        "harness_sha256": harness_sha256(),
        "created_at": datetime.now(UTC).isoformat(),
        "matrix_path": str(matrix_path.resolve()),
        "checkout": checkouts["candidate"],
        "checkouts": checkouts,
        "environment": environment_metadata(),
        "model": {"alias": matrix.model_alias, "name": matrix.model_name},
        "limits": {
            "client_rpm": matrix.client_rpm,
            "client_tpm": matrix.client_tpm,
            "cell_timeout_seconds": matrix.cell_timeout_seconds,
            "max_cost_usd": matrix.max_cost_usd,
        },
        "pricing": as_jsonable(matrix.pricing),
        "projected_requests": projected_requests(cells),
        "projected_cost_usd": projected_cost(matrix, cells),
        "max_cost_usd": matrix.max_cost_usd,
        "interleaving_seed": matrix.interleaving_seed,
        "cells": [as_jsonable(cell) for cell in cells],
        "output": str(output.resolve()),
    }


def write_json(path: Path, document: Any) -> None:
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def verify_plan(plan: dict[str, Any], cap: float) -> None:
    if plan["environment"]["host"] != socket.gethostname():
        raise RuntimeError("refusing run: plan was created on a different host")
    if file_sha256(Path(plan["matrix_path"])) != plan["matrix_sha256"]:
        raise RuntimeError("refusing run: matrix changed after planning")
    if harness_sha256() != plan["harness_sha256"]:
        raise RuntimeError("refusing run: benchmark harness changed after planning")
    checkouts = plan.get("checkouts", {"candidate": plan["checkout"]})
    for label, planned in checkouts.items():
        if planned["dirty"]:
            raise RuntimeError(
                f"refusing run: {label} checkout was dirty at planning time"
            )
        state = checkout_state(Path(planned["path"]), planned["expected_ref"])
        if state["dirty"]:
            raise RuntimeError(
                f"refusing run: {label} checkout became dirty after planning"
            )
        if state["head"] != planned["head"]:
            raise RuntimeError(
                f"refusing run: {label} checkout HEAD changed after planning"
            )
    if plan["projected_cost_usd"] > cap:
        raise RuntimeError(
            f"projected cost ${plan['projected_cost_usd']:.6f} exceeds supplied cap ${cap:.6f}"
        )


def _metrics_cost(receipt: dict[str, Any]) -> float:
    return float((receipt.get("lm_metrics") or {}).get("cost", 0) or 0)


def aggregate_receipts(receipts: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for receipt in receipts:
        key = (
            receipt.get("checkout"),
            receipt.get("tested_commit"),
            receipt.get("scenario_id"),
            receipt.get("rows"),
            receipt.get("unique_inputs"),
            receipt.get("batch_size"),
        )
        groups[key].append(receipt)
    summaries = []
    for key, items in sorted(groups.items(), key=repr):
        baseline = [item for item in items if item.get("arm") == "standard"]
        candidate = [item for item in items if item.get("arm") == "streaming"]
        baseline_times = [
            float(item["wall_clock_ms"])
            for item in baseline
            if item.get("wall_clock_ms") is not None
        ]
        candidate_times = [
            float(item["wall_clock_ms"])
            for item in candidate
            if item.get("wall_clock_ms") is not None
        ]
        hashes = {item.get("result_hash") for item in items}
        counts = {item.get("result_count") for item in items}
        request_counts_ok = all(
            int((item.get("lm_metrics") or {}).get("num_requests", -1))
            == int(item.get("physical_requests", -2))
            for item in items
        )
        lifecycle_checks: list[bool] = []
        for item in items:
            lifecycle = item.get("lifecycle") or {}
            available = bool(
                lifecycle.get("availability", {})
                .get("event_counts", {})
                .get("available")
            )
            if available:
                event_counts = lifecycle.get("event_counts") or {}
                expected = int(item.get("physical_requests", -1))
                lifecycle_checks.append(
                    int(event_counts.get("queued", -1)) == expected
                    and int(event_counts.get("settled", -1)) == expected
                    and int(event_counts.get("failed", 0)) == 0
                )
        lifecycle_counts_ok = all(lifecycle_checks) if lifecycle_checks else None
        correctness_ok = (
            len(hashes) == 1
            and len(counts) == 1
            and request_counts_ok
            and lifecycle_counts_ok is not False
        )
        observed_rate_limits = [item.get("rate_limit_events") for item in items]
        rate_limits = (
            None
            if any(value is None for value in observed_rate_limits)
            else sum(int(value) for value in observed_rate_limits)
        )
        if not baseline or not candidate:
            verdict = "INCONCLUSIVE"
        else:
            verdict = classify_comparison(
                candidate_times,
                baseline_times,
                cache_heavy=bool(
                    items[0].get("unique_inputs", 0) < items[0].get("rows", 0)
                ),
                correctness_ok=correctness_ok,
                rate_limit_events=rate_limits,
            )
        summaries.append(
            {
                "checkout": key[0],
                "tested_commit": key[1],
                "scenario_id": key[2],
                "rows": key[3],
                "unique_inputs": key[4],
                "batch_size": key[5],
                "repetitions": max(
                    (item.get("repetition", 0) for item in items), default=0
                ),
                "baseline_median_ms": sorted(baseline_times)[len(baseline_times) // 2]
                if baseline_times
                else None,
                "candidate_median_ms": sorted(candidate_times)[
                    len(candidate_times) // 2
                ]
                if candidate_times
                else None,
                "correctness_ok": correctness_ok,
                "request_counts_ok": request_counts_ok,
                "lifecycle_counts_ok": lifecycle_counts_ok,
                "rate_limit_events": rate_limits,
                "idle_gap_available": all(
                    bool(
                        (item.get("lifecycle") or {})
                        .get("availability", {})
                        .get("idle_gap", {})
                        .get("available")
                    )
                    for item in items
                ),
                "verdict": verdict,
            }
        )
    return summaries


def write_summary(
    output: Path, receipts: list[dict[str, Any]], projected: float, actual: float
) -> None:
    summaries = aggregate_receipts(receipts)
    lifecycle_available = bool(summaries) and all(
        row["idle_gap_available"] for row in summaries
    )
    measurement_note = (
        "Lifecycle event, rate-limit, and idle-gap measurements are available."
        if lifecycle_available
        else "Lifecycle event, rate-limit, or idle-gap measurements are unavailable for one or more cells; affected timing verdicts are REGIME_UNVERIFIED, and wall time is not a saturation claim. Request counts remain hard-gated."
    )
    document = {
        "generated_at": datetime.now(UTC).isoformat(),
        "projected_cost_usd": projected,
        "actual_cost_usd": actual,
        "receipts": len(receipts),
        "measurement_availability": measurement_note,
        "cells": summaries,
    }
    write_json(output / "summary.json", document)
    fields = [
        "checkout",
        "tested_commit",
        "scenario_id",
        "rows",
        "unique_inputs",
        "batch_size",
        "repetitions",
        "baseline_median_ms",
        "candidate_median_ms",
        "correctness_ok",
        "request_counts_ok",
        "lifecycle_counts_ok",
        "rate_limit_events",
        "idle_gap_available",
        "verdict",
    ]
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {field: row.get(field) for field in fields} for row in summaries
        )
    lines = [
        "# Streaming benchmark summary",
        "",
        f"Projected cost: `${projected:.6f}`",
        f"Actual cost: `${actual:.6f}`",
        "",
        measurement_note,
        "",
        "| Checkout | Scenario | Rows | Unique | Batch | Verdict |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    lines.extend(
        f"| {row['checkout']} | {row['scenario_id']} | {row['rows']} | {row['unique_inputs']} | {row['batch_size']} | {row['verdict']} |"
        for row in summaries
    )
    (output / "summary.md").write_text("\n".join(lines) + "\n")


def write_manifest(output: Path) -> None:
    entries = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        entries.append(
            {
                "path": str(path.relative_to(output)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    write_json(output / "manifest.json", {"algorithm": "sha256", "files": entries})


def run_plan(plan_path: Path, approve_provider_spend: bool, cap: float) -> None:
    plan = json.loads(plan_path.read_text())
    if not approve_provider_spend:
        raise SystemExit(
            "refusing to call the provider without --approve-provider-spend"
        )
    verify_plan(plan, cap)
    cells = [_cell_from_dict(raw) for raw in plan["cells"]]
    assert_interleaved_same_run(cells)
    if not any(cell.provider_execution for cell in cells):
        raise SystemExit(
            "provider execution is disabled for every scenario in this matrix; no calls made"
        )
    output = plan_path.parent
    cells_dir = output / "cells"
    cells_dir.mkdir(exist_ok=True)
    receipts: list[dict[str, Any]] = []
    actual = 0.0
    total = len(cells)
    matrix = load_matrix(Path(plan["matrix_path"]))
    checkouts = plan.get("checkouts", {"candidate": plan["checkout"]})
    checkout_commits = {label: state["head"] for label, state in checkouts.items()}
    for index, cell in enumerate(cells):
        remaining = projected_cost(matrix, cells[index:])
        if not cost_within_cap(actual, remaining, cap):
            raise SystemExit(
                f"stopping before {cell.id}: actual plus remaining estimate exceeds cap"
            )
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as temp:
            json.dump(as_jsonable(cell), temp)
            cell_path = Path(temp.name)
        try:
            checkout_path = Path(checkouts[cell.checkout]["path"])
            command = [
                "uv",
                "run",
                "--project",
                str(checkout_path),
                "--no-sync",
                "python",
                str(Path(__file__).with_name("run_case.py")),
                "--cell-json",
                str(cell_path),
                "--model-name",
                plan["model"]["name"],
                "--client-rpm",
                str(plan["limits"]["client_rpm"]),
                "--client-tpm",
                str(plan["limits"]["client_tpm"]),
                "--work-dir",
                str(output / "work"),
                "--allow-provider",
            ]
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(checkout_path / "src")
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            completed = subprocess.run(
                command,
                cwd=checkout_path,
                env=environment,
                text=True,
                capture_output=True,
                timeout=matrix.cell_timeout_seconds,
            )
            (output / f"{cell.id}.stdout.log").write_text(completed.stdout)
            (output / f"{cell.id}.stderr.log").write_text(completed.stderr)
            if completed.returncode:
                raise SystemExit(f"cell {cell.id} failed; see logs")
            receipt = json.loads(completed.stdout.strip().splitlines()[-1])
            source_path = Path(receipt["fenic_source"]).resolve()
            if not source_path.is_relative_to((checkout_path / "src").resolve()):
                raise SystemExit(
                    f"cell {cell.id} imported fenic outside its declared checkout: {source_path}"
                )
            actual += _metrics_cost(receipt)
            receipt = stamp_receipt(
                receipt,
                tested_commit=checkout_commits[cell.checkout],
                cumulative_actual_spend_usd=actual,
                physical_requests=cell.physical_requests,
            )
            receipt.update(
                {
                    "schema_version": plan["schema_version"],
                    "scenario_version": plan["scenario_version"],
                    "matrix_sha256": plan["matrix_sha256"],
                    "harness_sha256": plan["harness_sha256"],
                }
            )
            write_json(cells_dir / f"{cell.id}.json", receipt)
            receipts.append(receipt)
            if actual > cap:
                raise SystemExit(f"stopping after {cell.id}: actual spend exceeds cap")
        finally:
            cell_path.unlink(missing_ok=True)
        print(f"[{index + 1}/{total}] {cell.id}: ${actual:.6f} cumulative", flush=True)
    write_summary(output, receipts, float(plan["projected_cost_usd"]), actual)
    write_manifest(output)


def _cell_from_dict(raw: dict[str, Any]) -> Cell:
    return Cell(**raw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--matrix", type=Path, required=True)
    plan_parser.add_argument("--checkout", type=Path, required=True)
    plan_parser.add_argument("--expect-ref", required=True)
    plan_parser.add_argument("--baseline-checkout", type=Path)
    plan_parser.add_argument("--baseline-ref")
    plan_parser.add_argument("--output", type=Path, required=True)
    run_parser = subparsers.add_parser("run")
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
        if plan["projected_cost_usd"] > plan["max_cost_usd"]:
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
