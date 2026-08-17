# ruff: noqa: D103
"""Provider-free contract tests for the streaming benchmark."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.streaming import run_case, run_matrix
from benchmarks.streaming.models import (
    Cell,
    as_jsonable,
    assert_interleaved_same_run,
    cell_estimated_cost,
    classify_comparison,
    expand_cells,
    interleave_cells,
    load_matrix,
    median,
    median_absolute_deviation,
    projected_cost,
    projected_requests,
    require_metrics,
)

ROOT = Path(__file__).parents[3]
MATRIX_PATH = ROOT / "benchmarks/streaming/matrices/streaming-v1.json"


def test_matrix_executes_only_current_join_surface() -> None:
    matrix = load_matrix(MATRIX_PATH)
    cells = expand_cells(matrix)
    assert len(cells) == 6
    assert {cell.operation for cell in cells} == {"join"}
    assert {cell.execution_mode for cell in cells} == {"simulated"}
    assert {
        scenario.id
        for scenario in matrix.scenarios
        if scenario.execution_mode == "disabled"
    } == {
        "map-reserved",
        "predicate-reserved",
        "map-extract-reserved",
        "three-hop-reserved",
    }
    assert projected_requests(cells, provider_only=True) == 0
    assert projected_cost(matrix, cells) == 0


def test_schema_structurally_rejects_zero_pricing_and_unbounded_shapes(
    tmp_path: Path,
) -> None:
    raw = json.loads(MATRIX_PATH.read_text())
    raw["pricing"]["input_per_million_usd"] = 0
    bad = tmp_path / "zero-price.json"
    bad.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        load_matrix(bad)

    raw = json.loads(MATRIX_PATH.read_text())
    raw["workload"]["shapes"][0]["rows"] = 10001
    bad.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        load_matrix(bad)


def test_interleaving_is_deterministic_and_requires_both_arms() -> None:
    matrix = load_matrix(MATRIX_PATH)
    cells = interleave_cells(expand_cells(matrix), matrix.interleaving_seed)
    assert cells == interleave_cells(expand_cells(matrix), matrix.interleaving_seed)
    assert {cell.arm for cell in cells[:2]} == {"standard", "streaming"}
    assert_interleaved_same_run(cells)
    with pytest.raises(ValueError, match="does not contain both arms"):
        assert_interleaved_same_run(cells[:1])


def test_even_median_and_gate_use_the_same_center() -> None:
    assert median([1, 2, 100, 101]) == 51
    assert median_absolute_deviation([1, 2, 100, 101]) == 49.5
    assert (
        classify_comparison([100, 101, 102], [100, 100, 101], rate_limit_events=0)
        == "PASS"
    )
    assert (
        classify_comparison([130, 131, 132], [100, 100, 101], rate_limit_events=0)
        == "FAIL"
    )
    assert classify_comparison([100, 101, 102], [100, 100, 101]) == "REGIME_UNVERIFIED"


def test_provider_metrics_missing_or_zero_are_a_hard_stop() -> None:
    for metrics in (None, {"cost": 0}, {"cost": 0.001, "num_output_tokens": 1}):
        with pytest.raises(RuntimeError):
            require_metrics(metrics)
    with pytest.raises(RuntimeError):
        run_matrix._metrics_cost({"lm_metrics": None})
    assert (
        run_matrix._metrics_cost(
            {"lm_metrics": {"cost": 0.001, "num_requests": 1, "num_output_tokens": 1}}
        )
        == 0.001
    )


def _receipt(
    arm: str, repetition: int, *, rate_available: bool = True
) -> dict[str, object]:
    maximum = 128 if arm == "standard" else 100
    return {
        "run_id": "run",
        "plan_id": "plan",
        "checkout": "candidate",
        "tested_commit": "sha",
        "scenario_id": "bounded-join",
        "rows": 64,
        "right_rows": 2,
        "unique_inputs": 64,
        "batch_size": 100,
        "repetition": repetition,
        "arm": arm,
        "wall_clock_ms": 100 if arm == "standard" else 105,
        "result_hash": "same",
        "result_count": 128,
        "expected_result_count": 128,
        "physical_requests": 128,
        "lm_metrics": {"num_requests": 128},
        "path_evidence": {
            "streaming_enabled": arm == "streaming",
            "max_live_requests": maximum,
            "configured_watermark": 100,
        },
        "lifecycle": {
            "availability": {
                "event_counts": {"available": True},
                "idle_gap": {"available": False},
                "max_queue_depth": {"available": True},
                "rate_limit_events": {"available": rate_available},
            },
            "event_counts": {"queued": 128, "settled": 128},
            "idle_gap": None,
            "max_queue_depth": maximum,
            "rate_limit_events": 0 if rate_available else None,
        },
    }


def test_aggregation_requires_one_run_exact_results_and_path_engagement() -> None:
    receipts = [
        _receipt(arm, repetition)
        for repetition in (1, 2, 3)
        for arm in ("standard", "streaming")
    ]
    [summary] = run_matrix.aggregate_receipts(receipts)
    assert summary["verdict"] == "PASS"
    assert summary["path_engaged"] is True

    receipts[0]["result_count"] = 127
    [summary] = run_matrix.aggregate_receipts(receipts)
    assert summary["verdict"] == "FAIL"
    receipts[0]["result_count"] = 128
    receipts[0]["run_id"] = "stale-run"
    with pytest.raises(ValueError, match="one run"):
        run_matrix.aggregate_receipts(receipts)


def test_unavailable_rate_limit_measurement_never_substitutes_zero() -> None:
    receipts = [
        _receipt(arm, repetition, rate_available=False)
        for repetition in (1, 2, 3)
        for arm in ("standard", "streaming")
    ]
    [summary] = run_matrix.aggregate_receipts(receipts)
    assert summary["rate_limit_events"] is None
    assert summary["verdict"] == "REGIME_UNVERIFIED"


def test_execution_spec_requires_parent_token_matrix_cell_and_ack(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    matrix = load_matrix(MATRIX_PATH)
    cell = expand_cells(matrix)[0]
    spec = {
        "run_token": "secret",
        "run_id": "run",
        "plan_id": "plan",
        "run_state_path": str(tmp_path / "run-state.json"),
        "matrix_path": str(MATRIX_PATH),
        "cell": as_jsonable(cell),
        "estimated_cost_usd": 0,
        "cell_cost_cap_usd": 1,
        "approve_provider_spend": False,
        "work_dir": "/tmp",
    }
    (tmp_path / "run-state.json").write_text(
        json.dumps(
            {
                "run_id": "run",
                "plan_id": "plan",
                "active_reservation": {
                    "cell_id": cell.id,
                    "estimated_cost_usd": 0,
                },
            }
        )
    )
    monkeypatch.delenv("FENIC_BENCHMARK_RUN_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="not authorized"):
        run_case.validate_execution_spec(spec)
    monkeypatch.setenv("FENIC_BENCHMARK_RUN_TOKEN", "secret")
    run_case.validate_execution_spec(spec)
    spec["cell"]["rows"] = 65
    with pytest.raises(RuntimeError, match="not derived"):
        run_case.validate_execution_spec(spec)

    spec["cell"] = as_jsonable(replace(cell, execution_mode="provider"))
    monkeypatch.setattr(
        run_case, "expand_cells", lambda matrix, checkout: [Cell(**spec["cell"])]
    )
    spec["estimated_cost_usd"] = cell_estimated_cost(matrix, Cell(**spec["cell"]))
    run_state = json.loads(Path(spec["run_state_path"]).read_text())
    run_state["active_reservation"]["estimated_cost_usd"] = spec["estimated_cost_usd"]
    Path(spec["run_state_path"]).write_text(json.dumps(run_state))
    with pytest.raises(RuntimeError, match="acknowledgement"):
        run_case.validate_execution_spec(spec)


def test_current_join_cell_runs_end_to_end_without_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    matrix = load_matrix(MATRIX_PATH)
    cell = next(cell for cell in expand_cells(matrix) if cell.arm == "streaming")
    monkeypatch.setenv("FENIC_BENCHMARK_RUN_TOKEN", "proof")
    run_state = tmp_path / "run-state.json"
    run_state.write_text(
        json.dumps(
            {
                "run_id": "run",
                "plan_id": "plan",
                "active_reservation": {
                    "cell_id": cell.id,
                    "estimated_cost_usd": 0,
                },
            }
        )
    )
    payload = run_case.execute(
        {
            "run_token": "proof",
            "run_id": "run",
            "plan_id": "plan",
            "run_state_path": str(run_state),
            "matrix_path": str(MATRIX_PATH),
            "cell": as_jsonable(cell),
            "estimated_cost_usd": 0,
            "cell_cost_cap_usd": 1,
            "approve_provider_spend": False,
            "work_dir": str(tmp_path),
        }
    )
    assert payload["result_count"] == cell.expected_result_count
    assert payload["lm_metrics"]["num_requests"] == cell.physical_requests
    assert payload["provider_execution"] is False
    assert payload["path_evidence"]["streaming_enabled"] is True
    assert payload["path_evidence"]["iterator_calls"] > 0
    assert payload["path_evidence"]["list_calls"] == 0


def _init_git_repo(path: Path) -> str:
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@example.com"], check=True
    )
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Test"], check=True)
    (path / "tracked").write_text("ok")
    subprocess.run(["git", "-C", str(path), "add", "tracked"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "init"], check=True)
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def test_checkout_state_refuses_dirty_worktree(tmp_path: Path) -> None:
    head = _init_git_repo(tmp_path)
    assert run_matrix.checkout_state(tmp_path, head)["head"] == head
    (tmp_path / "tracked").write_text("dirty")
    with pytest.raises(ValueError, match="dirty"):
        run_matrix.checkout_state(tmp_path, head)


def test_verify_plan_rederives_cells_pricing_and_schema_hash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    head = "a" * 40
    state = {
        "path": str(tmp_path),
        "head": head,
        "expected_ref": head,
        "expected_commit": head,
        "dirty": False,
    }
    monkeypatch.setattr(run_matrix, "checkout_state", lambda *args: state)
    plan = run_matrix.plan_document(MATRIX_PATH, tmp_path, head, tmp_path / "run")
    run_matrix.verify_plan(plan, 1)
    plan["cells"][0]["rows"] += 1
    with pytest.raises(RuntimeError, match="cells do not match"):
        run_matrix.verify_plan(plan, 1)


def test_run_directory_is_single_use_and_manifest_hashes_schema_receipt(
    tmp_path: Path,
) -> None:
    (tmp_path / "plan.json").write_text("{}\n")
    _, state = run_matrix._create_run_state(tmp_path, {"plan_id": "p"}, 1)
    assert state["actual_cost_usd"] == 0
    with pytest.raises(RuntimeError, match="not a fresh"):
        run_matrix._create_run_state(tmp_path, {"plan_id": "p"}, 1)
    run_matrix.write_manifest(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert {item["path"] for item in manifest["files"]} == {
        "plan.json",
        "run-state.json",
    }
    assert (
        manifest["files"][0]["sha256"]
        == hashlib.sha256(
            (tmp_path / manifest["files"][0]["path"]).read_bytes()
        ).hexdigest()
    )


def test_run_requires_ack_before_provider_and_retains_failed_reservation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    matrix = load_matrix(MATRIX_PATH)
    provider = replace(expand_cells(matrix)[0], execution_mode="provider")
    plan = {
        "plan_id": "p",
        "checkouts": {"candidate": {"path": str(tmp_path), "head": "sha"}},
        "projected_cost_usd": 0.5,
        "matrix_sha256": "m",
        "schema_sha256": "s",
        "harness_sha256": "h",
        "cells": [as_jsonable(provider)],
        "matrix_path": str(MATRIX_PATH),
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    monkeypatch.setattr(run_matrix, "verify_plan", lambda *args: (matrix, [provider]))
    with pytest.raises(SystemExit, match="approve-provider-spend"):
        run_matrix.run_plan(plan_path, False, 1)
    assert list(tmp_path.iterdir()) == [plan_path]

    monkeypatch.setattr(run_matrix, "cell_estimated_cost", lambda *args: 0.25)
    monkeypatch.setattr(run_matrix, "projected_cost", lambda *args: 0.0)
    monkeypatch.setattr(
        run_matrix.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1, stdout="", stderr="failure"
        ),
    )
    with pytest.raises(SystemExit, match="reservation remains accounted"):
        run_matrix.run_plan(plan_path, True, 1)
    state = json.loads((tmp_path / "run-state.json").read_text())
    assert state["unreconciled_reserved_usd"] == 0.25
    assert state["status"] == "failed"


def test_pre_cell_cap_recheck_counts_prior_actual_and_remaining(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    matrix = load_matrix(MATRIX_PATH)
    cells = [
        replace(cell, execution_mode="provider") for cell in expand_cells(matrix)[:2]
    ]
    plan = {
        "plan_id": "p",
        "checkouts": {"candidate": {"path": str(tmp_path), "head": "sha"}},
        "projected_cost_usd": 2.0,
        "matrix_sha256": "m",
        "schema_sha256": "s",
        "harness_sha256": "h",
        "cells": [as_jsonable(cell) for cell in cells],
        "matrix_path": str(MATRIX_PATH),
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    monkeypatch.setattr(run_matrix, "verify_plan", lambda *args: (matrix, cells))
    monkeypatch.setattr(run_matrix, "cell_estimated_cost", lambda *args: 0.6)
    monkeypatch.setattr(run_matrix, "projected_cost", lambda *args: 0.6)
    with pytest.raises(SystemExit, match="actual, reserved, and remaining"):
        run_matrix.run_plan(plan_path, True, 1)
    assert (
        json.loads((tmp_path / "run-state.json").read_text())["status"]
        == "stopped_before_cap"
    )


def test_cap_formula_includes_actual_reserved_current_and_remaining() -> None:
    assert run_matrix.within_run_cap(0.1, 0.2, 0.3, 0.4, 1.0)
    assert not run_matrix.within_run_cap(0.11, 0.2, 0.3, 0.4, 1.0)


def test_successful_child_with_missing_metrics_is_charged_and_stopped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    matrix = load_matrix(MATRIX_PATH)
    provider = replace(expand_cells(matrix)[0], execution_mode="provider")
    plan = {
        "plan_id": "p",
        "checkouts": {"candidate": {"path": str(tmp_path), "head": "sha"}},
        "projected_cost_usd": 0.25,
        "matrix_sha256": "m",
        "schema_sha256": "s",
        "harness_sha256": "h",
        "cells": [as_jsonable(provider)],
        "matrix_path": str(MATRIX_PATH),
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    monkeypatch.setattr(run_matrix, "verify_plan", lambda *args: (matrix, [provider]))
    monkeypatch.setattr(run_matrix, "cell_estimated_cost", lambda *args: 0.25)
    monkeypatch.setattr(run_matrix, "projected_cost", lambda *args: 0.0)

    def completed_without_metrics(
        command: list[str], **kwargs: object
    ) -> SimpleNamespace:
        receipt_path = Path(command[command.index("--receipt") + 1])
        receipt_path.write_text(json.dumps({"lm_metrics": None}))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(run_matrix.subprocess, "run", completed_without_metrics)
    with pytest.raises(RuntimeError, match="no LMMetrics"):
        run_matrix.run_plan(plan_path, True, 1)
    state = json.loads((tmp_path / "run-state.json").read_text())
    assert state["unreconciled_reserved_usd"] == 0.25
    assert state["status"] == "failed_metrics"
