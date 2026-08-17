# ruff: noqa: D103
"""Provider-free contract tests for the streaming gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.streaming.models import (
    Cell,
    assert_interleaved_same_run,
    classify_comparison,
    cost_within_cap,
    expand_cells,
    interleave_cells,
    lifecycle_unavailable,
    load_matrix,
    median_absolute_deviation,
    projected_cost,
    projected_requests,
    require_metrics,
    stamp_receipt,
)
from benchmarks.streaming.run_matrix import aggregate_receipts, write_manifest

ROOT = Path(__file__).parents[3]
MATRIX_PATH = ROOT / "benchmarks/streaming/matrices/streaming-v1.json"


def test_matrix_schema_and_preserved_cell_count() -> None:
    matrix = load_matrix(MATRIX_PATH)
    cells = expand_cells(matrix)
    assert len(cells) == 72
    assert {cell.operation for cell in cells} == {"map", "predicate"}
    assert {cell.arm for cell in cells} == {"standard", "streaming"}
    assert {cell.batch_size for cell in cells} == {32, 100}
    assert all(cell.provider_execution for cell in cells)
    assert {
        scenario.id for scenario in matrix.scenarios if not scenario.provider_execution
    } == {
        "map-extract-reserved",
        "three-hop-reserved",
    }


def test_matrix_validation_rejects_missing_required_field(tmp_path: Path) -> None:
    raw = json.loads(MATRIX_PATH.read_text())
    del raw["model"]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(raw))
    with pytest.raises((ValueError, KeyError)):
        load_matrix(bad)


def test_interleaving_is_deterministic_and_alternates_arms() -> None:
    matrix = load_matrix(MATRIX_PATH)
    cells = interleave_cells(expand_cells(matrix), matrix.interleaving_seed)
    assert cells == interleave_cells(expand_cells(matrix), matrix.interleaving_seed)
    assert {cell.arm for cell in cells[:2]} == {"standard", "streaming"}
    assert_interleaved_same_run(cells)


def test_interleaving_requires_both_arms_for_each_checkout() -> None:
    matrix = load_matrix(MATRIX_PATH)
    candidate = expand_cells(matrix)[0]
    baseline = Cell(
        **{**candidate.__dict__, "checkout": "baseline", "arm": "streaming"}
    )
    with pytest.raises(ValueError, match="does not contain both arms"):
        assert_interleaved_same_run([candidate, baseline])


def test_cost_arithmetic_and_cap_stop() -> None:
    matrix = load_matrix(MATRIX_PATH)
    cells = expand_cells(matrix)
    assert projected_requests(cells) == 28992
    assert projected_cost(matrix, cells) > 0
    assert cost_within_cap(1.0, 2.0, 3.0)
    assert not cost_within_cap(1.0, 2.01, 3.0)


def test_mad_and_gate_dispositions() -> None:
    assert median_absolute_deviation([1, 2, 3]) == 1
    assert (
        classify_comparison([100, 101, 102], [100, 100, 101], rate_limit_events=0)
        == "PASS"
    )
    assert (
        classify_comparison([130, 131, 132], [100, 100, 101], rate_limit_events=0)
        == "FAIL"
    )
    assert (
        classify_comparison([135, 136, 137], [80, 110, 140], rate_limit_events=0)
        == "INCONCLUSIVE"
    )
    assert (
        classify_comparison(
            [200, 201, 202], [100, 100, 100], cache_heavy=True, rate_limit_events=0
        )
        == "OBSERVATIONAL"
    )
    assert (
        classify_comparison([101, 102, 103], [100, 100, 101], rate_limit_events=1)
        == "OUTSIDE_REGIME"
    )
    assert classify_comparison([100, 101, 102], [100, 100, 101]) == "REGIME_UNVERIFIED"


def test_zero_metrics_are_a_hard_stop() -> None:
    with pytest.raises(RuntimeError):
        require_metrics(None)
    with pytest.raises(RuntimeError):
        require_metrics({"cost": 0})
    with pytest.raises(RuntimeError):
        require_metrics({"cost": 0.001, "num_output_tokens": 1})
    require_metrics({"cost": 0.001, "num_requests": 1, "num_output_tokens": 1})


def test_lifecycle_availability_is_honest_and_receipts_are_stamped() -> None:
    lifecycle = lifecycle_unavailable()
    assert lifecycle["availability"]["idle_gap"]["available"] is False
    assert lifecycle["idle_gap"] is None
    assert lifecycle["rate_limit_events"] is None
    receipt = stamp_receipt(
        {"cell_id": "one"},
        tested_commit="abc",
        cumulative_actual_spend_usd=0.25,
        physical_requests=8,
    )
    assert receipt == {
        "cell_id": "one",
        "tested_commit": "abc",
        "cumulative_actual_spend_usd": 0.25,
        "physical_requests": 8,
    }


def _receipt(
    checkout: str, arm: str, repetition: int, *, requests: int = 2
) -> dict[str, object]:
    return {
        "checkout": checkout,
        "tested_commit": f"{checkout}-sha",
        "scenario_id": "map",
        "rows": 2,
        "unique_inputs": 2,
        "batch_size": 32,
        "repetition": repetition,
        "arm": arm,
        "wall_clock_ms": 100 if arm == "standard" else 105,
        "result_hash": "same",
        "result_count": 2,
        "physical_requests": 2,
        "lm_metrics": {"num_requests": requests},
        "rate_limit_events": 0,
        "lifecycle": {"availability": {"idle_gap": {"available": False}}},
    }


def test_aggregation_keeps_checkouts_separate_and_hard_gates_request_counts() -> None:
    receipts = [
        _receipt(checkout, arm, repetition)
        for checkout in ("candidate", "baseline")
        for repetition in (1, 2, 3)
        for arm in ("standard", "streaming")
    ]
    summaries = aggregate_receipts(receipts)
    assert len(summaries) == 2
    assert {summary["checkout"] for summary in summaries} == {"candidate", "baseline"}
    assert {summary["verdict"] for summary in summaries} == {"PASS"}

    receipts[0] = _receipt("candidate", "standard", 1, requests=3)
    summaries = aggregate_receipts(receipts)
    candidate = next(row for row in summaries if row["checkout"] == "candidate")
    assert candidate["request_counts_ok"] is False
    assert candidate["verdict"] == "FAIL"

    bad_lifecycle = _receipt("candidate", "standard", 1)
    bad_lifecycle["lifecycle"] = {
        "availability": {
            "event_counts": {"available": True},
            "idle_gap": {"available": True},
        },
        "event_counts": {"queued": 2, "settled": 1},
    }
    receipts[0] = bad_lifecycle
    summaries = aggregate_receipts(receipts)
    candidate = next(row for row in summaries if row["checkout"] == "candidate")
    assert candidate["lifecycle_counts_ok"] is False
    assert candidate["verdict"] == "FAIL"


def test_manifest_hashes_every_existing_evidence_file(tmp_path: Path) -> None:
    (tmp_path / "plan.json").write_text("{}\n")
    cells = tmp_path / "cells"
    cells.mkdir()
    (cells / "one.json").write_text('{"ok": true}\n')
    write_manifest(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest == {
        "algorithm": "sha256",
        "files": [
            {
                "path": "cells/one.json",
                "sha256": hashlib.sha256((cells / "one.json").read_bytes()).hexdigest(),
            },
            {
                "path": "plan.json",
                "sha256": hashlib.sha256(
                    (tmp_path / "plan.json").read_bytes()
                ).hexdigest(),
            },
        ],
    }
