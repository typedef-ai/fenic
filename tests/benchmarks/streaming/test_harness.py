# ruff: noqa: D103
"""Provider-free contract tests for the streaming benchmark."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import jsonschema
import pytest

from benchmarks.streaming import run_case, run_matrix
from benchmarks.streaming.models import (
    as_jsonable,
    assert_interleaved_same_run,
    classify_comparison,
    expand_cells,
    interleave_cells,
    load_matrix,
    median,
    median_absolute_deviation,
    parse_matrix,
)

ROOT = Path(__file__).parents[3]
MATRIX_PATH = ROOT / "benchmarks/streaming/matrices/streaming-v1.json"


def test_benchmark_package_imports() -> None:
    assert run_case.__name__ == "benchmarks.streaming.run_case"
    assert run_matrix.__name__ == "benchmarks.streaming.run_matrix"


def test_matrix_is_provider_free_and_uses_binding_adapter_geometry() -> None:
    matrix = load_matrix(MATRIX_PATH)
    cells = expand_cells(matrix)
    assert len(cells) == 6
    assert {cell.execution_mode for cell in cells} == {"simulated"}
    assert {cell.operation for cell in cells} == {"join"}
    assert {cell.physical_requests for cell in cells} == {1024}
    assert {cell.pair_block_size for cell in cells} == {256}
    schema = json.loads((MATRIX_PATH.parents[1] / "matrix.schema.json").read_text())
    assert schema["properties"]["scenarios"]["items"]["properties"][
        "execution_mode"
    ]["enum"] == ["disabled", "simulated"]
    assert {
        item.id for item in matrix.scenarios if item.execution_mode == "disabled"
    } == {
        "map-reserved",
        "predicate-reserved",
        "map-extract-reserved",
        "three-hop-reserved",
    }


def test_cell_ids_include_full_workload_identity() -> None:
    document = json.loads(MATRIX_PATH.read_text())
    second_shape = dict(document["workload"]["shapes"][0])
    second_shape["rpm"] = 99
    document["workload"]["shapes"].append(second_shape)

    cells = expand_cells(parse_matrix(document))

    assert len({cell.id for cell in cells}) == len(cells)


def test_full_json_schema_is_required_and_rejects_unknown_fields(
    tmp_path: Path,
) -> None:
    assert jsonschema.Draft202012Validator
    raw = json.loads(MATRIX_PATH.read_text())
    raw["pricing"] = {"input": 0}
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(raw))
    with pytest.raises(jsonschema.ValidationError, match="Additional properties"):
        load_matrix(bad)

    raw = json.loads(MATRIX_PATH.read_text())
    raw["scenarios"][0]["execution_mode"] = "provider"
    bad.write_text(json.dumps(raw))
    with pytest.raises(jsonschema.ValidationError):
        load_matrix(bad)


def test_interleaving_is_deterministic_and_requires_both_arms() -> None:
    matrix = load_matrix(MATRIX_PATH)
    cells = interleave_cells(expand_cells(matrix), matrix.interleaving_seed)
    assert cells == interleave_cells(expand_cells(matrix), matrix.interleaving_seed)
    assert {cell.arm for cell in cells[:2]} == {"standard", "streaming"}
    first_arms = [cells[index].arm for index in range(0, len(cells), 2)]
    assert abs(first_arms.count("standard") - first_arms.count("streaming")) <= 1
    assert_interleaved_same_run(cells)
    with pytest.raises(ValueError, match="does not contain both arms"):
        assert_interleaved_same_run(cells[:1])


def test_preservation_verdict_passes_indistinguishable_arms() -> None:
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
    assert (
        classify_comparison([135, 136, 137], [80, 110, 140], rate_limit_events=0)
        == "INCONCLUSIVE"
    )
    assert classify_comparison([100, 101, 102], [100, 100, 101]) == "REGIME_UNVERIFIED"


def _receipt(
    arm: str,
    repetition: int,
    *,
    run_id: str | None = "run",
    plan_id: str | None = "plan",
    result_hash: str = "content-hash",
    unique_inputs: int = 512,
    rate_available: bool = True,
    event_counts_available: bool = True,
    wrong_path: bool = False,
    wrong_streaming_path: bool = False,
    over_admitting: bool = False,
    slower: bool = False,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "plan_id": plan_id,
        "checkout": "candidate",
        "tested_commit": "sha",
        "scenario_id": "bounded-join",
        "operation": "join",
        "rows": 512,
        "right_rows": 2,
        "unique_inputs": unique_inputs,
        "pair_block_size": 256,
        "block_token_budget": 18_000,
        "rpm": 100,
        "latency_seconds": 0.01,
        "batch_size": 100,
        "repetition": repetition,
        "input_seed": 29,
        "arm": arm,
        "wall_clock_ms": 140 if slower and arm == "streaming" else 100,
        "result_hash": result_hash,
        "result_count": 1024,
        "expected_result_count": 1024,
        "physical_requests": 1024,
        "request_metrics": {"num_requests": 1024},
        "path_evidence": {
            "list_calls": (
                (0 if wrong_path else 8)
                if arm == "standard"
                else 8 if wrong_streaming_path else 0
            ),
            "iterator_calls": (8 if wrong_path else 0)
            if arm == "standard"
            else 0 if wrong_streaming_path else 8,
            "outstanding_admission_high_water": (
                100 if over_admitting and arm == "standard" else 128
                if arm == "standard"
                else 101 if over_admitting else 100
            ),
            "configured_watermark": 100,
        },
        "geometry": {
            "window_binds": True,
            "multiple_pair_blocks": True,
            "token_budget_splits": True,
        },
        "lifecycle": {
            "availability": {
                "event_counts": {"available": event_counts_available},
                "idle_gap": {"available": False},
                "max_queue_depth": {"available": False},
                "rate_limit_events": {"available": rate_available},
            },
            "event_counts": {"queued": 1024, "settled": 1024},
            "idle_gap": None,
            "max_queue_depth": 128 if arm == "standard" else 100,
            "rate_limit_events": 0 if rate_available else None,
        },
    }


def _receipts(**kwargs: object) -> list[dict[str, object]]:
    return [
        _receipt(arm, repetition, **kwargs)
        for repetition in (1, 2, 3)
        for arm in ("standard", "streaming")
    ]


def test_aggregation_rejects_missing_or_mixed_identity() -> None:
    with pytest.raises(ValueError, match="identify exactly one"):
        run_matrix.aggregate_receipts(_receipts(run_id=None))
    mixed = _receipts()
    mixed[0]["plan_id"] = "other"
    with pytest.raises(ValueError, match="identify exactly one"):
        run_matrix.aggregate_receipts(mixed)


def test_actual_content_parity_is_a_hard_gate() -> None:
    receipts = _receipts()
    [summary] = run_matrix.aggregate_receipts(receipts)
    assert summary["correctness_ok"] is True
    assert summary["verdict"] == "PASS"
    receipts[0]["result_hash"] = "different-content"
    [summary] = run_matrix.aggregate_receipts(receipts)
    assert summary["correctness_ok"] is False
    assert summary["verdict"] == "FAIL"


@pytest.mark.parametrize(
    "wrong_path,wrong_streaming_path,over_admitting",
    [(True, False, False), (False, True, False), (False, False, True)],
)
def test_path_and_exact_admission_ceiling_are_hard_gates(
    wrong_path: bool, wrong_streaming_path: bool, over_admitting: bool
) -> None:
    [summary] = run_matrix.aggregate_receipts(
        _receipts(
            wrong_path=wrong_path,
            wrong_streaming_path=wrong_streaming_path,
            over_admitting=over_admitting,
        )
    )
    assert summary["correctness_ok"] is False
    assert summary["path_engaged"] is (not (wrong_path or wrong_streaming_path))
    assert summary["admission_bound_ok"] is (not over_admitting)
    assert summary["verdict"] == "FAIL"


def test_repeated_input_join_keeps_timing_failure_active() -> None:
    [summary] = run_matrix.aggregate_receipts(_receipts(unique_inputs=8, slower=True))
    assert summary["verdict"] == "FAIL"


def test_missing_rate_limit_measurement_never_substitutes_zero() -> None:
    [summary] = run_matrix.aggregate_receipts(_receipts(rate_available=False))
    assert summary["rate_limit_events"] is None
    assert summary["verdict"] == "REGIME_UNVERIFIED"


def test_missing_lifecycle_event_counts_are_a_hard_correctness_failure() -> None:
    [summary] = run_matrix.aggregate_receipts(_receipts(event_counts_available=False))
    assert summary["lifecycle_counts_ok"] is False
    assert summary["correctness_ok"] is False
    assert summary["verdict"] == "FAIL"


def test_fail_summary_produces_nonzero_exit() -> None:
    with pytest.raises(SystemExit, match="hard gate failed"):
        run_matrix.raise_for_failed_verdicts([{"verdict": "PASS"}, {"verdict": "FAIL"}])
    run_matrix.raise_for_failed_verdicts([{"verdict": "PASS"}])
    run_matrix.raise_for_failed_verdicts([{"verdict": "OBSERVATIONAL"}])
    for verdict in ("FAIL", "REGIME_UNVERIFIED", "OUTSIDE_REGIME", "INCONCLUSIVE"):
        with pytest.raises(SystemExit, match="hard gate failed"):
            run_matrix.raise_for_failed_verdicts([{"verdict": verdict}])


def test_classification_covers_each_non_pass_verdict() -> None:
    assert (
        classify_comparison([100, 101, 102], [100, 100, 101])
        == "REGIME_UNVERIFIED"
    )
    assert (
        classify_comparison(
            [100, 101, 102], [100, 100, 101], rate_limit_events=1
        )
        == "OUTSIDE_REGIME"
    )
    assert (
        classify_comparison(
            [135, 136, 137], [80, 110, 140], rate_limit_events=0
        )
        == "INCONCLUSIVE"
    )
    assert (
        classify_comparison(
            [400, 401, 402],
            [100, 101, 102],
            cache_heavy=True,
            rate_limit_events=0,
        )
        == "OBSERVATIONAL"
    )


def test_current_join_cell_runs_real_content_and_binding_geometry() -> None:
    matrix = load_matrix(MATRIX_PATH)
    standard_cell = next(
        cell for cell in expand_cells(matrix) if cell.arm == "standard"
    )
    streaming_cell = next(
        cell for cell in expand_cells(matrix) if cell.arm == "streaming"
    )
    standard = run_case.execute(standard_cell)
    streaming = run_case.execute(streaming_cell)
    reseeded = run_case.execute(replace(standard_cell, input_seed=30))
    assert standard["result_hash"] == streaming["result_hash"]
    assert standard["result_hash"] != reseeded["result_hash"]
    assert len(standard["result_hash"]) == 64
    assert standard["result_count"] == 1024
    assert standard["geometry"]["multiple_pair_blocks"] is True
    assert standard["geometry"]["token_budget_splits"] is True
    assert standard["geometry"]["window_binds"] is True
    assert standard["path_evidence"]["list_calls"] > 0
    assert streaming["path_evidence"]["iterator_calls"] > 0
    assert standard["lifecycle"]["max_queue_depth"] is None
    assert (
        standard["lifecycle"]["availability"]["max_queue_depth"]["available"]
        is False
    )


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


def test_verify_plan_rederives_cells_and_schema_hash(
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
    run_matrix.verify_plan(plan)
    plan["cells"][0]["rows"] += 1
    with pytest.raises(RuntimeError, match="cells do not match"):
        run_matrix.verify_plan(plan)


def test_derived_cells_survive_a_two_checkout_json_roundtrip() -> None:
    matrix = load_matrix(MATRIX_PATH)
    checkouts = {"candidate": {}, "baseline": {}}
    before = [as_jsonable(cell) for cell in run_matrix._derived_cells(matrix, checkouts)]
    round_tripped = json.loads(json.dumps({"checkouts": checkouts}))["checkouts"]
    after = [
        as_jsonable(cell) for cell in run_matrix._derived_cells(matrix, round_tripped)
    ]
    assert before == after
    assert [cell["checkout"] for cell in before[:2]] == ["baseline", "baseline"]


def test_runner_owns_harness_for_every_checkout_and_only_injects_fenic_src(
    tmp_path: Path,
) -> None:
    matrix = load_matrix(MATRIX_PATH)
    cells = [
        replace(cell, checkout=checkout)
        for checkout in ("candidate", "baseline")
        for cell in expand_cells(matrix)[:1]
    ]
    for cell in cells:
        checkout = tmp_path / cell.checkout
        environment = run_matrix._case_environment(checkout)
        command = run_matrix._case_command(cell, tmp_path / f"{cell.id}.json")
        python_path = environment["PYTHONPATH"].split(":")
        assert python_path == [
            str(run_matrix._runner_root()),
            str(checkout / "src"),
        ]
        assert command[command.index("--project") + 1] == str(
            run_matrix._runner_root()
        )
        assert str(checkout) not in command


def test_copied_plan_cannot_run_from_a_different_output_directory(tmp_path: Path) -> None:
    planned = tmp_path / "planned"
    copied = tmp_path / "copied"
    planned.mkdir()
    copied.mkdir()
    plan = {"output": str(planned)}
    with pytest.raises(RuntimeError, match="planned output"):
        run_matrix._output_for_plan(copied / "plan.json", plan)


def test_timeout_log_text_normalizes_bytes_and_text() -> None:
    assert run_matrix._log_text(b"stdout\xff") == "stdout�"
    assert run_matrix._log_text("stderr") == "stderr"
    assert run_matrix._log_text(None) == ""


def test_run_directory_is_single_use_and_manifest_is_complete(tmp_path: Path) -> None:
    (tmp_path / "plan.json").write_text("{}\n")
    run_id = run_matrix.start_run(tmp_path, "plan")
    assert run_id
    with pytest.raises(RuntimeError, match="not a fresh"):
        run_matrix.start_run(tmp_path, "plan")
    run_matrix.write_manifest(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert {item["path"] for item in manifest["files"]} == {
        ".run-started.json",
        "plan.json",
    }
    for item in manifest["files"]:
        assert (
            item["sha256"]
            == hashlib.sha256((tmp_path / item["path"]).read_bytes()).hexdigest()
        )
