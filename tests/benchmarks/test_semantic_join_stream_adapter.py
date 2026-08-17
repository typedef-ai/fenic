from types import SimpleNamespace

import pytest

from benchmarks import semantic_join_stream_adapter as adapter
from benchmarks.semantic_join_stream_adapter import (
    SIMULATED_JOIN_STEP,
    PredicateSimulatedClient,
    Workload,
    _dataframes,
    assert_workload_geometry,
    run,
    run_arm,
    workload_geometry,
)
from benchmarks.streaming import run_case
from benchmarks.streaming.models import Cell


def test_benchmark_workload_binds_window_and_exercises_join_boundaries():
    geometry = workload_geometry(Workload())

    assert geometry["expected_requests"] == 1024
    assert geometry["effective_watermark"] == 100
    assert geometry["window_binds"] is True
    assert geometry["multiple_pair_blocks"] is True
    assert geometry["token_budget_splits"] is True
    assert geometry["pair_block_count"] == 4
    assert geometry["token_bounded_block_count"] == 8
    assert geometry["token_bounded_block_sizes"] == [128] * 8
    assert all(
        size > geometry["effective_watermark"]
        for size in geometry["token_bounded_block_sizes"]
    )
    assert geometry["all_token_blocks_within_pair_cap"] is True


def test_workload_guard_rejects_the_old_single_block_gate_shape():
    with pytest.raises(AssertionError, match="only one pair block"):
        assert_workload_geometry(
            Workload(
                left_rows=64,
                right_rows=2,
                pair_block_size=128,
                block_token_budget=18_000,
            )
        )


def test_arm_receipt_hashes_actual_join_content():
    workload = Workload(
        left_rows=2,
        right_rows=2,
        pair_block_size=4,
        block_token_budget=14_000,
        latency_seconds=0.001,
    )

    standard = run_arm(workload, streaming=False, repetition=1)
    streaming = run_arm(workload, streaming=True, repetition=1)

    assert standard["result_hash"] == streaming["result_hash"]
    assert len(standard["result_hash"]) == 64


def test_input_seed_changes_prompt_content_without_changing_local_draw_indices():
    seeded_left, seeded_right = _dataframes(
        Workload(left_rows=2, right_rows=2, input_seed=29)
    )
    reseeded_left, reseeded_right = _dataframes(
        Workload(left_rows=2, right_rows=2, input_seed=30)
    )

    assert seeded_left["left_on"].to_list()[0].startswith("left-0000 seed-29 ")
    assert seeded_right["right_on"].to_list()[0].startswith("right-00 seed-29 ")
    assert seeded_left["left_on"].to_list() != reseeded_left["left_on"].to_list()
    assert seeded_right["right_on"].to_list() != reseeded_right["right_on"].to_list()
    assert seeded_left["record_id"].to_list() != reseeded_left["record_id"].to_list()
    request = SimpleNamespace(
        messages=SimpleNamespace(user=seeded_left["left_on"].to_list()[0])
    )
    assert PredicateSimulatedClient._row_index(request) == 0


def test_case_receipt_uses_adapter_execution_time(monkeypatch):
    cell = Cell(
        id="bounded-join:standard:1",
        scenario_id="bounded-join",
        scenario_kind="benchmark",
        execution_mode="simulated",
        execution_shape="bounded-join",
        operation="join",
        step=SIMULATED_JOIN_STEP,
        arm="standard",
        rows=512,
        right_rows=2,
        unique_inputs=512,
        pair_block_size=256,
        block_token_budget=18_000,
        rpm=100,
        latency_seconds=0.01,
        batch_size=100,
        repetition=1,
        input_seed=29,
    )
    raw = {
        "wall_seconds": 1.25,
        "result_hash": "a" * 64,
        "result_rows": 1024,
        "request_count": 1024,
        "output_tokens": 2048,
        "lifecycle_counts": {"queued": 1024, "settled": 1024},
        "simulated_429": 0,
        "max_live_requests": 512,
    }
    monkeypatch.setattr(
        "benchmarks.semantic_join_stream_adapter.run_arm", lambda *_: raw
    )

    receipt = run_case.execute(cell)

    assert receipt["wall_clock_ms"] == 1250


def test_adapter_preservation_verdict_precedes_overlapping_bands(monkeypatch):
    monkeypatch.setattr(
        "benchmarks.semantic_join_stream_adapter.assert_workload_geometry",
        lambda _: {},
    )

    def fake_arm(workload, streaming, repetition):
        standard_seconds = {1: 1.0, 2: 1.2, 3: 0.8}[repetition]
        return {
            "arm": "streaming" if streaming else "standard",
            "wall_seconds": standard_seconds + (0.1 if streaming else 0),
            "max_live_requests": workload.watermark if streaming else 128,
        }

    monkeypatch.setattr(
        "benchmarks.semantic_join_stream_adapter.run_arm", fake_arm
    )

    result = run(Workload(repetitions=3))

    assert result["mad_bands_overlap"] is True
    assert result["streaming_delta_percent"] == pytest.approx(10)
    assert result["evidence_verdict"] == "PASS"


def test_adapter_cli_exits_nonzero_for_nonpassing_verdict(monkeypatch) -> None:
    monkeypatch.setattr(adapter, "run", lambda _: {"evidence_verdict": "FAIL"})
    monkeypatch.setattr("sys.argv", ["semantic_join_stream_adapter.py"])

    with pytest.raises(SystemExit, match="benchmark verdict: FAIL"):
        adapter.main()


def test_case_rejects_declared_step_that_differs_from_adapter() -> None:
    matrix_cell = Cell(
        id="bounded-join:standard:1",
        scenario_id="bounded-join",
        scenario_kind="benchmark",
        execution_mode="simulated",
        execution_shape="bounded-join",
        operation="join",
        step={**SIMULATED_JOIN_STEP, "prompt_template": "different"},
        arm="standard",
        rows=512,
        right_rows=2,
        unique_inputs=512,
        pair_block_size=256,
        block_token_budget=18_000,
        rpm=100,
        latency_seconds=0.01,
        batch_size=100,
        repetition=1,
        input_seed=29,
    )

    with pytest.raises(AssertionError, match="declared scenario step"):
        run_case.execute(matrix_cell)
