import polars as pl
import pytest

from benchmarks.semantic_join_stream_adapter import Workload, workload_geometry


@pytest.mark.parametrize("empty_side", ["left", "right", "both"])
def test_empty_benchmark_sides_keep_declared_schema(empty_side):
    from benchmarks.semantic_join_stream_adapter import Workload, _dataframes

    left, right = _dataframes(
        Workload(
            left_rows=0 if empty_side in ("left", "both") else 1,
            right_rows=0 if empty_side in ("right", "both") else 1,
        )
    )
    assert left.schema == {"left_on": pl.String, "record_id": pl.Int64}
    assert right.schema == {"right_on": pl.String, "right_id": pl.Int64}


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
