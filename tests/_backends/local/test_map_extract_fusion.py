import polars as pl
import pytest
from pydantic import BaseModel, Field

from fenic import col, semantic
from fenic._backends.local.physical_plan import FusedMapExtractExec, ProjectionExec
from fenic._backends.local.transpiler.plan_converter import PlanConverter
from fenic._inference.types import FenicCompletionsResponse


class _Signal(BaseModel):
    category: str = Field(description="A deterministic test category")


def _map_extract_chain(local_session, include_mapped_output=False, rows=101):
    source = local_session.create_dataframe(
        pl.DataFrame(
            {
                "record_id": pl.Series(range(rows), dtype=pl.Int64),
                "description": pl.Series(
                    [f"record-{index}" for index in range(rows)], dtype=pl.String
                ),
            }
        )
    )
    mapped = source.select(
        col("record_id"),
        semantic.map(
            "Normalize {{ description }}",
            description=col("description"),
        ).alias("normalized"),
    )
    projections = [col("record_id")]
    if include_mapped_output:
        projections.append(col("normalized"))
    projections.append(semantic.extract(col("normalized"), _Signal).alias("signal"))
    return mapped.select(*projections)


def test_fused_map_extract_pipelines_b0_blocks_without_legacy_completion_api(
    local_session,
    monkeypatch,
):
    result = _map_extract_chain(local_session)
    physical = PlanConverter(local_session._session_state).convert(result._logical_plan)
    assert isinstance(physical, FusedMapExtractExec)

    model = local_session._session_state.get_language_model()
    observed_batches = []

    monkeypatch.setattr(
        model,
        "get_completions",
        lambda *_args, **_kwargs: pytest.fail("B1 fusion must use B0's completion iterator"),
    )

    def fake_make_batch_requests(requests, operation_name, request_timeout=None):
        observed_batches.append((operation_name, list(requests), request_timeout))
        if operation_name == "semantic.map":
            return [
                FenicCompletionsResponse(completion=f"mapped-{index}", logprobs=None)
                if request is not None
                else None
                for index, request in enumerate(requests)
            ]
        assert operation_name == "semantic.extract"
        return [
            FenicCompletionsResponse(completion='{"category": "fused"}', logprobs=None)
            if request is not None
            else None
            for request in requests
        ]

    monkeypatch.setattr(model.client, "make_batch_requests", fake_make_batch_requests)

    actual = result.to_polars()

    assert actual["record_id"].to_list() == list(range(101))
    assert actual["signal"].struct.field("category").to_list() == ["fused"] * 101
    assert [(operation, len(requests)) for operation, requests, _ in observed_batches] == [
        ("semantic.map", 100),
        ("semantic.extract", 100),
        ("semantic.map", 1),
        ("semantic.extract", 1),
    ]
    assert all(timeout is None for _, _, timeout in observed_batches)


def test_map_extract_fusion_keeps_mapped_output_as_a_materialization_breaker(local_session):
    result = _map_extract_chain(local_session, include_mapped_output=True)
    physical = PlanConverter(local_session._session_state).convert(result._logical_plan)

    assert isinstance(physical, ProjectionExec)


def test_fused_map_extract_preserves_the_empty_input_boundary(local_session):
    result = _map_extract_chain(local_session, rows=0)
    physical = PlanConverter(local_session._session_state).convert(result._logical_plan)

    assert isinstance(physical, FusedMapExtractExec)
    actual = result.to_polars()
    assert actual.columns == ["record_id", "signal"]
    assert actual.height == 0


def test_fused_map_extract_uses_the_existing_projection_path_for_lineage(
    local_session,
    monkeypatch,
):
    result = _map_extract_chain(local_session, rows=1)
    model = local_session._session_state.get_language_model()

    def fake_iter_completions(messages, operation_name=None, **_kwargs):
        for message in messages:
            if message is None:
                yield None
            elif operation_name == "semantic.map":
                yield FenicCompletionsResponse(completion="mapped", logprobs=None)
            else:
                yield FenicCompletionsResponse(
                    completion='{"category": "lineage"}', logprobs=None
                )

    monkeypatch.setattr(model, "iter_completions", fake_iter_completions)

    assert result.lineage() is not None
