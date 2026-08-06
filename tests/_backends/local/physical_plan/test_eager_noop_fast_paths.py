import polars as pl

from fenic._backends.local.physical_plan.transform import (
    ProjectionExec,
    _align_union_right_dataframe,
)
from fenic._backends.local.physical_plan.utils import apply_ingestion_coercions


def test_ingestion_coercions_return_original_wide_dataframe_when_no_types_change():
    for df in (
        pl.DataFrame(
            {
                "id": [1, 2],
                "name": ["first", "second"],
                "score": [1.5, 2.5],
            }
        ),
        pl.DataFrame(schema={"id": pl.Int64, "name": pl.String}),
    ):
        result = apply_ingestion_coercions(df, coerce_array=True)

        assert result is df
        assert result.schema == df.schema
        assert result.equals(df)


def test_ingestion_coercions_still_materialize_when_array_normalization_is_needed():
    df = pl.DataFrame(
        {"values": [[1, 2], [3, 4]]},
        schema={"values": pl.Array(pl.Int64, 2)},
    )

    result = apply_ingestion_coercions(df, coerce_array=True)

    assert result is not df
    assert result.schema == {"values": pl.List(pl.Int64)}
    assert result.to_dicts() == df.to_dicts()


def test_same_order_union_keeps_right_dataframe_without_an_alignment_select():
    for left, right, expected in (
        (
            pl.DataFrame({"id": [1], "name": ["left"]}),
            pl.DataFrame({"id": [2], "name": ["right"]}),
            [{"id": 1, "name": "left"}, {"id": 2, "name": "right"}],
        ),
        (
            pl.DataFrame(schema={"id": pl.Int64, "name": pl.String}),
            pl.DataFrame(schema={"id": pl.Int64, "name": pl.String}),
            [],
        ),
    ):
        aligned = _align_union_right_dataframe(left, right)

        assert aligned is right
        assert pl.concat([left, aligned], how="vertical").to_dicts() == expected


def test_different_order_union_still_aligns_right_dataframe_to_left_order():
    left = pl.DataFrame({"id": [1], "name": ["left"]})
    right = pl.DataFrame({"name": ["right"], "id": [2]})

    aligned = _align_union_right_dataframe(left, right)

    assert aligned is not right
    assert aligned.columns == left.columns
    assert pl.concat([left, aligned], how="vertical").to_dicts() == [
        {"id": 1, "name": "left"},
        {"id": 2, "name": "right"},
    ]


def test_identity_projection_returns_original_dataframe_for_rows_and_empty_frames():
    for df in (
        pl.DataFrame({"id": [1, 2], "name": ["first", "second"]}),
        pl.DataFrame(schema={"id": pl.Int64, "name": pl.String}),
    ):
        plan = ProjectionExec(
            child=None,
            projections=[pl.col(column) for column in df.columns],
            cache_info=None,
            session_state=None,
        )

        result = plan.execute_node([df])

        assert result is df
        assert result.schema == df.schema
        assert result.equals(df)


def test_projection_keeps_select_for_aliases_and_computed_columns():
    df = pl.DataFrame({"id": [1, 2]})
    plan = ProjectionExec(
        child=None,
        projections=[(pl.col("id") + 1).alias("id")],
        cache_info=None,
        session_state=None,
    )

    result = plan.execute_node([df])

    assert result is not df
    assert result.to_dicts() == [{"id": 2}, {"id": 3}]
