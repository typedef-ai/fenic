import shutil
from pathlib import Path

import polars as pl
import pytest

from fenic import (
    ArrayType,
    ColumnField,
    DoubleType,
    EmbeddingType,
    FloatType,
    IntegerType,
    StringType,
    col,
    lit,
    semantic,
    text,
)
from fenic._backends.local.semantic_operators import sim_join as sim_join_module
from fenic._constants import VECTOR_INDEX_DIR
from fenic.core.error import TypeMismatchError


def _create_semantic_join_dataframe(local_session):
    left = local_session.create_dataframe(
        {
            "course_id": [1, 2, 3, 4, 5, 6],
            "course_name": [
                "History of The Atlantic World",
                "Riemann Geometry",
                "Operating Systems",
                "Food Science",
                "Compilers",
                "Intro to Computer Networks",
            ],
            "other_col_left": ["a", "b", "c", "d", "e", "f"],
        }
    )
    right = local_session.create_dataframe(
        {
            "skill_id": [1, 2],
            "skill": ["Math", "Computer Science"],
            "other_col_right": ["g", "h"],
        }
    )
    return left, right


def _create_semantic_join_dataframe_with_none(local_session):
    left = local_session.create_dataframe(
        {
            "course_id": [1, 2, 3, 4, 5, 6, 7],
            "course_name": [
                "History of The Atlantic World",
                "Riemann Geometry",
                "Operating Systems",
                "Food Science",
                "Compilers",
                "Intro to Computer Networks",
                None,
            ],
            "other_col_left": ["a", "b", "c", "d", "e", "f", "g"],
        }
    )
    right = local_session.create_dataframe(
        {
            "skill_id": [1, 2],
            "skill": ["Math", "Computer Science"],
            "other_col_right": ["h", "i"],
        }
    )
    return left, right


def _create_semantic_join_dataframe_with_right_none(local_session):
    left = local_session.create_dataframe(
        {
            "course_id": [1, 2, 3, 4, 5, 6],
            "course_name": [
                "History of The Atlantic World",
                "Riemann Geometry",
                "Operating Systems",
                "Food Science",
                "Compilers",
                "Intro to Computer Networks",
            ],
            "other_col_left": ["a", "b", "c", "d", "e", "f"],
        }
    )
    right = local_session.create_dataframe(
        {
            "skill_id": [1, 2, 3],
            "skill": ["Math", "Computer Science", None],
            "other_col_right": ["h", "i", "j"],
        }
    )
    return left, right


def _create_semantic_join_dataframe_invalid_custom_embeddings(local_session):
    left = local_session.create_dataframe(
        pl.DataFrame(
            {
                "course_id": [1, 2, 3, 4, 5, 6],
                "course_name": [
                    "History of The Atlantic World",
                    "Riemann Geometry",
                    "Operating Systems",
                    "Food Science",
                    "Compilers",
                    "Intro to Computer Networks",
                ],
                "other_col_left": ["a", "b", "c", "d", "e", "f"],
                "course_embeddings": [
                    [float(1.0)],
                    [None],
                    [float('nan')],
                    [float(3.0)],
                    [float(6.0)],
                    None,
                ],
            },
            schema={
                "course_id": pl.Int64,
                "course_name": pl.String,
                "other_col_left": pl.String,
                "course_embeddings": pl.List(pl.Float32),
            },
        )
    ).with_column("course_embeddings", col("course_embeddings").cast(EmbeddingType(dimensions=1, embedding_model="test")))
    right = local_session.create_dataframe(
        pl.DataFrame(
            {
                "skill_id": [1, 2, 3, 4],
                "skill": ["Math", "Computer Science", None, "Philosophy"],
                "other_col_right": ["h", "i", "j", "k"],
                "skill_embeddings": [
                    [float(1.0)],
                    [None],
                    [float('nan')],
                    None,
                ],
            },
            schema={
                "skill_id": pl.Int64,
                "skill": pl.String,
                "other_col_right": pl.String,
                "skill_embeddings": pl.List(pl.Float32),
            },
        )
    ).with_column("skill_embeddings", col("skill_embeddings").cast(EmbeddingType(dimensions=1, embedding_model="test")))
    return left, right


def _create_semantic_sim_join_supplement(local_session):
    df_supplement = local_session.create_dataframe(
        {
            "high_level_skill": ["Theoretical", "Applied", "Philosophical"],
            "other_derived_column": ["i", "j", "k"],
        }
    )
    return df_supplement

@pytest.mark.parametrize("metric", ["dot", "cosine", "l2"])
def test_semantic_sim_join(local_session, metric, embedding_model_name_and_dimensions):
    embedding_model_name, embedding_dimensions = embedding_model_name_and_dimensions
    left, right = _create_semantic_join_dataframe(local_session)
    df = (
        left.with_column("course_embeddings", semantic.embed(col("course_name")))
        .semantic.sim_join(
            right.with_column("skill_embeddings", semantic.embed(col("skill"))),
            left_on="course_embeddings",
            right_on="skill_embeddings",
            k=1,
            similarity_metric=metric,
        )
    )
    assert df.schema.column_fields == [
        ColumnField("course_id", IntegerType),
        ColumnField("course_name", StringType),
        ColumnField("other_col_left", StringType),
        ColumnField("course_embeddings", EmbeddingType(dimensions=embedding_dimensions, embedding_model=embedding_model_name)),
        ColumnField("skill_id", IntegerType),
        ColumnField("skill", StringType),
        ColumnField("other_col_right", StringType),
        ColumnField("skill_embeddings", EmbeddingType(dimensions=embedding_dimensions, embedding_model=embedding_model_name)),
    ]
    result = df.to_polars()
    assert result.schema == pl.Schema(
        {
            "course_id": pl.Int64,
            "course_name": pl.String,
            "other_col_left": pl.String,
            "course_embeddings": pl.Array(pl.Float32, 1536),
            "skill_id": pl.Int64,
            "skill": pl.String,
            "other_col_right": pl.String,
            "skill_embeddings": pl.Array(pl.Float32, 1536),
        }
    )


def _create_public_sim_join_key_policy_dataframes(local_session):
    left = local_session.create_dataframe(
        pl.DataFrame(
            {
                "left_id": [1, 2],
                "left_vec": [[0.0, 0.0], [10.0, 0.0]],
                "left_payload": ["x", "y"],
            },
            schema={
                "left_id": pl.Int64,
                "left_vec": pl.List(pl.Float32),
                "left_payload": pl.String,
            },
        )
    )
    right = local_session.create_dataframe(
        pl.DataFrame(
            {
                "right_id": [10, 20],
                "right_vec": [[1.0, 0.0], [9.0, 0.0]],
                "right_payload": ["near-x", "near-y"],
            },
            schema={
                "right_id": pl.Int64,
                "right_vec": pl.List(pl.Float32),
                "right_payload": pl.String,
            },
        )
    )
    return left, right


def _expected_public_sim_join_schema(
    *,
    include_left_embedding: bool,
    include_right_embedding: bool,
):
    fields = [
        ColumnField("left_id", IntegerType),
        ColumnField("left_vec", ArrayType(FloatType)),
        ColumnField("left_payload", StringType),
    ]
    if include_left_embedding:
        fields.append(
            ColumnField(
                "left_embedding", EmbeddingType(dimensions=2, embedding_model="test")
            )
        )
    fields.extend(
        [
            ColumnField("right_id", IntegerType),
            ColumnField("right_vec", ArrayType(FloatType)),
            ColumnField("right_payload", StringType),
        ]
    )
    if include_right_embedding:
        fields.append(
            ColumnField(
                "right_embedding", EmbeddingType(dimensions=2, embedding_model="test")
            )
        )
    return fields


def _expected_public_sim_join_polars_schema(
    *,
    include_left_embedding: bool,
    include_right_embedding: bool,
):
    schema = {
        "left_id": pl.Int64,
        "left_vec": pl.List(pl.Float32),
        "left_payload": pl.String,
    }
    if include_left_embedding:
        schema["left_embedding"] = pl.Array(pl.Float32, 2)
    schema.update(
        {
            "right_id": pl.Int64,
            "right_vec": pl.List(pl.Float32),
            "right_payload": pl.String,
        }
    )
    if include_right_embedding:
        schema["right_embedding"] = pl.Array(pl.Float32, 2)
    return pl.Schema(schema)


@pytest.mark.parametrize(
    (
        "left_key_kind",
        "right_key_kind",
        "include_left_embedding",
        "include_right_embedding",
    ),
    [
        ("named", "named", True, True),
        ("expression", "expression", False, False),
        ("named", "expression", True, False),
        ("expression", "named", False, True),
    ],
)
def test_semantic_sim_join_public_api_join_key_column_policy(
    local_session,
    left_key_kind,
    right_key_kind,
    include_left_embedding,
    include_right_embedding,
):
    """Named join columns are output columns; expression-derived join keys are not."""
    left, right = _create_public_sim_join_key_policy_dataframes(local_session)

    left_embedding = col("left_vec").cast(
        EmbeddingType(dimensions=2, embedding_model="test")
    )
    right_embedding = col("right_vec").cast(
        EmbeddingType(dimensions=2, embedding_model="test")
    )

    if left_key_kind == "named":
        left = left.with_column("left_embedding", left_embedding)
        left_on = "left_embedding"
    else:
        left_on = left_embedding

    if right_key_kind == "named":
        right = right.with_column("right_embedding", right_embedding)
        right_on = "right_embedding"
    else:
        right_on = right_embedding

    df = left.semantic.sim_join(
        right, left_on=left_on, right_on=right_on, k=1, similarity_metric="l2"
    )

    assert df.schema.column_fields == _expected_public_sim_join_schema(
        include_left_embedding=include_left_embedding,
        include_right_embedding=include_right_embedding,
    )

    result = df.to_polars()
    assert result.schema == _expected_public_sim_join_polars_schema(
        include_left_embedding=include_left_embedding,
        include_right_embedding=include_right_embedding,
    )
    assert len(result) == 2


def test_semantic_sim_join_empty_result(local_session):
    left, right = _create_public_sim_join_key_policy_dataframes(local_session)
    left_embedding = col("left_vec").cast(
        EmbeddingType(dimensions=2, embedding_model="test")
    )
    right_embedding = col("right_vec").cast(
        EmbeddingType(dimensions=2, embedding_model="test")
    )

    empty_left = left.filter(col("left_id") < 0)
    df = empty_left.semantic.sim_join(
        right,
        left_on=left_embedding,
        right_on=right_embedding,
        similarity_metric="l2",
    )
    assert df.schema.column_fields == _expected_public_sim_join_schema(
        include_left_embedding=False,
        include_right_embedding=False,
    )
    result = df.to_polars()
    assert result.is_empty()
    assert result.schema == _expected_public_sim_join_polars_schema(
        include_left_embedding=False,
        include_right_embedding=False,
    )

    empty_right = right.filter(col("right_id") < 0)
    df = left.semantic.sim_join(
        empty_right,
        left_on=left_embedding,
        right_on=right_embedding,
        similarity_metric="l2",
        similarity_score_column="similarity_score",
    )
    assert df.schema.column_fields == [
        *_expected_public_sim_join_schema(
            include_left_embedding=False,
            include_right_embedding=False,
        ),
        ColumnField("similarity_score", DoubleType),
    ]
    result = df.to_polars()
    assert result.is_empty()
    assert result.schema == pl.Schema(
        {
            **dict(
                _expected_public_sim_join_polars_schema(
                    include_left_embedding=False,
                    include_right_embedding=False,
                )
            ),
            "similarity_score": pl.Float64,
        }
    )


def test_semantic_sim_join_with_sim_scores(local_session):
    left, right = _create_semantic_join_dataframe(local_session)
    df = (
        left.with_column("course_embeddings", semantic.embed(col("course_name")))
        .semantic.sim_join(
            right.with_column("skill_embeddings", semantic.embed(col("skill"))),
            left_on=col("course_embeddings"),
            right_on=col("skill_embeddings"),
            k=1,
            similarity_score_column="similarity_score",
        )
        .drop("course_embeddings", "skill_embeddings")
    )
    result = df.to_polars()
    assert result.schema == pl.Schema(
        {
            "course_id": pl.Int64,
            "course_name": pl.String,
            "other_col_left": pl.String,
            "skill_id": pl.Int64,
            "skill": pl.String,
            "other_col_right": pl.String,
            "similarity_score": pl.Float64,
        }
    )
    assert result.columns[-1] == "similarity_score"

    result_score_selected = df.select(col("similarity_score"))
    result_score_selected_result = result_score_selected.to_polars()
    assert result_score_selected_result.schema == pl.Schema(
        {"similarity_score": pl.Float64}
    )
    assert len(result_score_selected_result["similarity_score"].to_list()) == len(
        result
    )


def test_semantic_sim_join_errors(local_session):
    left, right = _create_semantic_join_dataframe(local_session)
    with pytest.raises(
        TypeMismatchError,
        match="Cannot apply semantic.sim_join on non embeddings type",
    ):
        left.semantic.sim_join(
            right.with_column("skill_embeddings", semantic.embed(col("skill"))),
            left_on=col("course_name"),
            right_on=col("skill_embeddings"),
            k=1,
        )

    with pytest.raises(
        TypeMismatchError,
        match="Cannot apply semantic.sim_join with mismatched types",
    ):
        left.with_column(
            "course_embeddings", semantic.embed(col("course_name"))
        ).semantic.sim_join(
            right, left_on=col("course_embeddings"), right_on=col("skill"), k=1
        )


def test_semantic_sim_join_derived_columns(local_session):
    left, right = _create_semantic_join_dataframe(local_session)
    supplement = _create_semantic_sim_join_supplement(local_session)

    # derived left
    df = left.join(supplement, how="cross").semantic.sim_join(
        right,
        left_on=semantic.embed(
            text.concat(col("course_name"), lit(" "), col("high_level_skill"))
        ),
        right_on=semantic.embed(col("skill")),
        k=1,
    )
    result = df.to_polars()
    assert result.schema == pl.Schema(
        {
            "course_id": pl.Int64,
            "course_name": pl.String,
            "other_col_left": pl.String,
            "high_level_skill": pl.String,
            "other_derived_column": pl.String,
            "skill_id": pl.Int64,
            "skill": pl.String,
            "other_col_right": pl.String,
        }
    )


def test_semantic_sim_join_derived_columns_with_k_gt_1(local_session):
    left, right = _create_semantic_join_dataframe(local_session)
    supplement = _create_semantic_sim_join_supplement(local_session)
    df = (
        left.with_column("course_embeddings", semantic.embed(col("course_name")))
        .semantic.sim_join(
            right.join(supplement, how="cross").with_column(
                "derived_skill_embeddings",
                semantic.embed(
                    text.concat(col("skill"), lit(" "), col("high_level_skill"))
                ),
            ),
            left_on="course_embeddings",
            right_on="derived_skill_embeddings",
            k=3,
        )
        .drop("course_embeddings", "derived_skill_embeddings")
    )
    result = df.to_polars()
    assert result.schema == pl.Schema(
        {
            "course_id": pl.Int64,
            "course_name": pl.String,
            "other_col_left": pl.String,
            "skill_id": pl.Int64,
            "skill": pl.String,
            "other_col_right": pl.String,
            "high_level_skill": pl.String,
            "other_derived_column": pl.String,
        }
    )
    assert len(result) == 18  # len(left) * k


def test_semantic_sim_join_with_none(local_session):
    """Test that we can perform a sim join a dataframe with a None value."""
    left, right = _create_semantic_join_dataframe_with_none(local_session)
    df = (
        left.with_column("course_embeddings", semantic.embed(col("course_name")))
        .semantic.sim_join(
            right.with_column("skill_embeddings", semantic.embed(col("skill"))),
            left_on="course_embeddings",
            right_on="skill_embeddings",
            k=1,
        )
        .drop("course_embeddings", "skill_embeddings")
    )
    result = df.to_polars()
    assert result.schema == pl.Schema(
        {
            "course_id": pl.Int64,
            "course_name": pl.String,
            "other_col_left": pl.String,
            "skill_id": pl.Int64,
            "skill": pl.String,
            "other_col_right": pl.String,
        }
    )

    # Row with none results is dropped.
    assert len(result) == 6
    assert None not in result["course_name"].to_list()


def test_semantic_sim_join_with_right_none(local_session):
    """Test that we can perform a sim join a dataframe with a None value."""
    left, right = _create_semantic_join_dataframe_with_right_none(local_session)
    df = (
        left.with_column("course_embeddings", semantic.embed(col("course_name")))
        .semantic.sim_join(
            right.with_column("skill_embeddings", semantic.embed(col("skill"))),
            left_on="course_embeddings",
            right_on="skill_embeddings",
            k=1,
        )
        .drop("course_embeddings", "skill_embeddings")
    )
    result = df.to_polars()
    assert result.schema == pl.Schema(
        {
            "course_id": pl.Int64,
            "course_name": pl.String,
            "other_col_left": pl.String,
            "skill_id": pl.Int64,
            "skill": pl.String,
            "other_col_right": pl.String,
        }
    )

    # there should be no match with a None value on the right side.
    assert len(result) == 6
    assert None not in result["skill"].to_list()


def test_semantic_sim_join_with_invalid_custom_embeddings(local_session):
    """Test that we can perform a sim join where a user brings their own embeddings."""
    left, right = _create_semantic_join_dataframe_invalid_custom_embeddings(local_session)
    df = left.semantic.sim_join(
        right,
        left_on="course_embeddings",
        right_on="skill_embeddings",
        k=1,
    ).drop("course_embeddings", "skill_embeddings")
    result = df.to_polars()
    assert result.schema == pl.Schema(
        {
            "course_id": pl.Int64,
            "course_name": pl.String,
            "other_col_left": pl.String,
            "skill_id": pl.Int64,
            "skill": pl.String,
            "other_col_right": pl.String,
        }
    )

    # there should be no match with a None value on the right side.
    assert len(result) == 3
    assert None not in result["skill"].to_list()


def test_semantic_sim_join_cleans_up_vector_index_dir(local_session):
    shutil.rmtree(VECTOR_INDEX_DIR, ignore_errors=True)
    vector_index_dir = Path(VECTOR_INDEX_DIR)

    left = local_session.create_dataframe(
        pl.DataFrame(
            {
                "left_id": [1, 2],
                "left_vec": [[0.0, 0.0], [10.0, 0.0]],
            },
            schema={
                "left_id": pl.Int64,
                "left_vec": pl.List(pl.Float32),
            },
        )
    ).with_column(
        "left_vec",
        col("left_vec").cast(EmbeddingType(dimensions=2, embedding_model="test")),
    )
    right = local_session.create_dataframe(
        pl.DataFrame(
            {
                "right_id": [10, 20],
                "right_vec": [[1.0, 0.0], [9.0, 0.0]],
            },
            schema={
                "right_id": pl.Int64,
                "right_vec": pl.List(pl.Float32),
            },
        )
    ).with_column(
        "right_vec",
        col("right_vec").cast(EmbeddingType(dimensions=2, embedding_model="test")),
    )

    result = left.semantic.sim_join(
        right,
        left_on="left_vec",
        right_on="right_vec",
        k=1,
        similarity_metric="l2",
    ).to_polars()

    assert len(result) == 2
    assert not list(vector_index_dir.iterdir())


def test_semantic_sim_join_custom_embeddings_golden_output(local_session):
    left = local_session.create_dataframe(
        pl.DataFrame(
            {
                "left_id": [1, 2, 3, 4],
                "left_label": ["x", "y", "null-vector", "nan-vector"],
                "left_vec": [
                    [0.0, 0.0],
                    [10.0, 0.0],
                    None,
                    [float("nan"), 0.0],
                ],
            },
            schema={
                "left_id": pl.Int64,
                "left_label": pl.String,
                "left_vec": pl.List(pl.Float32),
            },
        )
    ).with_column("left_vec", col("left_vec").cast(EmbeddingType(dimensions=2, embedding_model="test")))
    right = local_session.create_dataframe(
        pl.DataFrame(
            {
                "right_id": [10, 20, 30],
                "right_label": ["near-x", "near-y", "null-vector"],
                "right_vec": [[1.0, 0.0], [9.0, 0.0], None],
            },
            schema={
                "right_id": pl.Int64,
                "right_label": pl.String,
                "right_vec": pl.List(pl.Float32),
            },
        )
    ).with_column("right_vec", col("right_vec").cast(EmbeddingType(dimensions=2, embedding_model="test")))

    df = left.semantic.sim_join(
        right,
        left_on="left_vec",
        right_on="right_vec",
        k=2,
        similarity_metric="l2",
        similarity_score_column="distance",
    )
    assert df.schema.column_fields == [
        ColumnField("left_id", IntegerType),
        ColumnField("left_label", StringType),
        ColumnField("left_vec", EmbeddingType(dimensions=2, embedding_model="test")),
        ColumnField("right_id", IntegerType),
        ColumnField("right_label", StringType),
        ColumnField("right_vec", EmbeddingType(dimensions=2, embedding_model="test")),
        ColumnField("distance", DoubleType),
    ]

    result = df.drop("left_vec", "right_vec").to_polars().sort(["left_id", "right_id"])
    assert result.schema == pl.Schema(
        {
            "left_id": pl.Int64,
            "left_label": pl.String,
            "right_id": pl.Int64,
            "right_label": pl.String,
            "distance": pl.Float64,
        }
    )
    assert len(result) == 4
    assert result.to_dicts() == [
        {"left_id": 1, "left_label": "x", "right_id": 10, "right_label": "near-x", "distance": 1.0},
        {"left_id": 1, "left_label": "x", "right_id": 20, "right_label": "near-y", "distance": 81.0},
        {"left_id": 2, "left_label": "y", "right_id": 10, "right_label": "near-x", "distance": 81.0},
        {"left_id": 2, "left_label": "y", "right_id": 20, "right_label": "near-y", "distance": 1.0},
    ]


def _create_direct_sim_join_inputs():
    left = pl.DataFrame(
        {
            sim_join_module.LEFT_ON_COL_NAME: [[0.0, 0.0], [10.0, 0.0]],
            "left_payload": ["x", "y"],
        },
        schema={
            sim_join_module.LEFT_ON_COL_NAME: pl.Array(pl.Float32, 2),
            "left_payload": pl.String,
        },
    )
    right = pl.DataFrame(
        {
            sim_join_module.RIGHT_ON_COL_NAME: [[1.0, 0.0], [9.0, 0.0]],
            "right_payload": ["near-x", "near-y"],
        },
        schema={
            sim_join_module.RIGHT_ON_COL_NAME: pl.Array(pl.Float32, 2),
            "right_payload": pl.String,
        },
    )
    return left, right


def test_semantic_sim_join_can_skip_normalized_vector_columns():
    left, right = _create_direct_sim_join_inputs()

    result = sim_join_module.SimJoin(
        left,
        right,
        k=1,
        similarity_metric="l2",
        include_left_on=False,
        include_right_on=False,
    ).execute()

    assert sim_join_module.LEFT_ON_COL_NAME not in result.columns
    assert sim_join_module.RIGHT_ON_COL_NAME not in result.columns
    assert result.columns == [
        "left_payload",
        "right_payload",
        sim_join_module.DISTANCE_COL_NAME,
    ]


def test_semantic_sim_join_empty_result_can_skip_normalized_vector_columns():
    left, right = _create_direct_sim_join_inputs()
    empty_left = left.filter(pl.lit(False))

    result = sim_join_module.SimJoin(
        empty_left,
        right,
        k=1,
        similarity_metric="l2",
        include_left_on=False,
        include_right_on=False,
    ).execute()

    assert result.is_empty()
    assert result.schema == pl.Schema(
        {
            "left_payload": pl.String,
            "right_payload": pl.String,
            sim_join_module.DISTANCE_COL_NAME: pl.Float64,
        }
    )


def test_semantic_sim_join_batches_left_searches_without_changing_matches(monkeypatch):
    left, right = _create_direct_sim_join_inputs()
    observed_batch_sizes = []
    original_search_batch = sim_join_module.SimJoin._search_left_batch

    def record_search_batch(self, left_batch, table):
        observed_batch_sizes.append(len(left_batch))
        return original_search_batch(self, left_batch, table)

    monkeypatch.setattr(
        sim_join_module.SimJoin, "_search_left_batch", record_search_batch
    )

    result = sim_join_module.SimJoin(
        left, right, k=2, similarity_metric="l2", left_batch_size=1
    ).execute()

    assert observed_batch_sizes == [1, 1]
    assert result.select(
        "left_payload", "right_payload", sim_join_module.DISTANCE_COL_NAME
    ).sort(["left_payload", "right_payload"]).to_dicts() == [
        {
            "left_payload": "x",
            "right_payload": "near-x",
            sim_join_module.DISTANCE_COL_NAME: 1.0,
        },
        {
            "left_payload": "x",
            "right_payload": "near-y",
            sim_join_module.DISTANCE_COL_NAME: 81.0,
        },
        {
            "left_payload": "y",
            "right_payload": "near-x",
            sim_join_module.DISTANCE_COL_NAME: 81.0,
        },
        {
            "left_payload": "y",
            "right_payload": "near-y",
            sim_join_module.DISTANCE_COL_NAME: 1.0,
        },
    ]


def test_semantic_sim_join_rejects_non_positive_left_batch_size():
    left, right = _create_direct_sim_join_inputs()

    with pytest.raises(ValueError, match="left_batch_size must be positive"):
        sim_join_module.SimJoin(
            left, right, k=1, similarity_metric="l2", left_batch_size=0
        )


def test_semantic_sim_join_with_incompatible_embeddings(local_session):
    df = local_session.create_dataframe(
        {
            "course_id": [1, 2, 3, 4, 5],
            "course_name": [
                "History of The Atlantic World",
                "Riemann Geometry",
                "Operating Systems",
                "Food Science",
                "Compilers",
            ],
            "course_embeddings": [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0],
            ]
        }
    )
    left = df.select(col("course_embeddings").cast(EmbeddingType(dimensions=3, embedding_model="oai-small")).alias("left_embeddings"))
    right = df.select(col("course_embeddings").cast(EmbeddingType(dimensions=3, embedding_model="oai-large")).alias("right_embeddings"))
    with pytest.raises(
        TypeMismatchError,
        match="Cannot apply semantic.sim_join with mismatched types",
    ):
        left.semantic.sim_join(right, left_on="left_embeddings", right_on="right_embeddings", k=1)
