import polars as pl
import pytest

from fenic import (
    EmbeddingType,
)
from fenic.api.functions import col, lit, semantic, text
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
def test_semantic_sim_join(local_session, metric):
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


def test_semantic_sim_join_empty_result(local_session):
    left, right = _create_semantic_join_dataframe(local_session)
    empty_left = left.filter(col("course_name").is_null())
    df = empty_left.semantic.sim_join(
        right,
        left_on=semantic.embed(col("course_name")),
        right_on=semantic.embed(col("skill")),
    )
    result = df.to_polars()
    assert result.is_empty()
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

    empty_right = right.filter(col("skill").is_null())
    df = left.semantic.sim_join(
        empty_right,
        left_on=semantic.embed(col("course_name")),
        right_on=semantic.embed(col("skill")),
        return_similarity_scores=True,
    )
    result = df.to_polars()
    assert result.is_empty()
    assert result.schema == pl.Schema(
        {
            "course_id": pl.Int64,
            "course_name": pl.String,
            "other_col_left": pl.String,
            "skill_id": pl.Int64,
            "skill": pl.String,
            "other_col_right": pl.String,
            "_similarity_score": pl.Float64,
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
            return_similarity_scores=True,
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
            "_similarity_score": pl.Float64,
        }
    )
    assert result.columns[-1] == "_similarity_score"

    result_score_selected = df.select(col("_similarity_score"))
    result_score_selected_result = result_score_selected.to_polars()
    assert result_score_selected_result.schema == pl.Schema(
        {"_similarity_score": pl.Float64}
    )
    assert len(result_score_selected_result["_similarity_score"].to_list()) == len(
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
