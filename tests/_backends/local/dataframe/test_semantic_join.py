import polars as pl
import pytest

from fenic import JoinExample, JoinExampleCollection, OpenAIEmbeddingModel, col
from fenic.api.session import (
    SemanticConfig,
    Session,
    SessionConfig,
)
from fenic.core.error import PlanError, ValidationError


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


def test_semantic_join(local_session):
    left, right = _create_semantic_join_dataframe(local_session)
    join_instruction = "Taking {course_name:left} will help me learn {skill:right}"
    result = left.semantic.join(right, join_instruction)
    result = result.to_polars()
    assert result.schema == {
        "course_id": pl.Int64,
        "course_name": pl.String,
        "other_col_left": pl.String,
        "skill_id": pl.Int64,
        "skill": pl.String,
        "other_col_right": pl.String,
    }


def test_semantic_join_with_none(local_session):
    """Test that we can join a dataframe with a None value.
    Note: this will produce the same result as the test above, but we'll evaluate
          2 additional rows (None, Computer Science) and (None, Math) which should
          not match anything and shouldn't be sent to the LLM for processing.
    """
    left, right = _create_semantic_join_dataframe_with_none(local_session)
    join_instruction = "Taking {course_name:left} will help me learn {skill:right}"
    result = left.semantic.join(right, join_instruction)
    result = result.to_polars()
    assert result.schema == {
        "course_id": pl.Int64,
        "course_name": pl.String,
        "other_col_left": pl.String,
        "skill_id": pl.Int64,
        "skill": pl.String,
        "other_col_right": pl.String,
    }


def test_semantic_join_with_right_none(local_session):
    """Test that we can join a dataframe a none value on the right of the joint.
    In this case although there are 3 rows in the right dataframe, only 2 will be
    used for the join.
    """
    left, right = _create_semantic_join_dataframe_with_right_none(local_session)
    join_instruction = "Taking {course_name:left} will help me learn {skill:right}"
    result = left.semantic.join(right, join_instruction)
    result = result.to_polars()
    assert result.schema == {
        "course_id": pl.Int64,
        "course_name": pl.String,
        "other_col_left": pl.String,
        "skill_id": pl.Int64,
        "skill": pl.String,
        "other_col_right": pl.String,
    }


def test_semantic_join_duplicate_columns(local_session):
    left = local_session.create_dataframe(
        {
            "id": [1, 2, 3, 4, 5, 6],
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
            "id": [1, 2],
            "skill": ["Math", "Computer Science"],
            "other_col_right": ["g", "h"],
        }
    )
    join_instruction = "Taking {course_name:left} will help me learn {skill:right}"
    with pytest.raises(PlanError, match="Duplicate column names"):
        left.semantic.join(right, join_instruction)


def test_semantic_join_with_examples(local_session):
    left, right = _create_semantic_join_dataframe(local_session)
    collection = JoinExampleCollection()
    collection.create_example(
        JoinExample(
            left="Linear Algebra",
            right="Math",
            output=True,
        ),
    ).create_example(
        JoinExample(
            left="Intensive Study of a Culture: Pirates",
            right="Computer Science",
            output=False,
        ),
    )
    join_instruction = "Taking {course_name:left} will help me learn {skill:right}"
    result = left.semantic.join(right, join_instruction, examples=collection)
    result = result.to_polars()
    assert result.schema == {
        "course_id": pl.Int64,
        "course_name": pl.String,
        "other_col_left": pl.String,
        "skill_id": pl.Int64,
        "skill": pl.String,
        "other_col_right": pl.String,
    }


def test_semantic_join_empty_result(local_session):
    left, right = _create_semantic_join_dataframe(local_session)
    empty_left = left.filter(col("course_name").is_null())
    join_instruction = "Taking {course_name:left} will help me learn {skill:right}"
    result = empty_left.semantic.join(right, join_instruction)
    result = result.to_polars()
    assert result.is_empty()
    assert result.schema == {
        "course_id": pl.Int64,
        "course_name": pl.String,
        "other_col_left": pl.String,
        "skill_id": pl.Int64,
        "skill": pl.String,
        "other_col_right": pl.String,
    }

    empty_right = right.filter(col("skill").is_null())
    result = left.semantic.join(empty_right, join_instruction)
    result = result.to_polars()
    assert result.is_empty()
    assert result.schema == {
        "course_id": pl.Int64,
        "course_name": pl.String,
        "other_col_left": pl.String,
        "skill_id": pl.Int64,
        "skill": pl.String,
        "other_col_right": pl.String,
    }

def test_semantic_join_without_models():
    """Test semantic.join() method without models."""
    session_config = SessionConfig(
        app_name="semantic_join_without_models",
    )
    session = Session.get_or_create(session_config)
    with pytest.raises(ValidationError, match="No language models configured."):
        session.create_dataframe({"notes1": ["hello"]}).semantic.join(session.create_dataframe({"notes2": ["hello"]}), "Taking {notes1:left} will help me learn {notes2:right}")
    session.stop()
    session_config = SessionConfig(
        app_name="semantic_join_with_models",
        semantic=SemanticConfig(
            embedding_models={"oai-small": OpenAIEmbeddingModel(model_name="text-embedding-3-small", rpm=3000, tpm=1_000_000)},
        ),
    )
    session = Session.get_or_create(session_config)
    with pytest.raises(ValidationError, match="No language models configured."):
        session.create_dataframe({"notes1": ["hello"]}).semantic.join(session.create_dataframe({"notes2": ["hello"]}), "Taking {notes1:left} will help me learn {notes2:right}")
    session.stop()
