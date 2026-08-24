import polars as pl
import pytest

from fenic import (
    ColumnField,
    IntegerType,
    JoinExample,
    JoinExampleCollection,
    OpenAIEmbeddingModel,
    StringType,
    col,
    text,
)
from fenic._constants import MAX_MODEL_CLIENT_TIMEOUT
from fenic.api.session import (
    SemanticConfig,
    Session,
    SessionConfig,
)
from fenic.core.error import InvalidExampleCollectionError, PlanError, ValidationError


def _create_semantic_join_dataframe(local_session: Session):
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


def _create_semantic_join_dataframe_with_none(local_session: Session):
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


def _create_semantic_join_dataframe_with_right_none(local_session: Session):
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
            "skill": ["Math", None],
            "other_col_right": ["h", "i"],
        }
    )
    return left, right


@pytest.mark.requires_provider_key
def test_semantic_join(local_session: Session):
    left, right = _create_semantic_join_dataframe(local_session)
    join_instruction = "Taking {{left_on}} will help me learn {{right_on}}"
    result = left.semantic.join(right, join_instruction, left_on=col("course_name"), right_on=col("skill"))
    assert result.schema.column_fields == [
        ColumnField(name="course_id", data_type=IntegerType),
        ColumnField(name="course_name", data_type=StringType),
        ColumnField(name="other_col_left", data_type=StringType),
        ColumnField(name="skill_id", data_type=IntegerType),
        ColumnField(name="skill", data_type=StringType),
        ColumnField(name="other_col_right", data_type=StringType),
    ]
    result = result.to_polars()
    assert result.schema == {
        "course_id": pl.Int64,
        "course_name": pl.String,
        "other_col_left": pl.String,
        "skill_id": pl.Int64,
        "skill": pl.String,
        "other_col_right": pl.String,
    }


@pytest.mark.requires_provider_key
def test_semantic_join_with_none(local_session: Session):
    """Test that we can join a dataframe with a None value.
    Note: this will produce the same result as the test above, but we'll evaluate
          2 additional rows (None, Computer Science) and (None, Math) which should
          not match anything and shouldn't be sent to the LLM for processing.
    """
    left, right = _create_semantic_join_dataframe_with_none(local_session)
    join_instruction = "Taking {{left_on}} will help me learn {{right_on}}"
    result = left.semantic.join(right, join_instruction, left_on=col("course_name"), right_on=col("skill"))
    assert result.schema.column_fields == [
        ColumnField(name="course_id", data_type=IntegerType),
        ColumnField(name="course_name", data_type=StringType),
        ColumnField(name="other_col_left", data_type=StringType),
        ColumnField(name="skill_id", data_type=IntegerType),
        ColumnField(name="skill", data_type=StringType),
        ColumnField(name="other_col_right", data_type=StringType),
    ]
    result = result.to_polars()
    assert result.schema == {
        "course_id": pl.Int64,
        "course_name": pl.String,
        "other_col_left": pl.String,
        "skill_id": pl.Int64,
        "skill": pl.String,
        "other_col_right": pl.String,
    }


@pytest.mark.requires_provider_key
def test_semantic_join_with_right_none(local_session: Session):
    """Test that we can join a dataframe a none value on the right of the joint.
    In this case although there are 3 rows in the right dataframe, only 2 will be
    used for the join.
    """
    left, right = _create_semantic_join_dataframe_with_right_none(local_session)
    join_instruction = "Taking {{left_on}} will help me learn {{right_on}}"
    result = left.semantic.join(right, join_instruction, left_on=col("course_name"), right_on=col("skill"))
    result = result.to_polars()
    assert result.schema == {
        "course_id": pl.Int64,
        "course_name": pl.String,
        "other_col_left": pl.String,
        "skill_id": pl.Int64,
        "skill": pl.String,
        "other_col_right": pl.String,
    }


def test_semantic_join_duplicate_columns(local_session: Session):
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
    join_instruction = "Taking {{left_on}} will help me learn {{right_on}}"
    with pytest.raises(PlanError, match="Duplicate column names"):
        left.semantic.join(right, join_instruction, left_on=col("course_name"), right_on=col("skill"))


@pytest.mark.requires_provider_key
def test_semantic_join_with_examples(local_session: Session):
    left, right = _create_semantic_join_dataframe(local_session)
    collection = JoinExampleCollection()
    collection.create_example(
        JoinExample(
            left_on="Linear Algebra",
            right_on="Math",
            output=True,
        ),
    ).create_example(
        JoinExample(
            left_on="Intensive Study of a Culture: Pirates",
            right_on="Computer Science",
            output=False,
        ),
    )
    join_instruction = "Taking {{left_on}} will help me learn {{right_on}}"
    result = left.semantic.join(right, join_instruction, left_on=col("course_name"), right_on=col("skill"), examples=collection)
    result = result.to_polars()
    assert result.schema == {
        "course_id": pl.Int64,
        "course_name": pl.String,
        "other_col_left": pl.String,
        "skill_id": pl.Int64,
        "skill": pl.String,
        "other_col_right": pl.String,
    }
    bad_examples = JoinExampleCollection()
    bad_examples.create_example(JoinExample(
        left_on=True,
        right_on="Math",
        output=True,
    ))
    with pytest.raises(InvalidExampleCollectionError, match="Field 'left_on' type mismatch: operator expects"):
        left.semantic.join(right, join_instruction, left_on=col("course_name"), right_on=col("skill"), examples=bad_examples)


def test_semantic_join_empty_result(local_session: Session):
    left, right = _create_semantic_join_dataframe(local_session)
    empty_left = left.filter(col("course_name").is_null())
    join_instruction = "Taking {{left_on}} will help me learn {{right_on}}"
    result = empty_left.semantic.join(right, join_instruction, left_on=col("course_name"), right_on=col("skill"))
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
    result = left.semantic.join(empty_right, join_instruction, left_on=col("course_name"), right_on=col("skill"))
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


def test_semantic_join_golden_output_with_nulls_and_explicit_sort(local_session: Session, monkeypatch):
    from fenic._backends.local.semantic_operators.predicate import Predicate

    left = local_session.create_dataframe(
        {
            "course_id": [3, 1, 2, 4],
            "course_name": ["Intro to Computer Networks", "Riemann Geometry", "Operating Systems", None],
            "left_payload": ["networks", "math", "systems", "missing"],
        }
    )
    right = local_session.create_dataframe(
        {
            "skill_id": [20, 10, 30],
            "skill": ["Computer Science", "Math", None],
            "right_payload": ["cs", "math", "missing"],
        }
    )

    def fake_execute(self):
        return pl.Series(
            [
                ("Riemann Geometry" in rendered and "Math" in rendered)
                or ("Operating Systems" in rendered and "Computer Science" in rendered)
                or ("Intro to Computer Networks" in rendered and "Computer Science" in rendered)
                for rendered in self.input.to_list()
            ]
        )

    monkeypatch.setattr(Predicate, "execute", fake_execute)

    result_df = left.semantic.join(
        right,
        "Does {{left_on}} belong to {{right_on}}?",
        left_on=col("course_name"),
        right_on=col("skill"),
    )
    assert result_df.schema.column_fields == [
        ColumnField(name="course_id", data_type=IntegerType),
        ColumnField(name="course_name", data_type=StringType),
        ColumnField(name="left_payload", data_type=StringType),
        ColumnField(name="skill_id", data_type=IntegerType),
        ColumnField(name="skill", data_type=StringType),
        ColumnField(name="right_payload", data_type=StringType),
    ]

    # semantic.join survivor order is tied to block iteration; sort explicitly so
    # this golden-output test pins correspondence, not incidental execution order.
    result = result_df.to_polars().sort(["course_id", "skill_id"])
    assert result.schema == {
        "course_id": pl.Int64,
        "course_name": pl.String,
        "left_payload": pl.String,
        "skill_id": pl.Int64,
        "skill": pl.String,
        "right_payload": pl.String,
    }
    assert len(result) == 3
    assert result.to_dicts() == [
        {
            "course_id": 1,
            "course_name": "Riemann Geometry",
            "left_payload": "math",
            "skill_id": 10,
            "skill": "Math",
            "right_payload": "math",
        },
        {
            "course_id": 2,
            "course_name": "Operating Systems",
            "left_payload": "systems",
            "skill_id": 20,
            "skill": "Computer Science",
            "right_payload": "cs",
        },
        {
            "course_id": 3,
            "course_name": "Intro to Computer Networks",
            "left_payload": "networks",
            "skill_id": 20,
            "skill": "Computer Science",
            "right_payload": "cs",
        },
    ]


def test_semantic_join_all_false_golden_empty_result(local_session: Session, monkeypatch):
    from fenic._backends.local.semantic_operators.predicate import Predicate

    monkeypatch.setattr(
        Predicate,
        "execute",
        lambda self: pl.Series([False for _ in self.input.to_list()]),
    )

    left, right = _create_semantic_join_dataframe(local_session)
    result = left.semantic.join(
        right,
        "Taking {{left_on}} will help me learn {{right_on}}",
        left_on=col("course_name"),
        right_on=col("skill"),
    ).to_polars()
    assert result.is_empty()
    assert result.schema == {
        "course_id": pl.Int64,
        "course_name": pl.String,
        "other_col_left": pl.String,
        "skill_id": pl.Int64,
        "skill": pl.String,
        "other_col_right": pl.String,
    }


def test_semantic_join_with_derived_columns(local_session: Session):
    left, right = _create_semantic_join_dataframe(local_session)
    join_instruction = "Taking {{left_on}} will help me learn {{right_on}}"
    result = left.semantic.join(right, join_instruction, left_on=text.upper(col("course_name")), right_on=text.upper(col("skill")).alias("skill"))
    assert result.schema.column_fields == [
        ColumnField(name="course_id", data_type=IntegerType),
        ColumnField(name="course_name", data_type=StringType),
        ColumnField(name="other_col_left", data_type=StringType),
        ColumnField(name="skill_id", data_type=IntegerType),
        ColumnField(name="skill", data_type=StringType),
        ColumnField(name="other_col_right", data_type=StringType),
    ]
    for skill in right.to_polars()["skill"]:
        skill_not_all_upper = not all(char.isupper() for char in skill)
        assert skill_not_all_upper


@pytest.mark.requires_provider_key
def test_semantic_join_without_models(tmp_path):
    """Test semantic.join() method without models."""
    session_config = SessionConfig(
        app_name="semantic_join_without_models",
        db_path=tmp_path,
    )
    session = Session.get_or_create(session_config)
    with pytest.raises(ValidationError, match="No language models configured."):
        session.create_dataframe({"notes1": ["hello"]}).semantic.join(session.create_dataframe({"notes2": ["hello"]}), "Taking {{left_on}} will help me learn {{right_on}}", left_on=col("notes1"), right_on=col("notes2"))
    session.stop(skip_usage_summary=True)
    session_config = SessionConfig(
        app_name="semantic_join_with_models",
        semantic=SemanticConfig(
            embedding_models={"oai-small": OpenAIEmbeddingModel(model_name="text-embedding-3-small", rpm=3000, tpm=1_000_000)},
        ),
        db_path=tmp_path,
    )
    session = Session.get_or_create(session_config)
    with pytest.raises(ValidationError, match="No language models configured."):
        session.create_dataframe({"notes1": ["hello"]}).semantic.join(session.create_dataframe({"notes2": ["hello"]}), "Taking {{left_on}} will help me learn {{right_on}}", left_on=col("notes1"), right_on=col("notes2"))
    session.stop(skip_usage_summary=True)

def test_semantic_join_invalid_prompt(local_session: Session):
    left, right = _create_semantic_join_dataframe(local_session)
    with pytest.raises(ValidationError, match="The `predicate` argument to `semantic.join` must contain exactly the variables 'left_on' and 'right_on'."):
        left.semantic.join(right, "", left_on=col("course_name"), right_on=col("skill"))

    with pytest.raises(ValidationError, match="The `predicate` argument to `semantic.join` must contain exactly the variables 'left_on' and 'right_on'."):
        left.semantic.join(right, "{{left_on}}", left_on=col("course_name"), right_on=col("skill"))

    with pytest.raises(ValidationError, match="The `predicate` argument to `semantic.join` must contain exactly the variables 'left_on' and 'right_on'."):
        left.semantic.join(right, "{{right_on}}", left_on=col("course_name"), right_on=col("skill"))

    with pytest.raises(ValidationError, match="The `predicate` argument to `semantic.join` must contain exactly the variables 'left_on' and 'right_on'."):
        left.semantic.join(right, "{{left_on}} {{right_on}} {{foo}}", left_on=col("course_name"), right_on=col("skill"))

def test_semantic_join_fails_with_bad_request_timeouts(local_session: Session):
    left, right = _create_semantic_join_dataframe(local_session)
    with pytest.raises(ValidationError, match="The `request_timeout` argument must be a positive number."):
        left.semantic.join(right, "Taking {{left_on}} will help me learn {{right_on}}", left_on=col("course_name"), right_on=col("skill"), request_timeout=0)
    with pytest.raises(ValidationError, match=f"The `request_timeout` argument can't be greater than the system's max timeout of {MAX_MODEL_CLIENT_TIMEOUT} seconds."):
        left.semantic.join(right, "Taking {{left_on}} will help me learn {{right_on}}", left_on=col("course_name"), right_on=col("skill"), request_timeout=MAX_MODEL_CLIENT_TIMEOUT + 1)