import pytest

from fenic.api.dataframe.dataframe import DataFrame
from fenic.api.functions import col, tool_param
from fenic.core.error import PlanError, TypeMismatchError
from fenic.core.mcp._binder import bind_parameters, collect_unresolved_parameters
from fenic.core.mcp.types import BoundToolParam
from fenic.core.types.datatypes import ArrayType, IntegerType, StringType


def test_bind_parameters_replaces_unresolved_and_executes(local_session):
    df = local_session.create_dataframe(
        {"name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35], "city": ["SF", "SF", "SEA"]}
    )
    unresolved_df = df.filter(
        (col("age") >= tool_param("min_age", IntegerType)) &
        ((col("city") == tool_param("city_name", StringType)) | (col("city").contains(tool_param("city_name", StringType))))
    )

    # Ensure placeholders are present before binding
    assert len(collect_unresolved_parameters(unresolved_df._logical_plan).keys()) == 2

    bound_plan = bind_parameters(unresolved_df._logical_plan, {"min_age": 30, "city_name": "SF"}, [])
    bound_df = DataFrame._from_logical_plan(bound_plan, local_session._session_state)

    assert len(collect_unresolved_parameters(bound_plan)) == 0
    result = bound_df.to_pylist()
    assert len(result) == 1
    assert result[0]["name"] == "Bob"


def test_bind_parameters_missing_param_raises(local_session):
    df = local_session.create_dataframe({"age": [1, 2, 3]})
    unresolved_df = df.filter(col("age") > tool_param("min_age", IntegerType))

    with pytest.raises(PlanError, match="Missing parameter values"):
        bind_parameters(unresolved_df._logical_plan, {}, [])


def test_bind_parameters_with_tool_param_default(local_session):
    df = local_session.create_dataframe({"age": [1, 2, 3]})
    unresolved_df = df.filter(col("age") > tool_param("min_age", IntegerType))
    bound_plan = bind_parameters(
        unresolved_df._logical_plan,
        {},
        [
            BoundToolParam(
                name="min_age",
                description="",
                data_type=IntegerType,
                required=True,
                has_default=True,
                allowed_values=None,
                default_value=2,
            )
        ],
    )

    assert len(collect_unresolved_parameters(bound_plan)) == 0
    bound_df = DataFrame._from_logical_plan(bound_plan, local_session._session_state)
    result = bound_df.to_pylist()
    # With default 2, all 3 rows have age > 2? Only 3
    assert {r["age"] for r in result} == {3}


def test_bind_parameters_type_mismatch_raises(local_session):
    df = local_session.create_dataframe({"age": [1, 2, 3]})
    unresolved_df = df.filter(col("age") > tool_param("min_age", IntegerType))

    with pytest.raises(TypeMismatchError, match="incompatible type"):
        bind_parameters(
            unresolved_df._logical_plan,
            {"min_age": "thirty"},
            [
                BoundToolParam(
                    name="min_age",
                    description="",
                    data_type=IntegerType,
                    required=True,
                    has_default=False,
                    allowed_values=None,
                    default_value=None,
                )
            ],
        )


def test_bind_parameters_none_value_raises(local_session):
    df = local_session.create_dataframe({"age": [1, 2, 3]})
    unresolved_df = df.filter(col("age") > tool_param("min_age", IntegerType))

    with pytest.raises(PlanError, match="Failed to infer type"):
        bind_parameters(
            unresolved_df._logical_plan,
            {"min_age": None},
            [
                BoundToolParam(
                    name="min_age",
                    description="",
                    data_type=IntegerType,
                    required=True,
                    has_default=False,
                    allowed_values=None,
                    default_value=None,
                )
            ],
        )


def test_bind_parameters_list_type_mismatch_raises(local_session):
    df = local_session.create_dataframe({"arr": [[1, 2], [3], [4]]})
    # Expect an array of integers
    unresolved_df = df.filter(col("arr") == tool_param("vals", ArrayType(IntegerType)))

    # Provide floats -> inferred ArrayType(FloatType) != ArrayType(IntegerType)
    with pytest.raises(TypeMismatchError, match="incompatible type"):
        bind_parameters(
            unresolved_df._logical_plan,
            {"vals": [1.0, 2.0]},
            [
                BoundToolParam(
                    name="vals",
                    description="",
                    data_type=ArrayType(IntegerType),
                    required=True,
                    has_default=False,
                    allowed_values=None,
                    default_value=None,
                )
            ],
        )


def test_bind_no_unresolved_params_is_noop(local_session):
    df = local_session.create_dataframe({"age": [1, 2, 3]})
    # No tool_param used
    plan = df.filter(col("age") > 1)._logical_plan

    assert len(collect_unresolved_parameters(plan)) == 0
    bound = bind_parameters(plan, {"extra": 1}, [])
    # No unresolved should remain; and function should be a no-op
    assert len(collect_unresolved_parameters(bound)) == 0


def test_bind_extra_params_ignored(local_session):
    df = local_session.create_dataframe({"age": [1, 2, 3]})
    unresolved_df = df.filter(col("age") > tool_param("min_age", IntegerType))

    # Provide the needed param plus an extra unused one; should succeed
    bound_plan = bind_parameters(
        unresolved_df._logical_plan,
        {"min_age": 2, "unused": 123},
        [],
    )
    assert len(collect_unresolved_parameters(bound_plan)) == 0
