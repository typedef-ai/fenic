import pytest

from fenic.api.dataframe.dataframe import DataFrame
from fenic.api.functions import col, param
from fenic.core._logical_plan import bind_parameters, collect_unresolved_parameter_names
from fenic.core.error import PlanError, TypeMismatchError
from fenic.core.types.datatypes import IntegerType, StringType


def test_bind_parameters_replaces_unresolved_and_executes(local_session):
    df = local_session.create_dataframe(
        {"name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35], "city": ["SF", "SF", "SEA"]}
    )
    unresolved_df = df.filter(
        (col("age") >= param("min_age", IntegerType)) & 
        ((col("city") == param("city_name", StringType)) | (col("city").contains(param("city_name", StringType)))))

    # Ensure placeholders are present before binding
    assert collect_unresolved_parameter_names(unresolved_df._logical_plan) == {"min_age", "city_name"}

    bound_plan = bind_parameters(unresolved_df._logical_plan, {"min_age": 30, "city_name": "SF"})
    bound_df = DataFrame._from_logical_plan(bound_plan, local_session._session_state)

    assert collect_unresolved_parameter_names(bound_plan) == set()
    result = bound_df.to_pylist()
    assert len(result) == 1
    assert result[0]["name"] == "Bob"


def test_bind_parameters_missing_param_raises(local_session):
    df = local_session.create_dataframe({"age": [1, 2, 3]})
    unresolved_df = df.filter(col("age") > param("min_age", IntegerType))

    with pytest.raises(PlanError, match="Missing parameter values"):
        bind_parameters(unresolved_df._logical_plan, {})


def test_bind_parameters_type_mismatch_raises(local_session):
    df = local_session.create_dataframe({"age": [1, 2, 3]})
    unresolved_df = df.filter(col("age") > param("min_age", IntegerType))

    with pytest.raises(TypeMismatchError, match="incompatible type"):
        bind_parameters(unresolved_df._logical_plan, {"min_age": "thirty"})
