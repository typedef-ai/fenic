import re

import pytest
from pydantic import BaseModel
from pydantic import ValidationError as PydValidationError

from fenic.api.functions import col, tool_param
from fenic.core.error import PlanError, ValidationError
from fenic.core.mcp._tools import bind_tool, create_pydantic_model_for_tool
from fenic.core.mcp.types import ToolParam, ToolParamConstraints
from fenic.core.types.datatypes import ArrayType, IntegerType, StringType


def test_toolparam_required_and_default_validation():
    # No default => required
    p = ToolParam(name="city", description="city filter")
    assert p.required is True

    # default provided => not required
    p2 = ToolParam(name="city", description="city filter", default_value="SF")
    assert p2.required is False

    # allowed_values with valid non-None default passes
    p3 = ToolParam(name="dept", description="dept", allowed_values=["Sales", "Eng"], default_value="Sales")
    assert p3.required is False


def test_resolve_tool_validates_unresolved_params(local_session):
    df = local_session.create_dataframe({"name": ["Alice", "Bob"], "age": [25, 30], "city": ["SF", "SEA"]})
    query = df.filter((col("age") >= tool_param("min_age", IntegerType)) & (col("city") == tool_param("city_name", StringType)))._logical_plan

    resolved = bind_tool(
        name="users_by_city",
        description="Filter users",
        params=[
            ToolParam(name="min_age", description="Minimum age"),
            ToolParam(name="city_name", description="City name"),
        ],
        result_limit=50,
        query=query,
    )

    assert {p.name for p in resolved.params} == {"min_age", "city_name"}

    # Missing param should raise PlanError
    with pytest.raises(PlanError):
        bind_tool(
            name="users_by_city",
            description="Filter users",
            params=[ToolParam(name="min_age", description="Minimum age")],
            result_limit=50,
            query=query,
        )

def test_resolve_tool_validates_mistyped_validators(local_session):
    df = local_session.create_dataframe({"name": ["Alice", "Bob"], "age": [25, 30], "city": ["SF", "SEA"]})
    query = df.filter((col("age") >= tool_param("min_age", IntegerType)) & (col("city") == tool_param("city_name", StringType)))._logical_plan

    with pytest.raises(PlanError, match="Param Validator `regex` supports data types \(StringType\), but the parameter `min_age` has data type IntegerType."):
        bind_tool(
            name="users_by_city",
            description="Filter users",
            params=[
                ToolParam(name="min_age", description="Minimum age", validator_names=["regex"]),
                ToolParam(name="city_name", description="City name", validator_names=["regex"]),
            ],
            result_limit=50,
            query=query,
        )

def test_resolve_tool_validates_missing_validators(local_session):
    df = local_session.create_dataframe({"name": ["Alice", "Bob"], "age": [25, 30], "city": ["SF", "SEA"]})
    query = df.filter((col("age") >= tool_param("min_age", IntegerType)) & (col("city") == tool_param("city_name", StringType)))._logical_plan

    with pytest.raises(PlanError, match="Could not find a ParamValidator for the following validator names: \['non_existent'\]"):
        bind_tool(
            name="users_by_city",
            description="Filter users",
            params=[
                ToolParam(name="min_age", description="Minimum age"),
                ToolParam(name="city_name", description="City name", validator_names=["non_existent"]),
            ],
            result_limit=50,
            query=query,
        )

def test_create_pydantic_model_for_tool_applies_validators(local_session):
    df = local_session.create_dataframe({"name": ["Alice", "Bob"], "age": [25, 30], "city": ["SF", "SEA"]})
    query = df.filter(
        (col("age") >= tool_param("min_age", IntegerType)) &
        (col("city") == tool_param("city_name", StringType))
    )._logical_plan

    tool = bind_tool(
        name="users_by_city",
        description="Filter users",
        params=[
            ToolParam(name="min_age", description="Minimum age"),
            ToolParam(name="city_name", description="City name", validator_names=["regex"]),
        ],
        result_limit=50,
        query=query,
    )

    Model: type[BaseModel] = create_pydantic_model_for_tool(tool)

    with pytest.raises(ValidationError, match="Unbalanced curly braces"):
        Model(city_name="{+---", min_age=25)

    with pytest.raises(ValidationError, match="Too many alternations \(21 > 20\)"):
        Model(city_name="SF|SEA|OAK|PHX|LAS|ORD|XRD|PRD|IAD|CRD|FRA|LON|UMEA|BOS|YYZ|DOG|BAT|BAN|LAP|LAX|TYO|HND", min_age=25)


def test_create_pydantic_model_for_tool_applies_field_validators(local_session):
    df = local_session.create_dataframe({"city": ["SF"], "age": [10], "user_name": ["Alice"]})
    query = df.filter(
        (col("city") == tool_param("city_name", StringType))
        & (col("age") >= tool_param("age", IntegerType))
        & (col("user_name").is_in(tool_param("user_names", ArrayType(StringType))))
    )._logical_plan

    tool = bind_tool(
        name="tool_x",
        description="",
        params=[
            ToolParam(name="city_name", description="City name", constraints=ToolParamConstraints(pattern="^SF$")),
            ToolParam(name="age", description="Age", constraints=ToolParamConstraints(gt=0, lt=120, multiple_of=2)),
            ToolParam(name="user_names", description="User names", constraints=ToolParamConstraints(min_length=1, max_length=5)),
        ],
        result_limit=10,
        query=query,
    )

    Model: type[BaseModel] = create_pydantic_model_for_tool(tool)
    #should pass validation
    Model(city_name="SF", age=10, user_names=["Alice", "Bob"])
    with pytest.raises(PydValidationError, match=re.escape("String should match pattern '^SF$'")):
        Model(city_name="SEA", age=10, user_names=["Alice", "Bob"])

    with pytest.raises(PydValidationError, match=re.escape("Input should be greater than 0")):
        Model(city_name="SF", age=0, user_names=["Alice", "Bob"])

    with pytest.raises(PydValidationError, match=re.escape("Input should be a multiple of 2")):
        Model(city_name="SF", age=11, user_names=["Alice", "Bob"])

    with pytest.raises(PydValidationError, match=re.escape("List should have at most 5 items after validation, not 6")):
        Model(city_name="SF", age=10, user_names=["Alice", "Bob", "Charlie", "David", "Eve", "Frank"])

    with pytest.raises(PydValidationError, match=re.escape("List should have at least 1 item after validation, not 0")):
        Model(city_name="SF", age=10, user_names=[])


def test_create_pydantic_model_for_tool_defaults_and_required(local_session):
    df = local_session.create_dataframe({"city": ["SF"], "age": [10]})
    query = df.filter(col("city") == tool_param("city_name", StringType))._logical_plan

    tool = bind_tool(
        name="tool_x",
        description="",
        params=[
            ToolParam(name="city_name", description="City name", default_value="SF"),
        ],
        result_limit=10,
        query=query,
    )

    Model: type[BaseModel] = create_pydantic_model_for_tool(tool)
    m = Model()  # default applies
    assert m.city_name == "SF"

    # Now a required param model: no default
    tool2 = bind_tool(
        name="tool_y",
        description="",
        params=[ToolParam(name="city_name", description="City name")],
        result_limit=10,
        query=query,
    )
    Model2: type[BaseModel] = create_pydantic_model_for_tool(tool2)

    with pytest.raises(PydValidationError):
        Model2()  # required field missing
    m2 = Model2(city_name="SEA")
    assert m2.city_name == "SEA"


def test_create_pydantic_model_with_allowed_values_default(local_session):
    df = local_session.create_dataframe({"dept": ["Sales"]})
    query = df.filter(col("dept") == tool_param("department", StringType))._logical_plan

    bound_tool = bind_tool(
        name="dept_tool",
        description="",
        params=[
            ToolParam(name="department", description="Department", allowed_values=["Sales", "Eng"], default_value="Sales"),
        ],
        result_limit=5,
        query=query,
    )

    Model: type[BaseModel] = create_pydantic_model_for_tool(bound_tool)

    # Default applies
    m = Model()
    assert m.department == "Sales"

    # Allowed override works
    m_ok = Model(department="Eng")
    assert m_ok.department == "Eng"

    # Disallowed value rejected
    with pytest.raises(PydValidationError):
        Model(department="HR")


def test_create_pydantic_model_with_allowed_values_required(local_session):
    df = local_session.create_dataframe({"dept": ["Sales"]})
    query = df.filter(col("dept") == tool_param("department", StringType))._logical_plan

    bound_tool = bind_tool(
        name="dept_tool_req",
        description="",
        params=[
            ToolParam(name="department", description="Department", allowed_values=["Sales", "Eng"]),
        ],
        result_limit=5,
        query=query,
    )

    Model: type[BaseModel] = create_pydantic_model_for_tool(bound_tool)

    # Missing required field rejected
    with pytest.raises(PydValidationError):
        Model()

    # Allowed value accepted
    m_ok = Model(department="Sales")
    assert m_ok.department == "Sales"

    # Disallowed value rejected
    with pytest.raises(PydValidationError):
        Model(department="HR")


def test_create_pydantic_model_with_allowed_values_ints(local_session):
    df = local_session.create_dataframe({"priority": [1]})
    query = df.filter(col("priority") == tool_param("priority", IntegerType))._logical_plan

    resolved = bind_tool(
        name="priority_tool",
        description="",
        params=[
            ToolParam(name="priority", description="Priority", allowed_values=[1, 2], default_value=1),
        ],
        result_limit=5,
        query=query,
    )
    Model: type[BaseModel] = create_pydantic_model_for_tool(resolved)

    # Default applies
    m = Model()
    assert m.priority == 1

    # Allowed override works
    m_ok = Model(priority=2)
    assert m_ok.priority == 2

    # Disallowed value rejected
    with pytest.raises(PydValidationError):
        Model(priority=3)
