import asyncio

import pytest

pytest.importorskip("fastmcp")
from fastmcp.tools import Tool

from fenic import SystemTool, SystemToolConfig
from fenic.api.functions import col, tool_param
from fenic.api.mcp._tool_generation_utils import auto_generate_system_tools_from_tables
from fenic.api.mcp.server import create_mcp_server
from fenic.api.session.session import Session
from fenic.core._utils.misc import to_snake_case
from fenic.core.mcp._tools import bind_tool
from fenic.core.mcp.types import ToolParam, ToolParamConstraints
from fenic.core.types.datatypes import ArrayType, IntegerType, StringType
from tests.api.mcp.utils import create_table_with_rows


def test_server_generation_with_parameterized_tools(local_session: Session):
    df = local_session.create_dataframe({"city": ["SF"], "age": [10], "user_name": ["Alice"]})
    query = df.filter(
        (col("city") == tool_param("city_name", StringType))
        & (col("age") >= tool_param("age", IntegerType))
        & (col("user_name").is_in(tool_param("user_names", ArrayType(StringType))))
    )._logical_plan

    parameterized_tool = bind_tool(
        name="tool_x",
        description="table one",
        params=[
            ToolParam(name="city_name", description="City name", constraints=ToolParamConstraints(pattern="^SF$")),
            ToolParam(name="age", description="Age", constraints=ToolParamConstraints(gt=0, lt=120, multiple_of=2)),
            ToolParam(name="user_names", description="User names", constraints=ToolParamConstraints(min_length=1, max_length=5)),
        ],
        result_limit=10,
        query=query,
    )

    server = create_mcp_server(local_session, "Test Server", user_defined_tools=[parameterized_tool])
    server_tools = asyncio.run(server.mcp.get_tools())
    assert len(server_tools) == 1
    parameter_schema = server_tools["tool_x"].parameters['properties']
    city_name_param = parameter_schema['city_name']
    assert city_name_param['type'] == 'string'
    assert city_name_param['pattern'] == '^SF$'
    assert city_name_param['description'] == "City name"
    age_param = parameter_schema['age']
    assert age_param['type'] == 'integer'
    assert age_param['exclusiveMinimum'] == 0
    assert age_param['exclusiveMaximum'] == 120
    assert age_param['multipleOf'] == 2
    assert age_param['description'] == "Age"
    user_names_param = parameter_schema['user_names']
    assert user_names_param['type'] == 'array'
    assert user_names_param['items']['type'] == 'string'
    assert user_names_param['maxItems'] == 5
    assert user_names_param['minItems'] == 1
    assert user_names_param['description'] == "User names"

def test_server_generation_with_system_tools(local_session: Session):
    create_table_with_rows(local_session, "t1", [1, 2, 3], description="table one")
    create_table_with_rows(local_session, "t2", [10, 20], description="table two")
    tools = auto_generate_system_tools_from_tables(["t1", "t2"], local_session, tool_namespace="Auto")
    server = create_mcp_server(local_session, "Test Server", system_tools=SystemToolConfig(
        table_names=["t1", "t2"],
        tool_namespace="Auto",
        max_result_rows=100
    ))
    server_tools = asyncio.run(server.mcp.get_tools())
    _validate_server_tools(server_tools, tools)


def test_catalog_tables_server_generation(local_session: Session):
    pytest.importorskip("fastmcp")
    create_table_with_rows(local_session, "t1", [1, 2, 3], description="table one")
    create_table_with_rows(local_session, "t2", [10, 20], description="table two")
    tools = auto_generate_system_tools_from_tables(["t1", "t2"], local_session, tool_namespace="Auto")
    server = create_mcp_server(local_session, "Test Server", system_tools=SystemToolConfig(
        table_names=local_session.catalog.list_tables(),
        tool_namespace="Auto",
        max_result_rows=100
    ))
    server_tools = asyncio.run(server.mcp.get_tools())
    _validate_server_tools(server_tools, tools)

def _validate_server_tools(server_tools: dict[str, Tool], reference_system_tools: list[SystemTool]):
    assert len(server_tools) == len(reference_system_tools)
    for expected_tool in reference_system_tools:
        snake_case_name = to_snake_case(expected_tool.name)
        assert snake_case_name in server_tools
        server_tool = server_tools[snake_case_name]
        assert server_tool.annotations.readOnlyHint == expected_tool.read_only
        assert server_tool.annotations.openWorldHint == expected_tool.open_world
        assert server_tool.annotations.destructiveHint == expected_tool.destructive
        assert server_tool.annotations.idempotentHint == expected_tool.idempotent
        assert server_tool.title == expected_tool.name
        assert server_tool.description == expected_tool.description
        # check that server added limit and table_format parameters
        tool_params = server_tool.parameters['properties']
        assert 'table_format' in tool_params
        assert tool_params['table_format']['default'] == expected_tool.default_table_format
        if expected_tool.add_limit_parameter and expected_tool.max_result_limit:
            assert 'limit' in tool_params
            assert tool_params['limit']['default'] == expected_tool.max_result_limit