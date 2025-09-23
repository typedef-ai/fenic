import asyncio

import pytest

from fenic.api.mcp.server import create_mcp_server
from fenic.api.mcp.tool_generation import auto_generate_system_tools_from_tables
from fenic.api.session.session import Session
from fenic.core._utils.misc import to_snake_case
from tests.api.mcp.utils import create_table_with_rows


def test_server_generation(local_session: Session):
    pytest.importorskip("fastmcp")
    create_table_with_rows(local_session, "t1", [1, 2, 3], description="table one")
    create_table_with_rows(local_session, "t2", [10, 20], description="table two")

    tools = auto_generate_system_tools_from_tables(["t1", "t2"], local_session, tool_group_name="Auto")

    server = create_mcp_server(local_session, "Test Server", system_tools=tools)
    server_tools = asyncio.run(server.mcp.get_tools())
    assert len(server_tools) == len(tools)
    for tool in tools:
        snake_case_name = to_snake_case(tool.name)
        assert snake_case_name in server_tools
        server_tool = server_tools[snake_case_name]
        assert server_tool.annotations.readOnlyHint == tool.read_only
        assert server_tool.annotations.openWorldHint == tool.open_world
        assert server_tool.annotations.destructiveHint == tool.destructive
        assert server_tool.annotations.idempotentHint == tool.idempotent
        assert server_tool.title == tool.name
        assert server_tool.description == tool.description
        # check that server added limit and table_format parameters
        tool_params = server_tool.parameters['properties']
        assert 'table_format' in tool_params
        assert tool_params['table_format']['default'] == tool.default_table_format
        if tool.add_limit_parameter and tool.max_result_limit:
            assert 'limit' in tool_params
            assert tool_params['limit']['default'] == tool.max_result_limit