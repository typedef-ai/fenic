import asyncio
import inspect
from inspect import iscoroutinefunction

import pytest

from fenic.api.mcp._tool_generation_utils import auto_generate_system_tools_from_tables
from fenic.core.error import ConfigurationError
from fenic.core.mcp.types import SystemTool
from tests.api.mcp.utils import create_table_with_rows


def test_auto_generate_core_tools_from_tables_missing_table_raises(local_session):
    with pytest.raises(ConfigurationError, match="do not exist"):
        auto_generate_system_tools_from_tables(["does_not_exist"], local_session, tool_namespace="TG")


def test_auto_generate_core_tools_from_tables_requires_descriptions(local_session):
    create_table_with_rows(local_session, "t_no_desc", [1, 2, 3], description=None)
    with pytest.raises(ConfigurationError, match="Missing descriptions"):
        auto_generate_system_tools_from_tables(["t_no_desc"], local_session, tool_namespace="TG")


def test_auto_generate_core_tools_from_tables_builds_tools(local_session):
    pytest.importorskip("fastmcp")
    create_table_with_rows(local_session, "t1", [1, 2, 3], description="table one")
    create_table_with_rows(local_session, "t2", [10, 20], description="table two")

    tools = auto_generate_system_tools_from_tables(["t1", "t2"], local_session, tool_namespace="Auto")

    # Expect core set: Schema, Describe, Read, Search Summary, Search Content, Analyze
    assert len(tools) == 6
    names = {t.name for t in tools}
    assert any(name.endswith("Schema") for name in names)
    assert any(name.endswith("Profile") for name in names)
    assert any(name.endswith("Read") for name in names)
    assert any(name.endswith("Search Summary") for name in names)
    assert any(name.endswith("Search Content") for name in names)
    assert any(name.endswith("Analyze") for name in names)

    for tool in tools:
        assert isinstance(tool, SystemTool)
        assert callable(tool.func)
        assert iscoroutinefunction(tool.func)
        func_signature = inspect.signature(tool.func)
        # limit and table_format are added by the MCP server wrapper
        assert "table_format" not in func_signature.parameters
        if tool.add_limit_parameter:
            assert "limit" not in func_signature.parameters

    # Sanity check: the Schema tool's callable returns a LogicalPlan we can collect
    schema_tool = next(t for t in tools if t.name.endswith("Schema"))
    plan = asyncio.run(schema_tool.func())  # type: ignore[call-arg]
    pl_df, _ = local_session._session_state.execution.collect(plan)
    assert set(pl_df.columns) == {"dataset", "schema"}
    assert sorted(pl_df.get_column("dataset").to_list()) == ["t1", "t2"]