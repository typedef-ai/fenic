import inspect

import pytest

from fenic.api.mcp._tool_generation_utils import auto_generate_system_tools_from_tables
from fenic.core.error import ConfigurationError, ValidationError
from fenic.core.mcp.types import SystemTool
from tests.api.mcp.utils import create_table_from_dict, create_table_with_rows


def _system_tool(local_session, table_names: list[str], suffix: str) -> SystemTool:
    tools = auto_generate_system_tools_from_tables(
        table_names, local_session, tool_namespace="Auto"
    )
    return next(t for t in tools if t.name.endswith(suffix))


def _search_content_tool(local_session, table_name: str) -> SystemTool:
    return _system_tool(local_session, [table_name], "Search Content")


def _collect_rows(local_session, plan):
    pl_df, _ = local_session._session_state.execution.collect(plan)
    return pl_df.to_dicts()


def test_auto_generate_core_tools_from_tables_missing_table_raises(local_session):
    with pytest.raises(ConfigurationError, match="do not exist"):
        auto_generate_system_tools_from_tables(
            ["does_not_exist"], local_session, tool_namespace="TG"
        )


def test_auto_generate_core_tools_from_tables_requires_descriptions(local_session):
    create_table_with_rows(local_session, "t_no_desc", [1, 2, 3], description=None)
    with pytest.raises(ConfigurationError, match="Missing descriptions"):
        auto_generate_system_tools_from_tables(
            ["t_no_desc"], local_session, tool_namespace="TG"
        )


def test_auto_generate_core_tools_from_tables_builds_tools(local_session):
    pytest.importorskip("fastmcp")
    create_table_with_rows(local_session, "t1", [1, 2, 3], description="table one")
    create_table_with_rows(local_session, "t2", [10, 20], description="table two")

    tools = auto_generate_system_tools_from_tables(
        ["t1", "t2"], local_session, tool_namespace="Auto"
    )

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
        func_signature = inspect.signature(tool.func)
        # limit and table_format are added by the MCP server wrapper
        assert "table_format" not in func_signature.parameters
        if tool.add_limit_parameter:
            assert "limit" not in func_signature.parameters

        if tool.name.endswith(("Search Summary", "Search Content")):
            assert func_signature.parameters["search_mode"].default == "regex"

    # Sanity check: the Schema tool's callable returns a LogicalPlan we can collect
    schema_tool = next(t for t in tools if t.name.endswith("Schema"))
    plan = schema_tool.func()
    pl_df, _ = local_session._session_state.execution.collect(plan)
    assert set(pl_df.columns) == {"dataset", "schema"}
    assert sorted(pl_df.get_column("dataset").to_list()) == ["t1", "t2"]


def test_search_content_literal_mode_treats_regex_metacharacters_literally(
    local_session,
):
    create_table_from_dict(
        local_session,
        "docs",
        {
            "id": [1, 2, 3],
            "body": ["a.b", "axb", "nomatch"],
        },
        description="docs table",
    )
    search_tool = _search_content_tool(local_session, "docs")

    literal_plan = search_tool.func(
        df_name="docs",
        pattern="a.b",
        search_mode="literal",
        order_by="id",
    )
    regex_plan = search_tool.func(
        df_name="docs",
        pattern="a.b",
        search_mode="regex",
        order_by="id",
    )

    assert [row["id"] for row in _collect_rows(local_session, literal_plan)] == [1]
    assert [row["id"] for row in _collect_rows(local_session, regex_plan)] == [1, 2]


def test_search_content_literal_mode_respects_search_columns(local_session):
    create_table_from_dict(
        local_session,
        "docs",
        {
            "id": [1, 2, 3],
            "title": ["needle", "other", "other"],
            "body": ["other", "needle", "other"],
        },
        description="docs table",
    )
    search_tool = _search_content_tool(local_session, "docs")

    plan = search_tool.func(
        df_name="docs",
        pattern="needle",
        search_mode="literal",
        search_columns="body",
        order_by="id",
    )

    assert [row["id"] for row in _collect_rows(local_session, plan)] == [2]


def test_search_content_literal_mode_supports_paging(local_session):
    create_table_from_dict(
        local_session,
        "docs",
        {
            "id": [3, 1, 2],
            "body": ["hit", "hit", "hit"],
            "rank": [30, 10, 20],
        },
        description="docs table",
    )
    search_tool = _search_content_tool(local_session, "docs")

    plan = search_tool.func(
        df_name="docs",
        pattern="hit",
        search_mode="literal",
        order_by="id",
        limit="1",
        offset="1",
    )

    assert [row["id"] for row in _collect_rows(local_session, plan)] == [2]


def test_search_content_preserves_existing_positional_optional_arguments(local_session):
    create_table_from_dict(
        local_session,
        "docs",
        {
            "id": [3, 1, 2],
            "body": ["hit", "hit", "hit"],
        },
        description="docs table",
    )
    search_tool = _search_content_tool(local_session, "docs")

    plan = search_tool.func("docs", "hit", "1", "1", "id")

    assert [row["id"] for row in _collect_rows(local_session, plan)] == [2]


def test_search_content_unknown_search_mode_raises(local_session):
    create_table_from_dict(
        local_session,
        "docs",
        {
            "id": [1],
            "body": ["text"],
        },
        description="docs table",
    )
    search_tool = _search_content_tool(local_session, "docs")

    with pytest.raises(
        ValidationError, match="search_mode must be one of: regex, literal"
    ):
        search_tool.func(df_name="docs", pattern="text", search_mode="unknown")


def test_search_content_unknown_search_mode_raises_without_string_columns(
    local_session,
):
    create_table_with_rows(local_session, "metrics", [1], description="metrics table")
    search_tool = _search_content_tool(local_session, "metrics")

    with pytest.raises(
        ValidationError, match="search_mode must be one of: regex, literal"
    ):
        search_tool.func(df_name="metrics", pattern="text", search_mode="unknown")


def test_search_summary_literal_mode_counts_literal_matches_across_datasets(
    local_session,
):
    create_table_from_dict(
        local_session,
        "docs",
        {
            "id": [1, 2, 3],
            "title": ["a.b", "other", "other"],
            "body": ["other", "a.b", "axb"],
        },
        description="docs table",
    )
    create_table_from_dict(
        local_session,
        "notes",
        {
            "id": [1, 2],
            "body": ["a.b note", "plain note"],
        },
        description="notes table",
    )
    summary_tool = _system_tool(local_session, ["docs", "notes"], "Search Summary")

    plan = summary_tool.func(pattern="a.b", search_mode="literal")

    rows = {
        row["dataset"]: row["total_matches"]
        for row in _collect_rows(local_session, plan)
    }
    assert rows == {"docs": 2, "notes": 1}


def test_search_summary_regex_mode_preserves_regex_behavior_and_no_string_rows(
    local_session,
):
    create_table_from_dict(
        local_session,
        "docs",
        {
            "id": [1, 2, 3],
            "title": ["a.b", "other", "other"],
            "body": ["other", "a.b", "axb"],
        },
        description="docs table",
    )
    create_table_with_rows(
        local_session, "metrics", [1, 2, 3], description="metrics table"
    )
    summary_tool = _system_tool(local_session, ["docs", "metrics"], "Search Summary")

    plan = summary_tool.func(pattern="a.b", search_mode="regex")

    rows = {
        row["dataset"]: row["total_matches"]
        for row in _collect_rows(local_session, plan)
    }
    assert rows == {"docs": 3, "metrics": 0}


def test_search_summary_unknown_search_mode_raises_without_string_columns(
    local_session,
):
    create_table_with_rows(local_session, "metrics", [1], description="metrics table")
    summary_tool = _system_tool(local_session, ["metrics"], "Search Summary")

    with pytest.raises(
        ValidationError, match="search_mode must be one of: regex, literal"
    ):
        summary_tool.func(pattern="text", search_mode="unknown")
