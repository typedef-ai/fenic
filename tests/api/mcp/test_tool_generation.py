import pytest

from fenic import ColumnField, IntegerType, Schema
from fenic.api.mcp.tool_generation import auto_generate_core_tools_from_tables
from fenic.core.error import ConfigurationError


def _create_table_with_rows(session, name: str, values: list[int], description: str | None = None) -> None:
    df = session.create_dataframe({"id": values})
    # Persist table and optional description through writer (threads description into TableSink)
    if description is not None:
        df.write.save_as_table(name, mode="overwrite", description=description)
    else:
        # No description: create an empty table with schema only
        session.catalog.create_table(name, Schema([ColumnField("id", IntegerType)]))


def test_auto_generate_core_tools_from_tables_missing_table_raises(local_session):
    with pytest.raises(ConfigurationError, match="do not exist"):
        auto_generate_core_tools_from_tables(["does_not_exist"], local_session, tool_group_name="TG")


def test_auto_generate_core_tools_from_tables_requires_descriptions(local_session):
    _create_table_with_rows(local_session, "t_no_desc", [1, 2, 3], description=None)
    with pytest.raises(ConfigurationError, match="Missing descriptions"):
        auto_generate_core_tools_from_tables(["t_no_desc"], local_session, tool_group_name="TG")


def test_auto_generate_core_tools_from_tables_builds_tools(local_session):
    _create_table_with_rows(local_session, "t1", [1, 2, 3], description="table one")
    _create_table_with_rows(local_session, "t2", [10, 20], description="table two")

    tools = auto_generate_core_tools_from_tables(["t1", "t2"], local_session, tool_group_name="Auto")

    # Expect core set: Schema, Describe, Read, Search Summary, Search Content, Analyze
    assert len(tools) == 6
    names = {t.name for t in tools}
    assert any(name.endswith("Schema") for name in names)
    assert any(name.endswith("Describe") for name in names)
    assert any(name.endswith("Read") for name in names)
    assert any(name.endswith("Search Summary") for name in names)
    assert any(name.endswith("Search Content") for name in names)
    assert any(name.endswith("Analyze") for name in names)

    # Sanity check: the Schema tool's callable returns a LogicalPlan we can collect
    schema_tool = next(t for t in tools if t.name.endswith("Schema"))
    plan = schema_tool.func()  # type: ignore[call-arg]
    pl_df, _ = local_session._session_state.execution.collect(plan)
    assert set(pl_df.columns) == {"dataset", "schema"}
    assert sorted(pl_df.get_column("dataset").to_list()) == ["t1", "t2"]


