"""Integration tests for the Fenic-native documentation server."""

import uuid

import pytest
from fastmcp import Client
from fenic_mcp.server.native import create_native_server, register_docs_tools

import fenic as fc


@pytest.fixture
def docs_session(tmp_path, monkeypatch):
    """Create a minimal persisted documentation catalog."""
    monkeypatch.chdir(tmp_path)
    session = fc.Session.get_or_create(
        fc.SessionConfig(app_name=f"docs-test-{uuid.uuid4().hex}")
    )
    session.create_dataframe(
        [
            {
                "type": "function",
                "name": "recursive_word_chunk",
                "qualified_name": "fenic.api.functions.text.recursive_word_chunk",
                "docstring": "Split text recursively into chunks by word count.",
                "annotation": "",
                "returns": "Column",
                "parameters": ["column", "chunk_size"],
                "parent_class": None,
                "is_public": True,
            },
            {
                "type": "method",
                "name": "select",
                "qualified_name": "fenic.api.dataframe.DataFrame.select",
                "docstring": "Select columns from this DataFrame.",
                "annotation": "",
                "returns": "DataFrame",
                "parameters": ["cols"],
                "parent_class": "DataFrame",
                "is_public": True,
            },
            {
                "type": "function",
                "name": "_private_helper",
                "qualified_name": "fenic.api.functions.text._private_helper",
                "docstring": "Not part of the public API.",
                "annotation": "",
                "returns": "Column",
                "parameters": ["column"],
                "parent_class": None,
                "is_public": False,
            },
        ]
    ).write.save_as_table("api_df", mode="overwrite")
    session.create_dataframe(
        [
            {
                "api_tree": "├─ [module] text\n  ├─ [function] recursive_word_chunk",
                "project_overview": "Fenic is a semantic DataFrame library.",
            }
        ]
    ).write.save_as_table("fenic_project_context", mode="overwrite")
    yield session
    session.stop()


@pytest.mark.asyncio
async def test_native_server_tools_are_agent_usable(docs_session):
    """Exercise discovery and detail tools through an in-process MCP client."""
    register_docs_tools(docs_session)
    server = create_native_server(docs_session)

    async with Client(server.mcp) as client:
        tools = await client.list_tools()
        assert {tool.name for tool in tools} == {  # nosec B101
            "get_api_tree",
            "get_entities",
            "get_entity",
            "get_project_overview",
            "search_fenic_api",
        }

        search = await client.call_tool(
            "search_fenic_api",
            {
                "query": "WORD",
                "element_type": "function",
                "table_format": "structured",
            },
        )
        search_result = search.structured_content
        assert search_result["returned_result_count"] == 1  # nosec B101
        assert (  # nosec B101
            search_result["rows"][0]["qualified_name"]
            == "fenic.api.functions.text.recursive_word_chunk"
        )

        multi_keyword_search = await client.call_tool(
            "search_fenic_api",
            {
                "query": "recursive chunks word count",
                "element_type": "function",
                "table_format": "structured",
            },
        )
        assert (  # nosec B101
            multi_keyword_search.structured_content["rows"][0]["qualified_name"]
            == "fenic.api.functions.text.recursive_word_chunk"
        )

        entity = await client.call_tool(
            "get_entity",
            {
                "qualified_name": (
                    "fenic.api.functions.text.recursive_word_chunk"
                ),
                "table_format": "structured",
            },
        )
        entity_result = entity.structured_content
        assert entity_result["rows"][0]["returns"] == "Column"  # nosec B101

        entities = await client.call_tool(
            "get_entities",
            {
                "qualified_names": [
                    "fenic.api.functions.text.recursive_word_chunk",
                    "fenic.api.dataframe.DataFrame.select",
                ],
                "table_format": "structured",
            },
        )
        assert entities.structured_content["returned_result_count"] == 2  # nosec B101

        overview = await client.call_tool("get_project_overview", {})
        assert (  # nosec B101
            "semantic DataFrame" in overview.structured_content["rows"]
        )

        tree = await client.call_tool("get_api_tree", {})
        assert (  # nosec B101
            "recursive_word_chunk" in tree.structured_content["rows"]
        )
