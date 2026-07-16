import fenic as fc
import pytest
from fenic_mcp.server.search import (
    get_entity_by_qualified_name,
    search_api_docs,
)


@pytest.fixture(scope="module")
def session() -> fc.Session:
    # Minimal session with defaults (no semantic/LLM needed for these tests)
    config = fc.SessionConfig(app_name="fenic_mcp_tests")
    return fc.Session.get_or_create(config)


def _write_api_df(session: fc.Session, rows: list[dict]) -> None:
    # Ensure no column is entirely None to avoid Polars inferring Null dtype
    sanitized: list[dict] = []
    for r in rows:
        r = dict(r)
        for k in ("docstring", "annotation", "returns"):
            if r.get(k) is None:
                r[k] = ""
        sanitized.append(r)
    df = session.create_dataframe(sanitized)
    df.write.save_as_table("api_df", mode="overwrite")


def test_search_excludes_private_segments_and_non_public(session: fc.Session) -> None:
    rows = [
        {
            "type": "function",
            "name": "public_fn",
            "qualified_name": "fenic.api.module.public_fn",
            "docstring": "A public function",
            "annotation": None,
            "returns": None,
            "is_public": True,
        },
        {
            # Public but in an underscore path segment → should be excluded by regex
            "type": "function",
            "name": "hidden",
            "qualified_name": "fenic.api._internal.hidden",
            "docstring": "hidden function",
            "annotation": None,
            "returns": None,
            "is_public": True,
        },
        {
            # Non-public → should be excluded by is_public filter
            "type": "function",
            "name": "non_public",
            "qualified_name": "fenic.api.module.non_public",
            "docstring": "not public",
            "annotation": None,
            "returns": None,
            "is_public": False,
        },
    ]
    _write_api_df(session, rows)

    result_df = search_api_docs(session, query="public")
    result = result_df.select("qualified_name").to_pylist()
    qualified_names = {r["qualified_name"] for r in result}

    assert "fenic.api.module.public_fn" in qualified_names
    assert "fenic.api._internal.hidden" not in qualified_names
    assert "fenic.api.module.non_public" not in qualified_names


def test_search_type_filter(session: fc.Session) -> None:
    rows = [
        {
            "type": "class",
            "name": "Widget",
            "qualified_name": "fenic.api.Widget",
            "docstring": "Widget class",
            "annotation": None,
            "returns": None,
            "is_public": True,
        },
        {
            "type": "function",
            "name": "make_widget",
            "qualified_name": "fenic.api.make_widget",
            "docstring": "Create a widget",
            "annotation": None,
            "returns": None,
            "is_public": True,
        },
    ]
    _write_api_df(session, rows)

    only_classes = (
        search_api_docs(session, query="widget", types=["class"])
        .select("type")
        .to_pylist()
    )
    assert {r["type"] for r in only_classes} == {"class"}

    only_functions = (
        search_api_docs(session, query="widget", types=["function"])
        .select("type")
        .to_pylist()
    )
    assert {r["type"] for r in only_functions} == {"function"}


def test_regex_matches_multiple_fields(session: fc.Session) -> None:
    rows = [
        {
            "type": "function",
            "name": "alpha",
            "qualified_name": "fenic.api.alpha",
            "docstring": "does beta things",
            "annotation": None,
            "returns": None,
            "is_public": True,
        },
        {
            "type": "function",
            "name": "gamma",
            "qualified_name": "fenic.api.gamma",
            "docstring": None,
            "annotation": "beta-annotation",
            "returns": None,
            "is_public": True,
        },
        {
            "type": "function",
            "name": "delta",
            "qualified_name": "fenic.api.delta",
            "docstring": None,
            "annotation": None,
            "returns": "beta-return",
            "is_public": True,
        },
    ]
    _write_api_df(session, rows)

    result_df = search_api_docs(session, query="beta")
    names = {r["name"] for r in result_df.select("name").to_pylist()}
    assert names == {"alpha", "gamma", "delta"}


def test_get_entity_by_qualified_name(session: fc.Session) -> None:
    rows = [
        {
            "type": "method",
            "name": "do_it",
            "qualified_name": "fenic.api.Widget.do_it",
            "docstring": "Does it",
            "annotation": None,
            "returns": "None",
            "is_public": True,
        },
        {
            "type": "function",
            "name": "other",
            "qualified_name": "fenic.api.other",
            "docstring": None,
            "annotation": None,
            "returns": None,
            "is_public": True,
        },
    ]
    _write_api_df(session, rows)

    df = get_entity_by_qualified_name(session, "fenic.api.Widget.do_it")
    rows = df.to_pylist()
    assert len(rows) == 1
    row = rows[0]
    assert row["type"] == "method"
    assert row["name"] == "do_it"
    assert row["qualified_name"] == "fenic.api.Widget.do_it"
    assert row["docstring"] == "Does it"
