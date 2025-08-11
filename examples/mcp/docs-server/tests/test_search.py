import pytest
from server import FenicAPIDocQuerySearch, FenicSession


@pytest.fixture
def session():
    return FenicSession().get_session()

def test_fenic_session(session):
    """Test that the session is the same."""
    session2 = FenicSession().get_session()
    assert session is session2 # nosec: B101

def test_search_regex(session):
    """Test regular expression search."""
    search_df = FenicAPIDocQuerySearch.search_api_docs(session, "semantic.*extract")
    assert search_df is not None # nosec: B101
    assert search_df.count() > 0 # nosec: B101
    
    search_dict = search_df.to_pydict()
    print(search_dict["qualified_name"])
    assert "fenic.api.functions.semantic.extract" in search_dict["qualified_name"] # nosec: B101

def test_search_keyword(session):
    """When searching for keywords, we should get a union of results from the different terms."""
    search_df = FenicAPIDocQuerySearch.search_api_docs(session, "semantic extract")
    assert search_df is not None # nosec: B101
    assert search_df.count() > 0 # nosec: B101
    
    search_dict = search_df.to_pydict()
    assert "fenic.core.types.enums.SemanticSimilarityMetric" in search_dict["qualified_name"] # nosec: B101
    assert "fenic.api.functions.semantic.extract" in search_dict["qualified_name"] # nosec: B101
