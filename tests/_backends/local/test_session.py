import polars as pl
import pytest

from fenic import (
    SemanticConfig,
    Session,
    SessionConfig,
)
from fenic.api.session.config import OpenAILanguageModel
from fenic.core.error import SessionError


def test_multiple_sessions(tmp_path, local_session_config):
    session = Session.get_or_create(local_session_config)
    session2 = Session.get_or_create(local_session_config)

    assert session._session_state == session2._session_state
    session_config3 = SessionConfig(
        app_name="semantic_test2",
        semantic=SemanticConfig(
            language_models={"mini": OpenAILanguageModel(model_name="gpt-4.1-mini", rpm=500, tpm=200_000)},
            default_language_model="mini"
        ),
        db_path=tmp_path,
    )
    session3 = Session.get_or_create(session_config3)

    # make sure multiple sessions can coexist
    session.stop(skip_usage_summary=True)
    session3.stop(skip_usage_summary=True)

    with pytest.raises(SessionError, match="This session 'test_app' has been stopped"):
        df = session.create_dataframe(
            pl.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
        )
        df.to_polars()

    with pytest.raises(SessionError, match="This session 'semantic_test2' has been stopped"):
        df = session3.create_dataframe(
            pl.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
        )
        df.to_polars()

    session4 = Session.get_or_create(local_session_config)

    assert session != session4

def test_combining_dataframes_from_different_sessions_raises(tmp_path):
    config1 = SessionConfig(
        app_name="test_session_1",
        semantic=SemanticConfig(
            language_models={
                "test_model": OpenAILanguageModel(model_name="gpt-4.1-mini", rpm=500, tpm=200_000),
            },
        ),
        db_path=tmp_path,
    )
    config2 = SessionConfig(
        app_name="test_session_2",
        semantic=SemanticConfig(
            language_models={
                "test_model": OpenAILanguageModel(model_name="gpt-4.1-mini", rpm=500, tpm=200_000),
            },
        ),
        db_path=tmp_path,
    )
    session1 = Session.get_or_create(config1)
    session2 = Session.get_or_create(config2)

    df1 = session1.create_dataframe(pl.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]}))
    df2 = session2.create_dataframe(pl.DataFrame({"name": ["Charlie", "David"], "age": [35, 40]}))

    with pytest.raises(SessionError):
        df1.union(df2)

    with pytest.raises(SessionError):
        df1.join(df2, on="name")

    with pytest.raises(SessionError):
        session2.sql("SELECT * FROM {df1} inner join {df2} on {df1}.name = {df2}.name", df1=df1, df2=df2)

    with pytest.raises(SessionError):
        config3 = SessionConfig(
            app_name="test_session_3",
            semantic=SemanticConfig(
                language_models={
                    "test_model": OpenAILanguageModel(model_name="gpt-4.1-mini", rpm=500, tpm=200_000),
                },
            ),
            db_path=tmp_path,
        )
        session3 = Session.get_or_create(config3)
        session3.sql("SELECT * FROM {df1} inner join {df2} on {df1}.name = {df2}.name", df1=df1, df2=df2)

def test_stopped_session_remains_stopped_after_new_session_same_name(tmp_path):
    """Test that a stopped session remains stopped even when a new session with the same app_name is created."""
    config = SessionConfig(
        app_name="reused_app_name",
        semantic=SemanticConfig(
            language_models={
                "test_model": OpenAILanguageModel(model_name="gpt-4.1-mini", rpm=500, tpm=200_000),
            },
        ),
        db_path=tmp_path,
    )

    # Create first session and keep a reference to it
    session1 = Session.get_or_create(config)
    df1 = session1.create_dataframe(pl.DataFrame({"x": [1, 2, 3]}))

    # Stop the first session
    session1.stop(skip_usage_summary=True)

    # Verify the first session is stopped
    with pytest.raises(SessionError, match="This session 'reused_app_name' has been stopped"):
        df1.to_polars()

    # Create a new session with the same app_name
    session2 = Session.get_or_create(config)
    df2 = session2.create_dataframe(pl.DataFrame({"x": [4, 5, 6]}))

    # The new session should work fine
    result = df2.to_polars()
    assert len(result) == 3

    # The old session should still be stopped
    with pytest.raises(SessionError, match="This session 'reused_app_name' has been stopped"):
        df1.to_polars()

    session2.stop(skip_usage_summary=True)
