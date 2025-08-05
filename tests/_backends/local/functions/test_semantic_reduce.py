
import polars as pl
import pytest

from fenic import (
    OpenAIEmbeddingModel,
    col,
    semantic,
    sum,
)
from fenic.api.session import (
    SemanticConfig,
    Session,
    SessionConfig,
)
from fenic.core.error import PlanError, ValidationError


def test_semantic_reduce(local_session):
    """Test semantic.reduce() method."""
    data = {
        "date": ["2024-01-01", "2024-01-01", "2024-01-02"],
        "notes": [
            "Q4 Sales Review Discussion: Revenue exceeded targets by 12%. John mentioned concerns about EMEA pipeline. Team agreed John will conduct deep-dive analysis by Friday. Alice suggested meeting with key clients to gather feedback.",
            "Product Planning: Discussed upcoming features for Q1. Team debated prioritization of mobile vs desktop improvements. Bob noted sprint board needs restructuring. Agreed to have product roadmap ready for next board meeting.",
            "Marketing Sync: Campaign performance trending well. Creative assets need final revisions before launch next week. Sarah raised concerns about Q1 budget - needs executive approval for additional spend.",
        ],
        "num_attendees": [10, 15, 20],
    }
    df = local_session.create_dataframe(data)

    result = df.group_by("date").agg(
        semantic.reduce("Summarize the main action items from the notes.", col("notes")).alias(
            "summary"
        ),
        sum("num_attendees").alias("num_attendees"),
    )
    result = result.to_polars()

    assert result.schema == {
        "date": pl.Utf8,
        "summary": pl.Utf8,
        "num_attendees": pl.Int64,
    }
    assert result.filter(pl.col("date") == "2024-01-01")["num_attendees"][0] == 25
    assert result.filter(pl.col("date") == "2024-01-02")["num_attendees"][0] == 20

    result = df.agg(
        semantic.reduce("Summarize the main action items from the notes.", col("notes")).alias(
            "summary"
        ),
        sum("num_attendees").alias("num_attendees"),
    )
    result = result.to_polars()

    assert result.schema == {
        "summary": pl.Utf8,
        "num_attendees": pl.Int64,
    }

def test_semantic_reduce_with_order_by(local_session):
    """Test semantic.reduce() method."""
    data = {
        "department": ["Sales", "Sales", "Engineering"],
        "date": ["2024-01-01", "2024-01-01", "2024-01-02"],
        "notes": [
            "Q4 Sales Review Discussion: Revenue exceeded targets by 12%. John mentioned concerns about EMEA pipeline. Team agreed John will conduct deep-dive analysis by Friday. Alice suggested meeting with key clients to gather feedback.",
            "Product Planning: Discussed upcoming features for Q1. Team debated prioritization of mobile vs desktop improvements. Bob noted sprint board needs restructuring. Agreed to have product roadmap ready for next board meeting.",
            "Marketing Sync: Campaign performance trending well. Creative assets need final revisions before launch next week. Sarah raised concerns about Q1 budget - needs executive approval for additional spend.",
        ],
        "num_attendees": [20, 15, 20],
    }
    df = local_session.create_dataframe(data)

    df = df.group_by("department").agg(
        semantic.reduce("Summarize the main action items from the notes.", col("notes"), order_by=[col("date"), col("num_attendees").desc_nulls_last()]).alias(
            "summary"
        ),
        sum("num_attendees").alias("num_attendees"),
    )
    df = df.to_polars()
    assert df.schema == {
        "department": pl.Utf8,
        "summary": pl.Utf8,
        "num_attendees": pl.Int64,
    }

def test_semantic_reduce_with_group_context(local_session):
    """Test semantic.reduce() method with group context."""
    data = {
        "date": ["2024-01-01", "2024-01-01", "2024-01-02"],
        "notes": [
            "Q4 Sales Review Discussion: Revenue exceeded targets by 12%. John mentioned concerns about EMEA pipeline. Team agreed John will conduct deep-dive analysis by Friday. Alice suggested meeting with key clients to gather feedback.",
            "Product Planning: Discussed upcoming features for Q1. Team debated prioritization of mobile vs desktop improvements. Bob noted sprint board needs restructuring. Agreed to have product roadmap ready for next board meeting.",
            "Marketing Sync: Campaign performance trending well. Creative assets need final revisions before launch next week. Sarah raised concerns about Q1 budget - needs executive approval for additional spend.",
        ],
        "num_attendees": [10, 15, 20],
    }
    df = local_session.create_dataframe(data)

    df.group_by("date").agg(
        semantic.reduce(
            "Summarize the main action items from the notes. FYI the notes are from {{date}}.",
            col("notes"),
            group_context={"date": col("date")},
        ).alias("summary"),
    )

    with pytest.raises(ValidationError, match="Template variable 'date' is not defined. Available columns: none."):
        df.group_by("date").agg(
            semantic.reduce(
                "Summarize the main action items from the notes. FYI the notes are from {{date}}.",
                col("notes"),
            ).alias("summary"),
        )

    with pytest.raises(PlanError, match="semantic.reduce context expression 'num_attendees' not found in group by. Available group by expressions: date."):
        df.group_by("date").agg(
            semantic.reduce(
                "Summarize the main action items from the notes. FYI the notes are from {{num_attendees}}.",
                col("notes"),
                group_context={"num_attendees": col("num_attendees")},
            ).alias("summary"),
        )

def test_semantic_reduce_with_group_context_and_order_by(local_session):
    """Test semantic.reduce() method with group context."""
    data = {
        "date": ["2024-01-01", "2024-01-01", "2024-01-02"],
        "notes": [
            "Q4 Sales Review Discussion: Revenue exceeded targets by 12%. John mentioned concerns about EMEA pipeline. Team agreed John will conduct deep-dive analysis by Friday. Alice suggested meeting with key clients to gather feedback.",
            "Product Planning: Discussed upcoming features for Q1. Team debated prioritization of mobile vs desktop improvements. Bob noted sprint board needs restructuring. Agreed to have product roadmap ready for next board meeting.",
            "Marketing Sync: Campaign performance trending well. Creative assets need final revisions before launch next week. Sarah raised concerns about Q1 budget - needs executive approval for additional spend.",
        ],
        "num_attendees": [10, 15, 20],
    }
    df = local_session.create_dataframe(data)

    df = df.group_by("date").agg(
        semantic.reduce(
            "Summarize the main action items from the notes. FYI the notes are from {{date}}.",
            col("notes"),
            group_context={"date": col("date")},
            order_by=[col("num_attendees")],
        ).alias("summary"),
    )


def test_semantic_reduce_without_models():
    """Test semantic.reduce() method without models."""
    session_config = SessionConfig(
        app_name="semantic_reduce_without_models",
    )
    session = Session.get_or_create(session_config)
    with pytest.raises(ValidationError, match="No language models configured."):
        session.create_dataframe({"notes": ["hello"]}).agg(semantic.reduce("Summarize the main action items from the notes.", col("notes")).alias("summary"))
    session.stop()

    session_config = SessionConfig(
        app_name="semantic_reduce_with_models",
        semantic=SemanticConfig(
            embedding_models={"oai-small": OpenAIEmbeddingModel(model_name="text-embedding-3-small", rpm=3000, tpm=1_000_000)},
        ),
    )
    session = Session.get_or_create(session_config)
    with pytest.raises(ValidationError, match="No language models configured."):
        session.create_dataframe({"notes": ["hello"]}).agg(semantic.reduce("Summarize the main action items from the notes.", col("notes")).alias("summary"))
    session.stop()

def test_semantic_reduce_invalid_prompt(local_session):
    with pytest.raises(ValidationError, match="The `prompt` argument to `semantic.reduce` cannot be empty."):
        local_session.create_dataframe({"notes": ["hello"]}).agg(semantic.reduce("", col("notes")).alias("summary"))

def test_semantic_reduce_agg_no_group_by(local_session):
    data = {
        "date": ["2024-01-01", "2024-01-01", "2024-01-02"],
        "notes": [
            "Q4 Sales Review Discussion: Revenue exceeded targets by 12%. John mentioned concerns about EMEA pipeline. Team agreed John will conduct deep-dive analysis by Friday. Alice suggested meeting with key clients to gather feedback.",
            "Product Planning: Discussed upcoming features for Q1. Team debated prioritization of mobile vs desktop improvements. Bob noted sprint board needs restructuring. Agreed to have product roadmap ready for next board meeting.",
            "Marketing Sync: Campaign performance trending well. Creative assets need final revisions before launch next week. Sarah raised concerns about Q1 budget - needs executive approval for additional spend.",
        ],
        "num_attendees": [10, 15, 20],
    }
    df = local_session.create_dataframe(data)
    df.agg(semantic.reduce("Summarize the main action items from the notes.", col("notes")).alias("summary"))
    with pytest.raises(PlanError, match="semantic.reduce context expression 'date' not found in group by. Available group by expressions: none."):
        df.agg(semantic.reduce("Summarize the main action items from the notes on {{date}}.", col("notes"), group_context={"date": col("date")}).alias("summary"))
