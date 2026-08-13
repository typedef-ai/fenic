import re

import polars as pl
import pytest

from fenic import (
    ColumnField,
    IntegerType,
    OpenAIEmbeddingModel,
    StringType,
    col,
    semantic,
    sum,
)
from fenic._inference.types import FenicCompletionsRequest, FenicCompletionsResponse
from fenic.api.session import (
    SemanticConfig,
    Session,
    SessionConfig,
)
from fenic.core.error import PlanError, ValidationError


def _install_deterministic_reduce_model(local_session, monkeypatch):
    model = local_session._session_state.get_language_model()

    monkeypatch.setattr(model, "count_tokens", lambda _: 1)

    def fake_get_completions(messages, **kwargs):
        responses = []
        for message in messages:
            docs = re.findall(r"<document\d+>\s*(.*?)\s*</document\d+>", message.user, re.DOTALL)
            responses.append(
                FenicCompletionsResponse(
                    completion=" | ".join(docs),
                    logprobs=None,
                )
            )
        return responses

    monkeypatch.setattr(model, "get_completions", fake_get_completions)


def test_semantic_reduce_calls_model_client_completion_api(local_session, monkeypatch):
    """Tripwire semantic.reduce's call into LanguageModel/ModelClient completions."""
    model = local_session._session_state.get_language_model()
    monkeypatch.setattr(model, "count_tokens", lambda _: 1)

    captured_batches = []

    def fake_make_batch_requests(requests, operation_name, request_timeout=None):
        captured_batches.append(
            {
                "requests": requests,
                "operation_name": operation_name,
                "request_timeout": request_timeout,
            }
        )
        return [FenicCompletionsResponse(completion="tripwire summary", logprobs=None)]

    monkeypatch.setattr(model.client, "make_batch_requests", fake_make_batch_requests)

    result = local_session.create_dataframe({"notes": ["alpha", "beta"]}).agg(
        semantic.reduce(
            "Summarize the tripwire notes.",
            col("notes"),
            max_output_tokens=37,
            temperature=0.25,
        ).alias("summary")
    ).to_polars()

    assert result.to_dicts() == [{"summary": "tripwire summary"}]
    assert len(captured_batches) == 1

    batch = captured_batches[0]
    assert batch["operation_name"] == "semantic.reduce(group=0)"
    assert batch["request_timeout"] is None

    requests = batch["requests"]
    assert len(requests) == 1
    request = requests[0]
    assert isinstance(request, FenicCompletionsRequest)
    assert request.max_completion_tokens == 37
    assert request.top_logprobs is None
    assert request.structured_output is None
    assert request.model_profile is None
    assert request.temperature in (0.25, None)
    user_message = request.messages.user
    assert user_message is not None
    assert "Summarize the tripwire notes." in user_message
    assert "<document1>\nalpha\n</document1>" in user_message
    assert "<document2>\nbeta\n</document2>" in user_message


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


def test_semantic_reduce_golden_output_ordering_and_nulls(local_session, monkeypatch):
    _install_deterministic_reduce_model(local_session, monkeypatch)

    df = local_session.create_dataframe(
        {
            "bucket": ["alpha", "alpha", "alpha", "empty-docs", "null-docs"],
            "sort_key": [2, 1, None, 1, 1],
            "notes": ["second", "first", None, "", None],
            "row_value": [10, 20, 30, 40, 50],
        }
    )

    result_df = df.group_by("bucket").agg(
        semantic.reduce(
            "Summarize notes in order.",
            col("notes"),
            order_by=[col("sort_key").asc_nulls_last()],
        ).alias("summary"),
        sum("row_value").alias("row_value_sum"),
    )
    assert result_df.schema.column_fields == [
        ColumnField(name="bucket", data_type=StringType),
        ColumnField(name="summary", data_type=StringType),
        ColumnField(name="row_value_sum", data_type=IntegerType),
    ]

    result = result_df.to_polars().sort("bucket")
    assert result.schema == {
        "bucket": pl.Utf8,
        "summary": pl.Utf8,
        "row_value_sum": pl.Int64,
    }
    assert len(result) == 3
    assert result.to_dicts() == [
        {"bucket": "alpha", "summary": "first | second", "row_value_sum": 60},
        {"bucket": "empty-docs", "summary": None, "row_value_sum": 40},
        {"bucket": "null-docs", "summary": None, "row_value_sum": 50},
    ]


def test_case_semantic_reduce_golden_empty_result(local_session, monkeypatch):
    _install_deterministic_reduce_model(local_session, monkeypatch)

    df = local_session.create_dataframe(
        {
            "bucket": ["alpha"],
            "notes": ["first"],
            "row_value": [1],
        }
    )
    empty_result = df.filter(col("bucket") == "missing").group_by("bucket").agg(
        semantic.reduce("Summarize notes in order.", col("notes")).alias("summary"),
        sum("row_value").alias("row_value_sum"),
    ).to_polars()
    assert empty_result.is_empty()
    assert empty_result.schema == {
        "bucket": pl.Utf8,
        "summary": pl.Utf8,
        "row_value_sum": pl.Int64,
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


def test_semantic_reduce_without_models(tmp_path):
    """Test semantic.reduce() method without models."""
    session_config = SessionConfig(
        app_name="semantic_reduce_without_models",
        db_path=tmp_path,
    )
    session = Session.get_or_create(session_config)
    with pytest.raises(ValidationError, match="No language models configured."):
        session.create_dataframe({"notes": ["hello"]}).agg(semantic.reduce("Summarize the main action items from the notes.", col("notes")).alias("summary"))
    session.stop(skip_usage_summary=True)

    session_config = SessionConfig(
        app_name="semantic_reduce_with_models",
        semantic=SemanticConfig(
            embedding_models={"oai-small": OpenAIEmbeddingModel(model_name="text-embedding-3-small", rpm=3000, tpm=1_000_000)},
        ),
        db_path=tmp_path,
    )
    session = Session.get_or_create(session_config)
    with pytest.raises(ValidationError, match="No language models configured."):
        session.create_dataframe({"notes": ["hello"]}).agg(semantic.reduce("Summarize the main action items from the notes.", col("notes")).alias("summary"))
    session.stop(skip_usage_summary=True)

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
