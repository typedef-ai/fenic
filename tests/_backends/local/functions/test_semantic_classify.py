
import polars as pl
import pytest

from fenic import (
    ClassifyExample,
    ClassifyExampleCollection,
    col,
    lit,
    semantic,
    text,
)
from fenic.api.session import OpenAIModelConfig, SemanticConfig, Session, SessionConfig
from fenic.core.error import InvalidExampleCollectionError, ValidationError
from fenic.core.types import ClassDefinition, ColumnField, StringType


def test_semantic_classification_simple(local_session):
    categories = ["Billing", "Tech Support", "General Inquiry"]

    comments_data = {
        "user_comments": [
            "My bill is too high",
            "The product doesn" "t work when I try to use a specific feature",
            "Where are you located?",
            None,
        ]
    }

    comments_df = local_session.create_dataframe(comments_data)
    categorized_comments_df = comments_df.select(
        col("user_comments"),
        semantic.classify(
            text.concat(col("user_comments"), lit(" ")), categories
        ).alias("category"),
    )
    assert categorized_comments_df.schema.column_fields == [
        ColumnField(name="user_comments", data_type=StringType),
        ColumnField(name="category", data_type=StringType),
    ]
    result = categorized_comments_df.to_polars()

    assert result.schema == {
        "user_comments": pl.String,
        "category": pl.String,
    }

    result_list = result.select(pl.col("category"))["category"].to_list()
    assert len(result_list) == 4
    assert result_list[3] is None

    for result in result_list[:3]:
        assert result in ["Billing", "Tech Support", "General Inquiry"]

def test_semantic_classification_with_definitions(local_session):
    categories = [
        ClassDefinition(
            label="Billing",
            description="Questions or issues related to payments, charges, invoices, subscriptions, or account billing."
        ),
        ClassDefinition(
            label="Tech Support",
            description="Requests for help with technical problems, system errors, troubleshooting, or product malfunctions."
        ),
        ClassDefinition(
            label="General Inquiry",
            description="All other questions or requests not related to billing or technical issues, such as product information, company policies, or general support."
        ),
    ]

    comments_data = {
        "user_comments": [
            "My bill is too high",
            "The product doesn" "t work when I try to use a specific feature",
            "Where are you located?",
            None,
        ]
    }

    comments_df = local_session.create_dataframe(comments_data)
    categorized_comments_df = comments_df.select(
        col("user_comments"),
        semantic.classify(
            text.concat(col("user_comments"), lit(" ")), categories
        ).alias("category"),
    )
    assert categorized_comments_df.schema.column_fields == [
        ColumnField(name="user_comments", data_type=StringType),
        ColumnField(name="category", data_type=StringType),
    ]
    result = categorized_comments_df.to_polars()

    assert result.schema == {
        "user_comments": pl.String,
        "category": pl.String,
    }

    result_list = result.select(pl.col("category"))["category"].to_list()
    assert len(result_list) == 4
    assert result_list[3] is None

    for result in result_list[:3]:
        assert result in ["Billing", "Tech Support", "General Inquiry"]


def test_semantic_classification_with_examples(local_session):
    categories = [
        ClassDefinition(label="Health", description="Health related inquiries"),
        ClassDefinition(label="Finance", description="Finance related inquiries"),
        ClassDefinition(label="Other", description="Other inquiries"),
    ]

    comments_data = {
        "user_comments": [
            "Call to department of health",
            "Call to finance department",
            "Connect to the money division",
            "I want to talk about my well-being",
            "Connect me to HR",
        ]
    }

    collection = ClassifyExampleCollection()
    collection.create_example(
        ClassifyExample(
            input="money related question",
            output="Finance",
        )
    ).create_example(
        ClassifyExample(
            input="Connect me to health department",
            output="Health",
        )
    ).create_example(
        ClassifyExample(
            input="Connect me to Human Resources",
            output="Other",
        )
    )

    comments_df = local_session.create_dataframe(comments_data)
    categorized_comments_df = comments_df.select(
        col("user_comments"),
        semantic.classify(
            col("user_comments"),
            categories,
            examples=collection,
        ).alias("category"),
    )

    result = categorized_comments_df.to_polars()

    assert result.schema == {
        "user_comments": pl.String,
        "category": pl.String,
    }

    for value in result.select(pl.col("category"))["category"].to_list():
        assert value in ["Health", "Finance", "Other"]


def test_semantic_classification_with_bad_examples(local_session):
    categories = [
        ClassDefinition(label="Health", description="Health related inquiries"),
        ClassDefinition(label="Other", description="Not health related inquiries"),
    ]

    comments_data = {
        "user_comments": [
            "Call to department of health",
            "Call to finance department",
        ]
    }

    collection = ClassifyExampleCollection()
    collection.create_example(
        ClassifyExample(
            input="Call to finance",
            output="General",
        )
    )

    comments_df = local_session.create_dataframe(comments_data)
    with pytest.raises(InvalidExampleCollectionError):
        comments_df.select(
            col("user_comments"),
            semantic.classify(
                col("user_comments"),
                categories,
                examples=collection,
            ).alias("category"),
        ).to_polars()



def test_semantic_classification_invalid_categories(local_session):
    categories = [
        ClassDefinition(label="Billing", description="Billing related inquiries"),
    ]
    comments_data = {
        "user_comments": [
            "My bill is too high",
        ]
    }

    comments_df = local_session.create_dataframe(comments_data)

    with pytest.raises(ValidationError):
        comments_df.select(
            col("user_comments"),
            semantic.classify(col("user_comments"), categories).alias("category"),
        )
    categories = []
    with pytest.raises(ValidationError):
        comments_df.select(
            col("user_comments"),
            semantic.classify(col("user_comments"), categories).alias("category"),
        )

    categories = [
        ClassDefinition(label="Billing", description="Description 1"),
        ClassDefinition(label="Billing", description="Description 2"),
    ]
    with pytest.raises(ValidationError):
        comments_df.select(
            col("user_comments"),
            semantic.classify(col("user_comments"), categories).alias("category"),
        )


def test_semantic_classification_err_handling_invalid_column(local_session):
    categories = ["Billing", "Tech Support"]
    comments_data = {
        "user_comments": [
            "My bill is too high",
        ]
    }

    comments_df = local_session.create_dataframe(comments_data)

    with pytest.raises(ValueError):
        comments_df.select(
            col("user_comments"),
            semantic.classify("invalid_column", categories).alias("category"),
        )

def test_semantic_classify_without_models():
    """Test that an error is raised if no language models are configured."""
    session_config = SessionConfig(
        app_name="semantic_classify_without_models",
    )
    session = Session.get_or_create(session_config)
    with pytest.raises(ValidationError, match="No language models configured."):
        session.create_dataframe({"text": ["hello"]}).select(semantic.classify(col("text"), ["hello", "world"]).alias("classified_text"))
    session.stop()

    session_config = SessionConfig(
        app_name="semantic_classify_with_models",
        semantic=SemanticConfig(
            embedding_models={"oai-small": OpenAIModelConfig(model_name="text-embedding-3-small", rpm=3000, tpm=1_000_000)},
        ),
    )
    session = Session.get_or_create(session_config)
    with pytest.raises(ValidationError, match="No language models configured."):
        session.create_dataframe({"text": ["hello"]}).select(semantic.classify(col("text"), ["hello", "world"]).alias("classified_text"))
    session.stop()
