from enum import Enum

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
from fenic.core.error import InvalidExampleCollectionError


def test_semantic_classification_simple(local_session):
    categories = ["Billing", "Tech Support", "General Inquiry"]
    comments_data = {
        "user_comments": [
            "My bill is too high",
            "The product doesn" "t work when I try to use a specific feature",
            "Where are you located?",
        ]
    }

    comments_df = local_session.create_dataframe(comments_data)
    categorized_comments_df = comments_df.select(
        col("user_comments"),
        semantic.classify(
            text.concat(col("user_comments"), lit(" ")), categories
        ).alias("category"),
    )
    result = categorized_comments_df.to_polars()

    assert result.schema == {
        "user_comments": pl.String,
        "category": pl.String,
    }

    result_list = result.select(pl.col("category"))["category"].to_list()

    # check that the categories are in the correct order
    assert result_list == ["Billing", "Tech Support", "General Inquiry"]

    # test more idiomatic approach.
    filter = result.select(pl.col("category")).filter(pl.col("category") == "Billing")
    assert len(filter.rows()) == 1


def test_semantic_classification_enum(local_session):
    Category = Enum(
        "Category",
        [
            ("BILLING", "Billing"),
            # On purpose not matching the enum name.
            ("TECH_SUPPORT", "Tech Support"),
            ("GENERAL_INQUIRY", "General Inquiry"),
        ],
    )

    comments_data = {
        "user_comments": [
            "My bill is too high",
            "The product doesn" "t work when I try to use a specific feature",
            "Where are you located?",
        ]
    }

    comments_df = local_session.create_dataframe(comments_data)
    categorized_comments_df = comments_df.select(
        col("user_comments"),
        semantic.classify(col("user_comments"), Category).alias("category"),
    )

    result = categorized_comments_df.to_polars()

    assert result.schema == {
        "user_comments": pl.String,
        "category": pl.String,
    }

    _check_results_in_enum(result, "category", Category)


def test_semantic_classification_enum_with_none(local_session):
    Category = Enum(
        "Category",
        [
            ("BILLING", "Billing"),
            # On purpose not matching the enum name.
            ("TECH_SUPPORT", "Support"),
            ("GENERAL_INQUIRY", "General Inquiry"),
        ],
    )

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
        semantic.classify(col("user_comments"), Category).alias("category"),
    )

    result = categorized_comments_df.to_polars()

    assert result.schema == {
        "user_comments": pl.String,
        "category": pl.String,
    }

    # none is allowed as a result.
    _check_results_in_enum(result, "category", Category, allow_none=True)

    assert result["category"].to_list()[3] is None


def test_semantic_classification_enum_2(local_session):
    """Test spam / not spam with strings that are more than 2 words."""
    Category = Enum(
        "EmailType",  # testing to ensure if a user puts a category not named `Category` it will still work
        [
            ("SPAM", "spam"),
            ("NOT_SPAM", "not spam"),
        ],
    )

    comments_data = {
        "user_comments": [
            "Buy cheap watches now!",
            "Meeting at 3 PM in the conference room",
            "You've won a free iPhone! Click here",
            "Can you pick up some milk on your way home?",
            "Increase your followers by 10000 overnight!",
        ]
    }

    comments_df = local_session.create_dataframe(comments_data)
    categorized_comments_df = comments_df.select(
        col("user_comments"),
        semantic.classify(col("user_comments"), Category).alias("category"),
    )

    result = categorized_comments_df.to_polars()

    assert result.schema == {
        "user_comments": pl.String,
        "category": pl.String,
    }

    _check_results_in_enum(result, "category", Category)


def test_semantic_classification_enum_with_examples(local_session):
    """Test spam / not spam with strings that are more than 2 words."""
    Category = Enum(
        "Category",
        [
            ("HEALTH", "Health"),
            ("FINANCE", "Finance"),
            ("OTHER", "Other"),
        ],
    )

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
            output=Category.FINANCE.value,
        )
    ).create_example(
        ClassifyExample(
            input="Connect me to health department",
            output=Category.HEALTH.value,
        )
    ).create_example(
        ClassifyExample(
            input="Connect me to Human Resources",
            output=Category.OTHER.value,
        )
    )

    comments_df = local_session.create_dataframe(comments_data)
    categorized_comments_df = comments_df.select(
        col("user_comments"),
        semantic.classify(
            col("user_comments"),
            Category,
            examples=collection,
        ).alias("category"),
    )

    result = categorized_comments_df.to_polars()

    assert result.schema == {
        "user_comments": pl.String,
        "category": pl.String,
    }

    _check_results_in_enum(result, "category", Category)


def test_semantic_classification_enum_with_bad_examples(local_session):
    """Test spam / not spam with strings that are more than 2 words."""
    Category = Enum(
        "Category",
        [
            ("HEALTH", "Health"),
            ("OTHER", "Other"),
        ],
    )

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
            output="RANDOM_ANSWER",
        )
    )

    comments_df = local_session.create_dataframe(comments_data)
    with pytest.raises(InvalidExampleCollectionError):
        comments_df.select(
            col("user_comments"),
            semantic.classify(
                col("user_comments"),
                Category,
                examples=collection,
            ).alias("category"),
        ).to_polars()


def test_semantic_classification_enum_type_error(local_session):
    Category = Enum(
        "Category", [("BILLING", 1), ("TECH_SUPPORT", 2), ("GENERAL_INQUIRY", 3)]
    )

    comments_data = {
        "user_comments": [
            "My bill is too high",
            "The product doesn" "t work when I try to use a specific feature",
            "Where are you located?",
        ]
    }

    comments_df = local_session.create_dataframe(comments_data)

    with pytest.raises(
        TypeError,
        match="Type mismatch: Cannot apply semantic.classify to an enum that is not a string. ",
    ):
        comments_df.select(
            col("user_comments"),
            semantic.classify(col("user_comments"), Category).alias("category"),
        )


def test_semantic_classification_err_handling_no_categories(local_session):
    categories = []
    comments_data = {
        "user_comments": [
            "My bill is too high",
        ]
    }

    comments_df = local_session.create_dataframe(comments_data)

    with pytest.raises(ValueError):
        comments_df.select(
            col("user_comments"),
            semantic.classify("user_comments", categories).alias("category"),
        )


def test_semantic_classification_err_handling_invalid_column(local_session):
    categories = ["Billing"]
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


def _check_results_in_enum(
    result: pl.DataFrame,
    col_name: str,
    possible_results: list[str] | Enum,
    allow_none: bool = False,
):
    possibilities = (
        possible_results
        if isinstance(possible_results, list)
        else [e.value for e in possible_results]
    )
    result_list = result.select(pl.col(col_name))[col_name].to_list()
    for result in result_list:
        if allow_none and result is None:
            continue
        elif result is None:
            raise ValueError("Result is None, but allow_none is False")

        assert result in possibilities
