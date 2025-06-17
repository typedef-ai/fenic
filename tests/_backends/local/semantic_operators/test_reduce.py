from unittest.mock import MagicMock

import polars as pl
import pytest

from fenic._backends.local.semantic_operators.reduce import Reduce
from fenic._backends.local.semantic_operators.utils import (
    uppercase_instruction_placeholder,
)


@pytest.fixture
def mock_language_model():
    """Pytest fixture to create a mock BatchClientLM object."""
    mock_language_model = MagicMock()
    mock_language_model.max_context_window_length = 3000
    model_parameters = MagicMock()
    model_parameters.max_context_window_length = 3000
    model_parameters.max_output_tokens = 100
    mock_language_model.model_parameters = model_parameters
    mock_language_model.count_tokens.return_value = 10
    return mock_language_model


@pytest.fixture
def reduce_instance(mock_language_model):
    """Pytest fixture to create an instance of the Reduce class with the mock manager."""
    user_instruction = "Summarize these documents using each document's title: {title} and body: {body}."
    reduce_instance = Reduce(
        input=pl.Series([]),
        user_instruction=user_instruction,
        model=mock_language_model,
        max_tokens=1024,
        temperature=0,
    )
    reduce_instance.prefix_tokens = 50
    return reduce_instance


def test_leaf_level_single_batch(reduce_instance, mock_language_model):
    docs = [
        {"TITLE": "Title 1", "BODY": "Body 1"},
        {"TITLE": "Title 2", "BODY": "Body 2"},
    ]

    # Determine the token count of the leaf instruction template
    mock_language_model.count_tokens.return_value = 10  # Set a default for template count
    leaf_template_tokens = mock_language_model.count_tokens(
        Reduce.LEAF_INSTRUCTION_TEMPLATE
    )

    # Now set the side_effect to account for:
    # 1. Template token count
    # 2. Token count of the first formatted document
    # 3. Token count of the second formatted document
    mock_language_model.count_tokens.side_effect = [leaf_template_tokens, 20, 25]

    batches = reduce_instance._build_request_messages_batch(docs, 0)
    assert len(batches) == 1
    messages_batch = batches[0].to_message_list()
    user_message = messages_batch[1]
    assert len(messages_batch) == 2
    assert (
        "Document 1:\n[TITLE]: «Title 1»\n[BODY]: «Body 1»" in user_message["content"]
    )
    assert (
        "Document 2:\n[TITLE]: «Title 2»\n[BODY]: «Body 2»" in user_message["content"]
    )
    assert (
        uppercase_instruction_placeholder(reduce_instance.user_instruction)
        in user_message["content"]
    )
    assert (
        Reduce.LEAF_INSTRUCTION_TEMPLATE.split("{user_instruction}")[0]
        in user_message["content"]
    )

def test_context_window_reduction_factor(reduce_instance, mock_language_model):
    """Tests that there is no combination of max_output_tokens and max_context_window_length that would result in a negative max_input_tokens."""
    mock_language_model.max_context_window_length = 1000
    mock_language_model.model_parameters.max_output_tokens = 640 # > 1/3 of the max context window length
    # the maximum allowed input tokens should be (1000 - 640) / 3 = 120, so the first request should be fine
    mock_language_model.count_tokens.side_effect = [10, 50]
    reduce_instance._build_request_messages_batch([{"TITLE": "Title 1", "BODY": "Body 1"}], 0)

    # the second request should be too large
    mock_language_model.count_tokens.side_effect = [10, 120]
    with pytest.raises(ValueError) as exc_info:
        reduce_instance._build_request_messages_batch([{"TITLE": "Title 1", "BODY": "Body 1"}], 0)
    assert "sem.reduce document is too large" in str(exc_info.value)


def test_node_level_multiple_batches(reduce_instance, mock_language_model):
    """Tests batching at the node level when documents require multiple batches."""
    docs = ["Summary 1", "Summary 2", "Summary 3"]

    node_template_tokens = 15
    max_allowed_content_tokens = (
            mock_language_model.max_context_window_length / 3
            - mock_language_model.model_parameters.max_output_tokens
            - reduce_instance.prefix_tokens
            - node_template_tokens
    )
    fitting_tokens = max_allowed_content_tokens - 5

    # Now set the side_effect to account for:
    # 1. Template token count
    # 2. Token count of "Document 1: Summary 1"
    # 3. Token count of "Document 2: Summary 2"
    # 4. Token count of "Document 3: Summary 3"
    # Docs 1 and 2 fit in one batch, doc 3 fits in the next
    mock_language_model.count_tokens.side_effect = [
        node_template_tokens,
        fitting_tokens // 2,
        fitting_tokens // 2 + 1,
        100,
    ]

    messages_batch = reduce_instance._build_request_messages_batch(docs, 1)
    assert len(messages_batch) == 2
    first_batch = messages_batch[0].to_message_list()
    assert len(first_batch) == 2
    first_batch_user_message = first_batch[1]
    assert (
        "Document 1:\nSummary 1\nDocument 2:\nSummary 2"
        in first_batch_user_message["content"]
    )
    assert (
        Reduce.NODE_INSTRUCTION_TEMPLATE.split("{user_instruction}")[0]
        in first_batch_user_message["content"]
    )
    assert len(messages_batch[1].to_message_list()) == 2
    second_batch = messages_batch[1].to_message_list()
    assert len(second_batch) == 2
    second_batch_user_message = second_batch[1]
    assert "Document 3:\nSummary 3" in second_batch_user_message["content"]
    assert (
        Reduce.NODE_INSTRUCTION_TEMPLATE.split("{user_instruction}")[0]
        in second_batch_user_message["content"]
    )
    assert (
        uppercase_instruction_placeholder(reduce_instance.user_instruction)
        in first_batch_user_message["content"]
    )
    assert (
        uppercase_instruction_placeholder(reduce_instance.user_instruction)
        in second_batch_user_message["content"]
    )


def test_single_document_exceeds_limit(reduce_instance, mock_language_model):
    """Tests the scenario where a single document exceeds the maximum token limit."""
    long_doc = {"TEXT": "This is a very long document"}
    mock_language_model.count_tokens.return_value = mock_language_model.max_context_window_length

    with pytest.raises(ValueError) as exc_info:
        reduce_instance._build_request_messages_batch([long_doc], 0)
    assert "sem.reduce document is too large" in str(exc_info.value)


def test_leaf_level_formatting(reduce_instance, mock_language_model):
    """Tests the formatting of documents at the leaf level."""
    doc = {"FIELD_A": "Value A", "FIELD_B": "Value B"}
    mock_language_model.count_tokens.return_value = 30
    batches = reduce_instance._build_request_messages_batch([doc], 0)
    assert len(batches) == 1
    messages_batch = batches[0].to_message_list()
    assert len(messages_batch) == 2
    user_message = messages_batch[1]
    assert (
        "Document 1:\n[FIELD_A]: «Value A»\n[FIELD_B]: «Value B»"
        in user_message["content"]
    )


def test_node_level_formatting(reduce_instance, mock_language_model):
    """Tests the formatting of documents at the node level."""
    doc = "Previous summary content"
    mock_language_model.count_tokens.return_value = 25
    batches = reduce_instance._build_request_messages_batch([doc], 1)
    assert len(batches) == 1
    messages_batch = batches[0].to_message_list()
    assert len(messages_batch) == 2
    user_message = messages_batch[1]
    assert "Document 1:\nPrevious summary content" in user_message["content"]


def test_leaf_level_null_handling(reduce_instance, mock_language_model):
    """Tests how _create_batches handles null or empty documents at the leaf level."""
    docs_with_nulls = [
        {"TITLE": "Title 1", "BODY": "Body 1"},
        None,
        {},
        {"TITLE": None, "BODY": "Body 2"},
        {"TITLE": "Title 3", "BODY": None},
        {"TITLE": "", "BODY": "Body 4"},
    ]
    mock_language_model.count_tokens.side_effect = [
        10,  # leaf template tokens
        20,  # token count of "Document 1..."
    ]
    batches = reduce_instance._build_request_messages_batch(docs_with_nulls, 0)
    assert len(batches) == 1
    messages_batch = batches[0].to_message_list()
    assert len(messages_batch) == 2
    user_message = messages_batch[1]
    assert (
        "Document 1:\n[TITLE]: «Title 1»\n[BODY]: «Body 1»" in user_message["content"]
    )
    # None document should be skipped
    assert "Document 2" not in user_message["content"]
    # Empty dict should be skipped
    assert "Document 3" not in user_message["content"]
    # Documents with None values should be skipped
    assert "Document 4" not in user_message["content"]
    # Empty string should be skipped
    assert "Document 5" not in user_message["content"]


def test_node_level_null_handling(reduce_instance, mock_language_model):
    """Tests how _create_batches handles null or empty documents at the node level."""
    docs_with_nulls = [
        "Summary 1",
        None,
        "",
        "Summary 2",
    ]
    mock_language_model.count_tokens.side_effect = [
        15,  # node template tokens
        10,  # token count of "Document 1: Summary 1"
        12,  # token count of "Document 2: Summary 2"
    ]
    batches = reduce_instance._build_request_messages_batch(docs_with_nulls, 1)
    assert len(batches) == 1
    messages_batch = batches[0].to_message_list()
    assert len(messages_batch) == 2
    user_message = messages_batch[1]
    assert "Document 1:\nSummary 1" in user_message["content"]
    assert "Document 2:\nSummary 2" in user_message["content"]
    # None and empty strings should be skipped


def test_empty_document_list(reduce_instance, mock_language_model):
    """Tests how _create_batches handles an empty list of documents."""
    mock_language_model.count_tokens.return_value = 10  # For template
    batches = reduce_instance._build_request_messages_batch([], 0)
    assert batches is None
