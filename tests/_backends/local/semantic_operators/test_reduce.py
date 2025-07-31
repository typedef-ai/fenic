from unittest.mock import MagicMock

import polars as pl
import pytest

from fenic._backends.local.semantic_operators.reduce import Reduce


@pytest.fixture
def mock_language_model():
    """Pytest fixture to create a mock LanguageModel object."""
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
    """Pytest fixture to create an instance of the Reduce class."""
    user_instruction = "Summarize the documents."
    reduce_instance = Reduce(
        input=pl.Series([]),
        user_instruction=user_instruction,
        model=mock_language_model,
        max_tokens=1024,
        temperature=0,
    )
    reduce_instance.prefix_tokens = 50
    return reduce_instance


def test_single_batch(reduce_instance, mock_language_model):
    """Test that documents fitting in context are batched together."""
    docs = ["Document one content", "Document two content"]

    # Token accounting:
    # - prefix_tokens = 50 (from fixture)
    # - instruction = "Summarize the documents." = 10 tokens
    # - user_message_tokens = 10 + 50 = 60
    # - max_input_tokens = floor((3000 - 100) * 0.7) = floor(2030) = 2030
    #
    # Token counting calls:
    # 1. count_tokens(instruction) -> 10
    # 2. count_tokens("Document one content") -> 20
    # 3. count_tokens("Document two content") -> 25
    #
    # Batch calculation:
    # - After doc 1: 60 + 20 = 80 (fits)
    # - After doc 2: 60 + 20 + 25 = 105 (still fits in 2030)

    mock_language_model.count_tokens.side_effect = [10, 20, 25]

    batches = reduce_instance._build_request_messages_batch(docs, 0)
    assert len(batches) == 1

    # Verify the formatted message contains XML-formatted documents
    messages = batches[0]
    user_content = messages.user

    assert "Summarize the documents." in user_content
    assert "<document1>\nDocument one content\n</document1>" in user_content
    assert "<document2>\nDocument two content\n</document2>" in user_content


def test_multiple_batches(reduce_instance, mock_language_model):
    """Test batching when documents require multiple batches."""
    docs = ["Summary 1", "Summary 2", "Summary 3"]

    # Adjust context window to force batching
    mock_language_model.max_context_window_length = 1000
    mock_language_model.model_parameters.max_output_tokens = 100

    # Token accounting:
    # - max_input_tokens = floor((1000 - 100) * 0.7) = floor(630) = 630
    # - user_message_tokens = 10 + 50 = 60
    #
    # Token counting calls:
    # 1. count_tokens(instruction) -> 10
    # 2. count_tokens("Summary 1") -> 250
    # 3. count_tokens("Summary 2") -> 250
    # 4. count_tokens("Summary 3") -> 250
    #
    # Batch calculation:
    # - After doc 1: 60 + 250 = 310 (fits in 630)
    # - After doc 2: 60 + 250 + 250 = 560 (fits in 630)
    # - After doc 3: 60 + 250 + 250 + 250 = 810 (exceeds 630!)
    #   -> Flush batch with docs 1&2, start new batch with doc 3

    mock_language_model.count_tokens.side_effect = [10, 250, 250, 250]

    messages_batch = reduce_instance._build_request_messages_batch(docs, 1)
    assert len(messages_batch) == 2

    # First batch: docs 1 and 2
    first_batch_content = messages_batch[0].user
    assert "<document1>\nSummary 1\n</document1>" in first_batch_content
    assert "<document2>\nSummary 2\n</document2>" in first_batch_content
    assert "<document3>" not in first_batch_content

    # Second batch: doc 3 (renumbered as document1)
    second_batch_content = messages_batch[1].user
    assert "<document1>\nSummary 3\n</document1>" in second_batch_content
    assert "Summary 1" not in second_batch_content
    assert "Summary 2" not in second_batch_content


def test_single_document_exceeds_limit(reduce_instance, mock_language_model):
    """Test error when a single document exceeds the maximum token limit."""
    long_doc = "This is a very long document that exceeds limits"

    # Token accounting:
    # - max_input_tokens = floor((3000 - 100) * 0.7) = floor(2029.99) = 2029
    # - user_message_tokens = 10 + 50 = 60
    #
    # Token counting calls:
    # 1. count_tokens(instruction) -> 10
    # 2. count_tokens(long_doc) -> 2500
    #
    # Check: 60 + 2500 = 2560 > 2029 (exceeds limit!)

    mock_language_model.count_tokens.side_effect = [10, 2500]

    with pytest.raises(ValueError) as exc_info:
        reduce_instance._build_request_messages_batch([long_doc], 0)

    error_msg = str(exc_info.value)
    assert "semantic.reduce document is too large" in error_msg
    assert "(2560 tokens)" in error_msg  # user_message_tokens + doc_tokens
    assert "(2029 tokens)" in error_msg  # max_input_tokens


def test_context_window_edge_case(reduce_instance, mock_language_model):
    """Test when max_output_tokens is very large relative to context window."""
    # Set up edge case where output tokens take up most of context
    mock_language_model.max_context_window_length = 1000
    mock_language_model.model_parameters.max_output_tokens = 640

    # Token accounting:
    # - theoretical_max = 1000 - 640 = 360
    # - max_input_tokens = floor(360 * 0.7) = floor(252) = 252
    # - user_message_tokens = 10 + 50 = 60
    #
    # Available for docs: 252 - 60 = 192 tokens

    # First doc fits
    mock_language_model.count_tokens.side_effect = [10, 150]
    batches = reduce_instance._build_request_messages_batch(["Small doc"], 0)
    assert len(batches) == 1

    # Second doc exceeds limit
    mock_language_model.count_tokens.side_effect = [10, 200]
    with pytest.raises(ValueError) as exc_info:
        reduce_instance._build_request_messages_batch(["Larger doc"], 0)
    assert "semantic.reduce document is too large" in str(exc_info.value)


def test_empty_document_handling(reduce_instance, mock_language_model):
    """Test how empty documents are handled."""
    docs_with_empties = [
        "Document 1 content",
        "",  # Empty string should be skipped
        "Document 2 content",
    ]

    # Token accounting:
    # Empty docs are skipped before token counting
    #
    # Token counting calls:
    # 1. count_tokens(instruction) -> 10
    # 2. count_tokens("Document 1 content") -> 20
    # 3. Skip empty string
    # 4. count_tokens("Document 2 content") -> 25

    mock_language_model.count_tokens.side_effect = [10, 20, 25]

    batches = reduce_instance._build_request_messages_batch(docs_with_empties, 0)
    assert len(batches) == 1

    user_content = batches[0].user
    # Documents should be renumbered after skipping empty
    assert "<document1>\nDocument 1 content\n</document1>" in user_content
    assert "<document2>\nDocument 2 content\n</document2>" in user_content


def test_empty_document_list(reduce_instance, mock_language_model):
    """Test handling of empty document list."""
    # No token counting should occur for empty list
    batches = reduce_instance._build_request_messages_batch([], 0)
    assert batches is None

    # Also test list with only empty strings
    batches = reduce_instance._build_request_messages_batch(["", "", ""], 0)
    assert batches is None


def test_hierarchical_reduction_logic(reduce_instance, mock_language_model):
    """Test that hierarchical reduction properly reduces through levels."""
    # Create a group with 4 documents
    group = pl.Series(["Doc 1", "Doc 2", "Doc 3", "Doc 4"])
    reduce_instance.input = pl.Series([group])

    # Force batching: make only 2 docs fit per batch
    mock_language_model.max_context_window_length = 500
    mock_language_model.model_parameters.max_output_tokens = 100

    # Token accounting:
    # - max_input_tokens = floor((500 - 100) * 0.7) = floor(280) = 280
    # - user_message_tokens = 10 + 50 = 60
    # - Available for docs = 280 - 60 = 220 tokens

    # To force 2 docs per batch:
    # - Need each doc small enough that 2 fit: 2 * doc_tokens <= 220
    # - So each doc should be <= 110 tokens
    # - But need 3 docs to NOT fit: 3 * doc_tokens > 220
    # - So each doc should be > 73 tokens
    # - Let's use 80 tokens per doc
    # - Batch check: 60 + 80 = 140 (1 fits)
    # - Batch check: 60 + 80 + 80 = 220 (2 fit exactly)
    # - Batch check: 60 + 80 + 80 + 80 = 300 > 280 (3 don't fit)

    # Track all get_completions calls to verify hierarchy
    call_count = 0
    token_call_count = 0

    def mock_count_tokens(text):
        nonlocal token_call_count
        token_call_count += 1

        if token_call_count == 1:
            return 10  # instruction
        elif token_call_count <= 5:  # 4 docs
            return 80  # each doc
        elif token_call_count == 6:
            return 10  # instruction again for level 1
        else:
            return 50  # summaries are smaller

    def mock_get_completions(messages, **kwargs):
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            # Level 0: Should receive 2 message batches (2 docs each)
            assert len(messages) == 2
            # Return 2 summaries
            return [
                MagicMock(completion="Summary A"),
                MagicMock(completion="Summary B")
            ]
        elif call_count == 2:
            # Level 1: Should receive 1 message batch (2 summaries)
            assert len(messages) == 1
            # Verify it's processing the summaries from level 0
            user_msg = messages[0].user
            assert "Summary A" in user_msg
            assert "Summary B" in user_msg
            return [MagicMock(completion="Final summary")]
        else:
            raise AssertionError("Unexpected call to get_completions")

    mock_language_model.count_tokens.side_effect = mock_count_tokens
    mock_language_model.get_completions.side_effect = mock_get_completions

    result = reduce_instance.execute()
    assert result[0] == "Final summary"
    assert call_count == 2  # Verify we went through 2 levels
    raise AssertionError()
