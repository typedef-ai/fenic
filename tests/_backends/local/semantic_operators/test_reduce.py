from unittest.mock import MagicMock

import polars as pl
import pytest

from fenic._backends.local.semantic_operators.reduce import (
    DATA_COLUMN_NAME,
    SORT_KEY_COLUMN_NAME,
    Reduce,
)


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
    user_instruction = "Summarize the documents."

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

    batches = reduce_instance._build_request_messages_batch(user_instruction, docs, 0)
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
    user_instruction = "Summarize the documents."

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

    messages_batch = reduce_instance._build_request_messages_batch(user_instruction, docs, 1)
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
    user_instruction = "Summarize the documents."

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
        reduce_instance._build_request_messages_batch(user_instruction, [long_doc], 0)

    error_msg = str(exc_info.value)
    assert "semantic.reduce document is too large" in error_msg
    assert "(2560 tokens)" in error_msg  # user_message_tokens + doc_tokens
    assert "(2029 tokens)" in error_msg  # max_input_tokens


def test_context_window_edge_case(reduce_instance, mock_language_model):
    """Test when max_output_tokens is very large relative to context window."""
    user_instruction = "Summarize the documents."

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
    batches = reduce_instance._build_request_messages_batch(user_instruction, ["Small doc"], 0)
    assert len(batches) == 1

    # Second doc exceeds limit
    mock_language_model.count_tokens.side_effect = [10, 200]
    with pytest.raises(ValueError) as exc_info:
        reduce_instance._build_request_messages_batch(user_instruction, ["Larger doc"], 0)
    assert "semantic.reduce document is too large" in str(exc_info.value)


def test_empty_document_handling(reduce_instance, mock_language_model):
    """Test how empty documents are handled."""
    user_instruction = "Summarize the documents."
    docs_with_empties = [
        "Document 1 content",
        "",  # Empty string should be skipped
        "Document 2 content",
        None,
    ]

    # Token accounting:
    # Empty docs are skipped before token counting
    #
    # Token counting calls:
    # 1. count_tokens(instruction) -> 10
    # 2. count_tokens("Document 1 content") -> 20
    # 3. Skip empty string
    # 4. count_tokens("Document 2 content") -> 25
    # 5. Skip None

    mock_language_model.count_tokens.side_effect = [10, 20, 25]

    batches = reduce_instance._build_request_messages_batch(user_instruction, docs_with_empties, 0)
    assert len(batches) == 1

    user_content = batches[0].user
    # Documents should be renumbered after skipping empty
    assert "<document1>\nDocument 1 content\n</document1>" in user_content
    assert "<document2>\nDocument 2 content\n</document2>" in user_content


def test_empty_document_list(reduce_instance, mock_language_model):
    """Test handling of empty document list."""
    user_instruction = "Summarize the documents."

    # No token counting should occur for empty list
    batches = reduce_instance._build_request_messages_batch(user_instruction, [], 0)
    assert batches is None

    # Also test list with only empty strings
    batches = reduce_instance._build_request_messages_batch(user_instruction, ["", "", ""], 0)
    assert batches is None


def test_hierarchical_reduction_logic(mock_language_model):
    """Test that hierarchical reduction properly reduces through levels."""
    # Create structured data
    docs_data = [
        {DATA_COLUMN_NAME: "Doc 1"},
        {DATA_COLUMN_NAME: "Doc 2"},
        {DATA_COLUMN_NAME: "Doc 3"},
        {DATA_COLUMN_NAME: "Doc 4"},
    ]
    group = pl.DataFrame(docs_data).to_struct()

    reduce_instance = Reduce(
        input=pl.Series([group]),
        user_instruction="Summarize the documents.",
        model=mock_language_model,
        max_tokens=1024,
        temperature=0,
    )
    reduce_instance.prefix_tokens = 50

    # Force batching: make only 2 docs fit per batch
    mock_language_model.max_context_window_length = 500
    mock_language_model.model_parameters.max_output_tokens = 100

    # Token accounting:
    # - max_input_tokens = floor((500 - 100) * 0.7) = floor(280) = 280
    # - user_message_tokens = 10 + 50 = 60
    # - Available for docs = 280 - 60 = 220 tokens

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


def test_group_context_injection(mock_language_model):
    """Test that group context variables are properly injected into instructions."""
    # Create groups with different contexts
    docs_data_sales = [
        {DATA_COLUMN_NAME: "Sales report Q1", "department": "Sales", "region": "North"},
        {DATA_COLUMN_NAME: "Sales report Q2", "department": "Sales", "region": "North"},
    ]
    docs_data_eng = [
        {DATA_COLUMN_NAME: "Engineering update", "department": "Engineering", "region": "West"},
        {DATA_COLUMN_NAME: "Tech roadmap", "department": "Engineering", "region": "West"},
    ]

    group_sales = pl.DataFrame(docs_data_sales).to_struct()
    group_eng = pl.DataFrame(docs_data_eng).to_struct()

    # Template with context variables
    user_instruction = "Summarize these {{department}} documents from the {{region}} region."

    reduce_instance = Reduce(
        input=pl.Series([group_sales, group_eng]),
        user_instruction=user_instruction,
        model=mock_language_model,
        max_tokens=1024,
        temperature=0,
        group_context_names=["department", "region"],
    )
    reduce_instance.prefix_tokens = 50

    # Mock simple responses
    mock_language_model.count_tokens.return_value = 10
    mock_language_model.get_completions.return_value = [
        MagicMock(completion="Summary of group")
    ]

    # Execute and verify the correct templates were used
    reduce_instance.execute()

    # Check that get_completions was called with the right instructions
    calls = mock_language_model.get_completions.call_args_list

    # First group should have Sales/North in the instruction
    first_call_messages = calls[0][1]['messages'][0]
    assert "Sales documents from the North region" in first_call_messages.user

    # Second group should have Engineering/West in the instruction
    second_call_messages = calls[1][1]['messages'][0]
    assert "Engineering documents from the West region" in second_call_messages.user


def test_sorting_ascending(mock_language_model):
    """Test that documents are sorted in ascending order when specified."""
    docs_data = [
        {DATA_COLUMN_NAME: "Doc from Feb", SORT_KEY_COLUMN_NAME: "2024-02-01"},
        {DATA_COLUMN_NAME: "Doc from Jan", SORT_KEY_COLUMN_NAME: "2024-01-01"},
        {DATA_COLUMN_NAME: "Doc from Mar", SORT_KEY_COLUMN_NAME: "2024-03-01"},
    ]
    group = pl.DataFrame(docs_data).to_struct()

    reduce_instance = Reduce(
        input=pl.Series([group]),
        user_instruction="Summarize the documents.",
        model=mock_language_model,
        max_tokens=1024,
        temperature=0,
        ascending=True,  # Sort by date ascending
    )
    reduce_instance.prefix_tokens = 50

    mock_language_model.count_tokens.return_value = 10
    mock_language_model.get_completions.return_value = [
        MagicMock(completion="Summary")
    ]

    reduce_instance.execute()

    # Check that documents were processed in sorted order
    calls = mock_language_model.get_completions.call_args_list
    first_call_messages = calls[0][1]['messages'][0]

    # Documents should appear in chronological order
    assert first_call_messages.user.index("Doc from Jan") < first_call_messages.user.index("Doc from Feb")
    assert first_call_messages.user.index("Doc from Feb") < first_call_messages.user.index("Doc from Mar")


def test_sorting_descending(mock_language_model):
    """Test that documents are sorted in descending order when specified."""
    docs_data = [
        {DATA_COLUMN_NAME: "Priority 2", SORT_KEY_COLUMN_NAME: 2},
        {DATA_COLUMN_NAME: "Priority 1", SORT_KEY_COLUMN_NAME: 1},
        {DATA_COLUMN_NAME: "Priority 3", SORT_KEY_COLUMN_NAME: 3},
    ]
    group = pl.DataFrame(docs_data).to_struct()

    reduce_instance = Reduce(
        input=pl.Series([group]),
        user_instruction="Summarize the documents.",
        model=mock_language_model,
        max_tokens=1024,
        temperature=0,
        ascending=False,
    )
    reduce_instance.prefix_tokens = 50

    mock_language_model.count_tokens.return_value = 10
    mock_language_model.get_completions.return_value = [
        MagicMock(completion="Summary")
    ]

    reduce_instance.execute()

    # Check that documents were processed in reverse priority order
    calls = mock_language_model.get_completions.call_args_list
    first_call_messages = calls[0][1]['messages'][0]

    # Documents should appear in descending priority order
    assert first_call_messages.user.index("Priority 3") < first_call_messages.user.index("Priority 2")
    assert first_call_messages.user.index("Priority 2") < first_call_messages.user.index("Priority 1")


def test_no_sorting_preserves_order(mock_language_model):
    """Test that original order is preserved when no sorting is specified."""
    docs_data = [
        {DATA_COLUMN_NAME: "First doc", SORT_KEY_COLUMN_NAME: 1},
        {DATA_COLUMN_NAME: "Second doc", SORT_KEY_COLUMN_NAME: 2},
        {DATA_COLUMN_NAME: "Third doc", SORT_KEY_COLUMN_NAME: 3},
    ]
    group = pl.DataFrame(docs_data).to_struct()

    reduce_instance = Reduce(
        input=pl.Series([group]),
        user_instruction="Summarize the documents.",
        model=mock_language_model,
        max_tokens=1024,
        temperature=0,
    )
    reduce_instance.prefix_tokens = 50

    mock_language_model.count_tokens.return_value = 10
    mock_language_model.get_completions.return_value = [
        MagicMock(completion="Summary")
    ]

    reduce_instance.execute()

    # Check that documents appear in original order
    calls = mock_language_model.get_completions.call_args_list
    first_call_messages = calls[0][1]['messages'][0]

    assert first_call_messages.user.index("First doc") < first_call_messages.user.index("Second doc")
    assert first_call_messages.user.index("Second doc") < first_call_messages.user.index("Third doc")


def test_sorting_with_group_context(mock_language_model):
    """Test that sorting and group context work together correctly."""
    # Create two groups with different contexts and sortable data
    docs_data_sales = [
        {DATA_COLUMN_NAME: "Sales Feb", "department": "Sales", SORT_KEY_COLUMN_NAME: "2024-02-01"},
        {DATA_COLUMN_NAME: "Sales Jan", "department": "Sales", SORT_KEY_COLUMN_NAME: "2024-01-01"},
    ]
    docs_data_eng = [
        {DATA_COLUMN_NAME: "Eng Mar", "department": "Engineering", SORT_KEY_COLUMN_NAME: "2024-03-01"},
        {DATA_COLUMN_NAME: "Eng Feb", "department": "Engineering", SORT_KEY_COLUMN_NAME: "2024-02-01"},
    ]

    group_sales = pl.DataFrame(docs_data_sales).to_struct()
    group_eng = pl.DataFrame(docs_data_eng).to_struct()

    reduce_instance = Reduce(
        input=pl.Series([group_sales, group_eng]),
        user_instruction="Summarize {{department}} documents in chronological order.",
        model=mock_language_model,
        max_tokens=1024,
        temperature=0,
        group_context_names=["department"],
        ascending=True,  # Sort by date ascending
    )
    reduce_instance.prefix_tokens = 50

    mock_language_model.count_tokens.return_value = 10
    mock_language_model.get_completions.return_value = [
        MagicMock(completion="Summary")
    ]

    reduce_instance.execute()

    calls = mock_language_model.get_completions.call_args_list

    # First group (Sales) should have correct context and sorting
    first_call_messages = calls[0][1]['messages'][0]
    assert "Sales documents" in first_call_messages.user
    assert first_call_messages.user.index("Sales Jan") < first_call_messages.user.index("Sales Feb")

    # Second group (Engineering) should have correct context and sorting
    second_call_messages = calls[1][1]['messages'][0]
    assert "Engineering documents" in second_call_messages.user
    assert second_call_messages.user.index("Eng Feb") < second_call_messages.user.index("Eng Mar")


def test_empty_group(mock_language_model):
    """Test handling of empty groups."""
    # Create an empty group
    empty_group = pl.DataFrame({DATA_COLUMN_NAME: []}).to_struct()

    reduce_instance = Reduce(
        input=pl.Series([empty_group]),
        user_instruction="Summarize the documents.",
        model=mock_language_model,
        max_tokens=1024,
        temperature=0,
    )

    result = reduce_instance.execute()
    assert result[0] is None  # Empty group should return None


def test_all_empty_documents_in_group(mock_language_model):
    """Test when all documents in a group are empty strings."""
    docs_data = [
        {DATA_COLUMN_NAME: ""},
        {DATA_COLUMN_NAME: ""},
    ]
    group = pl.DataFrame(docs_data).to_struct()

    reduce_instance = Reduce(
        input=pl.Series([group]),
        user_instruction="Summarize the documents.",
        model=mock_language_model,
        max_tokens=1024,
        temperature=0,
    )

    # No get_completions should be called for empty documents
    result = reduce_instance.execute()
    assert result[0] is None
    mock_language_model.get_completions.assert_not_called()
