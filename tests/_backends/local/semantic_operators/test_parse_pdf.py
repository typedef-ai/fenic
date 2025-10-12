import base64
import math
import os
import re
from textwrap import dedent
from unittest.mock import MagicMock

import jinja2
import polars as pl
import pytest

from fenic._backends.local.semantic_operators.parse_pdf import ParsePDF
from fenic._inference.common_openai.openai_utils import convert_messages
from fenic.core.error import FileLoaderError
from tests.conftest import _save_pdf_file


def _check_base64_string(s):
    """Check if a string is valid base64 encoded content."""
    if not isinstance(s, str):
        return False
    
    # Check if it starts with data:application/pdf;base64, pattern
    assert s.startswith("data:application/pdf;base64,")
    base64_part = s.split(",", 1)[1]
    
    # Check if it's valid base64
    base64.b64decode(base64_part, validate=True)

def compare_openai_converted_messages(actual, expected):
    """
    Compare two lists of converted messages.

    For the file_data field, just verify it's base64 encoded content.
    Args:
        actual: List of actual message dictionaries
        expected: List of expected message dictionaries  
    """
    assert len(actual) == len(expected)
    for actual_messages, expected_messages in zip(actual, expected, strict=False):
        for actual_msg, expected_msg in zip(actual_messages, expected_messages, strict=False):
            # Compare all fields except file_data
            assert "content" in actual_msg
            assert "role" in actual_msg
            assert actual_msg["role"] == expected_msg["role"]
            if actual_msg["role"] == "system":
                assert actual_msg["content"] == expected_msg["content"]
            elif actual_msg["role"] == "user":
                actual_content = actual_msg["content"][0]
                expected_content = expected_msg["content"][0]
                assert "file" in actual_msg["content"][0]
                assert "type" in actual_content
                assert actual_content["type"] == expected_content["type"]
                assert "filename" in actual_content["file"]
                assert "file_data" in actual_content["file"]
                assert actual_content["file"]["filename"] == expected_content["file"]["filename"]
                _check_base64_string(actual_content["file"]["file_data"])
    return True




@pytest.fixture
def mock_language_model():
    """Pytest fixture to create a mock LanguageModel object."""
    mock_language_model = MagicMock()
    mock_language_model.max_context_window_length = 3000
    model_parameters = MagicMock()
    model_parameters.max_context_window_length = 3000
    model_parameters.max_output_tokens = 1000
    mock_language_model.model_parameters = model_parameters
    mock_language_model.count_tokens.return_value = 10
    return mock_language_model

def mock_get_completions(messages, **kwargs):
    completions = [MagicMock(completion=f"Semantic Parse Output: start_page:'{message.user_file.page_range[0]}'") for message in messages]
    return completions

def check_chunk_content_and_order(result, chunks, chunk_max_size):
    # Extract all start_page numbers using regex
    pattern = r"Semantic Parse Output: start_page:'(\d+)'"
    matches = re.findall(pattern, result)

    # Convert to integers and check order
    start_page_numbers = [int(match) for match in matches]

    # Verify we have the expected number of chunks
    assert len(start_page_numbers) == chunks, f"Expected {chunks} chunks, got {len(start_page_numbers)}"

    # Check that pages follow the pattern: 0, chunk_max_size, chunk_max_size*2, ...
    expected_order = [i * chunk_max_size for i in range(chunks)]
    assert start_page_numbers == expected_order, f"Expected order {expected_order}, got {start_page_numbers}"

    return start_page_numbers

def check_chunk_page_separators(result, pages, chunk_max_size):
    pattern = r"--- PAGE (\d+) ---"
    end_page_numbers = re.findall(pattern, result)

    chunk_count = math.ceil(pages / chunk_max_size)

    assert len(end_page_numbers) == chunk_count - 1

    if chunk_count == 1:
        return

    expected_pages = [str(chunk_max_size * (i+1)) for i in range(chunk_count - 1)]
    assert end_page_numbers == expected_pages

class TestParsePDF:
    """Test cases for the ParsePDF operator."""

    expected_system_prompt = jinja2.Template(dedent("""\
        Transcribe the main content of this PDF document to clean, well-formatted markdown.
         - Output should be raw markdown, don't surround the whole output in code fences or backticks.
         - For each topic, create a markdown heading. For key terms, use bold text.
         - Preserve the structure, formatting, headings, lists, table of contents, and any tables using markdown syntax.
         - Format tables as github markdown tables, however:
             - for table headings, immediately add ' |' after the table heading
        {% if multiple_pages %}
        {% if page_separator and '{page}' in page_separator %}
         - Insert the page separator '{{ page_separator }}' as a markdown line for each page break, replacing the '{{ '{page}' }}' pattern with the current page number. If the document contains page numbers, do not include them in the output, instead replace them with this page separator.
        {% elif page_separator %}
         - Don't include the page numbers in the output, instead insert the page separator '{{ page_separator }}' as a markdown line for each page break
        {% endif %}
        {% endif %}
        {% if describe_images %}
         - For each image, describe them briefly in a markdown section with 'Image' in the title, preserving the output order.
        {% else %}
         - Ignore any images that aren't tables or charts that can be converted to markdown.
        {% endif %}""").strip()
    )
    def test_build_prompts_basic(self, local_session, temp_dir_with_test_files):
        """Test basic PDF parsing without any options."""
        file_path_1 = os.path.join(temp_dir_with_test_files, "file8.pdf")
        file_path_2 = os.path.join(temp_dir_with_test_files, "file9.pdf")
        input = pl.Series("input", [file_path_1, file_path_2])

        parse_pdf = ParsePDF(
            input=input,
            model=local_session._session_state.get_language_model(),
        )

        result = list(
            map(
                lambda x: convert_messages(x) if x else None,
                parse_pdf.build_request_messages_batch()[0],
            )
        )
        expected = [
            [
                {
                    "role": "system",
                    "content": self.expected_system_prompt.render(page_separator=None, describe_images=False, multiple_pages=False),
                },
                {
                    "role": "user",
                    "content": [
                    {
                        "type": "file",
                        "file": {
                            "filename": file_path_1,
                            "file_data": "dummy_value", # checked separately in _check_base64_string
                        },
                    }
                ]}
            ],
            [
                {
                    "role": "system",
                    "content": self.expected_system_prompt.render(page_separator=None, describe_images=False, multiple_pages=False),
                },
                {
                    "role": "user",
                    "content": [
                    {
                        "type": "file",
                        "file": {
                            "filename": file_path_2,
                            "file_data": "dummy_value", # checked separately in _check_base64_string
                        },
                    }
                ]}
            ],
        ]
        compare_openai_converted_messages(result, expected)

    def test_build_prompts_with_page_separator(self, local_session, temp_dir_just_one_file, monkeypatch):
        """Test PDF parsing with page separator."""
        file_path_1 = os.path.join(temp_dir_just_one_file, "file_one_page.pdf")
        file_path_2 = os.path.join(temp_dir_just_one_file, "file_two_pages.pdf")
        _save_pdf_file(os.path.join(temp_dir_just_one_file, "file_one_page.pdf"), page_count=1, text_content="dummy text")
        _save_pdf_file(os.path.join(temp_dir_just_one_file, "file_two_pages.pdf"), page_count=2, text_content="dummy text")

        # make sure we're chunking with 1 page sizes
        monkeypatch.setattr("fenic._backends.local.semantic_operators.parse_pdf.PDF_MAX_PAGES_CHUNK", 1)

        input = pl.Series("input", [file_path_1, file_path_2])

        page_separator = "--- PAGE BREAK ---"
        parse_pdf = ParsePDF(
            input=input,
            model=local_session._session_state.get_language_model(),
            page_separator=page_separator,
        )

        result = list(
            map(
                lambda x: convert_messages(x) if x else None,
                parse_pdf.build_request_messages_batch()[0],
            )
        )

        expected = [
            [
                {
                    "role": "system",
                    "content": self.expected_system_prompt.render(page_separator=page_separator, describe_images=False, multiple_pages=False),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "file",
                            "file": {
                                "filename": file_path_1,
                                "file_data": "dummy_value", # checked separately in _check_base64_string
                            },
                        }
                    ],
                },
            ],
            [
                {
                    "role": "system",
                    "content": self.expected_system_prompt.render(page_separator=page_separator, describe_images=False, multiple_pages=True),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "file",
                            "file": {
                                "filename": file_path_2,
                                "file_data": "dummy_value", # checked separately in _check_base64_string
                            },
                        }
                    ],
                },
            ],
                        [
                {
                    "role": "system",
                    "content": self.expected_system_prompt.render(page_separator=page_separator, describe_images=False, multiple_pages=True),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "file",
                            "file": {
                                "filename": file_path_2,
                                "file_data": "dummy_value", # checked separately in _check_base64_string
                            },
                        }
                    ],
                },
            ],
        ]
        compare_openai_converted_messages(result, expected)

    def test_build_prompts_with_page_number_placeholder(self, local_session, temp_dir_with_test_files):
        """Test PDF parsing with page number placeholder in separator."""
        file_path = os.path.join(temp_dir_with_test_files, "file8.pdf")
        input = pl.Series("input", [file_path])

        page_separator = "--- PAGE {page} ---"
        parse_pdf = ParsePDF(
            input=input,
            model=local_session._session_state.get_language_model(),
            page_separator=page_separator,
        )

        result = list(
            map(
                lambda x: convert_messages(x) if x else None,
                parse_pdf.build_request_messages_batch()[0],
            )
        )

        expected = [
            [
                {
                    "role": "system",
                    "content": self.expected_system_prompt.render(page_separator=None, describe_images=False, multiple_pages=False),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "file",
                            "file": {
                                "filename": file_path,
                                "file_data": "dummy_value", # checked separately in _check_base64_string
                            },
                        }
                    ],
                },
            ]
        ]

        compare_openai_converted_messages(result, expected)

    def test_build_prompts_with_describe_images(self, local_session, temp_dir_with_test_files):
        """Test PDF parsing with image description enabled."""
        file_path = os.path.join(temp_dir_with_test_files, "file8.pdf")
        input = pl.Series("input", [file_path])

        parse_pdf = ParsePDF(
            input=input,
            model=local_session._session_state.get_language_model(),
            describe_images=True,
        )

        result = list(
            map(
                lambda x: convert_messages(x) if x else None,
                parse_pdf.build_request_messages_batch()[0],
            )
        )

        expected = [
            [
                {
                    "role": "system",
                    "content": self.expected_system_prompt.render(page_separator=None, describe_images=True, multiple_pages=False),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "file",
                            "file": {
                                "filename": file_path,
                                "file_data": "dummy_value", # checked separately in _check_base64_string
                            },
                        }
                    ],
                },
            ]
        ]

        compare_openai_converted_messages(result, expected)

    def test_build_prompts_with_all_options(self, local_session, temp_dir_just_one_file, monkeypatch):
        """Test PDF parsing with page number placeholder in separator and image description enabled.

        Create a pdf with 2 pages."""
        file_path = os.path.join(temp_dir_just_one_file, "file_two_pages.pdf")
        _save_pdf_file(os.path.join(temp_dir_just_one_file, "file_two_pages.pdf"), page_count=2, text_content="dummy text")
        input = pl.Series("input", [file_path])

        page_separator = "--- PAGE {page} ---"

        # make sure we're chunking with 1 page sizes
        monkeypatch.setattr("fenic._backends.local.semantic_operators.parse_pdf.PDF_MAX_PAGES_CHUNK", 1)

        parse_pdf = ParsePDF(
            input=input,
            model=local_session._session_state.get_language_model(),
            page_separator=page_separator,
            describe_images=True,
        )

        result = list(
            map(
                lambda x: convert_messages(x) if x else None,
                parse_pdf.build_request_messages_batch()[0],
            )
        )

        expected = [
            [
                {
                    "role": "system",
                    "content": self.expected_system_prompt.render(page_separator=page_separator, describe_images=True, multiple_pages=True),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "file",
                            "file": {
                                "filename": file_path,
                                "file_data": "dummy_value", # checked separately in _check_base64_string
                            },
                        }
                    ],
                },
            ],
            [
                {
                    "role": "system",
                    "content": self.expected_system_prompt.render(page_separator=page_separator, describe_images=True, multiple_pages=True),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "file",
                            "file": {
                                "filename": file_path,
                                "file_data": "dummy_value", # checked separately in _check_base64_string
                            },
                        }
                    ],
                },
            ]
        ]

        compare_openai_converted_messages(result, expected)

    def test_handles_invalid_file_extensions(self, local_session, temp_dir_with_test_files):
        """Test handling of invalid file extensions in input."""
        input = pl.Series("input", [os.path.join(temp_dir_with_test_files, "file1.md")])

        with pytest.raises(FileLoaderError, match="Only files with the extension 'pdf' are supported in this plan."):
            _ = ParsePDF(
                input=input,
                model=local_session._session_state.get_language_model(),
            )

    def test_handles_non_existing_files(self, local_session, temp_dir_with_test_files):
        """Test handling of non-existing files in input."""
        input = pl.Series("input", ["dir/nonexistent.pdf"])

        with pytest.raises(FileNotFoundError, match="Path does not exist: dir/nonexistent.pdf"):
            _ = ParsePDF(
                input=input,
                model=local_session._session_state.get_language_model(),
            )



    def test_pdf_chunking_based_on_token_limit(self, temp_dir_just_one_file, mock_language_model):
        #create pdfs of varying page counts.  Mock token count for each page to be 50 tokens.
        page_counts = [1, 5, 10, 20]
        pdf_paths = []
        for i in range(len(page_counts)):
            path = os.path.join(temp_dir_just_one_file, f"file{i}.pdf")
            _save_pdf_file(path,
                   title="File Title {i}", author="File Author {i}", page_count=page_counts[i],
                   text_content="dummy text")
            pdf_paths.append(path)


        input = pl.Series("input", pdf_paths)


        mock_language_model.model_parameters.max_output_tokens = 150
        mock_language_model.count_tokens.return_value = 50
        mock_language_model.get_completions.side_effect = mock_get_completions

        parse_pdf = ParsePDF(
            input=input,
            model=mock_language_model,
        )

        result = parse_pdf.execute()
        assert result.shape == (len(page_counts),)

        check_chunk_content_and_order(result[0], chunks=1, chunk_max_size=2)
        check_chunk_content_and_order(result[1], chunks=3, chunk_max_size=2)
        check_chunk_content_and_order(result[2], chunks=5, chunk_max_size=2)
        check_chunk_content_and_order(result[3], chunks=10, chunk_max_size=2)

    def test_pdf_chunking_based_on_internal_limit(self, temp_dir_just_one_file, mock_language_model, monkeypatch):
        # create a pdfs with varying page counts.  Mock max_output_tokens to be something larger than the total number of tokens in the pdfs.
        page_counts = [1, 5, 10, 20]
        pdf_paths = []
        for i in range(len(page_counts)):
            path = os.path.join(temp_dir_just_one_file, f"file{i}.pdf")
            _save_pdf_file(path,
                   title="File Title {i}", author="File Author {i}", page_count=page_counts[i],
                   text_content="dummy text")
            pdf_paths.append(path)

        mock_language_model.max_output_tokens = 100_000
        mock_language_model.count_tokens.return_value = 50
        mock_language_model.get_completions.side_effect = mock_get_completions

        input = pl.Series("input", pdf_paths)

        parse_pdf = ParsePDF(
            input=input,
            model=mock_language_model,
        )

        test_chunk_max_size = 4
        monkeypatch.setattr("fenic._backends.local.semantic_operators.parse_pdf.PDF_MAX_PAGES_CHUNK", test_chunk_max_size)
        result = parse_pdf.execute()
        assert result.shape == (len(page_counts),)
        check_chunk_content_and_order(result[0], chunks=1, chunk_max_size=test_chunk_max_size)
        check_chunk_content_and_order(result[1], chunks=2, chunk_max_size=test_chunk_max_size)
        check_chunk_content_and_order(result[2], chunks=3, chunk_max_size=test_chunk_max_size)
        check_chunk_content_and_order(result[3], chunks=5, chunk_max_size=test_chunk_max_size)

        test_chunk_max_size = 1
        monkeypatch.setattr("fenic._backends.local.semantic_operators.parse_pdf.PDF_MAX_PAGES_CHUNK", test_chunk_max_size)
        result2 = parse_pdf.execute()
        assert result2.shape == (len(page_counts),)
        check_chunk_content_and_order(result2[0], chunks=1, chunk_max_size=test_chunk_max_size)
        check_chunk_content_and_order(result2[1], chunks=5, chunk_max_size=test_chunk_max_size)
        check_chunk_content_and_order(result2[2], chunks=10, chunk_max_size=test_chunk_max_size)
        check_chunk_content_and_order(result2[3], chunks=20, chunk_max_size=test_chunk_max_size)

        test_chunk_max_size = 10
        monkeypatch.setattr("fenic._backends.local.semantic_operators.parse_pdf.PDF_MAX_PAGES_CHUNK", test_chunk_max_size)
        result3 = parse_pdf.execute()
        assert result3.shape == (len(page_counts),)
        check_chunk_content_and_order(result3[0], chunks=1, chunk_max_size=test_chunk_max_size)
        check_chunk_content_and_order(result3[1], chunks=1, chunk_max_size=test_chunk_max_size)
        check_chunk_content_and_order(result3[2], chunks=1, chunk_max_size=test_chunk_max_size)
        check_chunk_content_and_order(result3[3], chunks=2, chunk_max_size=test_chunk_max_size)


    def test_pdf_chunking_with_page_separator(self, temp_dir_just_one_file, mock_language_model, monkeypatch):
       # create a pdfs with varying page counts.  Mock max_output_tokens to be something larger than the total number of tokens in the pdfs.
        page_counts = [1, 5, 10, 20]
        pdf_paths = []
        for i in range(len(page_counts)):
            path = os.path.join(temp_dir_just_one_file, f"file{i}.pdf")
            _save_pdf_file(path,
                   title="File Title {i}", author="File Author {i}", page_count=page_counts[i],
                   text_content="dummy text")
            pdf_paths.append(path)

        mock_language_model.max_output_tokens = 100_000
        mock_language_model.count_tokens.return_value = 50
        mock_language_model.get_completions.side_effect = mock_get_completions

        input = pl.Series("input", pdf_paths)

        parse_pdf = ParsePDF(
            input=input,
            model=mock_language_model,
            page_separator="--- PAGE {page} ---",
        )

        test_chunk_max_size = 4
        monkeypatch.setattr("fenic._backends.local.semantic_operators.parse_pdf.PDF_MAX_PAGES_CHUNK", test_chunk_max_size)
        result = parse_pdf.execute()
        assert result.shape == (len(page_counts),)
        check_chunk_page_separators(result[0], pages=page_counts[0], chunk_max_size=test_chunk_max_size)
        check_chunk_page_separators(result[1], pages=page_counts[1], chunk_max_size=test_chunk_max_size)
        check_chunk_page_separators(result[2], pages=page_counts[2], chunk_max_size=test_chunk_max_size)
        check_chunk_page_separators(result[3], pages=page_counts[3], chunk_max_size=test_chunk_max_size)

        test_chunk_max_size = 1
        monkeypatch.setattr("fenic._backends.local.semantic_operators.parse_pdf.PDF_MAX_PAGES_CHUNK", test_chunk_max_size)
        result2 = parse_pdf.execute()
        assert result2.shape == (len(page_counts),)
        check_chunk_page_separators(result2[0], pages=page_counts[0], chunk_max_size=test_chunk_max_size)
        check_chunk_page_separators(result2[1], pages=page_counts[1], chunk_max_size=test_chunk_max_size)
        check_chunk_page_separators(result2[2], pages=page_counts[2], chunk_max_size=test_chunk_max_size)
        check_chunk_page_separators(result2[3], pages=page_counts[3], chunk_max_size=test_chunk_max_size)

        test_chunk_max_size = 10
        monkeypatch.setattr("fenic._backends.local.semantic_operators.parse_pdf.PDF_MAX_PAGES_CHUNK", test_chunk_max_size)
        result3 = parse_pdf.execute()
        assert result3.shape == (len(page_counts),)
        check_chunk_page_separators(result3[0], pages=page_counts[0], chunk_max_size=test_chunk_max_size)
        check_chunk_page_separators(result3[1], pages=page_counts[1], chunk_max_size=test_chunk_max_size)
        check_chunk_page_separators(result3[2], pages=page_counts[2], chunk_max_size=test_chunk_max_size)
        check_chunk_page_separators(result3[3], pages=page_counts[3], chunk_max_size=test_chunk_max_size)