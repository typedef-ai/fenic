import os
from textwrap import dedent

import jinja2
import polars as pl
import pytest

from fenic._backends.local.semantic_operators.parse_pdf import ParsePDF
from fenic._inference.common_openai.openai_utils import convert_messages
from fenic.core.error import FileLoaderError


class TestParsePDF:
    """Test cases for the ParsePDF operator."""

    expected_system_prompt = jinja2.Template(dedent("""\
        Transcribe the main content of this PDF document to clean, well-formatted markdown.
         - Output should be raw markdown, don't surround in code fences or backticks.
         - Preserve the structure, formatting, headings, lists, and any tables to the best of your ability
         - Format tables as github markdown tables, however:
             - for table headings, immediately add ' |' after the table heading
        {% if page_separator and '{page}' in page_separator %}
         - Insert the page separator '{{ page_separator }}' as a markdown line for each page break, replacing the '{{ '{page}' }}' pattern with the current page number. If the document contains page numbers, do not include them in the output, instead replace them with this page separator.
        {% elif page_separator %}
         - Don't include the page numbers in the output, instead insert the page separator '{{ page_separator }}' as a markdown line for each page break
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
                parse_pdf.build_request_messages_batch(),
            )
        )

        expected = [
            [
                {
                    "role": "system",
                    "content": self.expected_system_prompt.render(page_separator=None, describe_images=None),
                },
                {
                    "role": "user",
                    "content": [
                    {
                        "type": "file",
                        "content": {
                            "filename": file_path_1,
                            "file_data": file_path_1,
                        },
                    }
                ]}
            ],
            [
                {
                    "role": "system",
                    "content": self.expected_system_prompt.render(page_separator=None, describe_images=None),
                },
                {
                    "role": "user",
                    "content": [
                    {
                        "type": "file",
                        "content": {
                            "filename": file_path_2,
                            "file_data": file_path_2,
                        },
                    }
                ]}
            ],
        ]

        assert result == expected

    def test_build_prompts_with_page_separator(self, local_session, temp_dir_with_test_files):
        """Test PDF parsing with page separator."""
        file_path = os.path.join(temp_dir_with_test_files, "file8.pdf")
        input = pl.Series("input", [file_path])

        page_separator = "--- PAGE BREAK ---"
        parse_pdf = ParsePDF(
            input=input,
            model=local_session._session_state.get_language_model(),
            page_separator=page_separator,
        )

        result = list(
            map(
                lambda x: convert_messages(x) if x else None,
                parse_pdf.build_request_messages_batch(),
            )
        )

        expected = [
            [
                {
                    "role": "system",
                    "content": self.expected_system_prompt.render(page_separator=page_separator, describe_images=None),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "file",
                            "content": {
                                "filename": file_path,
                                "file_data": file_path,
                            },
                        }
                    ],
                },
            ]
        ]

        assert result == expected

    def test_build_prompts_with_formatted_page_separator(self, local_session, temp_dir_with_test_files):
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
                parse_pdf.build_request_messages_batch(),
            )
        )

        expected = [
            [
                {
                    "role": "system",
                    "content": self.expected_system_prompt.render(page_separator=page_separator, describe_images=None),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "file",
                            "content": {
                                "filename": file_path,
                                "file_data": file_path,
                            },
                        }
                    ],
                },
            ]
        ]

        assert result == expected

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
                parse_pdf.build_request_messages_batch(),
            )
        )

        expected = [
            [
                {
                    "role": "system",
                    "content": self.expected_system_prompt.render(page_separator=None, describe_images=True),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "file",
                            "content": {
                                "filename": file_path,
                                "file_data": file_path,
                            },
                        }
                    ],
                },
            ]
        ]

        assert result == expected

    def test_build_prompts_with_all_options(self, local_session, temp_dir_with_test_files):
        """Test PDF parsing with all options enabled."""
        file_path = os.path.join(temp_dir_with_test_files, "file8.pdf")
        input = pl.Series("input", [file_path])

        page_separator = "--- PAGE {page} ---"
        parse_pdf = ParsePDF(
            input=input,
            model=local_session._session_state.get_language_model(),
            page_separator=page_separator,
            describe_images=True,
        )

        result = list(
            map(
                lambda x: convert_messages(x) if x else None,
                parse_pdf.build_request_messages_batch(),
            )
        )

        expected = [
            [
                {
                    "role": "system",
                    "content": self.expected_system_prompt.render(page_separator=page_separator, describe_images=True),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "file",
                            "content": {
                                "filename": file_path,
                                "file_data": file_path,
                            },
                        }
                    ],
                },
            ]
        ]

        assert result == expected

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
