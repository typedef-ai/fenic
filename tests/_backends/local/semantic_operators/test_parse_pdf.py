import os
from textwrap import dedent

import polars as pl
import pytest
from pydantic import ValidationError as PydanticValidationError

from fenic._backends.local.semantic_operators.parse_pdf import ParsePDF
from fenic._inference.common_openai.openai_utils import convert_messages
from fenic.core.error import FileLoaderError


class TestParsePDF:
    """Test cases for the ParsePDF operator."""

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
                    "content": dedent("""\
                        Convert the main content of this PDF document to clean, well-formatted markdown. Output should be raw markdown, don't surround in code fences or backticks. 
                        Preserve the structure, formatting, headings, lists, and any tables.
                        Never skip any text from the body of the document, even if it's repetitive.
                        Ignore any images that aren't tables or charts that can be converted to markdown.""").strip(),
                },
                {
                    "role": "user",
                    "content": [],
                },
            ],
            [
                {
                    "role": "system",
                    "content": dedent("""\
                        Convert the main content of this PDF document to clean, well-formatted markdown. Output should be raw markdown, don't surround in code fences or backticks. 
                        Preserve the structure, formatting, headings, lists, and any tables.
                        Never skip any text from the body of the document, even if it's repetitive.
                        Ignore any images that aren't tables or charts that can be converted to markdown.""").strip(),
                },
                {
                    "role": "user",
                    "content": [],
                },
            ],
        ]

        # Update user content with file references
        expected[0][1]["content"] = [
            {
                "type": "file",
                "content": {
                    "filename": file_path_1,
                    "file_data": file_path_1,
                },
            }
        ]
        expected[1][1]["content"] = [
            {
                "type": "file",
                "content": {
                    "filename": file_path_2,
                    "file_data": file_path_2,
                },
            }
        ]

        assert result == expected

    def test_build_prompts_with_page_separator(self, local_session, temp_dir_with_test_files):
        """Test PDF parsing with page separator."""
        file_path = os.path.join(temp_dir_with_test_files, "file8.pdf")
        input = pl.Series("input", [file_path])

        parse_pdf = ParsePDF(
            input=input,
            model=local_session._session_state.get_language_model(),
            page_separator="--- PAGE BREAK ---",
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
                    "content": dedent("""\
                        Convert the main content of this PDF document to clean, well-formatted markdown. Output should be raw markdown, don't surround in code fences or backticks. 
                        Preserve the structure, formatting, headings, lists, and any tables.
                        Never skip any text from the body of the document, even if it's repetitive.
                        Don't include the page numbers in the output, instead insert the page separator '--- PAGE BREAK ---' as a markdown line for each page break.
                        Ignore any images that aren't tables or charts that can be converted to markdown.""").strip(),
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

    def test_build_prompts_with_page_number_placeholder(self, local_session, temp_dir_with_test_files):
        """Test PDF parsing with page number placeholder in separator."""
        file_path = os.path.join(temp_dir_with_test_files, "file8.pdf")
        input = pl.Series("input", [file_path])

        parse_pdf = ParsePDF(
            input=input,
            model=local_session._session_state.get_language_model(),
            page_separator="--- PAGE {page} ---",
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
                    "content": dedent("""\
                        Convert the main content of this PDF document to clean, well-formatted markdown. Output should be raw markdown, don't surround in code fences or backticks. 
                        Preserve the structure, formatting, headings, lists, and any tables.
                        Never skip any text from the body of the document, even if it's repetitive.
                        Insert the page separator '--- PAGE {page} ---' as a markdown line for each page break, replacing the '{page}' pattern with the current page number.  If the document contains page numbers, do not include them in the output, instead replace them with this page separator.
                        Ignore any images that aren't tables or charts that can be converted to markdown.""").strip(),
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
                    "content": dedent("""\
                        Convert the main content of this PDF document to clean, well-formatted markdown. Output should be raw markdown, don't surround in code fences or backticks. 
                        Preserve the structure, formatting, headings, lists, and any tables.
                        Never skip any text from the body of the document, even if it's repetitive.
                        For each image, describe them briefly in a markdown section with 'Image' in the title, preserving the output order.""").strip(),
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

        parse_pdf = ParsePDF(
            input=input,
            model=local_session._session_state.get_language_model(),
            page_separator="Page {page}",
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
                    "content": dedent("""\
                        Convert the main content of this PDF document to clean, well-formatted markdown. Output should be raw markdown, don't surround in code fences or backticks. 
                        Preserve the structure, formatting, headings, lists, and any tables.
                        Never skip any text from the body of the document, even if it's repetitive.
                        Insert the page separator 'Page {page}' as a markdown line for each page break, replacing the '{page}' pattern with the current page number.  If the document contains page numbers, do not include them in the output, instead replace them with this page separator.
                        For each image, describe them briefly in a markdown section with 'Image' in the title, preserving the output order.""").strip(),
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

    def test_handles_none_values(self, local_session, temp_dir_with_test_files):
        """Test handling of None values in input."""
        input = pl.Series("input", [os.path.join(temp_dir_with_test_files, "file8.pdf"), None])

        with pytest.raises(PydanticValidationError, match="1 validation error for DocFolderLoader.check_file_extensions"):
            _ = ParsePDF(
                input=input,
                model=local_session._session_state.get_language_model(),
            )

    def test_handles_invalid_file_extensions(self, local_session, temp_dir_with_test_files):
        """Test handling of invalid file extensions in input."""
        input = pl.Series("input", [os.path.join(temp_dir_with_test_files, "file1.md")])

        with pytest.raises(FileLoaderError, match="Only files with the extension pdf are supported in this plan."):
            _ = ParsePDF(
                input=input,
                model=local_session._session_state.get_language_model(),
            )

    def test_handles_non_existing_files(self, local_session, temp_dir_with_test_files):
        """Test handling of non-existing files in input."""
        input = pl.Series("input", ["dir/nonexistent.pdf"])

        with pytest.raises(ValueError, match="Path does not exist: dir/nonexistent.pdf"):
            _ = ParsePDF(
                input=input,
                model=local_session._session_state.get_language_model(),
            )
