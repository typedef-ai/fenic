import os
from pathlib import Path
from textwrap import wrap
from typing import List, Optional

import pytest
from pydantic import BaseModel

from fenic import SemanticConfig, Session, SessionConfig, col, semantic
from fenic.api.session.config import GoogleDeveloperLanguageModel, OpenAILanguageModel
from fenic.core._inference.model_catalog import ModelProvider, model_catalog
from fenic.core.error import ValidationError
from fenic.core.types import ColumnField, MarkdownType
from tests.conftest import _save_pdf_file

basic_text_content = [
    "Some content about the fundamentals of data science and machine learning. This section covers basic statistical concepts, data preprocessing techniques, and common algorithms used in predictive modeling. It also includes practical examples and case studies to illustrate key concepts.",
    "Content about dragons and their role in medieval mythology and fantasy literature. This comprehensive guide explores different types of dragons, their cultural significance across various civilizations, and their representation in modern media. The text delves into dragon anatomy, behavior patterns, and the symbolic meaning they hold in different cultures.",
    "Content about war and peace, examining the complex relationship between conflict and harmony throughout human history. This analysis covers major historical conflicts, their causes and consequences, as well as the various peace movements and diplomatic efforts that have shaped our world. The discussion includes philosophical perspectives on violence, justice, and the pursuit of lasting peace.",
]

# keeping the more expensive models off by default
vlms_to_test = [
    (OpenAILanguageModel, "gpt-5-nano"),
    (OpenAILanguageModel, "gpt-4o-mini"),
    #(OpenAILanguageModel, "o3"),
    (OpenAILanguageModel, "o4-mini"),
    #(GoogleDeveloperLanguageModel, "gemini-2.5-pro"),
    (GoogleDeveloperLanguageModel, "gemini-2.0-flash-lite"),
    (GoogleDeveloperLanguageModel, "gemini-2.5-flash-lite"),
]


@pytest.mark.parametrize("pdf_chunk_size", [1, 0])
@pytest.mark.parametrize("test_model_class, test_model_name", vlms_to_test)
def test_semantic_parse_pdf_basic_markdown(request, temp_dir_just_one_file, test_model_class, test_model_name, pdf_chunk_size, monkeypatch):
    """Test basic PDF parsing functionality.

    By default, just parse the PDFS and make sure non-empty markdown is returned.
    In model evaluation mode, check that the text markdown is returned."""
    pytest.importorskip("google.genai")
    pdf_count = 2
    page_count = 3
    evaluate_response = False
    if request.config.getoption("--test-model-evaluation"):
        evaluate_response = True

    if pdf_chunk_size > 0:
        # Mock PDF_MAX_PAGES_CHUNK
        monkeypatch.setattr("fenic._backends.local.semantic_operators.parse_pdf.PDF_MAX_PAGES_CHUNK", pdf_chunk_size)

    local_session = _setup_session_with_vlm(test_model_class=test_model_class, model_name=test_model_name)
    pdf_paths = _make_test_pdf_paths(basic_text_content,
                                    temp_dir_just_one_file,
                                    pdf_count=pdf_count,
                                    page_count=page_count,
                                    include_images=[True, False])
    try:
        df = local_session.create_dataframe({"pdf_path": pdf_paths})
        parsed_df = df.select(
            semantic.parse_pdf(col("pdf_path")).alias("markdown_content")
        ).cache()
        assert parsed_df.schema.column_fields == [
            ColumnField(name="markdown_content", data_type=MarkdownType)
        ]
        markdown_result = parsed_df.collect()

        for i in range(2):
            markdown_content = markdown_result.data["markdown_content"][i]
            assert markdown_content is not None and markdown_content != ""
            assert isinstance(markdown_content, str)
            if evaluate_response:
                for text in basic_text_content:
                    text_wrapped = wrap(text, 80)
                    for line in text_wrapped:
                        assert line in markdown_result.data["markdown_content"][i]
                assert "Image" not in markdown_result.data["markdown_content"][i]
    finally:
        local_session.stop()


@pytest.mark.parametrize("pdf_chunk_size", [1, 0])
@pytest.mark.parametrize("test_model_class, test_model_name", vlms_to_test)
def test_semantic_parse_pdf_markdown_with_simple_page_break_and_images(request, temp_dir_just_one_file, test_model_class, test_model_name, pdf_chunk_size, monkeypatch):
    """Test basic PDF parsing functionality with page separators and image descriptions.

    By default, just parse the PDFS and make sure non-empty markdown is returned.
    In model evaluation mode, check that the text markdown is returned, and that the page separators and image descriptions are included."""
    pytest.importorskip("google.genai")
    pdf_count = 2
    page_count = 3
    evaluate_response = False
    if request.config.getoption("--test-model-evaluation"):
        evaluate_response = True

    if pdf_chunk_size > 0:
        # Mock PDF_MAX_PAGES_CHUNK
        monkeypatch.setattr("fenic._backends.local.semantic_operators.parse_pdf.PDF_MAX_PAGES_CHUNK", pdf_chunk_size)

    local_session = _setup_session_with_vlm(test_model_class=test_model_class, model_name=test_model_name)
    pdf_paths = _make_test_pdf_paths(basic_text_content,
                                    temp_dir_just_one_file,
                                    pdf_count=pdf_count,
                                    page_count=page_count,
                                    include_images=[True, False],
                                    include_small_images=[True, True])
    try:
        df = local_session.create_dataframe({"pdf_path": pdf_paths})
        markdown_result = df.select(
            semantic.parse_pdf(col("pdf_path"),
                page_separator="--- PAGE {page} ---",
                describe_images=True,
            ).alias("markdown_content")
        ).collect()

        for i in range(2):
            markdown_content = markdown_result.data["markdown_content"][i]
            assert markdown_content is not None and markdown_content != ""
            assert isinstance(markdown_content, str)
            if evaluate_response:
                for text in basic_text_content:
                    text_wrapped = wrap(text, 80)
                    for line in text_wrapped:
                        assert line in markdown_result.data["markdown_content"][i]
                assert "--- PAGE 1 ---" in markdown_result.data["markdown_content"][i]
                assert "--- PAGE 2 ---" in markdown_result.data["markdown_content"][i]
                # The model is very hit or miss on adding the image section
                #if not include_images[i]:
                #    assert "Image" not in markdown_result.data["markdown_content"][i]
                #else:
                #    assert "Image" in markdown_result.data["markdown_content"][i]
    finally:
        local_session.stop()


def test_semantic_parse_pdf_without_models():
    """Test that an error is raised if no language models are configured."""
    session_config = SessionConfig(
        app_name="semantic_parse_pdf_without_models",
    )
    session = Session.get_or_create(session_config)
    with pytest.raises(ValidationError, match="No language models configured."):
        session.create_dataframe({"pdf_path": ["test.pdf"]}).select(semantic.parse_pdf(col("pdf_path")).alias("markdown_content"))
    session.stop()

def _make_test_pdf_paths(text_content: list[str],
                        temp_dir: str,
                        pdf_count: int,
                        page_count: int,
                        include_images:Optional[List[bool]] = None,
                        include_small_images: Optional[List[bool]] = None,
                        include_signatures: Optional[List[bool]] = None):
    """Create PDFs with varying content in the given temporary directory."""
    pdf_paths = []
    for i in range(pdf_count):
        path = os.path.join(temp_dir, f"file{i}.pdf")
        _save_pdf_file(Path(path),
                   title=f"File {i} Title", author=f"File {i} Author", page_count=page_count,
                   text_content=text_content,
                   include_headers_and_footers=True,
                   include_images=False if not include_images else include_images[i],
                   include_small_images=False if not include_small_images else include_small_images[i],
                   include_signatures=False if not include_signatures else include_signatures[i])
        pdf_paths.append(path)

    return pdf_paths

# Test Utility Functions
def _setup_session_with_vlm(test_model_class: BaseModel, model_name: str):
    # Lookup the model provider and parameters

    model_parameters = model_catalog.get_completion_model_parameters(
        ModelProvider("openai" if test_model_class == OpenAILanguageModel else "google-developer"), model_name
    )
    # Set up the profile with the lowest reasoning effort allowed by the model
    if test_model_class == GoogleDeveloperLanguageModel:
        if model_parameters.supports_disabled_reasoning:
            profile = test_model_class.Profile(thinking_token_budget=0)
        else:
            # gemini-2.5-pro doesn't support disabled reasoning and needs a minimum of 128 tokens
            profile = test_model_class.Profile(thinking_token_budget=128)
    elif test_model_class == OpenAILanguageModel and model_parameters.supports_profiles:
        if model_parameters.supports_minimal_reasoning:
            profile = test_model_class.Profile(reasoning_effort="minimal")
        else:
            profile = test_model_class.Profile(reasoning_effort="low")
    config = SessionConfig(
        app_name="test_app_parse_pdf",
        semantic=SemanticConfig(
            language_models={"vlm": test_model_class(
                model_name=model_name,
                rpm=500,
                tpm=1_000_000,
                profiles={"low_reasoning": profile} if model_parameters.supports_profiles else None,
            )}
        ),
    )
    return Session.get_or_create(config)