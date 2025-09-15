import os
from pathlib import Path
from textwrap import wrap
from typing import List, Optional

import pytest
from pydantic import BaseModel

from fenic import SemanticConfig, Session, SessionConfig, col, semantic
from fenic.api.session.config import GoogleDeveloperLanguageModel, OpenAILanguageModel
from fenic.core.error import ValidationError
from tests.conftest import _save_pdf_file

basic_text_content = [
    "Some content about the fundamentals of data science and machine learning. This section covers basic statistical concepts, data preprocessing techniques, and common algorithms used in predictive modeling. It also includes practical examples and case studies to illustrate key concepts.",
    "Content about dragons and their role in medieval mythology and fantasy literature. This comprehensive guide explores different types of dragons, their cultural significance across various civilizations, and their representation in modern media. The text delves into dragon anatomy, behavior patterns, and the symbolic meaning they hold in different cultures.",
    "Content about war and peace, examining the complex relationship between conflict and harmony throughout human history. This analysis covers major historical conflicts, their causes and consequences, as well as the various peace movements and diplomatic efforts that have shaped our world. The discussion includes philosophical perspectives on violence, justice, and the pursuit of lasting peace.",
]

def make_test_pdf_paths(text_content: list[str], temp_dir: str, include_images:Optional[List[bool]] = None, include_small_images: Optional[List[bool]] = None, include_signatures: Optional[List[bool]] = None):
    """Create a temporary directory with 3 PDFs of varying content."""
    pdf_paths = []
    for i in range(2):
        path = os.path.join(temp_dir, f"file{i}.pdf")
        _save_pdf_file(Path(path),
                   title=f"File {i} Title", author=f"File {i} Author", page_count=3,
                   text_content=text_content,
                   include_headers_and_footers=True,
                   include_images=False if not include_images else include_images[i],
                   include_small_images=False if not include_small_images else include_small_images[i],
                   include_signatures=False if not include_signatures else include_signatures[i])
        pdf_paths.append(path)

    return pdf_paths

def setup_session_with_vlm(provider: BaseModel, model_name: str):
    config = SessionConfig(
        app_name="test_pdf_parsing_with_vlms",
        semantic=SemanticConfig(
            language_models={"vlm": provider(model_name=model_name, rpm=1000, tpm=100000, thinking_token_budget=1024)}
        ),
    )
    return Session.get_or_create(config)

vlms_to_test = [
    (GoogleDeveloperLanguageModel, "gemini-2.5-flash"),
    #(GoogleDeveloperLanguageModel, "gemini-2.5-pro"), # TODO: fix bug where gemini-2.5-pro requires reasoning to be enabled
    (GoogleDeveloperLanguageModel, "gemini-2.0-flash"),
    (OpenAILanguageModel, "o3"),
    (OpenAILanguageModel, "o1"),
    (OpenAILanguageModel, "o4-mini"),
    (OpenAILanguageModel, "o3-mini"),
    (OpenAILanguageModel, "o1-mini"),
]

@pytest.mark.parametrize("test_provider, test_model_name", vlms_to_test)
def test_semantic_parse_pdf_basic_markdown(request, temp_dir_just_one_file, test_provider, test_model_name):
    """Test basic PDF parsing functionality with different page lengths."""
    if test_provider == GoogleDeveloperLanguageModel:
        pytest.importorskip("google.genai")
    evaluate_response = False
    if request.config.getoption("--test-model-evaluation"):
        evaluate_response = True

    include_images = [True, False]
    pdf_paths = make_test_pdf_paths(basic_text_content, temp_dir_just_one_file, include_images=include_images)
    local_session = setup_session_with_vlm(provider=test_provider, model_name=test_model_name)
    try:
        df = local_session.create_dataframe({"pdf_path": pdf_paths})
        markdown_result = df.select(
            semantic.parse_pdf(col("pdf_path")).alias("markdown_content")
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
                assert "Image" not in markdown_result.data["markdown_content"][i]
    finally:
        local_session.stop()

@pytest.mark.parametrize("test_provider, test_model_name", vlms_to_test)
def test_semantic_parse_pdf_markdown_with_simple_page_break_and_images(request,temp_dir_just_one_file, test_provider, test_model_name):
    """Test basic PDF parsing functionality with different page lengths."""
    if test_provider == GoogleDeveloperLanguageModel:
        pytest.importorskip("google.genai")
    evaluate_response = False
    if request.config.getoption("--test-model-evaluation"):
        evaluate_response = True

    include_images = [True, False]
    include_small_images = [True, True]
    local_session = setup_session_with_vlm(provider=test_provider, model_name=test_model_name)
    try:
        pdf_paths = make_test_pdf_paths(basic_text_content, temp_dir_just_one_file, include_images=include_images, include_small_images=include_small_images)
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