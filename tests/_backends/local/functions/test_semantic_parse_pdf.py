import os
from pathlib import Path
from textwrap import wrap
from typing import List, Optional

import pytest
from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from fenic import SemanticConfig, Session, SessionConfig, col, lit, semantic
from fenic.api.session.config import (
    GoogleDeveloperLanguageModel,
    OpenAILanguageModel,
    OpenRouterLanguageModel,
)
from fenic.core._inference.model_catalog import ModelProvider, model_catalog
from fenic.core.error import ExecutionError, ValidationError
from fenic.core.types import ColumnField, MarkdownType
from tests.conftest import _save_pdf_file

basic_text_content = [
    "Some content about the fundamentals of data science and machine learning. This section covers basic statistical concepts, data preprocessing techniques, and common algorithms used in predictive modeling. It also includes practical examples and case studies to illustrate key concepts.",
    "Content about dragons and their role in medieval mythology and fantasy literature. This comprehensive guide explores different types of dragons, their cultural significance across various civilizations, and their representation in modern media. The text delves into dragon anatomy, behavior patterns, and the symbolic meaning they hold in different cultures.",
    "Content about war and peace, examining the complex relationship between conflict and harmony throughout human history. This analysis covers major historical conflicts, their causes and consequences, as well as the various peace movements and diplomatic efforts that have shaped our world. The discussion includes philosophical perspectives on violence, justice, and the pursuit of lasting peace.",
]

# Test matrix of models to test:
# - tuple of (test_model_class, test_model_name, test_processing_engine)
# - forcing the page chunking to different page ranges
# keeping the more expensive models off by default
# test_processing_engine is an OpenRouter tool choice for processing PDFs
vlms_to_test = [
    (OpenRouterLanguageModel, "google/gemini-2.0-flash-lite-001", "native"),
    (OpenRouterLanguageModel, "openai/gpt-4.1-nano", "mistral-ocr"),
    (OpenRouterLanguageModel, "openai/gpt-4.1-nano", "pdf-text"),
    #(OpenAILanguageModel, "gpt-5-nano", None),
    (OpenAILanguageModel, "gpt-4o-mini", None),
    #(OpenAILanguageModel, "o3", None),
    (OpenAILanguageModel, "o4-mini", None),
    #(GoogleDeveloperLanguageModel, "gemini-2.5-pro", None),
    (GoogleDeveloperLanguageModel, "gemini-2.0-flash-lite", None),
    (GoogleDeveloperLanguageModel, "gemini-2.5-flash-lite", None),
]

@pytest.mark.parametrize("pdf_chunk_size", [1, 0])
@pytest.mark.parametrize("test_model_class, test_model_name, test_processing_engine", vlms_to_test)
def test_semantic_parse_pdf_basic_markdown(request, temp_dir_just_one_file, test_model_class, test_model_name, test_processing_engine, pdf_chunk_size, monkeypatch):
    """Test basic PDF parsing functionality.

    By default, just parse the PDFS and make sure non-empty markdown is returned.
    In model evaluation mode, check that the text markdown is returned."""
    if test_model_class == GoogleDeveloperLanguageModel:
        pytest.importorskip("google.genai")
    pdf_count = 2
    page_count = 3
    evaluate_response = False
    if request.config.getoption("--test-model-evaluation"):
        evaluate_response = True

    if pdf_chunk_size > 0:
        # Mock PDF_MAX_PAGES_CHUNK
        monkeypatch.setattr("fenic._backends.local.semantic_operators.parse_pdf.PDF_MAX_PAGES_CHUNK", pdf_chunk_size)

    local_session = _setup_session_with_vlm(test_model_class=test_model_class, model_name=test_model_name, processing_engine=test_processing_engine)
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
@pytest.mark.parametrize("test_model_class, test_model_name, test_processing_engine", vlms_to_test)
def test_semantic_parse_pdf_markdown_with_simple_page_break_and_images(request, temp_dir_just_one_file, test_model_class, test_model_name, test_processing_engine, pdf_chunk_size, monkeypatch):
    """Test basic PDF parsing functionality with page separators and image descriptions.

    By default, just parse the PDFS and make sure non-empty markdown is returned.
    In model evaluation mode, check that the text markdown is returned, and that the page separators and image descriptions are included."""
    if test_model_class == GoogleDeveloperLanguageModel:
        pytest.importorskip("google.genai")
    pdf_count = 2
    page_count = 3
    evaluate_response = False
    if request.config.getoption("--test-model-evaluation"):
        evaluate_response = True

    if pdf_chunk_size > 0:
        # Mock PDF_MAX_PAGES_CHUNK
        monkeypatch.setattr("fenic._backends.local.semantic_operators.parse_pdf.PDF_MAX_PAGES_CHUNK", pdf_chunk_size)

    local_session = _setup_session_with_vlm(test_model_class=test_model_class, model_name=test_model_name, processing_engine=test_processing_engine)
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


def test_semantic_parse_pdf_invalid_pages(temp_dir_just_one_file):
    """Test that invalid pages argument raises ValidationError."""

    # Session with models for column tests
    session_config = SessionConfig(
        app_name="semantic_parse_pdf_invalid_pages_with_models",
        semantic=SemanticConfig(
            language_models={"test_model": GoogleDeveloperLanguageModel(
                model_name="gemini-2.0-flash-lite",
                rpm=10,
                tpm=1_000_000,
            )}
        ),
    )
    session = Session.get_or_create(session_config)

    # Create a dummy PDF file for testing
    dummy_pdf = os.path.join(temp_dir_just_one_file, "dummy.pdf")
    _save_pdf_file(Path(dummy_pdf), page_count=10, text_content=["Test content"])

    try:
        df = session.create_dataframe({"pdf_path": [dummy_pdf]})

        # Test 1: Negative page number (static)
        with pytest.raises(ValidationError, match="Page numbers must be positive integers"):
            df.select(semantic.parse_pdf(col("pdf_path"), pages=-1))

        # Test 1b: Negative page number (column)
        with pytest.raises(ExecutionError, match="Page numbers must be positive integers"):
            df.select(semantic.parse_pdf(col("pdf_path"), pages=lit(-1))).collect()

        # Test 2: Zero page number (static)
        with pytest.raises(ValidationError, match="Page numbers must be positive integers"):
            df.select(semantic.parse_pdf(col("pdf_path"), pages=0))

        # Test 2b: Zero page number (column)
        with pytest.raises(ExecutionError, match="Page numbers must be positive integers"):
            df.select(semantic.parse_pdf(col("pdf_path"), pages=lit(0))).collect()

        # Test 3: Invalid range - end < start (static)
        with pytest.raises(ValidationError, match="end page must be >= start page"):
            df.select(semantic.parse_pdf(col("pdf_path"), pages=[[5, 3]]))

        # Test 3b: Invalid range - end < start (column)[
        df_with_pages = session.create_dataframe({"pages":[[[5, 3]]], "pdf_path": [dummy_pdf]})
        with pytest.raises(ExecutionError, match="end page must be >= start page"):
            df_with_pages.select(semantic.parse_pdf(col("pdf_path"), pages=col("pages"))).collect()

        # Test 4: Invalid range - single element (static)
        with pytest.raises(ValidationError, match="Page ranges must be pairs of two numbers"):
            df.select(semantic.parse_pdf(col("pdf_path"), pages=[[5]]))

        # Test 4b: Invalid range - single element (column)
        df_with_pages = session.create_dataframe({"pages":[[[5]]], "pdf_path": [dummy_pdf]})
        with pytest.raises(ExecutionError, match="Page ranges must be pairs of two numbers"):
            df_with_pages.select(semantic.parse_pdf(col("pdf_path"), pages=col("pages"))).collect()

        # Test 5: Invalid range - three elements (static)
        with pytest.raises(ValidationError, match="Page ranges must be pairs of two numbers"):
            df.select(semantic.parse_pdf(col("pdf_path"), pages=[[1, 2, 3]]))

        # Test 5b: Invalid range - three elements (column)
        df_with_pages = session.create_dataframe({"pages":[[[1, 2, 3]]], "pdf_path": [dummy_pdf]})
        with pytest.raises(ExecutionError, match="Page ranges must be pairs of two numbers"):
            df_with_pages.select(semantic.parse_pdf(col("pdf_path"), pages=col("pages"))).collect()

        # Test 6: Negative page in range (static)
        with pytest.raises(ValidationError, match="Page numbers must be positive integers"):
            df.select(semantic.parse_pdf(col("pdf_path"), pages=[[-1, 3]]))

        # Test 6b: Negative page in range (column)
        df_with_pages = session.create_dataframe({"pages":[[[-1, 3]]], "pdf_path": [dummy_pdf]})
        with pytest.raises(ExecutionError, match="Page numbers must be positive integers"):
            df_with_pages.select(semantic.parse_pdf(col("pdf_path"), pages=col("pages"))).collect()

        # Test 7: Empty page range (static)
        with pytest.raises(PydanticValidationError):
            df.select(semantic.parse_pdf(col("pdf_path"), pages=[[[1,2], []]]))
        with pytest.raises(PydanticValidationError):
            df.select(semantic.parse_pdf(col("pdf_path"), pages=[[[1,2], None]]))

        # Test 7b: Empty page range (column)
        df_with_pages = session.create_dataframe({"pages":[[[1,2], []]], "pdf_path": [dummy_pdf]})
        with pytest.raises(ExecutionError, match="Page ranges must be pairs of two numbers"):
            df_with_pages.select(semantic.parse_pdf(col("pdf_path"), pages=col("pages"))).collect()
        df_with_pages = session.create_dataframe({"pages":[[[1,2], None]], "pdf_path": [dummy_pdf]})
        with pytest.raises(ExecutionError, match="Invalid pages element type: NoneType. Expected int or list of two ints"):
            df_with_pages.select(semantic.parse_pdf(col("pdf_path"), pages=col("pages"))).collect()

        # Test 8: Invalid pages type (static)
        with pytest.raises(PydanticValidationError):
            df.select(semantic.parse_pdf(col("pdf_path"), pages="test"))

        # Test 8b: Invalid pages type (column)
        df_with_pages = session.create_dataframe({"pages":["test"], "pdf_path": [dummy_pdf]})
        with pytest.raises(ExecutionError, match="Invalid pages type: str. Expected int, list, or Column"):
            df_with_pages.select(semantic.parse_pdf(col("pdf_path"), pages=col("pages"))).collect()

        # Test 9: Invalid pages element type (static)
        with pytest.raises(PydanticValidationError):
            df.select(semantic.parse_pdf(col("pdf_path"), pages=[[1,2], [3,"test"]]))

    finally:
        session.stop()


def test_semantic_parse_pdf_with_static_page_ranges(temp_dir_just_one_file):
    """Test PDF parsing with static page ranges (mix of int and list)."""
    # Just test with one model (not with google api)
    test_model_class, test_model_name, _ = vlms_to_test[0] 

    # Create a PDF with 10 pages, each with different content
    page_contents = [f"Page {i+1} content: This is unique text for page {i+1}." for i in range(10)]
    pdf_path = os.path.join(temp_dir_just_one_file, "test_10_pages.pdf")
    _save_pdf_file(Path(pdf_path),
                   title="Test PDF",
                   author="Test Author",
                   page_count=10,
                   text_content=page_contents)

    local_session = _setup_session_with_vlm(test_model_class=test_model_class, model_name=test_model_name)

    try:
        df = local_session.create_dataframe({"pdf_path": [pdf_path]})

        # Parse only pages 1, 3-5, 7 (1-indexed)
        # Should get content from pages 1, 3, 4, 5, 7
        result = df.select(
            semantic.parse_pdf(col("pdf_path"), pages=[1, [3, 5], 7]).alias("markdown_content")
        ).collect()

        markdown = result.data["markdown_content"][0]
        assert markdown is not None and markdown != ""

        # Check that we got the right pages
        assert "Page 1 content" in markdown
        assert "Page 3 content" in markdown
        assert "Page 4 content" in markdown
        assert "Page 5 content" in markdown
        assert "Page 7 content" in markdown

        # Check that we didn't get other pages
        assert "Page 2 content" not in markdown
        assert "Page 6 content" not in markdown
        assert "Page 8 content" not in markdown
        assert "Page 9 content" not in markdown
        assert "Page 10 content" not in markdown
    finally:
        local_session.stop()


def test_semantic_parse_pdf_with_column_page_lists(temp_dir_just_one_file):
    """Test PDF parsing with column page ranges (mix of int and list)."""
    # Just test with one model (not with google api)
    test_model_class, test_model_name, _ = vlms_to_test[0]

    # Create two PDFs with different page counts
    page_contents_1 = [f"PDF1 Page {i+1}: Unique content for first PDF page {i+1}." for i in range(8)]
    page_contents_2 = [f"PDF2 Page {i+1}: Unique content for second PDF page {i+1}." for i in range(8)]
    page_contents_3 = [f"PDF3 Page {i+1}: Unique content for third PDF page {i+1}." for i in range(3)]

    pdf_path_1 = os.path.join(temp_dir_just_one_file, "test_pdf_1.pdf")
    pdf_path_2 = os.path.join(temp_dir_just_one_file, "test_pdf_2.pdf")
    pdf_path_3 = os.path.join(temp_dir_just_one_file, "test_pdf_3.pdf")

    _save_pdf_file(Path(pdf_path_1),
                   title="Test PDF 1",
                   author="Test Author",
                   page_count=10,
                   text_content=page_contents_1)

    _save_pdf_file(Path(pdf_path_2),
                   title="Test PDF 2",
                   author="Test Author",
                   page_count=8,
                   text_content=page_contents_2)

    _save_pdf_file(Path(pdf_path_3),
                   title="Test PDF 3",
                   author="Test Author",
                   page_count=5,
                   text_content=page_contents_3)
    local_session = _setup_session_with_vlm(test_model_class=test_model_class, model_name=test_model_name)

    try:
        # Row 1: Parse pages 2, 5-7 from PDF 1
        # Row 2: Parse page 3 from PDF 2
        df = local_session.create_dataframe({
            "pdf_path": [pdf_path_1, pdf_path_2, pdf_path_3, pdf_path_3],
            "pages": [[2, 5, 7], [3], [], None]  # Column with different page specs per row
        })

        result = df.select(
            semantic.parse_pdf(col("pdf_path"), pages=col("pages")).alias("markdown_content")
        ).collect()

        # Check first row (PDF 1, pages 2, 5-7)
        markdown_1 = result.data["markdown_content"][0]
        assert markdown_1 is not None and markdown_1 != ""
        assert "PDF1 Page 2:" in markdown_1
        assert "PDF1 Page 5:" in markdown_1
        assert "PDF1 Page 7:" in markdown_1
        # Should not contain other pages
        assert "PDF1 Page 1:" not in markdown_1
        assert "PDF1 Page 3:" not in markdown_1
        assert "PDF1 Page 4:" not in markdown_1
        assert "PDF1 Page 6:" not in markdown_1
        assert "PDF1 Page 8:" not in markdown_1

        # Check second row (PDF 2, page 3)
        markdown_2 = result.data["markdown_content"][1]
        assert markdown_2 is not None and markdown_2 != ""
        assert "PDF2 Page 3:" in markdown_2
        # Should not contain other pages
        assert "PDF2 Page 1:" not in markdown_2
        assert "PDF2 Page 2:" not in markdown_2
        assert "PDF2 Page 4:" not in markdown_2
        assert "PDF2 Page 5:" not in markdown_2
        assert "PDF2 Page 6:" not in markdown_2
        assert "PDF2 Page 7:" not in markdown_2
        assert "PDF2 Page 8:" not in markdown_2

        # Check third row (empty list) should get no pages
        markdown_3 = result.data["markdown_content"][2]
        assert markdown_3 == ""

        # Check fourth row (None), should get all pages
        markdown_4 = result.data["markdown_content"][3]
        assert markdown_4 is not None and markdown_4 != ""
        assert "PDF3 Page 1:" in markdown_4
        assert "PDF3 Page 2:" in markdown_4
        assert "PDF3 Page 3:" in markdown_4
    finally:
        local_session.stop()

def test_semantic_parse_pdf_with_column_page_ranges(temp_dir_just_one_file):
    """Test PDF parsing with column page ranges (mix of int and list)."""
    # Just test with one model (not with google api)
    test_model_class, test_model_name, _ = vlms_to_test[0]

    # Create two PDFs with different page counts
    page_contents_1 = [f"PDF1 Page {i+1}: Unique content for first PDF page {i+1}." for i in range(8)]
    page_contents_2 = [f"PDF2 Page {i+1}: Unique content for second PDF page {i+1}." for i in range(8)]
    page_contents_3 = [f"PDF3 Page {i+1}: Unique content for third PDF page {i+1}." for i in range(3)]

    pdf_path_1 = os.path.join(temp_dir_just_one_file, "test_pdf_1.pdf")
    pdf_path_2 = os.path.join(temp_dir_just_one_file, "test_pdf_2.pdf")
    pdf_path_3 = os.path.join(temp_dir_just_one_file, "test_pdf_3.pdf")

    _save_pdf_file(Path(pdf_path_1),
                   title="Test PDF 1",
                   author="Test Author",
                   page_count=10,
                   text_content=page_contents_1)

    _save_pdf_file(Path(pdf_path_2),
                   title="Test PDF 2",
                   author="Test Author",
                   page_count=8,
                   text_content=page_contents_2)

    _save_pdf_file(Path(pdf_path_3),
                   title="Test PDF 3",
                   author="Test Author",
                   page_count=5,
                   text_content=page_contents_3)
    local_session = _setup_session_with_vlm(test_model_class=test_model_class, model_name=test_model_name)

    try:
        # Row 1: Parse pages 2, 5-7 from PDF 1
        # Row 2: Parse page 3 from PDF 2
        df = local_session.create_dataframe({
            "pdf_path": [pdf_path_1, pdf_path_2, pdf_path_3, pdf_path_3],
            "pages": [[[2,2], [5, 7]], [[3,3]], [], None]  # Column with different page specs per row
        })

        result = df.select(
            semantic.parse_pdf(col("pdf_path"), pages=col("pages")).alias("markdown_content")
        ).collect()

        # Check first row (PDF 1, pages 2, 5-7)
        markdown_1 = result.data["markdown_content"][0]
        assert markdown_1 is not None and markdown_1 != ""
        assert "PDF1 Page 2:" in markdown_1
        assert "PDF1 Page 5:" in markdown_1
        assert "PDF1 Page 6:" in markdown_1
        assert "PDF1 Page 7:" in markdown_1
        # Should not contain other pages
        assert "PDF1 Page 1:" not in markdown_1
        assert "PDF1 Page 3:" not in markdown_1
        assert "PDF1 Page 4:" not in markdown_1
        assert "PDF1 Page 8:" not in markdown_1

        # Check second row (PDF 2, page 3)
        markdown_2 = result.data["markdown_content"][1]
        assert markdown_2 is not None and markdown_2 != ""
        assert "PDF2 Page 3:" in markdown_2
        # Should not contain other pages
        assert "PDF2 Page 1:" not in markdown_2
        assert "PDF2 Page 2:" not in markdown_2
        assert "PDF2 Page 4:" not in markdown_2
        assert "PDF2 Page 5:" not in markdown_2
        assert "PDF2 Page 6:" not in markdown_2
        assert "PDF2 Page 7:" not in markdown_2
        assert "PDF2 Page 8:" not in markdown_2

        # Check third row (empty list) should get no pages
        markdown_3 = result.data["markdown_content"][2]
        assert markdown_3 == ""

        # Check fourth row (None), should get all pages
        markdown_4 = result.data["markdown_content"][3]
        assert markdown_4 is not None and markdown_4 != ""
        assert "PDF3 Page 1:" in markdown_4
        assert "PDF3 Page 2:" in markdown_4
        assert "PDF3 Page 3:" in markdown_4

    finally:
        local_session.stop()

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
def _setup_session_with_vlm(test_model_class: BaseModel, model_name: str, processing_engine: Optional[str] = None):
    # Lookup the model provider and parameters

    if test_model_class == OpenRouterLanguageModel:
        model_provider = ModelProvider("openrouter")
    elif test_model_class == OpenAILanguageModel:
        model_provider = ModelProvider("openai")
    elif test_model_class == GoogleDeveloperLanguageModel:
        model_provider = ModelProvider("google-developer")
    else:
        raise ValueError(f"Unsupported language model class for pdf parsing test: {test_model_class}")
    model_parameters = model_catalog.get_completion_model_parameters(model_provider, model_name)

    # Set up the profile with the lowest reasoning effort allowed by the model
    profile = None
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
    elif test_model_class == OpenRouterLanguageModel:
        profile = test_model_class.Profile(parsing_engine=processing_engine)

    config = SessionConfig(
        app_name="test_app_parse_pdf",
        semantic=SemanticConfig(
            language_models={"vlm": test_model_class(
                model_name=model_name,
                rpm=500,
                tpm=1_000_000,
                profiles={"low_reasoning": profile} if profile else None,
            )}
        ),
    )
    return Session.get_or_create(config)