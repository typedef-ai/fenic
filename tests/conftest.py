import datetime
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Protocol, Tuple
from urllib.parse import urlparse

import boto3
import fitz  # PyMuPDF
import pytest
import requests

from fenic import (
    GoogleVertexLanguageModel,
    OpenAIEmbeddingModel,
    SemanticConfig,
    Session,
    SessionConfig,
    configure_logging,
)
from fenic.api.session.config import (
    AnthropicLanguageModel,
    CohereEmbeddingModel,
    EmbeddingModel,
    GoogleDeveloperEmbeddingModel,
    GoogleDeveloperLanguageModel,
    LanguageModel,
    OpenAILanguageModel,
)
from fenic.core._inference.model_catalog import ModelProvider, model_catalog
from fenic.core._inference.model_provider import ModelProviderClass

LANGUAGE_MODEL_PROVIDER_ARG = "--language-model-provider"
LANGUAGE_MODEL_NAME_ARG = "--language-model-name"
EMBEDDING_MODEL_PROVIDER_ARG = "--embedding-model-provider"
EMBEDDING_MODEL_NAME_ARG = "--embedding-model-name"

class TestPath(Protocol):
    """Protocol for test paths that can be either local or S3."""

    def cleanup(self) -> None: ...

    def __str__(self) -> str: ...


class LocalTestPath(TestPath):
    """Local path."""

    def __init__(self, path: str):
        self.is_s3 = False
        self.path = path
        already_exists = os.path.isdir(path)
        if not already_exists:
            os.makedirs(path)
        self.should_cleanup = not already_exists

    def cleanup(self) -> None:
        if self.should_cleanup:
            import shutil

            shutil.rmtree(self.path)

    def __repr__(self) -> str:
        return f"LocalTestPath({self.path})"


class S3TestPath(TestPath):
    """S3 path."""

    def __init__(self, path: str):
        self.is_s3 = True
        self.path = path.rstrip("/")
        self.setup_s3()
        # Check if directory exists in S3
        try:
            self.s3.head_object(Bucket=self.bucket, Key=self.prefix)
            self.should_cleanup = False
        except self.s3.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                self.should_cleanup = True
            else:
                raise e

    def cleanup(self) -> None:
        if not self.should_cleanup:
            return
        # List all objects with the prefix
        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            if "Contents" in page:
                # Delete all objects in the prefix
                self.s3.delete_objects(
                    Bucket=self.bucket,
                    Delete={
                        "Objects": [{"Key": obj["Key"]} for obj in page["Contents"]]
                    },
                )

    def setup_s3(self) -> Tuple[boto3.client, str, str]:
        from urllib.parse import urlparse

        self.parsed = urlparse(self.path)
        self.bucket = self.parsed.netloc
        self.prefix = self.parsed.path.lstrip("/")
        self.s3 = boto3.client("s3")


@pytest.fixture
def app_name():
    return "test_app"


def pytest_addoption(parser):
    """Command line options for test suite."""
    parser.addoption(
        "--test-output-dir",
        action="store",
        default=None,
        help="Path to use for test output files instead of temp_dir. Can be a local path or an s3 path (starts with s3://).",
    )
    parser.addoption(
        LANGUAGE_MODEL_PROVIDER_ARG,
        action="store",
        default="openai",
        help="Model Provider to run tests against",
    )
    parser.addoption(
        LANGUAGE_MODEL_NAME_ARG,
        action="store",
        default="gpt-4.1-nano",
        help="Model Name to run tests against",
    )
    parser.addoption(
        EMBEDDING_MODEL_PROVIDER_ARG,
        action="store",
        default="openai",
        help="Model Provider to run embeddings tests against",
    )
    parser.addoption(
        EMBEDDING_MODEL_NAME_ARG,
        action="store",
        default="text-embedding-3-small",
        help="Model Name to run embeddings tests against",
    )
    parser.addoption(
        "--test-huggingface-reads",
        action="store",
        default=None,
        help="If set, will run reader tests that read from HuggingFace.",
    )

@pytest.fixture
def embedding_model_name_and_dimensions(local_session) -> Tuple[str, int]:
    """Returns the embedding model name and dimensions for the default embedding model."""
    embedding_model = local_session._session_state.get_embedding_model()
    embedding_model_name = f"{embedding_model.model_provider.value}/{embedding_model.model}"
    embedding_dimensions = embedding_model.model_parameters.default_dimensions
    return embedding_model_name, embedding_dimensions

@pytest.fixture
def examples_session_config(app_name, request) -> SessionConfig:
    """Creates a test session config."""
    language_model_provider = ModelProvider(request.config.getoption(LANGUAGE_MODEL_PROVIDER_ARG))
    embedding_model_provider = ModelProvider(request.config.getoption(EMBEDDING_MODEL_PROVIDER_ARG))
    language_model = configure_language_model(language_model_provider, request.config.getoption(LANGUAGE_MODEL_NAME_ARG))
    embedding_model = configure_embedding_model(embedding_model_provider, request.config.getoption(EMBEDDING_MODEL_NAME_ARG))

    return SessionConfig(
        app_name=app_name,
        semantic=SemanticConfig(
            language_models={
                "default": language_model,
            },
            embedding_models={"default": embedding_model},
        ),
    )


@pytest.fixture
def multi_model_local_session_config(app_name, request) -> SessionConfig:
    """Creates a test session config."""
    language_model_provider = ModelProvider(request.config.getoption(LANGUAGE_MODEL_PROVIDER_ARG))
    embedding_model_provider = ModelProvider(request.config.getoption(EMBEDDING_MODEL_PROVIDER_ARG))
    embedding_model = configure_embedding_model(embedding_model_provider, request.config.getoption(EMBEDDING_MODEL_NAME_ARG))
    nano = OpenAILanguageModel(model_name="gpt-4.1-nano", rpm=250, tpm=50_000)

    # these limits are purposely low so we don't consume our entire project limit while running multiple tests in multiple branches
    if language_model_provider == ModelProvider.OPENAI:
        language_models = {
            "model_1": nano,
            "model_2": OpenAILanguageModel(
                model_name="gpt-4.1-mini", rpm=250, tpm=50_000
            ),
        }
    elif language_model_provider == ModelProvider.ANTHROPIC:
        language_models = {
            "model_1": nano,
            "model_2": AnthropicLanguageModel(
                model_name=request.config.getoption(LANGUAGE_MODEL_NAME_ARG),
                rpm=500,
                input_tpm=50_000,
                output_tpm=20_000,
            ),
        }
    elif language_model_provider == ModelProvider.GOOGLE_DEVELOPER:
        language_models = {
            "model_1": nano,
            "model_2": GoogleDeveloperLanguageModel(
                model_name=request.config.getoption(LANGUAGE_MODEL_NAME_ARG),
                rpm=1000,
                tpm=500_000,
            ),
        }
    elif language_model_provider == ModelProvider.GOOGLE_VERTEX:
        language_models = {
            "model_1": nano,
            "model_2": GoogleVertexLanguageModel(
                model_name=request.config.getoption(LANGUAGE_MODEL_NAME_ARG),
                rpm=1000,
                tpm=500_000,
            ),
        }
    else:
        raise ValueError(f"Unsupported language model provider: {language_model_provider}")
    return SessionConfig(
        app_name=app_name,
        semantic=SemanticConfig(
            language_models=language_models,
            default_language_model="model_1",
            embedding_models={"small": embedding_model},
            default_embedding_model="small",
        ),
    )


@pytest.fixture
def multi_model_local_session(multi_model_local_session_config, request):
    """Creates a test session."""
    configure_logging()
    session = Session.get_or_create(multi_model_local_session_config)
    yield session
    session.stop()
    if os.path.exists(f"{multi_model_local_session_config.app_name}.duckdb"):
        os.remove(f"{multi_model_local_session_config.app_name}.duckdb")


@pytest.fixture
def local_session_config(app_name, request, monkeypatch) -> SessionConfig:
    """Creates a test session config.

    Notes:
        We mock the api key validation to avoid the noticeable delay of validating our api key in every test.
    """
    async def mock_validate_provider_api_keys(providers: set[ModelProviderClass]):
        return
    monkeypatch.setattr("fenic._backends.local.model_registry._validate_provider_api_keys", mock_validate_provider_api_keys)

    language_model_provider = ModelProvider(request.config.getoption(LANGUAGE_MODEL_PROVIDER_ARG))
    embedding_model_provider = ModelProvider(request.config.getoption(EMBEDDING_MODEL_PROVIDER_ARG))
    language_model = configure_language_model(language_model_provider, request.config.getoption(LANGUAGE_MODEL_NAME_ARG))
    embedding_model = configure_embedding_model(embedding_model_provider, request.config.getoption(EMBEDDING_MODEL_NAME_ARG))
    return SessionConfig(
        app_name=app_name,
        semantic=SemanticConfig(
            language_models={
                "test_model": language_model,
            },
            default_language_model="test_model",
            embedding_models={"embedding": embedding_model},
            default_embedding_model="embedding",
        ),
    )


def configure_language_model(model_provider: ModelProvider, model_name: str) -> LanguageModel:
    model_parameters = model_catalog.get_completion_model_parameters(
        model_provider, model_name
    )
    # these limits are purposely low so we don't consume our entire project limit while running multiple tests in multiple branches
    if model_provider == ModelProvider.OPENAI:
        if model_parameters.supports_reasoning and model_parameters.supports_verbosity:
            language_model = OpenAILanguageModel(
                model_name=model_name,
                rpm=500,
                tpm=100_000,
                profiles={
                    "minimal": OpenAILanguageModel.Profile(reasoning_effort="minimal", verbosity="low"),
                    "low": OpenAILanguageModel.Profile(reasoning_effort="low", verbosity="low"),
                    "medium": OpenAILanguageModel.Profile(reasoning_effort="medium", verbosity="low"),
                    "high": OpenAILanguageModel.Profile(reasoning_effort="high", verbosity="low"),
                },
                default_profile="minimal",
            )
        elif model_parameters.supports_reasoning:
            language_model = OpenAILanguageModel(
                model_name=model_name,
                rpm=500,
                tpm=100_000,
                profiles={
                    "low": OpenAILanguageModel.Profile(reasoning_effort="low"),
                    "medium": OpenAILanguageModel.Profile(reasoning_effort="medium"),
                    "high": OpenAILanguageModel.Profile(reasoning_effort="high"),
                },
                default_profile="low",
            )
        else:
            language_model = OpenAILanguageModel(
                model_name=model_name,
                rpm=500,
                tpm=100_000,
            )
    elif model_provider == ModelProvider.ANTHROPIC:
        if model_parameters.supports_reasoning:
            language_model = AnthropicLanguageModel(
                model_name=model_name,
                rpm=500,
                input_tpm=100_000,
                output_tpm=75_000,
                profiles={
                    "thinking_disabled": AnthropicLanguageModel.Profile(),
                    "low": AnthropicLanguageModel.Profile(thinking_token_budget=1024),
                    "medium": AnthropicLanguageModel.Profile(thinking_token_budget=4096),
                    "high": AnthropicLanguageModel.Profile(thinking_token_budget=8192),
                },
                default_profile="low",
            )
        else:
            language_model = AnthropicLanguageModel(
                model_name=model_name,
                rpm=500,
                input_tpm=100_000,
                output_tpm=75_000,
            )
    elif model_provider == ModelProvider.GOOGLE_DEVELOPER:
        if model_parameters.supports_reasoning:
            profiles = {
                "auto": GoogleDeveloperLanguageModel.Profile(thinking_token_budget=-1),
                "low": GoogleDeveloperLanguageModel.Profile(thinking_token_budget=1024),
                "medium": GoogleDeveloperLanguageModel.Profile(thinking_token_budget=4096),
                "high": GoogleDeveloperLanguageModel.Profile(thinking_token_budget=8192),
            }
            if model_parameters.supports_disabled_reasoning:
                profiles["thinking_disabled"] = GoogleDeveloperLanguageModel.Profile()
            language_model = GoogleDeveloperLanguageModel(
                model_name=model_name,
                rpm=1000,
                tpm=500_000,
                profiles=profiles,
                default_profile="thinking_disabled" if "thinking_disabled" in profiles else "auto",
            )
        else:
            language_model = GoogleDeveloperLanguageModel(
                model_name=model_name,
                rpm=1000,
                tpm=500_000,
            )
    elif model_provider == ModelProvider.GOOGLE_VERTEX:
        if model_parameters.supports_reasoning:
            language_model = GoogleVertexLanguageModel(
                model_name=model_name,
                rpm=1000,
                tpm=500_000,
                profiles={
                    "thinking_disabled": GoogleVertexLanguageModel.Profile(),
                    "auto": GoogleVertexLanguageModel.Profile(thinking_token_budget=-1),
                    "low": GoogleVertexLanguageModel.Profile(thinking_token_budget=1024),
                    "medium": GoogleVertexLanguageModel.Profile(thinking_token_budget=4096),
                    "high": GoogleVertexLanguageModel.Profile(thinking_token_budget=8192),
                },
                default_profile="auto",
            )
        else:
            language_model = GoogleVertexLanguageModel(
                model_name=model_name,
                rpm=1000,
                tpm=500_000,
            )
    else:
        raise ValueError(f"Unsupported language model provider: {model_provider}")
    return language_model

def configure_embedding_model(model_provider: ModelProvider, model_name: str) -> EmbeddingModel:
    """ Configure an embedding model for the test session.

    Note: Don't configure profiles that change dimension defaults, or it won't be consistent with embedding_model_name_and_dimensions
    and test_embed.py will fail. """
    if model_provider == ModelProvider.OPENAI:
        embedding_model = OpenAIEmbeddingModel(
            model_name=model_name, rpm=3000, tpm=1_000_000
        )
    elif model_provider == ModelProvider.GOOGLE_DEVELOPER or model_provider == ModelProvider.GOOGLE_VERTEX:
        embedding_model = GoogleDeveloperEmbeddingModel(
            model_name=model_name, rpm=3000, tpm=1_000_000
        )
    elif model_provider == ModelProvider.COHERE:
        embedding_model = CohereEmbeddingModel(
            model_name=model_name, rpm=3000, tpm=1_000_000
        )
    else:
        raise ValueError(f"Unsupported embedding model provider: {model_provider}")
    return embedding_model


@pytest.fixture
def local_session(local_session_config, request):
    """Creates a test session."""
    configure_logging()
    session = Session.get_or_create(local_session_config)
    yield session
    session.stop()
    if os.path.exists(f"{local_session_config.app_name}.duckdb"):
        os.remove(f"{local_session_config.app_name}.duckdb")


@pytest.fixture
def temp_dir(request):
    """Provides a temporary directory for test files.
    - By default, the test will use a temporary directory that is cleaned up after the test.
    - User can optionally provide a custom directory, either a local path or s3 path (starts with s3://)
    - if the custom path doesn't exist before the test, it will be created then deleted after the test.
    - if the custom path exists before the test, the test will NOT clean it up.
    """
    custom_dir = request.config.getoption("--test-output-dir")
    if custom_dir:
        if urlparse(custom_dir).scheme == "s3":
            path = S3TestPath(custom_dir)
        else:
            path = LocalTestPath(custom_dir)
        yield path
        path.cleanup()
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            yield LocalTestPath(tmpdir)


@pytest.fixture
def sample_df(local_session):
    """Creates a sample DataFrame for testing."""
    data = {
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35],
        "city": ["San Francisco", "San Francisco", "Seattle"],
    }
    return local_session.create_dataframe(data)


@pytest.fixture
def sample_df_dups_and_nulls(local_session):
    """Creates a sample DataFrame for testing."""
    data = {
        "name": ["Alice", "Bob", "Charlie", "David", None, "Alice"],
        "age": [None, 30, 30, 95, 25, 20],
        "group": [100, 300, 300, 100, 200, 300],
        "city": [
            "Product with Null",
            "San Francisco",
            "Seattle",
            "Largest Product",
            "San Francisco",
            "Denver",
        ],
    }
    return local_session.create_dataframe(data)


@pytest.fixture
def extract_data_df(local_session):
    """Creates a sample DataFrame for testing."""
    data = {
        "review": [
            "The iPhone 13 has a great camera but average battery life.",
            "I love my Samsung Galaxy S21! It performs well and the screen is amazing.",
            "The Google Pixel 6 heats up during gaming but has a solid build. I find that the screen flickers when brightness is low.",
        ],
    }
    return local_session.create_dataframe(data)


@pytest.fixture
def large_text_df(local_session):
    """Creates a sample DataFrame for testing."""
    pp_url = "https://typedef-assets.s3.us-west-2.amazonaws.com/example_texts/pride_and_prejudice"
    response = requests.get(pp_url)
    response.raise_for_status()
    pp_content = response.text

    cap_url = "https://typedef-assets.s3.us-west-2.amazonaws.com/example_texts/crime_and_punishment"
    response = requests.get(cap_url)
    response.raise_for_status()
    cap_content = response.text

    return local_session.create_dataframe({"text": [pp_content, cap_content]})

def _save_pdf_file(
    path,
    title="Test PDF",
    author="UnitTest",
    text_content=None,
    page_count=5,
    encrypt=False,
    include_forms=False,
    include_signatures=False,
    include_images=False,
    include_vectors=False,
):
    """Generate a PDF with optional features randomly assigned to pages."""
    doc = fitz.open()
    added_vectors = False
    added_images = False
    added_forms = False
    added_signatures = False
    for i in range(page_count):
        page = doc.new_page()

        # Always text

        page_text = f"Page {i+1} of {page_count}\nRandom Number: {random.randint(0, 100)}",
        if page_text is not None:
            if isinstance(text_content, str):
                page_text = page_text
            elif isinstance(page_text, list):
                page_text = text_content[i % len(text_content)]
        page.insert_text(
            (50,50),
            page_text,
            fontsize=12,
        )

        if include_vectors and (random.choice([True, False]) or (not added_vectors and i == page_count-1)):
            rect = fitz.Rect(100, 100, 200, 200) # x0, y0, x1, y1
            page.draw_rect(rect, color=(0, 0, 1), fill=(0.8, 0.8, 0.95))
            added_vectors = True

        if include_images and (random.choice([True, False]) or (not added_images and i == page_count-1)):
            rect = fitz.Rect(100, 200, 200, 300) # x0, y0, x1, y1
            page.insert_image(rect, pixmap=fitz.Pixmap(fitz.csRGB, fitz.IRect(0,0,100,100)))
            added_images = True

        if include_forms and (random.choice([True, False]) or (not added_forms and i == page_count-1)):
            rect = fitz.Rect(100, 400, 200, 500) # x0, y0, x1, y1
            widget = fitz.Widget()
            widget.field_name = f"field_{i}"
            widget.field_type = random.choice([fitz.PDF_WIDGET_TYPE_BUTTON, fitz.PDF_WIDGET_TYPE_TEXT])
            widget.rect = rect
            page.add_widget(widget)
            added_forms = True

        if include_signatures and (random.choice([True, False]) or (not added_signatures and i == page_count-1)):
            rect = fitz.Rect(100, 500, 200, 600) # x0, y0, x1, y1
            widget = fitz.Widget()
            widget.field_name = f"sig_{i}"
            widget.field_type = fitz.PDF_WIDGET_TYPE_SIGNATURE
            widget.rect = rect
            page.add_widget(widget)
            added_signatures = True

    # Metadata
    now = datetime.datetime.now().strftime("D:%Y%m%d%H%M%S")
    doc.set_metadata({
        "title": title,
        "author": author,
        "creationDate": now,
        "modDate": now,
    })

    # Save (optionally encrypted)
    if encrypt:
        doc.save(path, encryption=fitz.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="user")
    else:
        doc.save(path)

@pytest.fixture
def temp_dir_with_test_files():
    """Create a temporary directory with test files."""
    with tempfile.TemporaryDirectory() as _dir:
        temp_path = Path(_dir)

        # Create subdirectories
        (temp_path / "subdir1").mkdir()
        (temp_path / "subdir2").mkdir()
        (temp_path / "temp").mkdir()

        # Create test files
        test_files = [
            "file1.md",
            "file2.md",
            "file3.txy",
            "file8.pdf",
            "file9.pdf",
            "subdir1/file4.md",
            "subdir2/file5.md",
            "subdir1/file6.pdf",
            "subdir2/file7.pdf",
            "temp/temp_file.md",
            "temp/temp_file.pdf",
            "backup.md.bak",
            "file.tmp",
            "file_json.json"
        ]

        for file_name in test_files:
            file_path = temp_path / file_name
            file_path.parent.mkdir(parents=True, exist_ok=True)
            if file_name.endswith(".md"):
                _save_md_file(file_path)
            elif file_name.endswith(".json"):
                # TODO: Create a better sample json file.
                file_path.write_text(json.dumps({"name": file_name, "content": "sample content"}))
            elif file_name.endswith(".pdf"):
                _save_pdf_file(file_path, page_count=1)
            else:
                file_path.write_text(f"sample content for {file_name}")

        yield str(temp_path)


@pytest.fixture
def temp_dir_just_one_file():
    """Create a temporary directory with test files."""
    with tempfile.TemporaryDirectory() as _dir:
        temp_path = Path(_dir)
        _save_md_file(temp_path / "file1.md")

        yield str(temp_path)

def _save_md_file(file_path: Path):
    """Save a sample markdown file to the given path"""
    md_file_contents = """
# title
some text

# 1 Introduction
intro

## 2 Background
more text

### 2.1 More background
more background

## 3 Methods
some more text
"""
    file_path.write_text(md_file_contents)
