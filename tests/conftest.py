import os
import tempfile
from typing import Protocol, Tuple
from urllib.parse import urlparse

import boto3
import pytest
import requests

from fenic import (
    SemanticConfig,
    Session,
    SessionConfig,
    configure_logging,
)
from fenic._inference.model_catalog import ModelProvider
from fenic.api.session.config import (
    AnthropicModelConfig,
    GoogleGLAModelConfig,
    OpenAIModelConfig,
)

MODEL_NAME_ARG = "--model-name"

MODEL_PROVIDER_ARG = "--model-provider"

AVAILABLE_MODEL_PROVIDERS = "--configure-model"


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
        MODEL_PROVIDER_ARG,
        action="store",
        default="openai",
        help="Model Provider to run tests against",
    )
    parser.addoption(
        MODEL_NAME_ARG,
        action="store",
        default="gpt-4.1-nano",
        help="Model Name to run tests against",
    )


@pytest.fixture
def examples_session_config(app_name) -> SessionConfig:
    """Creates a test session config."""
    embedding_model = OpenAIModelConfig(
        model_name="text-embedding-3-small",
        rpm=3000,
        tpm=1_000_000
    )
    # limits are small so we can run the examples in parallel
    flash_lite_model = GoogleGLAModelConfig(
        model_name="gemini-2.0-flash-lite",
        rpm=250,
        tpm=125_000,
    )
    return SessionConfig(
        app_name=app_name,
        semantic=SemanticConfig(
            language_models={
                "flash-lite": flash_lite_model,
            },
            embedding_models={"oai-small": embedding_model},
        ),
    )

@pytest.fixture
def multi_model_local_session_config(app_name, request) -> SessionConfig:
    """Creates a test session config."""
    model_provider = ModelProvider(request.config.getoption(MODEL_PROVIDER_ARG))
    # these limits are purposely low so we don't consume our entire project limit while running multiple tests in multiple branches
    if model_provider == ModelProvider.OPENAI:
        language_models = {
            "model_1": OpenAIModelConfig(
                model_name="gpt-4.1-nano",
                rpm=250,
                tpm=50_000
            ),
            "model_2": OpenAIModelConfig(
                model_name="gpt-4.1-mini",
                rpm=250,
                tpm=50_000
            )
        }
    elif model_provider == ModelProvider.ANTHROPIC:
        language_models = {
            "model_1": OpenAIModelConfig(
                model_name="gpt-4.1-nano",
                rpm=500,
                tpm=100_000
            ),
            "model_2" : AnthropicModelConfig(
                model_name="claude-3-5-haiku-latest",
                rpm=500,
                input_tpm=50_000,
                output_tpm=20_000,
            )
        }
    elif model_provider == ModelProvider.GOOGLE_GLA:
        language_models = {
            "model_1": OpenAIModelConfig(
                model_name="gpt-4.1-nano",
                rpm=500,
                tpm=100_000
            ),
            "model_2" : GoogleGLAModelConfig(
                model_name=request.config.getoption(MODEL_NAME_ARG),
                rpm=1000,
                tpm=500_000,
                reasoning_effort="none" # will not be applied if model doesn't require it
            )
        }
    else:
        raise ValueError(f"Unsupported model provider: {model_provider}")
    embedding_model = OpenAIModelConfig(
        model_name="text-embedding-3-small",
        rpm=3000,
        tpm=1_000_000
    )
    return SessionConfig(
        app_name=app_name,
        semantic=SemanticConfig(
            language_models=language_models,
            default_language_model="model_1",
            embedding_models={"oai-small": embedding_model},
            default_embedding_model="oai-small",
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
def local_session_config(app_name, request) -> SessionConfig:
    """Creates a test session config."""
    model_provider = ModelProvider(request.config.getoption(MODEL_PROVIDER_ARG))
    # these limits are purposely low so we don't consume our entire project limit while running multiple tests in multiple branches
    if model_provider == ModelProvider.OPENAI:
        language_model = OpenAIModelConfig(
            model_name=request.config.getoption(MODEL_NAME_ARG),
            rpm=500,
            tpm=100_000
        )
    elif model_provider == ModelProvider.ANTHROPIC:
        language_model = AnthropicModelConfig(
            model_name=request.config.getoption(MODEL_NAME_ARG),
            rpm=500,
            input_tpm=50_000,
            output_tpm=10_000,
        )
    elif model_provider == ModelProvider.GOOGLE_GLA:
        language_model = GoogleGLAModelConfig(
            model_name=request.config.getoption(MODEL_NAME_ARG),
            rpm=1000,
            tpm=500_000,
            reasoning_effort="none"  # will not be applied if model doesn't require it
        )
    else:
        raise ValueError(f"Unsupported model provider: {model_provider}")
    embedding_model = OpenAIModelConfig(
        model_name="text-embedding-3-small",
        rpm=3000,
        tpm=1_000_000
    )
    return SessionConfig(
        app_name=app_name,
        semantic=SemanticConfig(
            language_models={
                "test_model": language_model,
            },
            default_language_model="test_model",
            embedding_models={"oai-small": embedding_model},
            default_embedding_model="oai-small",
        ),
    )


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
