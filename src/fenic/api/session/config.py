"""Session configuration classes for Fenic."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional, Union

from pydantic import BaseModel, Field, model_validator

from fenic._inference.model_catalog import (
    ANTHROPIC_AVAILABLE_LANGUAGE_MODELS,
    GOOGLE_GLA_AVAILABLE_MODELS,
    GOOGLE_VERTEX_AVAILABLE_MODELS,
    OPENAI_AVAILABLE_EMBEDDING_MODELS,
    OPENAI_AVAILABLE_LANGUAGE_MODELS,
    ModelProvider,
    model_catalog,
)
from fenic.core._resolved_session_config import (
    ResolvedAnthropicModelConfig,
    ResolvedCloudConfig,
    ResolvedEmbeddingModelConfig,
    ResolvedGoogleModelConfig,
    ResolvedLanguageModelConfig,
    ResolvedModelConfig,
    ResolvedOpenAIModelConfig,
    ResolvedSemanticConfig,
    ResolvedSessionConfig,
)
from fenic.core.error import ConfigurationError


class GoogleGLAModelConfig(BaseModel):
    """Configuration for Google GenerativeLAnguage (GLA) models.

    This class defines the configuration settings for models available in Google Developer AI Studio,
    including model selection and rate limiting parameters. These models are accessible using a GEMINI_API_KEY environment variable.
    """
    model_name: GOOGLE_GLA_AVAILABLE_MODELS
    rpm: int = Field(..., gt=0, description="Requests per minute; must be > 0")
    tpm: int = Field(..., gt=0, description="Tokens per minute; must be > 0")
    default_thinking_budget: Optional[int] = Field(
        default=None,
        description="""
            If configuring a reasoning model, provide a default thinking budget in tokens. If not provided, we will defer to
            the model's default settings. To have the model automatically determine a thinking budget based on the complexity of
            the prompt, set this to -1. To disable thinking for the model, set this to 0 (not supported on gemini-2.5-pro).
        """, ge=-1)


class GoogleVertexModelConfig(BaseModel):
    """Configuration for Google Vertex models.

    This class defines the configuration settings for models available in Google Vertex AI,
    including model selection and rate limiting parameters. In order to use these models, you must have a
    Google Cloud service account, or use the `gcloud` cli tool to authenticate your local environment.
    """
    model_name: GOOGLE_VERTEX_AVAILABLE_MODELS = Field(..., description="The name of the Google Vertex model to use")
    rpm: int = Field(..., gt=0, description="Requests per minute; must be > 0")
    tpm: int = Field(..., gt=0, description="Tokens per minute; must be > 0")
    default_thinking_budget: Optional[int] = Field(
        default=None,
        description="""
            If configuring a reasoning model, provide a default thinking budget in tokens. If not provided, we will defer to
            the model's default settings. To have the model automatically determine a thinking budget based on the complexity of
            the prompt, set this to -1. To disable thinking for the model, set this to 0 (not supported on gemini-2.5-pro).
        """, ge=-1)

class OpenAIModelConfig(BaseModel):
    """Configuration for OpenAI models.

    This class defines the configuration settings for OpenAI language and embedding models,
    including model selection and rate limiting parameters.

    Attributes:
        model_name: The name of the OpenAI model to use.
        rpm: Requests per minute limit; must be greater than 0.
        tpm: Tokens per minute limit; must be greater than 0.

    Examples:
        Configuring an OpenAI Language model with rate limits:

        ```python
        config = OpenAIModelConfig(model_name="gpt-4.1-nano", rpm=100, tpm=100)
        ```

        Configuring an OpenAI Embedding model with rate limits:

        ```python
        config = OpenAIModelConfig(model_name="text-embedding-3-small", rpm=100, tpm=100)
        ```
    """
    # TODO(bc): add support for model snapshot versions.
    model_name: Union[OPENAI_AVAILABLE_LANGUAGE_MODELS, OPENAI_AVAILABLE_EMBEDDING_MODELS] = Field(..., description="The name of the OpenAI model to use")
    rpm: int = Field(..., gt=0, description="Requests per minute; must be > 0")
    tpm: int = Field(..., gt=0, description="Tokens per minute; must be > 0")


class AnthropicModelConfig(BaseModel):
    """Configuration for Anthropic models.

    This class defines the configuration settings for Anthropic language models,
    including model selection and separate rate limiting parameters for input and output tokens.

    Attributes:
        model_name: The name of the Anthropic model to use.
        rpm: Requests per minute limit; must be greater than 0.
        input_tpm: Input tokens per minute limit; must be greater than 0.
        output_tpm: Output tokens per minute limit; must be greater than 0.

    Examples:
        Configuring an Anthropic model with separate input/output rate limits:

        ```python
        config = AnthropicModelConfig(
            model_name="claude-3-5-haiku-latest",
            rpm=100,
            input_tpm=100,
            output_tpm=100
        )
        ```
    """
    model_name: ANTHROPIC_AVAILABLE_LANGUAGE_MODELS = Field(..., description="The name of the Anthropic model to use")
    rpm: int = Field(..., gt=0, description="Requests per minute; must be > 0")
    input_tpm: int = Field(..., gt=0, description="Input tokens per minute; must be > 0")
    output_tpm: int = Field(..., gt=0, description="Output tokens per minute; must be > 0")


ModelConfig = Union[OpenAIModelConfig, AnthropicModelConfig, GoogleGLAModelConfig, GoogleVertexModelConfig]


class SemanticConfig(BaseModel):
    """Configuration for semantic language and embedding models.

    This class defines the configuration for both language models and optional
    embedding models used in semantic operations. It ensures that all configured
    models are valid and supported by their respective providers.

    Attributes:
        language_models: Mapping of model aliases to language model configurations.
        default_language_model: The alias of the default language model to use for semantic operations. Not required
            if only one language model is configured.
        embedding_models: Optional mapping of model aliases to embedding model configurations.
        default_embedding_model: The alias of the default embedding model to use for semantic operations.

    Note:
        The embedding model is optional and only required for operations that
        need semantic search or embedding capabilities.
    """
    language_models: Optional[dict[str, ModelConfig]] = None
    default_language_model: Optional[str] = None
    embedding_models: Optional[dict[str, ModelConfig]] = None
    default_embedding_model: Optional[str] = None

    def model_post_init(self, __context) -> None:
        """Post initialization hook to set defaults.

        This hook runs after the model is initialized and validated.
        It sets the default language and embedding models if they are not set
        and there is only one model available.
        """
        # Set default language model if not set and only one model exists
        if self.language_models and self.default_language_model is None and len(self.language_models) == 1:
            self.default_language_model = list(self.language_models.keys())[0]
        # Set default embedding model if not set and only one model exists
        if self.embedding_models is not None and self.default_embedding_model is None and len(self.embedding_models) == 1:
            self.default_embedding_model = list(self.embedding_models.keys())[0]

    @model_validator(mode="after")
    def validate_models(self) -> SemanticConfig:
        """Validates that the selected models are supported by the system.

        This validator checks that both the language model and embedding model (if provided)
        are valid and supported by their respective providers.

        Returns:
            The validated SemanticConfig instance.

        Raises:
            ConfigurationError: If any of the models are not supported.
        """
        # Skip validation if no models configured (embedding-only or empty config)
        if not self.language_models and not self.embedding_models:
            return self

        # Validate language models if provided
        if self.language_models:
            available_language_model_aliases = list(self.language_models.keys())
            if self.default_language_model is None and len(self.language_models) > 1:
                raise ConfigurationError(f"default_language_model is not set, and multiple language models are configured. Please specify one of: {available_language_model_aliases} as a default_language_model.")

            if self.default_language_model is not None and self.default_language_model not in self.language_models:
                raise ConfigurationError(f"default_language_model {self.default_language_model} is not in configured map of language models. Available models: {available_language_model_aliases} .")

            for model_alias, language_model in self.language_models.items():
                if isinstance(language_model, OpenAIModelConfig):
                    language_model_provider = ModelProvider.OPENAI
                    language_model_name = language_model.model_name
                elif isinstance(language_model, AnthropicModelConfig):
                    language_model_provider = ModelProvider.ANTHROPIC
                    language_model_name = language_model.model_name
                elif isinstance(language_model, GoogleGLAModelConfig):
                    language_model_provider = ModelProvider.GOOGLE_GLA
                    language_model_name = language_model.model_name
                elif isinstance(language_model, GoogleVertexModelConfig):
                    language_model_provider = ModelProvider.GOOGLE_VERTEX
                    language_model_name = language_model.model_name
                else:
                    raise ConfigurationError(
                        f"Invalid language model: {model_alias}: {language_model} unsupported model type.")

            completion_model = model_catalog.get_completion_model_parameters(language_model_provider,
                                                                             language_model_name)
            if completion_model is None:
                raise ConfigurationError(
                    model_catalog.generate_unsupported_completion_model_error_message(
                        language_model_provider,
                        language_model_name
                    )
                )
        if self.embedding_models is not None:
            available_embedding_model_aliases = list(self.embedding_models.keys())
            if self.default_embedding_model is None and len(self.embedding_models) > 1:
                raise ConfigurationError("default_embedding_model is not set, and multiple embedding models are configured. Please specify one of: {available_embedding_model_aliases} as a default_embedding_model.")

            if self.default_embedding_model is not None and self.default_embedding_model not in self.embedding_models:
                raise ConfigurationError(
                    f"default_embedding_model {self.default_embedding_model} is not in configured map of embedding models. Available models: {available_embedding_model_aliases} .")
            for model_alias, embedding_model in self.embedding_models.items():
                if isinstance(embedding_model, OpenAIModelConfig):
                    embedding_model_provider = ModelProvider.OPENAI
                    embedding_model_name = embedding_model.model_name
                else:
                    raise ConfigurationError(
                        f"Invalid embedding model: {model_alias}: {embedding_model} unsupported model type")
                embedding_model_parameters = model_catalog.get_embedding_model_parameters(embedding_model_provider,
                                                                                     embedding_model_name)
                if embedding_model_parameters is None:
                    raise ConfigurationError(model_catalog.generate_unsupported_embedding_model_error_message(
                        embedding_model_provider,
                        embedding_model_name
                    ))

        return self


class CloudExecutorSize(str, Enum):
    """Enum defining available cloud executor sizes.

    This enum represents the different size options available for cloud-based
    execution environments.

    Attributes:
        SMALL: Small instance size.
        MEDIUM: Medium instance size.
        LARGE: Large instance size.
        XLARGE: Extra large instance size.
    """
    SMALL = "INSTANCE_SIZE_S"
    MEDIUM = "INSTANCE_SIZE_M"
    LARGE = "INSTANCE_SIZE_L"
    XLARGE = "INSTANCE_SIZE_XL"


class CloudConfig(BaseModel):
    """Configuration for cloud-based execution.

    This class defines settings for running operations in a cloud environment,
    allowing for scalable and distributed processing of language model operations.

    Attributes:
        size: Size of the cloud executor instance.
            If None, the default size will be used.
    """
    size: Optional[CloudExecutorSize] = None


class SessionConfig(BaseModel):
    """Configuration for a user session.

    This class defines the complete configuration for a user session, including
    application settings, model configurations, and optional cloud settings.
    It serves as the central configuration object for all language model operations.

    Attributes:
        app_name: Name of the application using this session. Defaults to "default_app".
        db_path: Optional path to a local database file for persistent storage.
        semantic: Configuration for semantic models (optional).
        cloud: Optional configuration for cloud execution.

    Note:
        The semantic configuration is optional. When not provided, only non-semantic operations
        are available. The cloud configuration is optional and only needed for
        distributed processing.
    """
    app_name: str = "default_app"
    db_path: Optional[Path] = None
    semantic: Optional[SemanticConfig] = None
    cloud: Optional[CloudConfig] = None

    def _to_resolved_config(self) -> ResolvedSessionConfig:
        def resolve_model(model: ModelConfig) -> ResolvedModelConfig:
            if isinstance(model, OpenAIModelConfig):
                return ResolvedOpenAIModelConfig(
                    model_name=model.model_name,
                    rpm=model.rpm,
                    tpm=model.tpm
                )
            elif isinstance(model, GoogleGLAModelConfig):
                return ResolvedGoogleModelConfig(
                    model_provider=ModelProvider.GOOGLE_GLA,
                    model_name=model.model_name,
                    rpm=model.rpm,
                    tpm=model.tpm,
                    default_thinking_budget=model.default_thinking_budget,
                )
            elif isinstance(model, GoogleVertexModelConfig):
                return ResolvedGoogleModelConfig(
                    model_name=model.model_name,
                    model_provider=ModelProvider.GOOGLE_VERTEX,
                    rpm=model.rpm,
                    tpm=model.tpm,
                    default_thinking_budget=model.default_thinking_budget,
                )
            else:
                return ResolvedAnthropicModelConfig(
                    model_name=model.model_name,
                    rpm=model.rpm,
                    input_tpm=model.input_tpm,
                    output_tpm=model.output_tpm
                )

        language_models = (
            ResolvedLanguageModelConfig(
                model_configs={alias: resolve_model(cfg) for alias, cfg in self.semantic.language_models.items()},
                default_model=self.semantic.default_language_model,
            )
            if self.semantic and self.semantic.language_models else None
        )

        embedding_models = (
            ResolvedEmbeddingModelConfig(
                model_configs={alias: resolve_model(cfg) for alias, cfg in self.semantic.embedding_models.items()},
                default_model=self.semantic.default_embedding_model,
            )
            if self.semantic and self.semantic.embedding_models else None
        )

        resolved_semantic = ResolvedSemanticConfig(
            language_models=language_models,
            embedding_models=embedding_models,
        )

        resolved_cloud = (
            ResolvedCloudConfig(size=self.cloud.size)
            if self.cloud else None
        )

        return ResolvedSessionConfig(
            app_name=self.app_name,
            db_path=self.db_path,
            semantic=resolved_semantic,
            cloud=resolved_cloud
        )
