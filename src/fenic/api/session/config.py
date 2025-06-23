"""Session configuration classes for Fenic."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional, Union

from pydantic import BaseModel, Field, model_validator

from fenic.core._inference.model_catalog import (
    ANTHROPIC_AVAILABLE_LANGUAGE_MODELS,
    GOOGLE_GLA_AVAILABLE_MODELS,
    GOOGLE_VERTEX_AVAILABLE_MODELS,
    OPENAI_AVAILABLE_EMBEDDING_MODELS,
    OPENAI_AVAILABLE_LANGUAGE_MODELS,
    ModelProvider,
    model_catalog,
)
from fenic.core._resolved_session_config import (
    ReasoningEffort,
    ResolvedAnthropicModelConfig,
    ResolvedAnthropicModelPreset,
    ResolvedCloudConfig,
    ResolvedGoogleModelConfig,
    ResolvedGoogleModelPreset,
    ResolvedModelConfig,
    ResolvedOpenAIModelConfig,
    ResolvedOpenAIModelPreset,
    ResolvedSemanticConfig,
    ResolvedSessionConfig,
)
from fenic.core.error import ConfigurationError

presets_desc = """
            Allow the same model configuration to be used with different presets, currently used to set thinking budget/reasoning effort
            for reasoning models. To use a preset of a given model alias in a semantic operator, reference the model as <model_alias>.<preset_name>.
        """

default_preset_desc = """
            If presets are configured, which should be used by default?
        """

class GoogleModelPreset(BaseModel):
    """Configuration for a preset reasoning budget."""
    thinking_token_budget: Optional[int] = Field(
        default=None,
        description="""
            If configuring a reasoning model, provide a thinking budget in tokens.
            If not provided, or if set to 0, thinking will be disabled for the preset (not supported on gemini-2.5-pro).
            To have the model automatically determine a thinking budget based on the complexity of
            the prompt, set this to -1.

            Note that Gemini models take this as a suggestion -- and not a hard limit. It is very possible for the model to
            generate far more thinking tokens than the suggested budget.
        """, ge=-1, lt=32768)


class GoogleGLAModelConfig(BaseModel):
    """Configuration for Google GenerativeLAnguage (GLA) models.
    
    This class defines the configuration settings for models available in Google Developer AI Studio,
    including model selection and rate limiting parameters. These models are accessible using a GOOGLE_API_KEY environment variable.
    """
    model_name: GOOGLE_GLA_AVAILABLE_MODELS
    rpm: int = Field(..., gt=0, description="Requests per minute; must be > 0")
    tpm: int = Field(..., gt=0, description="Tokens per minute; must be > 0")
    presets: Optional[dict[str, GoogleModelPreset]] = Field(default=None, description=presets_desc)
    default_preset: Optional[str] = Field(default=None, description=default_preset_desc)


class GoogleVertexModelConfig(BaseModel):
    """Configuration for Google Vertex models.

        This class defines the configuration settings for models available in Google Vertex AI,
    including model selection and rate limiting parameters. In order to use these models, you must have a
    Google Cloud service account, or use the `gcloud` cli tool to authenticate your local environment.
    """
    model_name: GOOGLE_VERTEX_AVAILABLE_MODELS = Field(..., description="The name of the Google Vertex model to use")
    rpm: int = Field(..., gt=0, description="Requests per minute; must be > 0")
    tpm: int = Field(..., gt=0, description="Tokens per minute; must be > 0")
    presets: Optional[dict[str, GoogleModelPreset]] = Field(default=None, description=presets_desc)
    default_preset: Optional[str] = Field(default=None, description=default_preset_desc)

class OpenAIModelPreset(BaseModel):
    """Configuration for a preset reasoning effort."""
    reasoning_effort: Optional[ReasoningEffort] = Field(
        default=None,
        description="""
            If configuring a reasoning model, provide a reasoning effort. If not provided, we will defer to
            the model's default settings. For OpenAI reasoning models, this is typically 'medium'. When using an o-series
            reasoning model, the `temperature` cannot be customized -- any changes to temperature will be ignored.
        """)

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
    model_name: Union[OPENAI_AVAILABLE_LANGUAGE_MODELS, OPENAI_AVAILABLE_EMBEDDING_MODELS] = Field(..., description="The name of the OpenAI model to use")
    rpm: int = Field(..., gt=0, description="Requests per minute; must be > 0")
    tpm: int = Field(..., gt=0, description="Tokens per minute; must be > 0")
    presets: Optional[dict[str, OpenAIModelPreset]] = Field(default=None, description=presets_desc)
    default_preset: Optional[str] = Field(default=None, description=default_preset_desc)


class AnthropicModelPreset(BaseModel):
    """Configuration for a preset reasoning effort."""
    thinking_token_budget: Optional[int] = Field(
        default=None,
        description="""
            If configuring a model that supports reasoning, provide a default thinking budget in tokens. If not provided,
            thinking will be disabled for the preset. The minimum token budget supported by Anthropic is 1024 tokens.
            If thinking_token_budget is set, temperature cannot be customized -- any changes to temperature will be ignored.
        """, ge=1024)

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
    presets: Optional[dict[str, AnthropicModelPreset]] = Field(default=None, description=presets_desc)
    default_preset: Optional[str] = Field(default=None, description=default_preset_desc)

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
    language_models: dict[str, ModelConfig]
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
        if self.default_language_model is None and len(self.language_models) == 1:
            self.default_language_model = list(self.language_models.keys())[0]

        # Set default preset for each model if not set and only one preset exists
        for model_config in self.language_models.values():
            if model_config.presets is not None:
                preset_names = list(model_config.presets.keys())
                if model_config.default_preset is None and len(preset_names) == 1:
                    model_config.default_preset = preset_names[0]

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
        if len(self.language_models) == 0:
            raise ConfigurationError("You must specify at least one language model configuration.")
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

            if language_model.presets is not None:
                preset_names = list(language_model.presets.keys())
                if language_model.default_preset is None and len(preset_names) > 0:
                    raise ConfigurationError(f"default_preset is not set for model {model_alias}, but multiple presets are configured. Please specify one of: {preset_names} as a default_preset.")
                if language_model.default_preset is not None and language_model.default_preset not in preset_names:
                    raise ConfigurationError(f"default_preset {language_model.default_preset} is not in configured presets for model {model_alias}. Available presets: {preset_names}")

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
            if self.default_embedding_model is None and len(self.embedding_models) > 1:
                raise ConfigurationError("embedding_models is set but default_embedding_model is missing (ambiguous).")

            if self.default_embedding_model is not None and self.default_embedding_model not in self.embedding_models:
                raise ConfigurationError(
                    f"default_embedding_model {self.default_embedding_model} is not in embedding_models")
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
        semantic: Configuration for semantic models (required).
        cloud: Optional configuration for cloud execution.

    Note:
        The semantic configuration is required as it defines the language models
        that will be used for processing. The cloud configuration is optional and
        only needed for distributed processing.
    """
    app_name: str = "default_app"
    db_path: Optional[Path] = None
    semantic: SemanticConfig
    cloud: Optional[CloudConfig] = None

    def _to_resolved_config(self) -> ResolvedSessionConfig:
        def resolve_model(model: ModelConfig) -> ResolvedModelConfig:
            if isinstance(model, OpenAIModelConfig):
                presets = {
                    preset: ResolvedOpenAIModelPreset(reasoning_effort=preset_config.reasoning_effort) for preset, preset_config in model.presets.items()
                } if model.presets else None
                return ResolvedOpenAIModelConfig(
                    model_name=model.model_name,
                    rpm=model.rpm,
                    tpm=model.tpm,
                    presets=presets,
                    default_preset=model.default_preset
                )
            elif isinstance(model, GoogleGLAModelConfig):
                presets = {
                    preset: ResolvedGoogleModelPreset(thinking_token_budget=preset_config.thinking_token_budget) for preset, preset_config in model.presets.items()
                } if model.presets else None
                return ResolvedGoogleModelConfig(
                    model_name=model.model_name,
                    model_provider=ModelProvider.GOOGLE_GLA,
                    rpm=model.rpm,
                    tpm=model.tpm,
                    presets=presets,
                    default_preset=model.default_preset,
                )
            elif isinstance(model, GoogleVertexModelConfig):
                presets = {
                    preset: ResolvedGoogleModelPreset(thinking_token_budget=preset_config.thinking_token_budget) for preset, preset_config in model.presets.items()
                } if model.presets else None
                return ResolvedGoogleModelConfig(
                    model_name=model.model_name,
                    model_provider=ModelProvider.GOOGLE_VERTEX,
                    rpm=model.rpm,
                    tpm=model.tpm,
                    presets=presets,
                    default_preset=model.default_preset,
                )
            else:
                presets = {
                    preset: ResolvedAnthropicModelPreset(thinking_token_budget=preset_config.thinking_token_budget) for preset, preset_config in model.presets.items()
                } if model.presets else None
                return ResolvedAnthropicModelConfig(
                    model_name=model.model_name,
                    rpm=model.rpm,
                    input_tpm=model.input_tpm,
                    output_tpm=model.output_tpm,
                    presets=presets,
                    default_preset=model.default_preset
                )

        resolved_language_models = {
            alias: resolve_model(cfg)
            for alias, cfg in self.semantic.language_models.items()
        }

        resolved_embedding_models = (
            {
                alias: resolve_model(cfg)  # type: ignore[arg-type]
                for alias, cfg in self.semantic.embedding_models.items()
            }
            if self.semantic.embedding_models
            else None
        )
        all_language_model_aliases = set(resolved_language_models.keys())
        for alias, model in resolved_language_models.items():
            if model.presets:
                for preset_name in model.presets:
                    all_language_model_aliases.add(f"{alias}.{preset_name}")

        resolved_semantic = ResolvedSemanticConfig(
            language_models=resolved_language_models,
            all_language_model_aliases=all_language_model_aliases,
            default_language_model=self.semantic.default_language_model,
            embedding_models=resolved_embedding_models,
            default_embedding_model=self.semantic.default_embedding_model,
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
