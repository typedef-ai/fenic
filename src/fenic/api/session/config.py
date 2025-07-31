"""Session configuration classes for Fenic."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator

from fenic.core._inference.model_catalog import (
    AnthropicLanguageModelName,
    EmbeddingModelParameters,
    GoogleDeveloperLanguageModelName,
    GoogleVertexLanguageModelName,
    ModelProvider,
    OpenAIEmbeddingModelName,
    OpenAILanguageModelName,
    model_catalog,
)
from fenic.core._resolved_session_config import (
    ReasoningEffort,
    ResolvedAnthropicModelConfig,
    ResolvedAnthropicModelProfile,
    ResolvedCloudConfig,
    ResolvedEmbeddingModelConfig,
    ResolvedGoogleModelConfig,
    ResolvedGoogleModelProfile,
    ResolvedLanguageModelConfig,
    ResolvedModelConfig,
    ResolvedOpenAIModelConfig,
    ResolvedOpenAIModelProfile,
    ResolvedSemanticConfig,
    ResolvedSessionConfig,
)
from fenic.core.error import ConfigurationError, InternalError

profiles_desc = """
            Allow the same model configuration to be used with different profiles, currently used to set thinking budget/reasoning effort
            for reasoning models. To use a profile of a given model alias in a semantic operator, reference the model as `ModelAlias(name="<model_alias>", profile="<profile_name>")`.
        """

default_profiles_desc = """
            If profiles are configured, which should be used by default?
        """

GoogleEmbeddingTaskType = Literal[
            "SEMANTIC_SIMILARITY",
            "CLASSIFICATION",
            "CLUSTERING",
            "RETRIEVAL_DOCUMENT",
            "RETRIEVAL_QUERY",
            "CODE_RETRIEVAL_QUERY",
            "QUESTION_ANSWERING",
            "FACT_VERIFICATION"
]

class GoogleDeveloperEmbeddingModel(BaseModel):
    """Configuration for Google Developer embedding models.

    This class defines the configuration settings for Google embedding models available in Google Developer AI Studio,
    including model selection and rate limiting parameters. These models are accessible using a GOOGLE_API_KEY environment variable.

    Attributes:
        model_name: The name of the Google Developer embedding model to use.
        rpm: Requests per minute limit; must be greater than 0.
        tpm: Tokens per minute limit; must be greater than 0.
        profiles: Optional mapping of profile names to profile configurations.
        default_profile: The name of the default profile to use if profiles are configured.

    Example:
        Configuring a Google Developer embedding model with rate limits:

        ```python
        config = GoogleDeveloperEmbeddingModelConfig(
            model_name="gemini-embedding-001",
            rpm=100,
            tpm=1000
        )
        ```

        Configuring a Google Developer embedding model with profiles:

        ```python
        config = GoogleDeveloperEmbeddingModelConfig(
            model_name="gemini-embedding-001",
            rpm=100,
            tpm=1000,
            profiles={
                "default": GoogleDeveloperEmbeddingModelConfig.Profile(),
                "high_dim": GoogleDeveloperEmbeddingModelConfig.Profile(output_dimensionality=3072)
            },
            default_profile="default"
        )
        ```
    """
    model_name: str
    model_provider: ModelProvider = Field(default=ModelProvider.GOOGLE_DEVELOPER)
    rpm: int = Field(..., gt=0, description="Requests per minute; must be > 0")
    tpm: int = Field(..., gt=0, description="Tokens per minute; must be > 0")
    profiles: Optional[dict[str, Profile]] = Field(default=None, description=profiles_desc)
    default_profile: Optional[str] = Field(default=None, description=default_profiles_desc)

    class Profile(BaseModel):
        """Profile configurations for Google Developer embedding models.

        This class defines profile configurations for Google embedding models, allowing
        different output dimensionality and task type settings to be applied to the same model.

        Attributes:
            output_dimensionality: The dimensionality of the embedding created by this model.
                If not provided, the model will use its default dimensionality.
            task_type: The type of task for the embedding model.

        Example:
            Configuring a profile with custom dimensionality:

            ```python
            profile = GoogleDeveloperEmbeddingModelConfig.Profile(output_dimensionality=3072)
            ```

            Configuring a profile with default settings:

            ```python
            profile = GoogleDeveloperEmbeddingModelConfig.Profile()
            ```
        """
        output_dimensionality: Optional[int] = Field(default=None, gt=0, le=3072, description="Dimensionality of the embedding created by this model")
        task_type: GoogleEmbeddingTaskType = Field(default="SEMANTIC_SIMILARITY", description="Type of the task")



class GoogleDeveloperLanguageModel(BaseModel):
    """Configuration for Gemini models accessible through Google Developer AI Studio.

    This class defines the configuration settings for Google Gemini models available in Google Developer AI Studio,
    including model selection and rate limiting parameters. These models are accessible using a GOOGLE_API_KEY environment variable.

    Attributes:
        model_name: The name of the Google Developer model to use.
        rpm: Requests per minute limit; must be greater than 0.
        tpm: Tokens per minute limit; must be greater than 0.
        profiles: Optional mapping of profile names to profile configurations.
        default_profile: The name of the default profile to use if profiles are configured.

    Example:
        Configuring a Google Developer model with rate limits:

        ```python
        config = GoogleDeveloperLanguageModel(
            model_name="gemini-2.0-flash",
            rpm=100,
            tpm=1000
        )
        ```

        Configuring a reasoning Google Developer model with profiles:

        ```python
        config = GoogleDeveloperLanguageModel(
            model_name="gemini-2.5-flash",
            rpm=100,
            tpm=1000,
            profiles={
                "thinking_disabled": GoogleDeveloperLanguageModel.Profile(),
                "fast": GoogleDeveloperLanguageModel.Profile(thinking_token_budget=1024),
                "thorough": GoogleDeveloperLanguageModel.Profile(thinking_token_budget=8192)
            },
            default_profile="fast"
        )
        ```
    """
    model_name: GoogleDeveloperLanguageModelName
    rpm: int = Field(..., gt=0, description="Requests per minute; must be > 0")
    tpm: int = Field(..., gt=0, description="Tokens per minute; must be > 0")
    profiles: Optional[dict[str, Profile]] = Field(default=None, description=profiles_desc)
    default_profile: Optional[str] = Field(default=None, description=default_profiles_desc)

    class Profile(BaseModel):
        """Profile configurations for Google Developer models.

        This class defines profile configurations for Google Gemini models, allowing
        different thinking/reasoning settings to be applied to the same model.

        Attributes:
            thinking_token_budget: If configuring a reasoning model, provide a thinking budget in tokens.
                If not provided, or if set to 0, thinking will be disabled for the profile (not supported on gemini-2.5-pro).
                To have the model automatically determine a thinking budget based on the complexity of
                the prompt, set this to -1. Note that Gemini models take this as a suggestion -- and not a hard limit.
                It is very possible for the model to generate far more thinking tokens than the suggested budget, and for the
                model to generate reasoning tokens even if thinking is disabled.

        Example:
            Configuring a profile with a fixed thinking budget:

            ```python
            profile = GoogleDeveloperLanguageModel.Profile(thinking_token_budget=4096)
            ```

            Configuring a profile with automatic thinking budget:

            ```python
            profile = GoogleDeveloperLanguageModel.Profile(thinking_token_budget=-1)
            ```

            Configuring a profile with thinking disabled:

            ```python
            profile = GoogleDeveloperLanguageModel.Profile(thinking_token_budget=0)
            ```
        """

        thinking_token_budget: Optional[int] = Field(
            default=None, description="The thinking budget in tokens.", ge=-1, lt=32768
        )

class GoogleVertexEmbeddingModel(BaseModel):
    """Configuration for Google Vertex AI embedding models.

    This class defines the configuration settings for Google embedding models available in Google Vertex AI,
    including model selection and rate limiting parameters. These models are accessible using Google Cloud credentials.

    Attributes:
        model_name: The name of the Google Vertex embedding model to use.
        rpm: Requests per minute limit; must be greater than 0.
        tpm: Tokens per minute limit; must be greater than 0.
        profiles: Optional mapping of profile names to profile configurations.
        default_profile: The name of the default profile to use if profiles are configured.

    Example:
        Configuring a Google Vertex embedding model with rate limits:

        ```python
        embedding_model = GoogleVertexEmbeddingModel(
            model_name="gemini-embedding-001",
            rpm=100,
            tpm=1000
        )
        ```

        Configuring a Google Vertex embedding model with profiles:

        ```python
        embedding_model = GoogleVertexEmbeddingModel(
            model_name="gemini-embedding-001",
            rpm=100,
            tpm=1000,
            profiles={
                "default": GoogleVertexEmbeddingModel.Profile(),
                "high_dim": GoogleVertexEmbeddingModel.Profile(output_dimensionality=3072)
            },
            default_profile="default"
        )
        ```
    """
    model_name: str
    model_provider: ModelProvider = Field(default=ModelProvider.GOOGLE_VERTEX)
    rpm: int = Field(..., gt=0, description="Requests per minute; must be > 0")
    tpm: int = Field(..., gt=0, description="Tokens per minute; must be > 0")
    profiles: Optional[dict[str, Profile]] = Field(default=None, description=profiles_desc)
    default_profile: Optional[str] = Field(default=None, description=default_profiles_desc)

    class Profile(BaseModel):
        """Profile configurations for Google Vertex embedding models.

        This class defines profile configurations for Google embedding models, allowing
        different output dimensionality and task type settings to be applied to the same model.

        Attributes:
            output_dimensionality: The dimensionality of the embedding created by this model.
                If not provided, the model will use its default dimensionality.
            task_type: The type of task for the embedding model.

        Example:
            Configuring a profile with custom dimensionality:

            ```python
            profile = GoogleVertexEmbeddingModelConfig.Profile(output_dimensionality=3072)
            ```

            Configuring a profile with default settings:

            ```python
            profile = GoogleVertexEmbeddingModelConfig.Profile()
            ```
        """
        output_dimensionality: Optional[int] = Field(default=None, gt=0, le=3072, description="Dimensionality of the embedding created by this model")
        task_type: GoogleEmbeddingTaskType = Field(default="SEMANTIC_SIMILARITY", description="Type of the task")


class GoogleVertexLanguageModel(BaseModel):
    """Configuration for Google Vertex AI models.

    This class defines the configuration settings for Google Gemini models available in Google Vertex AI,
    including model selection and rate limiting parameters. These models are accessible using Google Cloud credentials.

    Attributes:
        model_name: The name of the Google Vertex model to use.
        rpm: Requests per minute limit; must be greater than 0.
        tpm: Tokens per minute limit; must be greater than 0.
        profiles: Optional mapping of profile names to profile configurations.
        default_profile: The name of the default profile to use if profiles are configured.

    Example:
        Configuring a Google Vertex model with rate limits:

        ```python
        config = GoogleVertexLanguageModel(
            model_name="gemini-2.0-flash",
            rpm=100,
            tpm=1000
        )
        ```

        Configuring a reasoning Google Vertex model with profiles:

        ```python
        config = GoogleVertexLanguageModel(
            model_name="gemini-2.5-flash",
            rpm=100,
            tpm=1000,
            profiles={
                "thinking_disabled": GoogleVertexLanguageModel.Profile(),
                "fast": GoogleVertexLanguageModel.Profile(thinking_token_budget=1024),
                "thorough": GoogleVertexLanguageModel.Profile(thinking_token_budget=8192)
            },
            default_profile="fast"
        )
        ```
    """
    model_name: GoogleVertexLanguageModelName
    rpm: int = Field(..., gt=0, description="Requests per minute; must be > 0")
    tpm: int = Field(..., gt=0, description="Tokens per minute; must be > 0")
    profiles: Optional[dict[str, Profile]] = Field(default=None, description=profiles_desc)
    default_profile: Optional[str] = Field(default=None, description=default_profiles_desc)

    class Profile(BaseModel):
        """Profile configurations for Google Vertex models.

        This class defines profile configurations for Google Gemini models, allowing
        different thinking/reasoning settings to be applied to the same underlying model.

        Attributes:
            thinking_token_budget: If configuring a reasoning model, provide a thinking budget in tokens.
                If not provided, or if set to 0, thinking will be disabled for the profile (not supported on gemini-2.5-pro).
                To have the model automatically determine a thinking budget based on the complexity of
                the prompt, set this to -1. Note that Gemini models take this as a suggestion -- and not a hard limit.
                It is very possible for the model to generate far more thinking tokens than the suggested budget, and for the
                model to generate reasoning tokens even if thinking is disabled.

        Example:
            Configuring a profile with a fixed thinking budget:

            ```python
            profile = GoogleVertexLanguageModel.Profile(thinking_token_budget=4096)
            ```

            Configuring a profile with automatic thinking budget:

            ```python
            profile = GoogleVertexLanguageModel.Profile(thinking_token_budget=-1)
            ```

            Configuring a profile with thinking disabled:

            ```python
            profile = GoogleVertexLanguageModel.Profile(thinking_token_budget=0)
            ```
        """

        thinking_token_budget: Optional[int] = Field(
            default=None, description="The thinking budget in tokens.", ge=-1, lt=32768
        )

class OpenAILanguageModel(BaseModel):
    """Configuration for OpenAI language models.

    This class defines the configuration settings for OpenAI language models,
    including model selection and rate limiting parameters.

    Attributes:
        model_name: The name of the OpenAI model to use.
        rpm: Requests per minute limit; must be greater than 0.
        tpm: Tokens per minute limit; must be greater than 0.
        profiles: Optional mapping of profile names to profile configurations.
        default_profile: The name of the default profile to use if profiles are configured.

    Example:
        Configuring an OpenAI language model with rate limits:

        ```python
        config = OpenAILanguageModel(
            model_name="gpt-4.1-nano",
            rpm=100,
            tpm=100
        )
        ```

        Configuring an OpenAI model with profiles:

        ```python
        config = OpenAILanguageModel(
            model_name="o4-mini",
            rpm=100,
            tpm=100,
            profiles={
                "fast": OpenAILanguageModel.Profile(reasoning_effort="low"),
                "thorough": OpenAILanguageModel.Profile(reasoning_effort="high")
            },
            default_profile="fast"
        )
        ```

        Using a profile in a semantic operation:

        ```python
        config = SemanticConfig(
            language_models={
                "o4": OpenAILanguageModel(
                    model_name="o4-mini",
                    rpm=1_000,
                    tpm=1_000_000,
                    profiles={
                        "fast": OpenAILanguageModel.Profile(reasoning_effort="low"),
                        "thorough": OpenAILanguageModel.Profile(reasoning_effort="high")
                    },
                    default_profile="fast"
                )
            },
            default_language_model="o4"
        )

        # Will use the default "fast" profile for the "o4" model
        semantic.map(instruction="Construct a formal proof of the {hypothesis}.", model_alias="o4")

        # Will use the "thorough" profile for the "o4" model
        semantic.map(instruction="Construct a formal proof of the {hypothesis}.", model_alias=ModelAlias(name="o4", profile="thorough"))
        ```
    """
    model_name: OpenAILanguageModelName = Field(..., description="The name of the OpenAI model to use")
    rpm: int = Field(..., gt=0, description="Requests per minute; must be > 0")
    tpm: int = Field(..., gt=0, description="Tokens per minute; must be > 0")
    profiles: Optional[dict[str, Profile]] = Field(default=None, description=profiles_desc)
    default_profile: Optional[str] = Field(default=None, description=default_profiles_desc)

    class Profile(BaseModel):
        """OpenAI-specific profile configurations.

        This class defines profile configurations for OpenAI models, allowing a user to reference
        the same underlying model in semantic operations with different settings. For now, only
        the reasoning effort can be customized.

        Attributes:
            reasoning_effort: If configuring a reasoning model, provide a reasoning effort. OpenAI has separate o-series reasoning models,
                for which thinking cannot be disabled. If an o-series model is specified, but no `reasoning_effort` is provided,
                the `reasoning_effort` will be set to `low`.

        Note:
            When using an o-series reasoning model, the `temperature` cannot be customized -- any changes to `temperature` will be ignored.

        Example:
            Configuring a profile with medium reasoning effort:

            ```python
            profile = OpenAILanguageModel.Profile(reasoning_effort="medium")
            ```
        """

        reasoning_effort: Optional[ReasoningEffort] = Field(
            default=None, description="The reasoning effort level for the profile"
        )

class OpenAIEmbeddingModel(BaseModel):
    """Configuration for OpenAI embedding models.

    This class defines the configuration settings for OpenAI embedding models,
    including model selection and rate limiting parameters.

    Attributes:
        model_name: The name of the OpenAI embedding model to use.
        rpm: Requests per minute limit; must be greater than 0.
        tpm: Tokens per minute limit; must be greater than 0.

    Example:
        Configuring an OpenAI embedding model with rate limits:

        ```python
        config = OpenAIEmbeddingModel(
            model_name="text-embedding-3-small",
            rpm=100,
            tpm=100
        )
        ```
    """
    model_name: OpenAIEmbeddingModelName = Field(..., description="The name of the OpenAI embedding model to use")
    rpm: int = Field(..., gt=0, description="Requests per minute; must be > 0")
    tpm: int = Field(..., gt=0, description="Tokens per minute; must be > 0")


class AnthropicLanguageModel(BaseModel):
    """Configuration for Anthropic language models.

    This class defines the configuration settings for Anthropic language models,
    including model selection and separate rate limiting parameters for input and output tokens.

    Attributes:
        model_name: The name of the Anthropic model to use.
        rpm: Requests per minute limit; must be greater than 0.
        input_tpm: Input tokens per minute limit; must be greater than 0.
        output_tpm: Output tokens per minute limit; must be greater than 0.
        profiles: Optional mapping of profile names to profile configurations.
        default_profile: The name of the default profile to use if profiles are configured.

    Example:
        Configuring an Anthropic model with separate input/output rate limits:

        ```python
        config = AnthropicLanguageModel(
            model_name="claude-3-5-haiku-latest",
            rpm=100,
            input_tpm=100,
            output_tpm=100
        )
        ```

        Configuring an Anthropic model with profiles:

        ```python
        config = SessionConfig(
            semantic=SemanticConfig(
                language_models={
                    "claude": AnthropicLanguageModel(
                        model_name="claude-opus-4-0",
                        rpm=100,
                        input_tpm=100,
                        output_tpm=100,
                        profiles={
                            "thinking_disabled": AnthropicLanguageModel.Profile(),
                            "fast": AnthropicLanguageModel.Profile(thinking_token_budget=1024),
                            "thorough": AnthropicLanguageModel.Profile(thinking_token_budget=4096)
                        },
                        default_profile="fast"
                    )
                },
                default_language_model="claude"
        )

        # Using the default "fast" profile for the "claude" model
        semantic.map(instruction="Construct a formal proof of the {hypothesis}.", model_alias="claude")

        # Using the "thorough" profile for the "claude" model
        semantic.map(instruction="Construct a formal proof of the {hypothesis}.", model_alias=ModelAlias(name="claude", profile="thorough"))
        ```
    """
    model_name: AnthropicLanguageModelName = Field(..., description="The name of the Anthropic model to use")
    rpm: int = Field(..., gt=0, description="Requests per minute; must be > 0")
    input_tpm: int = Field(..., gt=0, description="Input tokens per minute; must be > 0")
    output_tpm: int = Field(..., gt=0, description="Output tokens per minute; must be > 0")
    profiles: Optional[dict[str, Profile]] = Field(default=None, description=profiles_desc)
    default_profile: Optional[str] = Field(default=None, description=default_profiles_desc)

    class Profile(BaseModel):
        """Anthropic-specific profile configurations.

        This class defines profile configurations for Anthropic models, allowing
        different thinking token budget settings to be applied to the same model.

        Attributes:
            thinking_token_budget: If configuring a model that supports reasoning, provide a default thinking budget in tokens. If not provided,
                thinking will be disabled for the profile. The minimum token budget supported by Anthropic is 1024 tokens.

        Note:
            If `thinking_token_budget` is set, `temperature` cannot be customized -- any changes to `temperature` will be ignored.

        Example:
            Configuring a profile with a thinking budget:

            ```python
            profile = AnthropicLanguageModel.Profile(thinking_token_budget=2048)
            ```

            Configuring a profile with a large thinking budget:

            ```python
            profile = AnthropicLanguageModel.Profile(thinking_token_budget=8192)
            ```
        """

        thinking_token_budget: Optional[int] = Field(
            default=None,
            description="The thinking budget in tokens for the profile",
            ge=1024,
        )

EmbeddingModel = Union[OpenAIEmbeddingModel, GoogleVertexEmbeddingModel, GoogleDeveloperEmbeddingModel]
LanguageModel = Union[OpenAILanguageModel, AnthropicLanguageModel, GoogleDeveloperLanguageModel, GoogleVertexLanguageModel]
ModelConfig = Union[EmbeddingModel, LanguageModel]

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

    Example:
        Configuring semantic models with a single language model:

        ```python
        config = SemanticConfig(
            language_models={
                "gpt4": OpenAILanguageModel(
                    model_name="gpt-4.1-nano",
                    rpm=100,
                    tpm=100
                )
            }
        )
        ```

        Configuring semantic models with multiple language models and an embedding model:

        ```python
        config = SemanticConfig(
            language_models={
                "gpt4": OpenAILanguageModel(
                    model_name="gpt-4.1-nano",
                    rpm=100,
                    tpm=100
                ),
                "claude": AnthropicLanguageModel(
                    model_name="claude-3-5-haiku-latest",
                    rpm=100,
                    input_tpm=100,
                    output_tpm=100
                ),
                "gemini": GoogleDeveloperLanguageModel(
                    model_name="gemini-2.0-flash",
                    rpm=100,
                    tpm=1000
                )
            },
            default_language_model="gpt4",
            embedding_models={
                "openai_embeddings": OpenAIEmbeddingModel(
                    model_name="text-embedding-3-small",
                    rpm=100,
                    tpm=100
                )
            },
            default_embedding_model="openai_embeddings"
        )
        ```

        Configuring models with profiles:

        ```python
        config = SemanticConfig(
            language_models={
                "gpt4": OpenAILanguageModel(
                    model_name="gpt-4o-mini",
                    rpm=100,
                    tpm=100,
                    profiles={
                        "fast": OpenAILanguageModel.Profile(reasoning_effort="low"),
                        "thorough": OpenAILanguageModel.Profile(reasoning_effort="high")
                    },
                    default_profile="fast"
                ),
                "claude": AnthropicLanguageModel(
                    model_name="claude-3-5-haiku-latest",
                    rpm=100,
                    input_tpm=100,
                    output_tpm=100,
                    profiles={
                        "fast": AnthropicLanguageModel.Profile(thinking_token_budget=1024),
                        "thorough": AnthropicLanguageModel.Profile(thinking_token_budget=4096)
                    },
                    default_profile="fast"
                )
            },
            default_language_model="gpt4"
        )
        ```
    """
    language_models: Optional[dict[str, LanguageModel]] = None
    default_language_model: Optional[str] = None
    embedding_models: Optional[dict[str, EmbeddingModel]] = None
    default_embedding_model: Optional[str] = None

    def model_post_init(self, __context) -> None:
        """Post initialization hook to set defaults.

        This hook runs after the model is initialized and validated.
        It sets the default language and embedding models if they are not set
        and there is only one model available.
        """
        if self.language_models:
            # Set default language model if not set and only one model exists
            if self.default_language_model is None and len(self.language_models) == 1:
                self.default_language_model = list(self.language_models.keys())[0]

            # Set default profile for each model if not set and only one profile exists
            for model_config in self.language_models.values():
                if model_config.profiles is not None:
                    profile_names = list(model_config.profiles.keys())
                    if model_config.default_profile is None and len(profile_names) == 1:
                        model_config.default_profile = profile_names[0]

        # Set default embedding model if not set and only one model exists
        if self.embedding_models:
            if self.default_embedding_model is None and len(self.embedding_models) == 1:
                self.default_embedding_model = list(self.embedding_models.keys())[0]
            # Set default profile for each model if not set and only one preset exists
            for model_config in self.embedding_models.values():
                if hasattr(model_config, "profiles") and model_config.profiles is not None:
                    preset_names = list(model_config.profiles.keys())
                    if (
                        model_config.default_profile is None
                        and len(preset_names) == 1
                    ):
                        model_config.default_profile = preset_names[0]

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
                raise ConfigurationError(
                    f"default_language_model is not set, and multiple language models are configured. Please specify one of: {available_language_model_aliases} as a default_language_model.")

            if self.default_language_model is not None and self.default_language_model not in self.language_models:
                raise ConfigurationError(
                    f"default_language_model {self.default_language_model} is not in configured map of language models. Available models: {available_language_model_aliases} .")

            for model_alias, language_model in self.language_models.items():
                language_model_name = language_model.model_name
                language_model_provider = _get_model_provider_for_model_config(language_model)

                if language_model.profiles is not None:
                    profile_names = list(language_model.profiles.keys())
                    if language_model.default_profile is None and len(profile_names) > 0:
                        raise ConfigurationError(
                            f"default_profile is not set for model {model_alias}, but multiple profiles are configured. Please specify one of: {profile_names} as a default_profile.")
                    if language_model.default_profile is not None and language_model.default_profile not in profile_names:
                        raise ConfigurationError(
                            f"default_profile {language_model.default_profile} is not in configured profiles for model {model_alias}. Available profiles: {profile_names}")

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
                raise ConfigurationError(
                    f"default_embedding_model is not set, and multiple embedding models are configured. Please specify one of: {available_embedding_model_aliases} as a default_embedding_model.")

            if self.default_embedding_model is not None and self.default_embedding_model not in self.embedding_models:
                raise ConfigurationError(
                    f"default_embedding_model {self.default_embedding_model} is not in configured map of embedding models. Available models: {available_embedding_model_aliases} .")
            for model_alias, embedding_model in self.embedding_models.items():
                embedding_model_provider = _get_model_provider_for_model_config(embedding_model)
                embedding_model_name = embedding_model.model_name
                embedding_model_parameters = model_catalog.get_embedding_model_parameters(embedding_model_provider,
                                                                                          embedding_model_name)
                if embedding_model_parameters is None:
                    raise ConfigurationError(model_catalog.generate_unsupported_embedding_model_error_message(
                        embedding_model_provider,
                        embedding_model_name
                    ))
                if hasattr(embedding_model, "profiles") and embedding_model.profiles:
                    profile_names = list(embedding_model.profiles.keys())
                    if embedding_model.default_profile is None and len(profile_names) > 0:
                        raise ConfigurationError(
                            f"default_profile is not set for model {model_alias}, but multiple profiles are configured. Please specify one of: {profile_names} as a default_profile.")
                    if embedding_model.default_profile is not None and embedding_model.default_profile not in profile_names:
                        raise ConfigurationError(
                            f"default_profile {embedding_model.default_profile} is not in configured profiles for model {model_alias}. Available profiles: {profile_names}")

                    for profile_alias, profile in embedding_model.profiles.items():
                        _validate_embedding_profile(embedding_model_parameters, profile_alias, profile)


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

    Example:
        Configuring cloud execution with a specific size:

        ```python
        config = CloudConfig(size=CloudExecutorSize.MEDIUM)
        ```

        Using default cloud configuration:

        ```python
        config = CloudConfig()
        ```
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

    Example:
        Configuring a basic session with a single language model:

        ```python
        config = SessionConfig(
            app_name="my_app",
            semantic=SemanticConfig(
                language_models={
                    "gpt4": OpenAILanguageModel(
                        model_name="gpt-4.1-nano",
                        rpm=100,
                        tpm=100
                    )
                }
            )
        )
        ```

        Configuring a session with multiple models and cloud execution:

        ```python
        config = SessionConfig(
            app_name="production_app",
            db_path=Path("/path/to/database.db"),
            semantic=SemanticConfig(
                language_models={
                    "gpt4": OpenAILanguageModel(
                        model_name="gpt-4.1-nano",
                        rpm=100,
                        tpm=100
                    ),
                    "claude": AnthropicLanguageModel(
                        model_name="claude-3-5-haiku-latest",
                        rpm=100,
                        input_tpm=100,
                        output_tpm=100
                    )
                },
                default_language_model="gpt4",
                embedding_models={
                    "openai_embeddings": OpenAIEmbeddingModel(
                        model_name="text-embedding-3-small",
                        rpm=100,
                        tpm=100
                    )
                },
                default_embedding_model="openai_embeddings"
            ),
            cloud=CloudConfig(size=CloudExecutorSize.MEDIUM)
        )
        ```
    """
    app_name: str = "default_app"
    db_path: Optional[Path] = None
    semantic: Optional[SemanticConfig] = None
    cloud: Optional[CloudConfig] = None

    def _to_resolved_config(self) -> ResolvedSessionConfig:
        def resolve_model(model: ModelConfig) -> ResolvedModelConfig:
            if isinstance(model, OpenAIEmbeddingModel):
                return ResolvedOpenAIModelConfig(
                    model_name=model.model_name,
                    rpm=model.rpm,
                    tpm=model.tpm,
                )
            elif isinstance(model, OpenAILanguageModel):
                profiles = {
                    profile: ResolvedOpenAIModelProfile(reasoning_effort=profile_config.reasoning_effort) for
                    profile, profile_config in model.profiles.items()
                } if hasattr(model, "profiles") and model.profiles else None
                return ResolvedOpenAIModelConfig(
                    model_name=model.model_name,
                    rpm=model.rpm,
                    tpm=model.tpm,
                    profiles=profiles,
                    default_profile=model.default_profile
                )
            elif isinstance(model, (GoogleDeveloperLanguageModel, GoogleVertexLanguageModel)):
                profiles = {
                    profile: ResolvedGoogleModelProfile(thinking_token_budget=profile_config.thinking_token_budget) for
                    profile, profile_config in model.profiles.items()
                } if model.profiles else None
                return ResolvedGoogleModelConfig(
                    model_name=model.model_name,
                    model_provider=_get_model_provider_for_model_config(model),
                    rpm=model.rpm,
                    tpm=model.tpm,
                    profiles=profiles,
                    default_profile=model.default_profile,
                )
            elif isinstance(model, (GoogleDeveloperEmbeddingModel, GoogleVertexEmbeddingModel)):
                resolved_profiles = {
                    profile_name: ResolvedGoogleModelProfile(
                        embedding_dimensionality=profile.output_dimensionality,
                        embedding_task_type=profile.task_type,
                    ) for
                    profile_name, profile in model.profiles.items()
                }
                return ResolvedGoogleModelConfig(
                    model_name=model.model_name,
                    model_provider=model.model_provider,
                    rpm=model.rpm,
                    tpm=model.tpm,
                    profiles=resolved_profiles,
                    default_profile=model.default_profile,
                )
            elif isinstance(model, AnthropicLanguageModel):
                profiles = {
                    profile: ResolvedAnthropicModelProfile(thinking_token_budget=profile_config.thinking_token_budget) for
                    profile, profile_config in model.profiles.items()
                } if model.profiles else None
                return ResolvedAnthropicModelConfig(
                    model_name=model.model_name,
                    rpm=model.rpm,
                    input_tpm=model.input_tpm,
                    output_tpm=model.output_tpm,
                    profiles=profiles,
                    default_profile=model.default_profile
                )
            else:
                raise InternalError(f"Unknown model type: {type(model)}")

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

def _validate_embedding_profile(
    embedding_model_parameters: EmbeddingModelParameters,
    profile_alias: str,
    profile: EmbeddingModel.Profile
):
    """Validate Embedding profile against embedding model parameters."""
    if hasattr(profile, "output_dimensionality") and not embedding_model_parameters.supports_dimensions(profile.output_dimensionality):
        raise ConfigurationError(
            f"The dimensionality of the Embeddings model profile {profile_alias} is invalid."
            f"Requested dimensionality: {profile.embedding_dimensionality}"
            f"Available Options: {embedding_model_parameters.get_possible_dimensions()}")

def _get_model_provider_for_model_config(model_config: ModelConfig) -> ModelProvider:
    """Determine the ModelProvider for the given model configuration."""
    if isinstance(model_config, (OpenAILanguageModel, OpenAIEmbeddingModel)):
        return ModelProvider.OPENAI
    elif isinstance(model_config, (GoogleDeveloperLanguageModel, GoogleDeveloperEmbeddingModel)):
        return ModelProvider.GOOGLE_DEVELOPER
    elif isinstance(model_config, (GoogleVertexLanguageModel, GoogleVertexEmbeddingModel)):
        return ModelProvider.GOOGLE_VERTEX
    elif isinstance(model_config, AnthropicLanguageModel):
        return ModelProvider.ANTHROPIC
    else :
        raise InternalError(f"Unknown model type: {type(model_config)}")
