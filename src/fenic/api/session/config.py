"""Session configuration classes for Fenic."""

from __future__ import annotations

import re
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from fenic.core._inference.model_catalog import (
    AnthropicLanguageModelName,
    CohereEmbeddingModelName,
    CompletionModelParameters,
    EmbeddingModelParameters,
    GoogleDeveloperEmbeddingModelName,
    GoogleDeveloperLanguageModelName,
    GoogleVertexEmbeddingModelName,
    GoogleVertexLanguageModelName,
    MediaResolutionType,
    ModelProvider,
    OpenAIEmbeddingModelName,
    OpenAILanguageModelName,
    ThinkingLevelType,
    model_catalog,
)
from fenic.core._resolved_session_config import (
    ReasoningEffort,
    ResolvedAnthropicModelConfig,
    ResolvedAnthropicModelProfile,
    ResolvedCacheConfig,
    ResolvedCloudConfig,
    ResolvedCohereModelConfig,
    ResolvedCohereModelProfile,
    ResolvedEmbeddingModelConfig,
    ResolvedGoogleModelConfig,
    ResolvedGoogleModelProfile,
    ResolvedLanguageModelConfig,
    ResolvedModelConfig,
    ResolvedOpenAIModelConfig,
    ResolvedOpenAIModelProfile,
    ResolvedOpenRouterModelConfig,
    ResolvedOpenRouterModelProfile,
    ResolvedOpenRouterProviderRouting,
    ResolvedSemanticConfig,
    ResolvedSessionConfig,
    Verbosity,
)
from fenic.core.error import ConfigurationError, InternalError
from fenic.core.types.enums import CacheBackend
from fenic.core.types.provider_routing import (
    DataCollection,
    ModelQuantization,
    ProviderSort,
    StructuredOutputStrategy,
)

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
    "FACT_VERIFICATION",
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
            model_name="gemini-embedding-001", rpm=100, tpm=1000
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
                "high_dim": GoogleDeveloperEmbeddingModelConfig.Profile(
                    output_dimensionality=3072
                ),
            },
            default_profile="default",
        )
        ```
    """

    model_name: GoogleDeveloperEmbeddingModelName
    model_provider: ModelProvider = Field(default=ModelProvider.GOOGLE_DEVELOPER)
    rpm: int = Field(..., gt=0, description="Requests per minute; must be > 0")
    tpm: int = Field(..., gt=0, description="Tokens per minute; must be > 0")
    profiles: Optional[dict[str, Profile]] = Field(
        default=None, description=profiles_desc
    )
    default_profile: Optional[str] = Field(
        default=None, description=default_profiles_desc
    )

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
            profile = GoogleDeveloperEmbeddingModelConfig.Profile(
                output_dimensionality=3072
            )
            ```

            Configuring a profile with default settings:

            ```python
            profile = GoogleDeveloperEmbeddingModelConfig.Profile()
            ```
        """

        model_config = ConfigDict(extra="forbid")

        output_dimensionality: Optional[int] = Field(
            default=None,
            gt=0,
            le=3072,
            description="Dimensionality of the embedding created by this model",
        )
        task_type: GoogleEmbeddingTaskType = Field(
            default="SEMANTIC_SIMILARITY", description="Type of the task"
        )


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
                "fast": GoogleDeveloperLanguageModel.Profile(
                    thinking_token_budget=1024
                ),
                "thorough": GoogleDeveloperLanguageModel.Profile(
                    thinking_token_budget=8192
                ),
            },
            default_profile="fast",
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
                Note: For gemini-3 models, use thinking_level instead.
            thinking_level: For gemini-3+ models, set the thinking level to high, medium, low, or minimal.
                This parameter is mutually exclusive with thinking_token_budget.
            media_resolution: For gemini-3+ models, set the media resolution for PDF processing.
                Can be "low", "medium", "high", or "ultra_high". Affects token cost per page.

        Raises:
            ConfigurationError: If a profile is set with parameters that are not supported by the model.

        Example:
            Configuring a profile with a fixed thinking budget (gemini-2.5 and earlier):

            ```python
            profile = GoogleDeveloperLanguageModel.Profile(thinking_token_budget=4096)
            ```

            Configuring a profile with thinking level (gemini-3+):

            ```python
            profile = GoogleDeveloperLanguageModel.Profile(thinking_level="high")
            ```
        """

        model_config = ConfigDict(extra="forbid")

        thinking_token_budget: Optional[int] = Field(
            default=None, description="The thinking budget in tokens.", ge=-1, lt=32768
        )
        thinking_level: Optional[ThinkingLevelType] = Field(
            default=None, description="The thinking level for gemini-3+ models (high, medium, low, or minimal)."
        )
        media_resolution: Optional[MediaResolutionType] = Field(
            default=None, description="The media resolution for PDF processing in gemini-3+ models."
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
            model_name="gemini-embedding-001", rpm=100, tpm=1000
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
                "high_dim": GoogleVertexEmbeddingModel.Profile(
                    output_dimensionality=3072
                ),
            },
            default_profile="default",
        )
        ```
    """

    model_name: GoogleVertexEmbeddingModelName
    model_provider: ModelProvider = Field(default=ModelProvider.GOOGLE_VERTEX)
    rpm: int = Field(..., gt=0, description="Requests per minute; must be > 0")
    tpm: int = Field(..., gt=0, description="Tokens per minute; must be > 0")
    profiles: Optional[dict[str, Profile]] = Field(
        default=None, description=profiles_desc
    )
    default_profile: Optional[str] = Field(
        default=None, description=default_profiles_desc
    )

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
            profile = GoogleVertexEmbeddingModelConfig.Profile(
                output_dimensionality=3072
            )
            ```

            Configuring a profile with default settings:

            ```python
            profile = GoogleVertexEmbeddingModelConfig.Profile()
            ```
        """

        model_config = ConfigDict(extra="forbid")

        output_dimensionality: Optional[int] = Field(
            default=None,
            ge=768,
            le=3072,
            description="Dimensionality of the embedding created by this model",
        )
        task_type: GoogleEmbeddingTaskType = Field(
            default="SEMANTIC_SIMILARITY", description="Type of the task"
        )


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
            model_name="gemini-2.0-flash", rpm=100, tpm=1000
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
                "thorough": GoogleVertexLanguageModel.Profile(
                    thinking_token_budget=8192
                ),
            },
            default_profile="fast",
        )
        ```
    """

    model_name: GoogleVertexLanguageModelName
    rpm: int = Field(..., gt=0, description="Requests per minute; must be > 0")
    tpm: int = Field(..., gt=0, description="Tokens per minute; must be > 0")
    profiles: Optional[dict[str, Profile]] = Field(
        default=None, description=profiles_desc
    )
    default_profile: Optional[str] = Field(
        default=None, description=default_profiles_desc
    )

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
                Note: For gemini-3 models, use thinking_level instead.
            thinking_level: For gemini-3+ models, set the thinking level to high, medium, low, or minimal.
                This parameter is mutually exclusive with thinking_token_budget.
            media_resolution: For gemini-3+ models, set the media resolution for PDF processing.
                Can be "low", "medium", "high", or "ultra_high". Affects token cost per page.

        Raises:
            ConfigurationError: If a profile is set with parameters that are not supported by the model.

        Example:
            Configuring a profile with a fixed thinking budget (gemini-2.5 and earlier):

            ```python
            profile = GoogleVertexLanguageModel.Profile(thinking_token_budget=4096)
            ```

            Configuring a profile with thinking level (gemini-3+):

            ```python
            profile = GoogleVertexLanguageModel.Profile(thinking_level="high")
            ```
        """

        model_config = ConfigDict(extra="forbid")

        thinking_token_budget: Optional[int] = Field(
            default=None, description="The thinking budget in tokens.", ge=-1, lt=32768
        )
        thinking_level: Optional[ThinkingLevelType] = Field(
            default=None, description="The thinking level for gemini-3+ models (high, medium, low, or minimal)."
        )
        media_resolution: Optional[MediaResolutionType] = Field(
            default=None, description="The media resolution for PDF processing in gemini-3+ models."
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

    Note:
        When using an o-series or gpt5 reasoning model without specifying a reasoning effort in
        a Profile, the `reasoning_effort` will default to `low` (for o-series models) or `minimal`
        (for gpt5 models).

    Example:
        Configuring an OpenAI language model with rate limits:

        ```python
        config = OpenAILanguageModel(model_name="gpt-4.1-nano", rpm=100, tpm=100)
        ```

        Configuring an OpenAI model with profiles:

        ```python
        config = OpenAILanguageModel(
            model_name="o4-mini",
            rpm=100,
            tpm=100,
            profiles={
                "fast": OpenAILanguageModel.Profile(reasoning_effort="low"),
                "thorough": OpenAILanguageModel.Profile(reasoning_effort="high"),
            },
            default_profile="fast",
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
                        "thorough": OpenAILanguageModel.Profile(
                            reasoning_effort="high"
                        ),
                    },
                    default_profile="fast",
                )
            },
            default_language_model="o4",
        )

        # Will use the default "fast" profile for the "o4" model
        semantic.map(
            instruction="Construct a formal proof of the {hypothesis}.",
            model_alias="o4",
        )

        # Will use the "thorough" profile for the "o4" model
        semantic.map(
            instruction="Construct a formal proof of the {hypothesis}.",
            model_alias=ModelAlias(name="o4", profile="thorough"),
        )
        ```
    """

    model_name: OpenAILanguageModelName = Field(
        ..., description="The name of the OpenAI model to use"
    )
    rpm: int = Field(..., gt=0, description="Requests per minute; must be > 0")
    tpm: int = Field(..., gt=0, description="Tokens per minute; must be > 0")
    profiles: Optional[dict[str, Profile]] = Field(
        default=None, description=profiles_desc
    )
    default_profile: Optional[str] = Field(
        default=None, description=default_profiles_desc
    )

    class Profile(BaseModel):
        """OpenAI-specific profile configurations.

        This class defines profile configurations for OpenAI models, allowing a user to reference
        the same underlying model in semantic operations with different settings.

        Attributes:
            reasoning_effort: Provide a reasoning effort. Only for gpt5 and o-series models.
                Valid values: 'none', 'minimal', 'low', 'medium', 'high'.
                - For gpt-5.1 models: defaults to 'none' (disabled reasoning), does NOT support 'minimal'
                - For gpt-5 models: defaults to 'minimal', does NOT support 'none'
                - For o-series models: defaults to 'low', does NOT support 'none' or 'minimal'
            verbosity: Provide a verbosity level. Only for gpt5/gpt5.1 models.

        Raises:
            ConfigurationError: If a profile is set with parameters that are not supported by the model.

        Note:
            When using an o-series or gpt5 reasoning model with reasoning enabled, the `temperature` cannot be customized.
            For gpt-5.1 models with reasoning_effort='none', temperature CAN be customized.

        Example:
            Configuring a profile with medium reasoning effort:

            ```python
            profile = OpenAILanguageModel.Profile(reasoning_effort="medium")
            ```
        """

        model_config = ConfigDict(extra="forbid")

        reasoning_effort: Optional[ReasoningEffort] = Field(
            default=None, description="The reasoning effort level for the profile"
        )
        verbosity: Optional[Verbosity] = Field(
            default=None, description="The verbosity level for the profile"
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
            model_name="text-embedding-3-small", rpm=100, tpm=100
        )
        ```
    """

    model_name: OpenAIEmbeddingModelName = Field(
        ..., description="The name of the OpenAI embedding model to use"
    )
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
            model_name="claude-3-5-haiku-latest", rpm=100, input_tpm=100, output_tpm=100
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

    model_name: AnthropicLanguageModelName = Field(
        ..., description="The name of the Anthropic model to use"
    )
    rpm: int = Field(..., gt=0, description="Requests per minute; must be > 0")
    input_tpm: int = Field(
        ..., gt=0, description="Input tokens per minute; must be > 0"
    )
    output_tpm: int = Field(
        ..., gt=0, description="Output tokens per minute; must be > 0"
    )
    profiles: Optional[dict[str, Profile]] = Field(
        default=None, description=profiles_desc
    )
    default_profile: Optional[str] = Field(
        default=None, description=default_profiles_desc
    )

    class Profile(BaseModel):
        """Anthropic-specific profile configurations.

        This class defines profile configurations for Anthropic models, allowing
        different thinking token budget settings to be applied to the same model.

        Attributes:
            thinking_token_budget: Provide a default thinking budget in tokens. If not provided,
                thinking will be disabled for the profile. The minimum token budget supported by Anthropic is 1024 tokens.

        Raises:
            ConfigurationError: If a profile is set with parameters that are not supported by the model.

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

        model_config = ConfigDict(extra="forbid")

        thinking_token_budget: Optional[int] = Field(
            default=None,
            description="The thinking budget in tokens for the profile",
            ge=1024,
        )

ParsingEngine = Literal["mistral-ocr", "pdf-text", "native"]

class OpenRouterLanguageModel(BaseModel):
    """Configuration for OpenRouter language models.

    This class defines the configuration settings for OpenRouter language models,
    including model selection and rate limiting parameters. When fetching available models from OpenRouter, results
    will be filtered to only include models from providers that are not in the user’s ignored providers list and are either
    in the user’s allowed providers list (if configured) or from any provider (if no allowed providers are specified).

    Attributes:
        model_name: `{family}/{model}` identifier (e.g., `anthropic/claude-3-5-sonnet`).
        profiles: Mapping of profile names to profile configurations.
        default_profile: The key in `profiles` to select by default.
        structured_output_strategy: The strategy to use for structured output if a model supports both tool calling and structured outputs.

            - `prefer_tools`: prefer using tools over response format.
            - `prefer_response_format`: prefer using response format over tools.

    Requirements:
        - Set `OPENROUTER_API_KEY` in your environment.

    Example:
    ```python
    OpenRouterLanguageModel(
        model_name="openai/gpt-oss-20b",
        profiles={
            "default": OpenRouterLanguageModel.Profile(
                provider=OpenRouterLanguageModel.Provider(
                    sort="price"  # Routes to the cheapest available provider
                )
            )
        },
    )
    ```
    Example:
    ```python
    OpenRouterLanguageModel(
        model_name="anthropic/claude-sonnet-4-0-latest",
        profiles={
            "default": OpenRouterLanguageModel.Profile(
                provider=OpenRouterLanguageModel.Provider(
                    only=[
                        "Anthropic"
                    ]  # ensures the request will only be routed to Anthropic and not AWS Bedrock or Google Vertex
                )
            )
        },
    )
    ```
    Example:
    ```python
    OpenRouterLanguageModel(
        model_name="qwen/qwen3-next-80b-a3b-instruct",
        profiles={
            "default": OpenRouterLanguageModel.Profile(
                provider=OpenRouterLanguageModel.Provider(
                    sort="throughput", # routes to the provider with the highest overall throughput
                    data_collection="deny" # eliminates providers that retain prompt data (would only route to DeepInfra/AtlasCloud, in this example)
                    # Eliminate providers that offer an fp8 quantized version of the model, only allowing bf16.
                    # Note that many providers have an `unknown` quantization, so you may be excluding more providers than you expect.
                    quantizations=["bf16"]
                )
            )
        }
    )
    ```
    """

    model_name: str = Field(
        ...,
        description="The name of the OpenRouter model to use, typically `{provider}/{model_name}`",
    )
    profiles: Optional[dict[str, Profile]] = Field(
        default=None, description=profiles_desc
    )
    default_profile: Optional[str] = Field(
        default=None, description=default_profiles_desc
    )
    structured_output_strategy: Optional[StructuredOutputStrategy] = Field(
        default=None,
        description="The strategy to use for structured output if a model supports both tool calling and structured outputs. `prefer_tools`: prefer using tools over response format. `prefer_response_format`: prefer using response format over tools.",
    )

    class Provider(BaseModel):
        """Provider routing configuration for OpenRouter language models.

        [Provider Routing Documentation](https://openrouter.ai/docs/features/provider-routing)

        Attributes:
            order: List of providers to try in order (e.g. ['Anthropic', 'Amazon Bedrock']).
            sort: Provider routing preference (e.g. 'price', 'throughput', 'latency').
                "price" will route to the cheapest available provider first, progressing through the list of providers in order of price.
                "throughput" will route to the provider with the highest overall recent throughput, progressing through the list of providers in order of throughput.
                "latency" will route to the provider with the lowest overall recent latency, progressing through the list of providers in order of latency.
            quantizations: Allowed quantizations. Note: many providers report `unknown`.
            data_collection: Data collection preference. `allow`: allows the use of providers which store prompt data
                non-transiently and may train on it. `deny`: use only providers which do not collect/store prompt data.
            only: Only include these providers when performing provider routing.
            exclude: Exclude these providers when performing provider routing.
            max_prompt_price: Maximum prompt price ($USD per 1M tokens).
            max_completion_price: Maximum completion price ($USD per 1M tokens).
        """

        model_config = ConfigDict(extra="forbid")

        order: Optional[list[str]] = Field(
            default=None,
            description="List of providers to try in order",
        )
        sort: Optional[ProviderSort] = Field(
            default=None, description="Sort providers by preference"
        )
        quantizations: Optional[list[ModelQuantization]] = Field(
            default=None,
            description="Allowed model quantizations (e.g. ['fp16', 'fp8'])",
        )
        data_collection: Optional[DataCollection] = Field(
            default=None,
            description="Data collection preference. `allow`: allows the use of providers which store user data non-transiently and may train on it. `deny`: use only providers which do not collect/store user data.",
        )
        only: Optional[list[str]] = Field(
            default=None,
            description="Only include these providers when performing provider routing",
        )
        ignore: Optional[list[str]] = Field(
            default=None,
            description="Exclude these providers when performing provider routing",
        )
        max_prompt_price: Optional[float] = Field(
            default=None, description="Maximum prompt price per 1M tokens."
        )
        max_completion_price: Optional[float] = Field(
            default=None, description="Maximum completion price per 1M tokens."
        )

    class Profile(BaseModel):
        """Profile configurations for OpenRouter language models.

        Attributes:
            models: A list of fallback models to use if the primary model is unavailable.
                ([OpenRouter Documentation](https://openrouter.ai/docs/features/model-routing#the-models-parameter)).
            provider: Provider routing preferences (include/exclude specific providers, set provider ranking method preference)
                ([OpenRouter Documentation](https://openrouter.ai/docs/features/provider-routing)).
            reasoning_effort: OpenAI Style reasoning effort configuration (low, medium, high).
                If the model does support reasoning, but not `reasoning_effort`, a `reasoning_max_tokens` will be calculated
                that is roughly equivalent as a percentage of the model's maximum output size
                ([OpenRouter Documentation](https://openrouter.ai/docs/use-cases/reasoning-tokens#reasoning-effort-level))
            reasoning_max_tokens: Supported by Anthropic, Gemini, etc., sets a token budget for reasoning
                If the model does support reasoning, but not `reasoning_max_tokens`, a `reasoning_effort_ will be automatically
                calculated based on `reasoning_max_tokens` as a percentage of the model's maximum output size
                ([OpenRouter Documentation](https://openrouter.ai/docs/use-cases/reasoning-tokens#max-tokens-for-reasoning))
            parsing_engine: The parsing engine to use for processing PDF files.  By default, the model's native parsing engine will be used.  If the model doesn't support PDF processing and the parsing engine is not provided, an error will be raised.  Note: 'mistral-ocr' incurs additional costs.
                ([OpenRouter Documentation](https://openrouter.ai/docs/features/multimodal/pdfs))
        """

        model_config = ConfigDict(extra="forbid")

        reasoning_effort: Optional[Literal["high", "medium", "low"]] = Field(
            default=None, description="OpenAI-style reasoning effort"
        )
        reasoning_max_tokens: Optional[int] = Field(
            default=None,
            gt=0,
            description="Non-OpenAI-style reasoning effort (max tokens)",
        )
        models: Optional[list[str]] = Field(
            default=None,
            description="Alternate models to fall back to if the primary model is not available",
            max_length=3,
        )
        provider: Optional[OpenRouterLanguageModel.Provider] = Field(
            default=None, description="Provider routing configuration"
        )
        parsing_engine: Optional[ParsingEngine] = Field(
            default=None,
            description="The parsing engine to use for processing PDF files. By default, the model's native parsing engine will be used."
        )


CohereEmbeddingTaskType = Literal[
    "search_document",
    "search_query",
    "classification",
    "clustering",
]


class CohereEmbeddingModel(BaseModel):
    """Configuration for Cohere embedding models.

    This class defines the configuration settings for Cohere embedding models,
    including model selection and rate limiting parameters.

    Attributes:
        model_name: The name of the Cohere model to use.
        rpm: Requests per minute limit for the model.
        tpm: Tokens per minute limit for the model.
        profiles: Optional dictionary of profile configurations.
        default_profile: Default profile name to use if none specified.

    Example:
        Configuring a Cohere embedding model with profiles:
        ```python
        cohere_config = CohereEmbeddingModel(
            model_name="embed-v4.0",
            rpm=100,
            tpm=50_000,
            profiles={
                "high_dim": CohereEmbeddingModel.Profile(
                    embedding_dimensionality=1536, embedding_task_type="search_document"
                ),
                "classification": CohereEmbeddingModel.Profile(
                    embedding_dimensionality=1024, embedding_task_type="classification"
                ),
            },
            default_profile="high_dim",
        )
        ```
    """

    model_name: CohereEmbeddingModelName = Field(
        ..., description="The name of the Cohere model to use"
    )
    rpm: int = Field(..., gt=0, description="Requests per minute; must be > 0")
    tpm: int = Field(..., gt=0, description="Tokens per minute; must be > 0")
    profiles: Optional[dict[str, Profile]] = Field(
        default=None, description=profiles_desc
    )
    default_profile: Optional[str] = Field(
        default=None, description=default_profiles_desc
    )

    class Profile(BaseModel):
        """Profile configurations for Cohere embedding models.

        This class defines profile configurations for Cohere embedding models, allowing
        different output dimensionality and task type settings to be applied to the same model.

        Attributes:
            output_dimensionality: The dimensionality of the embedding created by this model.
                If not provided, the model will use its default dimensionality.
            input_type: The type of input text (search_query, search_document, classification, clustering)

        Example:
            Configuring a profile with custom dimensionality:

            ```python
            profile = CohereEmbeddingModel.Profile(output_dimensionality=1536)
            ```

            Configuring a profile with default settings:

            ```python
            profile = CohereEmbeddingModel.Profile()
            ```
        """

        model_config = ConfigDict(extra="forbid")

        output_dimensionality: Optional[int] = Field(
            default=None,
            gt=0,
            le=1536,
            description="Dimensionality of the embedding created by this model",
        )
        input_type: CohereEmbeddingTaskType = Field(
            default="search_document", description="Type of input"
        )


EmbeddingModel = Union[
    OpenAIEmbeddingModel,
    GoogleVertexEmbeddingModel,
    GoogleDeveloperEmbeddingModel,
    CohereEmbeddingModel,
]
LanguageModel = Union[
    OpenAILanguageModel,
    AnthropicLanguageModel,
    GoogleDeveloperLanguageModel,
    GoogleVertexLanguageModel,
    OpenRouterLanguageModel,
]
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
                "gpt4": OpenAILanguageModel(model_name="gpt-4.1-nano", rpm=100, tpm=100)
            }
        )
        ```

        Configuring semantic models with multiple language models and an embedding model:

        ```python
        config = SemanticConfig(
            language_models={
                "gpt4": OpenAILanguageModel(
                    model_name="gpt-4.1-nano", rpm=100, tpm=100
                ),
                "claude": AnthropicLanguageModel(
                    model_name="claude-3-5-haiku-latest",
                    rpm=100,
                    input_tpm=100,
                    output_tpm=100,
                ),
                "gemini": GoogleDeveloperLanguageModel(
                    model_name="gemini-2.0-flash", rpm=100, tpm=1000
                ),
            },
            default_language_model="gpt4",
            embedding_models={
                "openai_embeddings": OpenAIEmbeddingModel(
                    model_name="text-embedding-3-small", rpm=100, tpm=100
                )
            },
            default_embedding_model="openai_embeddings",
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
                        "thorough": OpenAILanguageModel.Profile(
                            reasoning_effort="high"
                        ),
                    },
                    default_profile="fast",
                ),
                "claude": AnthropicLanguageModel(
                    model_name="claude-3-5-haiku-latest",
                    rpm=100,
                    input_tpm=100,
                    output_tpm=100,
                    profiles={
                        "fast": AnthropicLanguageModel.Profile(
                            thinking_token_budget=1024
                        ),
                        "thorough": AnthropicLanguageModel.Profile(
                            thinking_token_budget=4096
                        ),
                    },
                    default_profile="fast",
                ),
            },
            default_language_model="gpt4",
        )
        ```
    """

    language_models: Optional[dict[str, LanguageModel]] = None
    default_language_model: Optional[str] = None
    embedding_models: Optional[dict[str, EmbeddingModel]] = None
    default_embedding_model: Optional[str] = None
    llm_response_cache: Optional[LLMResponseCacheConfig] = None

    def model_post_init(self, __context) -> None:
        """Post initialization hook to set defaults.

        This hook runs after the model is initialized and validated.
        It sets the default language and embedding models if they are not set
        and there is only one model available. For Google models that support
        thinking_level, it auto-creates "low" and "high" profiles if no profiles
        are configured.
        """
        if self.language_models:
            # Set default language model if not set and only one model exists
            if self.default_language_model is None and len(self.language_models) == 1:
                self.default_language_model = list(self.language_models.keys())[0]

            # Auto-create profiles for Google models that support thinking_level
            for model_config in self.language_models.values():
                if isinstance(model_config, (GoogleDeveloperLanguageModel, GoogleVertexLanguageModel)):
                    model_provider = _get_model_provider_for_model_config(model_config)
                    model_params = model_catalog.get_completion_model_parameters(
                        model_provider, model_config.model_name
                    )
                    if model_params and model_params.supported_thinking_levels and model_config.profiles is None:
                        # Auto-create profiles for each supported thinking level
                        if isinstance(model_config, GoogleDeveloperLanguageModel):
                            model_config.profiles = {
                                level: GoogleDeveloperLanguageModel.Profile(thinking_level=level)
                                for level in model_params.supported_thinking_levels
                            }
                        else:
                            model_config.profiles = {
                                level: GoogleVertexLanguageModel.Profile(thinking_level=level)
                                for level in model_params.supported_thinking_levels
                            }
                        model_config.default_profile = "low"

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
                if (
                    hasattr(model_config, "profiles")
                    and model_config.profiles is not None
                ):
                    preset_names = list(model_config.profiles.keys())
                    if model_config.default_profile is None and len(preset_names) == 1:
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
                    f"default_language_model is not set, and multiple language models are configured. Please specify one of: {available_language_model_aliases} as a default_language_model."
                )

            if (
                self.default_language_model is not None
                and self.default_language_model not in self.language_models
            ):
                raise ConfigurationError(
                    f"default_language_model {self.default_language_model} is not in configured map of language models. Available models: {available_language_model_aliases} ."
                )

            for model_alias, language_model in self.language_models.items():
                language_model_name = language_model.model_name
                language_model_provider = _get_model_provider_for_model_config(
                    language_model
                )

                completion_model_params = model_catalog.get_completion_model_parameters(
                    language_model_provider, language_model_name
                )
                if completion_model_params is None:
                    raise ConfigurationError(
                        model_catalog.generate_unsupported_completion_model_error_message(
                            language_model_provider, language_model_name
                        )
                    )
                if language_model.profiles is not None:
                    if not completion_model_params.supports_profiles:
                        raise ConfigurationError(
                            f"Model '{model_alias}' does not support parameter profiles. Please remove the Profile configuration."
                        )
                    profile_names = list(language_model.profiles.keys())
                    if (
                        language_model.default_profile is None
                        and len(profile_names) > 0
                    ):
                        raise ConfigurationError(
                            f"default_profile is not set for model {model_alias}, but multiple profiles are configured. Please specify one of: {profile_names} as a default_profile."
                        )
                    if (
                        language_model.default_profile is not None
                        and language_model.default_profile not in profile_names
                    ):
                        raise ConfigurationError(
                            f"default_profile {language_model.default_profile} is not in configured profiles for model {model_alias}. Available profiles: {profile_names}"
                        )
                    for profile_alias, profile in language_model.profiles.items():
                        _validate_language_profile(
                            language_model,
                            model_alias,
                            completion_model_params,
                            profile,
                            profile_alias,
                        )

        if self.embedding_models is not None:
            available_embedding_model_aliases = list(self.embedding_models.keys())
            if self.default_embedding_model is None and len(self.embedding_models) > 1:
                raise ConfigurationError(
                    f"default_embedding_model is not set, and multiple embedding models are configured. Please specify one of: {available_embedding_model_aliases} as a default_embedding_model."
                )

            if (
                self.default_embedding_model is not None
                and self.default_embedding_model not in self.embedding_models
            ):
                raise ConfigurationError(
                    f"default_embedding_model {self.default_embedding_model} is not in configured map of embedding models. Available models: {available_embedding_model_aliases} ."
                )
            for model_alias, embedding_model in self.embedding_models.items():
                embedding_model_provider = _get_model_provider_for_model_config(
                    embedding_model
                )
                embedding_model_name = embedding_model.model_name
                embedding_model_parameters = (
                    model_catalog.get_embedding_model_parameters(
                        embedding_model_provider, embedding_model_name
                    )
                )
                if embedding_model_parameters is None:
                    raise ConfigurationError(
                        model_catalog.generate_unsupported_embedding_model_error_message(
                            embedding_model_provider, embedding_model_name
                        )
                    )
                if hasattr(embedding_model, "profiles") and embedding_model.profiles:
                    profile_names = list(embedding_model.profiles.keys())
                    if (
                        embedding_model.default_profile is None
                        and len(profile_names) > 0
                    ):
                        raise ConfigurationError(
                            f"default_profile is not set for model {model_alias}, but multiple profiles are configured. Please specify one of: {profile_names} as a default_profile."
                        )
                    if (
                        embedding_model.default_profile is not None
                        and embedding_model.default_profile not in profile_names
                    ):
                        raise ConfigurationError(
                            f"default_profile {embedding_model.default_profile} is not in configured profiles for model {model_alias}. Available profiles: {profile_names}"
                        )

                    for profile_alias, profile in embedding_model.profiles.items():
                        _validate_embedding_profile(
                            embedding_model_parameters,
                            model_alias,
                            profile_alias,
                            profile,
                        )

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


class LLMResponseCacheConfig(BaseModel):
    """Configuration for LLM response caching.

    LLM response caching stores the results of language model API calls to reduce
    costs and improve performance for repeated queries. This is distinct from
    DataFrame caching (the `.cache()` operator).

    Attributes:
        enabled: Whether caching is enabled (default: True).
        backend: Cache backend to use (default: LOCAL).
        ttl: Time-to-live duration string (default: "1h").
            Format: <number><unit> where unit is s/m/h/d.
            Examples: "30s", "15m", "2h", "7d".
            Maximum: 30 days, Minimum: 1 second.
        max_size_mb: Maximum cache size in MB before LRU eviction (default: 128MB).
        namespace: Cache namespace for isolation (default: "default").

    Example:
        Basic configuration within SemanticConfig:

        ```python
        config = SessionConfig(
            app_name="my_app",
            semantic=SemanticConfig(
                language_models={
                    "gpt": OpenAILanguageModel(model_name="gpt-4o-mini", rpm=100, tpm=1000)
                },
                llm_response_cache=LLMResponseCacheConfig(
                    enabled=True,
                    ttl="1h",
                    max_size_mb=1000,
                )
            )
        )
        ```

        Custom TTL and larger cache:

        ```python
        llm_response_cache=LLMResponseCacheConfig(
            enabled=True,
            ttl="7d",  # 7 days
            max_size_mb=5000,
        )
        ```

        Disabled caching:

        ```python
        llm_response_cache=LLMResponseCacheConfig(enabled=False)
        ```
    """

    backend: CacheBackend = Field(default=CacheBackend.LOCAL)
    ttl: str = Field(default="1h")
    max_size_mb: int = Field(default=128, gt=0, le=100000)
    namespace: str = Field(default="default")

    @field_validator("ttl")
    @classmethod
    def validate_ttl(cls, v: str) -> str:
        """Validate TTL duration string format.

        Format: <number><unit> where unit is s/m/h/d
        Examples: "30s", "15m", "2h", "7d"

        Args:
            v: TTL duration string to validate.

        Returns:
            The validated TTL string.

        Raises:
            ValueError: If format is invalid or value is out of range.
        """
        pattern = r"^(\d+)([smhd])$"
        match = re.match(pattern, v.lower())

        if not match:
            raise ValueError(
                f"Invalid TTL format: '{v}'. "
                "Expected: <number><unit> where unit is s/m/h/d. "
                "Examples: '30m', '2h', '1d'"
            )

        value, unit = match.groups()
        value = int(value)

        # Validate ranges
        if unit == "s" and value < 1:
            raise ValueError("TTL must be at least 1 second")
        if unit == "h" and value > 720:  # 30 days
            raise ValueError("TTL cannot exceed 720 hours (30 days)")
        if unit == "d" and value > 30:
            raise ValueError("TTL cannot exceed 30 days")

        return v

    def ttl_seconds(self) -> int:
        """Convert TTL string to seconds.

        Returns:
            TTL duration in seconds.

        Raises:
            ValueError: If TTL format is invalid.
        """
        pattern = r"^(\d+)([smhd])$"
        match = re.match(pattern, self.ttl.lower())

        if not match:
            raise ValueError(f"Invalid TTL format: '{self.ttl}'")

        value, unit = match.groups()
        value = int(value)

        multipliers = {"s": 1, "m": 60, "h": 3600, "d": 86400}
        return value * multipliers[unit]


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
        cache: Optional configuration for LLM response caching.

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
                        model_name="gpt-4.1-nano", rpm=100, tpm=100
                    )
                }
            ),
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
                        model_name="gpt-4.1-nano", rpm=100, tpm=100
                    ),
                    "claude": AnthropicLanguageModel(
                        model_name="claude-3-5-haiku-latest",
                        rpm=100,
                        input_tpm=100,
                        output_tpm=100,
                    ),
                },
                default_language_model="gpt4",
                embedding_models={
                    "openai_embeddings": OpenAIEmbeddingModel(
                        model_name="text-embedding-3-small", rpm=100, tpm=100
                    )
                },
                default_embedding_model="openai_embeddings",
            ),
            cloud=CloudConfig(size=CloudExecutorSize.MEDIUM),
        )
        ```
    """

    app_name: str = "default_app"
    db_path: Optional[Path] = None
    semantic: Optional[SemanticConfig] = None
    cloud: Optional[CloudConfig] = None

    def to_json(self) -> str:
        """Export the session config to a JSON string."""
        return self.model_dump_json(indent=2)

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
                    profile: ResolvedOpenAIModelProfile(reasoning_effort=profile_config.reasoning_effort, verbosity=profile_config.verbosity) for
                    profile, profile_config in model.profiles.items()
                } if model.profiles else None
                return ResolvedOpenAIModelConfig(
                    model_name=model.model_name,
                    rpm=model.rpm,
                    tpm=model.tpm,
                    profiles=profiles,
                    default_profile=model.default_profile,
                )
            elif isinstance(model, (GoogleDeveloperLanguageModel, GoogleVertexLanguageModel)):
                profiles = {
                    profile: ResolvedGoogleModelProfile(
                        thinking_token_budget=profile_config.thinking_token_budget,
                        thinking_level=profile_config.thinking_level,
                        media_resolution=profile_config.media_resolution,
                    ) for
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
                } if model.profiles else None
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
                    profile_name: ResolvedAnthropicModelProfile(thinking_token_budget=profile.thinking_token_budget) for
                    profile_name, profile in model.profiles.items()
                } if model.profiles else None
                return ResolvedAnthropicModelConfig(
                    model_name=model.model_name,
                    rpm=model.rpm,
                    input_tpm=model.input_tpm,
                    output_tpm=model.output_tpm,
                    profiles=profiles,
                    default_profile=model.default_profile,
                )
            elif isinstance(model, CohereEmbeddingModel):
                profiles = {
                    profile_name: ResolvedCohereModelProfile(
                        embedding_dimensionality=profile.output_dimensionality,
                        embedding_task_type=profile.input_type
                    ) for
                    profile_name, profile in model.profiles.items()
                } if model.profiles else None
                return ResolvedCohereModelConfig(
                    model_name=model.model_name,
                    rpm=model.rpm,
                    tpm=model.tpm,
                    profiles=profiles,
                    default_profile=model.default_profile,
                )
            elif isinstance(model, OpenRouterLanguageModel):
                profiles = (
                    {
                        profile_name: ResolvedOpenRouterModelProfile(
                            reasoning_effort=profile.reasoning_effort,
                            reasoning_max_tokens=profile.reasoning_max_tokens,
                            models=profile.models,
                            parsing_engine=profile.parsing_engine,
                            provider=(
                                ResolvedOpenRouterProviderRouting(
                                    **(profile.provider.model_dump())
                                )
                                if profile.provider is not None
                                else None
                            ),
                            structured_output_strategy=model.structured_output_strategy,
                        )
                        for profile_name, profile in model.profiles.items()
                    }
                    if model.profiles
                    else None
                )
                return ResolvedOpenRouterModelConfig(
                    model_name=model.model_name,
                    profiles=profiles,
                    default_profile=model.default_profile,
                )
            else:
                raise InternalError(f"Unknown model type: {type(model)}")

        language_models = (
            ResolvedLanguageModelConfig(
                model_configs={
                    alias: resolve_model(cfg)
                    for alias, cfg in self.semantic.language_models.items()
                },
                default_model=self.semantic.default_language_model,
            )
            if self.semantic and self.semantic.language_models
            else None
        )

        embedding_models = (
            ResolvedEmbeddingModelConfig(
                model_configs={
                    alias: resolve_model(cfg)
                    for alias, cfg in self.semantic.embedding_models.items()
                },
                default_model=self.semantic.default_embedding_model,
            )
            if self.semantic and self.semantic.embedding_models
            else None
        )

        # Resolve cache config from semantic config
        resolved_cache = None
        if self.semantic and self.semantic.llm_response_cache:

            cache_cfg = self.semantic.llm_response_cache
            resolved_cache = ResolvedCacheConfig(
                backend=cache_cfg.backend,
                ttl=cache_cfg.ttl,
                ttl_seconds=cache_cfg.ttl_seconds(),
                max_size_mb=cache_cfg.max_size_mb,
                namespace=cache_cfg.namespace,
            )

        resolved_semantic = ResolvedSemanticConfig(
            language_models=language_models,
            embedding_models=embedding_models,
            llm_response_cache=resolved_cache,
        )

        resolved_cloud = (
            ResolvedCloudConfig(size=self.cloud.size) if self.cloud else None
        )

        return ResolvedSessionConfig(
            app_name=self.app_name,
            db_path=self.db_path,
            semantic=resolved_semantic,
            cloud=resolved_cloud,
        )


def _validate_language_profile(
    language_model: LanguageModel,
    model_alias: str,
    completion_model_params: CompletionModelParameters,
    profile: Any,
    profile_alias: str,
) -> None:
    """Validate the language profile against the language model."""
    if isinstance(language_model, OpenAILanguageModel):
        if not completion_model_params.supports_disabled_reasoning and profile.reasoning_effort == "none":
            minimal_str = "'minimal', " if completion_model_params.supports_minimal_reasoning else "" # can't nest quotes in python < 3.12
            raise ConfigurationError(f"Model '{model_alias}' does not support 'none' (disabled) reasoning. Please set reasoning_effort on '{profile_alias}' to {minimal_str}'low', 'medium', or 'high' instead.")
        if not completion_model_params.supports_minimal_reasoning and profile.reasoning_effort == "minimal":
            raise ConfigurationError(f"Model '{model_alias}' does not support 'minimal' reasoning. Please set reasoning_effort on '{profile_alias}' to 'low', 'medium', or 'high' instead.")
        if not completion_model_params.supports_verbosity and profile.verbosity is not None:
            raise ConfigurationError(f"Model '{model_alias}' does not support verbosity. Please remove verbosity from '{profile_alias}'.")
    elif isinstance(language_model, GoogleDeveloperLanguageModel) or isinstance(language_model, GoogleVertexLanguageModel):
        if completion_model_params.supported_thinking_levels:
            # For gemini-3+ models, thinking_level must be used instead of thinking_token_budget
            if profile.thinking_token_budget is not None:
                raise ConfigurationError(
                    f"Model '{model_alias}' uses thinking_level instead of thinking_token_budget. "
                    f"Please set thinking_level on '{profile_alias}' instead."
                )
            # Validate that the thinking_level is in supported set
            if profile.thinking_level and profile.thinking_level not in completion_model_params.supported_thinking_levels:
                raise ConfigurationError(
                    f"Model '{model_alias}' does not support thinking_level='{profile.thinking_level}'. "
                    f"Supported levels: {sorted(completion_model_params.supported_thinking_levels)}"
                )
        else:
            # For older models, thinking_token_budget is used
            if profile.thinking_level is not None:
                raise ConfigurationError(
                    f"Model '{model_alias}' does not support thinking_level. "
                    f"Please use thinking_token_budget on '{profile_alias}' instead."
                )
            if (not profile.thinking_token_budget or profile.thinking_token_budget == 0) and not completion_model_params.supports_disabled_reasoning:
                raise ConfigurationError(f"Model '{model_alias}' does not support disabling reasoning. Please set thinking_token_budget on '{profile_alias}' to a non-zero value.")
    elif isinstance(language_model, OpenRouterLanguageModel):
        # For OpenRouter, validate pass-through parameters against capabilities
        if (
            profile.reasoning_effort or profile.reasoning_max_tokens
        ) and not completion_model_params.supports_reasoning:
            raise ConfigurationError(
                f"Model '{model_alias}' does not support reasoning. Remove 'reasoning' from '{profile_alias}'."
            )


def _validate_embedding_profile(
    embedding_model_parameters: EmbeddingModelParameters,
    model_alias: str,
    profile_alias: str,
    profile: EmbeddingModel.Profile,
):
    """Validate Embedding profile against embedding model parameters."""
    if (
        hasattr(profile, "output_dimensionality")
        and profile.output_dimensionality is not None
        and not embedding_model_parameters.supports_dimensions(
            profile.output_dimensionality
        )
    ):
        raise ConfigurationError(
            f"The dimensionality of the Embeddings model profile {profile_alias} is invalid. "
            f"Requested dimensionality: {profile.output_dimensionality}. "
            f"Available Options: {embedding_model_parameters.get_possible_dimensions()}"
        )


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
    elif isinstance(model_config, CohereEmbeddingModel):
        return ModelProvider.COHERE
    elif isinstance(model_config, OpenRouterLanguageModel):
        return ModelProvider.OPENROUTER
    else:
        raise InternalError(f"Unknown model type: {type(model_config)}")
