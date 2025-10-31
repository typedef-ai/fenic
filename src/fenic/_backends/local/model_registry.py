import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

from fenic._backends.local.async_utils import EventLoopManager
from fenic._inference import (
    EmbeddingModel,
    LanguageModel,
    OpenAIBatchChatCompletionsClient,
    OpenAIBatchEmbeddingsClient,
)
from fenic._inference.openrouter.openrouter_batch_chat_completions_client import (
    OpenRouterBatchChatCompletionsClient,
)
from fenic._inference.rate_limit_strategy import (
    AdaptiveBackoffRateLimitStrategy,
    SeparatedTokenRateLimitStrategy,
    UnifiedTokenRateLimitStrategy,
)
from fenic.core._inference.model_provider import ModelProviderClass
from fenic.core._logical_plan.resolved_types import ResolvedModelAlias
from fenic.core._resolved_session_config import (
    ResolvedAnthropicModelConfig,
    ResolvedCohereModelConfig,
    ResolvedGoogleModelConfig,
    ResolvedModelConfig,
    ResolvedOpenAIModelConfig,
    ResolvedOpenRouterModelConfig,
    ResolvedSemanticConfig,
)
from fenic.core.error import ConfigurationError, InternalError, SessionError
from fenic.core.metrics import LMMetrics, RMMetrics

logger = logging.getLogger(__name__)


@dataclass
class LanguageModelRegistry:
    models: dict[str, LanguageModel]
    default_model: LanguageModel


@dataclass
class EmbeddingModelRegistry:
    models: dict[str, EmbeddingModel]
    default_model: EmbeddingModel


class SessionModelRegistry:
    """Registry for managing language and embedding models in a session.

    This class maintains a collection of language models and embedding models, along with
    their default instances. It provides methods for registering, retrieving, and managing
    these models, as well as tracking their usage metrics.

    Attributes:
        language_model_registry (LanguageModelRegistry): Registry for language models.
        embedding_model_registry (EmbeddingModelRegistry): Registry for embedding models.
    """

    language_model_registry: Optional[LanguageModelRegistry] = None
    embedding_model_registry: Optional[EmbeddingModelRegistry] = None

    def __init__(self, config: ResolvedSemanticConfig, cache=None):
        """Initialize the model registry with configuration.

        Args:
            config (ResolvedSemanticConfig): Configuration containing model settings and defaults.
            cache: Optional LLM response cache instance to pass to language models.
        """
        validate_providers: set[ModelProviderClass] = set()
        if config.language_models:
            language_model_config = config.language_models
            models: dict[str, LanguageModel] = {}
            for alias, model_config in language_model_config.model_configs.items():
                model = self._initialize_language_model(model_config, cache)
                models[alias] = model
                validate_providers.add(model.client.model_provider_class)
            self.language_model_registry = LanguageModelRegistry(
                models=models,
                default_model=models[language_model_config.default_model],
            )

        if config.embedding_models:
            embedding_model_config = config.embedding_models
            models: dict[str, EmbeddingModel] = {}
            for alias, model_config in embedding_model_config.model_configs.items():
                model = self._initialize_embedding_model(model_config)
                models[alias] = model
                validate_providers.add(model.client.model_provider_class)
            self.embedding_model_registry = EmbeddingModelRegistry(
                models=models,
                default_model=models[embedding_model_config.default_model],
            )
        if len(validate_providers) > 0:
            with EventLoopManager().loop_context() as loop:
                future = asyncio.run_coroutine_threadsafe(
                    _validate_provider_api_keys(validate_providers), loop
                )
                future.result()

    def get_language_model_metrics(self) -> LMMetrics:
        """Get aggregated metrics for all language models.

        Returns:
            LMMetrics: Combined metrics from all registered language models.
        """
        if not self.language_model_registry:
            return LMMetrics()
        total_metrics = LMMetrics()
        for language_model in self.language_model_registry.models.values():
            total_metrics += language_model.get_metrics()
        return total_metrics

    def reset_language_model_metrics(self):
        """Reset metrics for all registered language models."""
        if self.language_model_registry:
            for language_model in self.language_model_registry.models.values():
                language_model.reset_metrics()

    def get_embedding_model_metrics(self) -> RMMetrics:
        """Get aggregated metrics for all embedding models.

        Returns:
            RMMetrics: Combined metrics from all registered embedding models.
        """
        if not self.embedding_model_registry:
            return RMMetrics()
        total_metrics = RMMetrics()
        for embedding_model in self.embedding_model_registry.models.values():
            total_metrics += embedding_model.get_metrics()
        return total_metrics

    def reset_embedding_model_metrics(self):
        """Reset metrics for all registered embedding models."""
        if self.embedding_model_registry:
            for embedding_model in self.embedding_model_registry.models.values():
                embedding_model.reset_metrics()

    def get_language_model(
        self, alias: Optional[ResolvedModelAlias] = None
    ) -> LanguageModel:
        """Get a language model by alias or return the default model.

        Args:
            alias (Optional[ResolvedModelAlias], optional): ResolvedModelAlias containing name and optional profile. Defaults to None.

        Returns:
            LanguageModel: The requested language model.

        Raises:
            SessionError: If the requested model is not found.
        """
        if not self.language_model_registry:
            raise InternalError(
                "Requested language model, but no language models are configured."
            )
        if alias is None:
            return self.language_model_registry.default_model
        language_model_for_alias = self.language_model_registry.models.get(alias.name)
        if language_model_for_alias is None:
            raise InternalError(
                f"Language Model with alias '{alias.name}' not found in configured models: {sorted(list(self.language_model_registry.models.keys()))}"
            )
        return language_model_for_alias

    def get_embedding_model(
        self, alias: Optional[ResolvedModelAlias] = None
    ) -> EmbeddingModel:
        """Get an embedding model by alias or return the default model.

        Args:
            alias (Optional[str], optional): Alias of the embedding model to retrieve. Defaults to None.

        Returns:
            EmbeddingModel: The requested embedding model.

        Raises:
            SessionError: If no embedding models are configured or if the requested model is not found.
        """
        if not self.embedding_model_registry:
            raise InternalError(
                "Requested embedding model, but no embedding models are configured."
            )
        if alias is None:
            return self.embedding_model_registry.default_model
        embedding_model_for_model_alias = self.embedding_model_registry.models.get(
            alias.name
        )
        if embedding_model_for_model_alias is None:
            raise InternalError(
                f"Embedding Model with model name '{alias.name}' not found in configured models: {sorted(list(self.embedding_model_registry.models.keys()))}"
            )
        return embedding_model_for_model_alias

    def shutdown_models(self):
        """Shutdown all registered language and embedding models."""
        if self.language_model_registry:
            for alias, language_model in self.language_model_registry.models.items():
                try:
                    language_model.client.shutdown()
                except Exception as e:
                    logger.warning(
                        f"Failed graceful shutdown of language model client {alias}: {e}"
                    )
        if self.embedding_model_registry:
            for alias, embedding_model in self.embedding_model_registry.models.items():
                try:
                    embedding_model.client.shutdown()
                except Exception as e:
                    logger.warning(
                        f"Failed graceful shutdown of embedding model client {alias}: {e}"
                    )

    def _initialize_embedding_model(
        self, model_config: ResolvedModelConfig
    ) -> EmbeddingModel:
        """Initialize an embedding model with the given configuration.

        Args:
            model_config (ModelConfig): Configuration for the embedding client model.

        Returns:
            EmbeddingModel: Initialized embedding client model.

        Raises:
            SessionError: If model initialization fails.
        """
        try:
            if isinstance(model_config, ResolvedOpenAIModelConfig):
                rate_limit_strategy = UnifiedTokenRateLimitStrategy(
                    rpm=model_config.rpm, tpm=model_config.tpm
                )
                client = OpenAIBatchEmbeddingsClient(
                    rate_limit_strategy=rate_limit_strategy,
                    model=model_config.model_name,
                )
            elif isinstance(model_config, ResolvedGoogleModelConfig):
                try:
                    from fenic._inference.google.gemini_batch_embeddings_client import (
                        GoogleBatchEmbeddingsClient,
                    )
                except ImportError as err:
                    raise ImportError(
                        "To use Google models, please install the required dependencies by running: pip install fenic[google]"
                    ) from err
                rate_limit_strategy = UnifiedTokenRateLimitStrategy(
                    rpm=model_config.rpm, tpm=model_config.tpm
                )
                client = GoogleBatchEmbeddingsClient(
                    rate_limit_strategy=rate_limit_strategy,
                    model=model_config.model_name,
                    model_provider=model_config.model_provider,
                    profiles=model_config.profiles,
                    default_profile_name=model_config.default_profile,
                )
            elif isinstance(model_config, ResolvedCohereModelConfig):
                rate_limit_strategy = UnifiedTokenRateLimitStrategy(
                    rpm=model_config.rpm, tpm=model_config.tpm
                )
                try:
                    from fenic._inference.cohere.cohere_batch_embeddings_client import (
                        CohereBatchEmbeddingsClient,
                    )
                except ImportError as err:
                    raise ImportError(
                        "To use Cohere models, please install the required dependencies by running: pip install fenic[cohere]"
                    ) from err
                client = CohereBatchEmbeddingsClient(
                    rate_limit_strategy=rate_limit_strategy,
                    model=model_config.model_name,
                    profile_configurations=model_config.profiles,
                    default_profile_name=model_config.default_profile,
                )
            else:
                raise ConfigurationError(
                    f"Unsupported model configuration: {model_config}"
                )

        except Exception as e:
            raise SessionError(f"Failed to create embedding model client: {e}") from e

        return EmbeddingModel(client=client)

    def _initialize_language_model(
        self, model_config: ResolvedModelConfig, cache=None
    ) -> LanguageModel:
        """Initialize a language client model with the given configuration.

        Args:
            model_config (ModelConfig): Configuration for the language model.
            cache: Optional LLM response cache instance.

        Returns:
            LanguageModel: Initialized language model.

        Raises:
            SessionError: If model initialization fails.
            ConfigurationError: If the model configuration is not supported.
            ImportError: If required dependencies for Anthropic models are not installed.
        """
        try:
            if isinstance(model_config, ResolvedOpenAIModelConfig):
                rate_limit_strategy = UnifiedTokenRateLimitStrategy(
                    rpm=model_config.rpm, tpm=model_config.tpm
                )
                client = OpenAIBatchChatCompletionsClient(
                    model=model_config.model_name,
                    rate_limit_strategy=rate_limit_strategy,
                    profiles=model_config.profiles,
                    default_profile_name=model_config.default_profile,
                    cache=cache,
                )

            elif isinstance(model_config, ResolvedAnthropicModelConfig):
                try:
                    from fenic._inference.anthropic.anthropic_batch_chat_completions_client import (
                        AnthropicBatchCompletionsClient,
                    )
                except ImportError as err:
                    raise ImportError(
                        "To use Anthropic models, please install the required dependencies by running: pip install fenic[anthropic]"
                    ) from err
                rate_limit_strategy = SeparatedTokenRateLimitStrategy(
                    rpm=model_config.rpm,
                    input_tpm=model_config.input_tpm,
                    output_tpm=model_config.output_tpm,
                )
                client = AnthropicBatchCompletionsClient(
                    model=model_config.model_name,
                    rate_limit_strategy=rate_limit_strategy,
                    profiles=model_config.profiles,
                    default_profile_name=model_config.default_profile,
                    cache=cache,
                )

            elif isinstance(model_config, ResolvedGoogleModelConfig):
                try:
                    from fenic._inference.google.gemini_native_chat_completions_client import (
                        GeminiNativeChatCompletionsClient,
                    )
                except ImportError as err:
                    raise ImportError(
                        "To use Google models, please install the required dependencies by running: pip install fenic[google]"
                    ) from err
                rate_limit_strategy = UnifiedTokenRateLimitStrategy(
                    rpm=model_config.rpm, tpm=model_config.tpm
                )
                client = GeminiNativeChatCompletionsClient(
                    model=model_config.model_name,
                    model_provider=model_config.model_provider,
                    rate_limit_strategy=rate_limit_strategy,
                    profiles=model_config.profiles,
                    default_profile_name=model_config.default_profile,
                    cache=cache,
                )
            elif isinstance(model_config, ResolvedOpenRouterModelConfig):
                rate_limit_strategy = AdaptiveBackoffRateLimitStrategy()
                client = OpenRouterBatchChatCompletionsClient(
                    model=model_config.model_name,
                    rate_limit_strategy=rate_limit_strategy,
                    profiles=model_config.profiles,
                    default_profile_name=model_config.default_profile,
                    cache=cache,
                )
            else:
                raise ConfigurationError(
                    f"Unsupported model configuration: {model_config}"
                )
            return LanguageModel(client=client)

        except Exception as e:
            raise SessionError(f"Failed to create language model client: {e}") from e


async def _validate_provider_api_keys(providers: set[ModelProviderClass]):
    tasks = [asyncio.create_task(provider.validate_api_key()) for provider in providers]
    try:
        await asyncio.gather(*tasks)
    except Exception as e:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise ConfigurationError(f"Error during API key validation: {e}") from e
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
