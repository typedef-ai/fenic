import logging
from typing import Optional

from fenic._inference import (
    EmbeddingModel,
    LanguageModel,
    OpenAIBatchChatCompletionsClient,
    OpenAIBatchEmbeddingsClient,
)
from fenic._inference.model_client import (
    SeparatedTokenRateLimitStrategy,
    UnifiedTokenRateLimitStrategy,
)
from fenic.core._resolved_session_config import (
    ResolvedAnthropicModelConfig,
    ResolvedGoogleModelConfig,
    ResolvedModelConfig,
    ResolvedOpenAIModelConfig,
    ResolvedSemanticConfig,
)
from fenic.core.error import ConfigurationError, SessionError
from fenic.core.metrics import LMMetrics, RMMetrics

logger = logging.getLogger(__name__)


class SessionModelRegistry:
    """Registry for managing language and embedding models in a session.

    This class maintains a collection of language models and embedding models, along with
    their default instances. It provides methods for registering, retrieving, and managing
    these models, as well as tracking their usage metrics.

    Attributes:
        language_models (dict[str, LanguageModel]): Dictionary mapping aliases to language models.
        embedding_models (Optional[dict[str, EmbeddingModel]]): Dictionary mapping aliases to embedding models.
        default_language_model (LanguageModel): The default language model to use.
        default_embedding_model (Optional[EmbeddingModel]): The default embedding model to use.
    """

    language_models: dict[str, LanguageModel]
    embedding_models: Optional[dict[str, EmbeddingModel]] = None
    default_language_model: LanguageModel
    default_embedding_model: Optional[EmbeddingModel] = None

    def __init__(self, config: ResolvedSemanticConfig):
        """Initialize the model registry with configuration.

        Args:
            config (ResolvedSemanticConfig): Configuration containing model settings and defaults.
        """
        self.language_models = {
            alias: self._initialize_language_model(config)
            for alias, config in config.language_models.items()
        }
        self.default_language_model = self.language_models[config.default_language_model]

        if config.embedding_models:
            self.embedding_models = {
                alias: self._initialize_embedding_model(config)
                for alias, config in config.embedding_models.items()
            }
            self.default_embedding_model = self.embedding_models[config.default_embedding_model]
        else:
            self.embedding_models = {}
            self.default_embedding_model = None

    def get_language_model_metrics(self) -> LMMetrics:
        """Get aggregated metrics for all language models.

        Returns:
            LMMetrics: Combined metrics from all registered language models.
        """
        total_metrics = LMMetrics()
        for language_model in self.language_models.values():
            total_metrics += language_model.get_metrics()
        return total_metrics

    def reset_language_model_metrics(self):
        """Reset metrics for all registered language models."""
        for language_model in self.language_models.values():
            language_model.reset_metrics()

    def get_embedding_model_metrics(self) -> RMMetrics:
        """Get aggregated metrics for all embedding models.

        Returns:
            RMMetrics: Combined metrics from all registered embedding models.
        """
        total_metrics = RMMetrics()
        if self.embedding_models:
            for embedding_model in self.embedding_models.values():
                total_metrics += embedding_model.get_metrics()
        return total_metrics

    def reset_embedding_model_metrics(self):
        """Reset metrics for all registered embedding models."""
        for embedding_model in self.embedding_models.values():
            embedding_model.reset_metrics()

    def get_language_model(self, alias: Optional[str] = None) -> LanguageModel:
        """Get a language model by alias or return the default model.

        Args:
            alias (Optional[str], optional): Alias of the language model to retrieve. Defaults to None.

        Returns:
            LanguageModel: The requested language model.

        Raises:
            SessionError: If the requested model is not found.
        """
        if alias is None:
            return self.default_language_model
        language_model_for_alias = self.language_models.get(alias)
        if language_model_for_alias is None:
            raise SessionError(f"Language Model with alias '{alias}' not found in configured models: {sorted(list(self.language_models.keys()))}")
        return language_model_for_alias

    def get_embedding_model(self, alias: Optional[str] = None) -> EmbeddingModel:
        """Get an embedding model by alias or return the default model.

        Args:
            alias (Optional[str], optional): Alias of the embedding model to retrieve. Defaults to None.

        Returns:
            EmbeddingModel: The requested embedding model.

        Raises:
            SessionError: If no embedding models are configured or if the requested model is not found.
        """
        if not self.embedding_models:
            raise SessionError("No embedding models configured.")
        if alias is None:
            return self.default_embedding_model
        embedding_model_for_model_alias = self.embedding_models.get(alias)
        if embedding_model_for_model_alias is None:
            raise SessionError(f"Embedding Model with model name '{alias}' not found.")
        return embedding_model_for_model_alias

    def shutdown_models(self):
        """Shutdown all registered language and embedding models."""
        for alias, language_model in self.language_models.items():
            try:
                language_model.client.shutdown()
            except Exception as e:
                logger.warning(f"Failed graceful shutdown of language model client {alias}: {e}")
        if self.embedding_models:
            for alias, embedding_model in self.embedding_models.items():
                try:
                    embedding_model.client.shutdown()
                except Exception as e:
                    logger.warning(f"Failed graceful shutdown of embedding model client {alias}: {e}")

    def _initialize_embedding_model(self, model_config: ResolvedModelConfig) -> EmbeddingModel:
        """Initialize an embedding model with the given configuration.

        Args:
            model_config (ModelConfig): Configuration for the embedding model.

        Returns:
            EmbeddingModel: Initialized embedding model.

        Raises:
            SessionError: If model initialization fails.
        """
        try:
            rate_limit_strategy = UnifiedTokenRateLimitStrategy(rpm=model_config.rpm, tpm=model_config.tpm)
            client = OpenAIBatchEmbeddingsClient(
                rate_limit_strategy=rate_limit_strategy,
                model=model_config.model_name,
            )
        except Exception as e:
            raise SessionError(f"Failed to create retrieval model client: {e}") from e

        return EmbeddingModel(client=client)

    def _initialize_language_model(self, model_config: ResolvedModelConfig) -> LanguageModel:
        """Initialize a language model with the given configuration.

        Args:
            alias (str): Alias for the language model.
            model_config (ModelConfig): Configuration for the language model.

        Returns:
            LanguageModel: Initialized language model.

        Raises:
            SessionError: If model initialization fails.
            ConfigurationError: If the model configuration is not supported.
            ImportError: If required dependencies for Anthropic models are not installed.
        """
        try:
            if isinstance(model_config, ResolvedOpenAIModelConfig):
                rate_limit_strategy = UnifiedTokenRateLimitStrategy(rpm=model_config.rpm, tpm=model_config.tpm)
                client = OpenAIBatchChatCompletionsClient(rate_limit_strategy=rate_limit_strategy,
                                                          model=model_config.model_name)
            elif isinstance(model_config, ResolvedAnthropicModelConfig):
                try:
                    from fenic._inference.anthropic.anthropic_batch_chat_completions_client import (
                        AnthropicBatchCompletionsClient,
                    )
                except ImportError as err:
                    raise ImportError(
                        "To use Anthropic models, please install the required dependencies by running: pip install fenic[anthropic]"
                    ) from err
                rate_limit_strategy = SeparatedTokenRateLimitStrategy(rpm=model_config.rpm,
                                                                      input_tpm=model_config.input_tpm,
                                                                      output_tpm=model_config.output_tpm)
                client = AnthropicBatchCompletionsClient(rate_limit_strategy=rate_limit_strategy,
                                                         model=model_config.model_name)
            elif isinstance(model_config, ResolvedGoogleModelConfig):
                try:
                    from fenic._inference.google.gemini_native_chat_completions_client import (
                        GeminiNativeChatCompletionsClient,
                    )
                except ImportError as err:
                    raise ImportError(
                        "To use Google models, please install the required dependencies by running: pip install fenic[google]"
                    ) from err
                rate_limit_strategy = UnifiedTokenRateLimitStrategy(rpm=model_config.rpm, tpm=model_config.tpm)
                client = GeminiNativeChatCompletionsClient(
                    rate_limit_strategy=rate_limit_strategy,
                    model_provider=model_config.model_provider,
                    model=model_config.model_name,
                    default_thinking_budget=model_config.default_thinking_budget,
                )
            else:
                raise ConfigurationError(f"Unsupported model configuration: {model_config}")
            return LanguageModel(client=client)

        except Exception as e:
            raise SessionError(f"Failed to create language model client: {e}") from e
