from enum import Enum
from typing import Dict, Literal, Optional, TypeAlias, Union


class ModelProvider(Enum):
    """Enum representing different model providers supported by the system."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE_GLA = "google-gla"
    GOOGLE_VERTEX = "google-vertex"

class TieredTokenCost:

    def __init__(
        self,
        input_token_cost: float,
        cached_input_token_read_cost: float,
        output_token_cost: float,
        cached_input_token_write_cost: float = 0.0,
    ):
        self.input_token_cost = input_token_cost
        self.cached_input_token_read_cost = cached_input_token_read_cost
        self.cached_input_token_write_cost = cached_input_token_write_cost
        self.output_token_cost = output_token_cost

class CompletionModelParameters:
    """Parameters for completion models including costs and context window size.

    Attributes:
        input_token_cost: Cost per input token in USD
        cached_input_token_read_cost: Cost per cached input token read in USD
        cached_input_token_write_cost: Cost per cached input token write in USD
        output_token_cost: Cost per output token in USD
        context_window_length: Maximum number of tokens in the context window
        max_output_tokens: Maximum number of tokens the model can generate in a single request.
    """
    def __init__(
        self,
        input_token_cost: float,
        output_token_cost: float,
        context_window_length: int,
        max_output_tokens: int,
        max_temperature: float = 1,
        cached_input_token_write_cost: float = 0.0,
        cached_input_token_read_cost: float= 0.0,
        tiered_token_costs: Optional[Dict[int, TieredTokenCost]] = None,
        supports_reasoning = False,
    ):
        self.input_token_cost = input_token_cost
        self.cached_input_token_read_cost = cached_input_token_read_cost
        self.cached_input_token_write_cost = cached_input_token_write_cost
        self.output_token_cost = output_token_cost
        self.context_window_length = context_window_length
        self.has_tiered_input_token_costs = tiered_token_costs is not None
        self.tiered_input_token_costs = tiered_token_costs
        self.max_output_tokens = max_output_tokens
        self.max_temperature = max_temperature
        self.supports_reasoning = supports_reasoning


class EmbeddingModelParameters:
    """Parameters for embedding models including costs and output dimensions.

    Attributes:
        input_token_cost: Cost per input token in USD
        output_dimensions: Number of dimensions in the embedding output
        max_input_size: Maximum number of tokens in the input string
    """
    def __init__(self, input_token_cost: float, output_dimensions: int, max_input_size: int):
        self.input_token_cost = input_token_cost
        self.output_dimensions = output_dimensions
        self.max_input_size = max_input_size

CompletionModelCollection: TypeAlias = Dict[str, CompletionModelParameters]
EmbeddingModelCollection: TypeAlias = Dict[str, EmbeddingModelParameters]
OPENAI_AVAILABLE_LANGUAGE_MODELS = Literal[
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4.1-2025-04-14",
    "gpt-4.1-mini-2025-04-14",
    "gpt-4.1-nano-2025-04-14",
    "gpt-4o",
    "gpt-4o-2024-11-20",
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-05-13",
    "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18",
    "gpt-4-turbo",
    "gpt-4-turbo-2024-04-09",
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-0613",
]

OPENAI_AVAILABLE_EMBEDDING_MODELS = Literal[
    "text-embedding-3-small",
    "text-embedding-3-large"
]

ANTHROPIC_AVAILABLE_LANGUAGE_MODELS = Literal[
    "claude-3-7-sonnet-latest",
    "claude-3-7-sonnet-20250219",
    "claude-3-5-haiku-latest",
    "claude-3-5-haiku-20241022",
    "claude-sonnet-4-20250514",
    "claude-sonnet-4-0",
    "claude-4-sonnet-20250514",
    "claude-3-5-sonnet-latest",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-20240620",
    "claude-opus-4-0",
    "claude-opus-4-20250514",
    "claude-4-opus-20250514",
    "claude-3-opus-latest",
    "claude-3-opus-20240229",
    "claude-3-haiku-20240307",
]

GOOGLE_GLA_AVAILABLE_MODELS = Literal[
    "gemini-2.5-pro",
    "gemini-2.5-pro-preview-06-05",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite-preview-06-17",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-lite-001",
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
    "gemini-2.0-flash-exp",
]


GOOGLE_VERTEX_AVAILABLE_MODELS = Union[GOOGLE_GLA_AVAILABLE_MODELS]

AVAILABLE_LANGUAGE_MODELS = Union[OPENAI_AVAILABLE_LANGUAGE_MODELS, ANTHROPIC_AVAILABLE_LANGUAGE_MODELS, GOOGLE_GLA_AVAILABLE_MODELS]
AVAILABLE_EMBEDDING_MODELS = Union[OPENAI_AVAILABLE_EMBEDDING_MODELS]

class ModelCatalog:
    """Catalog of supported models and their parameters for different providers.

    This class maintains a registry of all supported models across different providers,
    including their costs, context windows, and other parameters. It provides methods
    to query model information and calculate costs for different operations.
    """
    def __init__(self):
        # Base model collections
        self._anthropic_models = {
            "claude-opus-4-0": CompletionModelParameters(
                input_token_cost=15.00 / 1_000_000,  # $15 per 1M tokens
                cached_input_token_write_cost=18.75 / 1_000_000,  # $18.75 per 1M tokens
                cached_input_token_read_cost=1.50 / 1_000_000,  # $1.50 per 1M tokens
                output_token_cost=75.00 / 1_000_000,  # $75 per 1M tokens
                context_window_length=200_000,
                max_output_tokens=32_000,
            ),
            "claude-sonnet-4-0": CompletionModelParameters(
                input_token_cost=3.00 / 1_000_000,  # $3 per 1M tokens
                cached_input_token_write_cost=3.75 / 1_000_000,  # $3.75 per 1M tokens
                cached_input_token_read_cost=0.30 / 1_000_000,  # $0.30 per 1M tokens
                output_token_cost=15.00 / 1_000_000,  # $15 per 1M tokens
                context_window_length=200_000,
                max_output_tokens=64_000,
            ),
            "claude-3-7-sonnet-latest": CompletionModelParameters(
                input_token_cost=3.0 / 1_000_000,  # $3 per 1M tokens
                cached_input_token_write_cost=3.75 / 1_000_000,  # $3.75 per 1M tokens
                cached_input_token_read_cost=0.30 / 1_000_000,  # $0.30 per 1M tokens
                output_token_cost=15.00 / 1_000_000,  # $15 per 1M tokens
                context_window_length=200_000,
                max_output_tokens=128_000,
            ),
            "claude-3-5-sonnet-latest": CompletionModelParameters(
                input_token_cost=3 / 1_000_000,  # $3 per 1M tokens
                cached_input_token_write_cost=3.75 / 1_000_000,  # $3.75 per 1M tokens
                cached_input_token_read_cost=0.30 / 1_000_000,  # $0.30 per 1M tokens
                output_token_cost=15.00 / 1_000_000,  # $15 per 1M tokens
                context_window_length=200_000,
                max_output_tokens=8_192,
            ),
            "claude-3-5-haiku-latest": CompletionModelParameters(
                input_token_cost=0.80 / 1_000_000,  # $0.80 per 1M tokens
                cached_input_token_write_cost=1.00 / 1_000_000,  # $1.00 per 1M tokens
                cached_input_token_read_cost=0.08 / 1_000_000,  # $0.08 per 1M tokens
                output_token_cost=4.00 / 1_000_000,  # $4 per 1M tokens
                context_window_length=200_000,
                max_output_tokens=8_000,
            ),
            "claude-3-opus-latest": CompletionModelParameters(
                input_token_cost=15.00 / 1_000_000,  # $15 per 1M tokens
                cached_input_token_write_cost=18.75 / 1_000_000,  # $18.75 per 1M tokens
                cached_input_token_read_cost=1.50 / 1_000_000,  # $1.50 per 1M tokens
                output_token_cost=75.00 / 1_000_000,  # $75 per 1M tokens
                context_window_length=200_000,
                max_output_tokens=4_096,
            ),
            "claude-3-haiku-20240307" : CompletionModelParameters(
                input_token_cost=0.25 / 1_000_000,  # $0.25 per 1M tokens
                cached_input_token_write_cost=0.30 / 1_000_000,  # $0.30 per 1M tokens
                cached_input_token_read_cost=0.03 / 1_000_000,  # $0.03 per 1M tokens
                output_token_cost=1.25 / 1_000_000,  # $1.25 per 1M tokens
                context_window_length=200_000,
                max_output_tokens=4_096,
            ),
        }
        self._openai_models = {
            "gpt-4": CompletionModelParameters(
                input_token_cost=30 / 1_000_000,  # $30 per 1M tokens
                cached_input_token_read_cost=0.0,  # N/A
                output_token_cost=60 / 1_000_000,  # $60 per 1M tokens
                context_window_length=8_192,
                max_output_tokens=8_192,
                max_temperature=2,
            ),
            "gpt-4-turbo" : CompletionModelParameters(
                input_token_cost=10 / 1_000_000,  # $10 per 1M tokens
                cached_input_token_read_cost=0.0,  # N/A
                output_token_cost=30 / 1_000_000,  # $30 per 1M tokens
                context_window_length=128_000,
                max_output_tokens=4_096,
                max_temperature=2,
            ),
            "gpt-4o-mini": CompletionModelParameters(
                input_token_cost=0.300 / 1_000_000,  # $0.300 per 1M tokens
                cached_input_token_read_cost=0.150 / 1_000_000,  # $0.150 per 1M tokens
                output_token_cost=1.200 / 1_000_000,  # $1.200 per 1M tokens
                context_window_length=128_000,
                max_output_tokens=16_384,
                max_temperature=2,
            ),
            "gpt-4o": CompletionModelParameters(
                input_token_cost=3.750 / 1_000_000,  # $3.750 per 1M tokens
                cached_input_token_read_cost=1.875 / 1_000_000,  # $1.875 per 1M tokens
                output_token_cost=15.00 / 1_000_000,  # $15.00 per 1M tokens
                context_window_length=128_000,
                max_output_tokens=16_384,
                max_temperature=2,
            ),
            "gpt-4.1-nano": CompletionModelParameters(
                input_token_cost=0.100 / 1_000_000,  # $0.100 per 1M tokens
                cached_input_token_read_cost=0.025 / 1_000_000,  # $0.025 per 1M tokens
                output_token_cost=0.400 / 1_000_000,  # $0.400 per 1M tokens
                context_window_length=1_000_000,
                max_output_tokens=32_768,
                max_temperature=2,
            ),
            "gpt-4.1-mini": CompletionModelParameters(
                input_token_cost=0.400 / 1_000_000,  # $0.400 per 1M tokens
                cached_input_token_read_cost=0.100 / 1_000_000,  # $0.100 per 1M tokens
                output_token_cost=1.600 / 1_000_000,  # $1.600 per 1M tokens
                context_window_length=1_000_000,
                max_output_tokens=32_768,
                max_temperature=2,
            ),
            "gpt-4.1": CompletionModelParameters(
                input_token_cost=2.00 / 1_000_000,  # $2.00 per 1M tokens
                cached_input_token_read_cost=0.500 / 1_000_000,  # $0.500 per 1M tokens
                output_token_cost=8.00 / 1_000_000,  # $8.00 per 1M tokens
                context_window_length=1_000_000,
                max_output_tokens=32_768,
                max_temperature=2,
            ),
        }
        self._openai_embedding_models = {
            "text-embedding-3-small": EmbeddingModelParameters(
                input_token_cost=0.02 / 1_000_000,  # $0.02 per 1M tokens
                output_dimensions=1536,
                max_input_size=8192,
            ),
            "text-embedding-3-large": EmbeddingModelParameters(
                input_token_cost=0.13 / 1_000_000,  # $0.13 per 1M tokens
                output_dimensions=3072,
                max_input_size=8192,
            ),
        }

        self._google_vertex_completion_models = {
            "gemini-2.5-pro": CompletionModelParameters(
                input_token_cost=1.25 / 1_000_000,  # $1.25 per 1M tokens
                cached_input_token_read_cost=0.03125 / 1_000_000,  # $0.03125 per 1M tokens
                output_token_cost=10 / 1_000_000,  # $10 per 1M tokens
                context_window_length=1_048_576,
                max_output_tokens=65_536,
                supports_reasoning=True,
                max_temperature=2.0,
                tiered_token_costs={
                    200_000: TieredTokenCost(
                        input_token_cost=2.50 / 1_000_000,  # $2.50 per 1M tokens
                        cached_input_token_read_cost=0.0625 / 1_000_000,  # $0.0625 per 1M tokens
                        output_token_cost=15 / 1_000_000,  # $15.00 per 1M tokens
                    )
                }
            ),
            "gemini-2.5-flash": CompletionModelParameters(
                input_token_cost=0.30 / 1_000_000,  # $1.25 per 1M tokens
                cached_input_token_read_cost=0.075 / 1_000_000,  # $0.0375 per 1M tokens
                output_token_cost=2.50 / 1_000_000,  # $2.50 per 1M tokens
                context_window_length=1_048_576,
                max_output_tokens=65_536,
                max_temperature=2.0,
                supports_reasoning=True,
            ),
            "gemini-2.5-flash-lite-preview-06-17": CompletionModelParameters(
                input_token_cost=0.10 / 1_000_000,  # $0.075 per 1M tokens
                output_token_cost=0.40 / 1_000_000,  # $0.30 per 1M tokens
                context_window_length=1_000_000,
                max_output_tokens=64_000,
                max_temperature=2.0,
                supports_reasoning=True,
            ),
            "gemini-2.0-flash-lite": CompletionModelParameters(
                input_token_cost=0.075 / 1_000_000,  # $0.075 per 1M tokens
                output_token_cost=0.30 / 1_000_000,  # $0.30 per 1M tokens
                context_window_length=1_048_576,
                max_output_tokens=8_192,
                max_temperature=2.0,
            ),
            "gemini-2.0-flash": CompletionModelParameters(
                input_token_cost=0.15 / 1_000_000,  # $0.15 per 1M tokens
                cached_input_token_read_cost=0.0375 / 1_000_000,  # $0.01875 per 1M tokens
                output_token_cost=0.60 / 1_000_000,  # $0.60 per 1M tokens
                context_window_length=1_048_576,
                max_output_tokens=8_192,
                max_temperature=2.0,
            ),
        }

        self._google_developer_completion_models = {
            "gemini-2.5-pro" : CompletionModelParameters(
                input_token_cost=1.25 / 1_000_000,  # $1.25 per 1M tokens
                cached_input_token_read_cost=0.03125 / 1_000_000,  # $0.03125 per 1M tokens
                output_token_cost=10 / 1_000_000,  # $10 per 1M tokens
                context_window_length=1_048_576,
                max_output_tokens=65_536,
                max_temperature=2.0,
                supports_reasoning=True,
                tiered_token_costs={
                    200_000: TieredTokenCost(
                        input_token_cost=2.50 / 1_000_000,  # $2.50 per 1M tokens
                        cached_input_token_read_cost=0.0625 / 1_000_000,  # $0.0625 per 1M tokens
                        output_token_cost=15 / 1_000_000,  # $15.00 per 1M tokens
                    )
                }
            ),
            "gemini-2.5-flash" : CompletionModelParameters(
                input_token_cost=0.15 / 1_000_000, # $1.25 per 1M tokens
                cached_input_token_read_cost= 0.075/ 1_000_000, # $0.0375 per 1M tokens
                output_token_cost=2.50 / 1_000_000, # $2.50 per 1M tokens
                context_window_length=1_048_576,
                max_output_tokens=65_536,
                max_temperature=2.0,
                supports_reasoning=True,
            ),
            "gemini-2.5-flash-lite-preview-06-17" : CompletionModelParameters(
                input_token_cost=0.10 / 1_000_000, # $0.075 per 1M tokens
                output_token_cost=0.40 / 1_000_000, # $0.30 per 1M tokens
                context_window_length=1_000_000,
                max_output_tokens=64_000,
                max_temperature=2.0,
                supports_reasoning=True,
            ),
            "gemini-2.0-flash-lite" : CompletionModelParameters(
                input_token_cost=0.075 / 1_000_000, # $0.075 per 1M tokens
                output_token_cost=0.30 / 1_000_000, # $0.30 per 1M tokens
                context_window_length=1_048_576,
                max_output_tokens=8_192,
                max_temperature=2.0,
            ),
            "gemini-2.0-flash": CompletionModelParameters(
                input_token_cost=0.10 / 1_000_000, # $0.15 per 1M tokens
                cached_input_token_read_cost= 0.0375 / 1_000_000, # $0.01875 per 1M tokens
                output_token_cost=0.40 / 1_000_000, # $0.60 per 1M tokens
                context_window_length=1_048_576,
                max_output_tokens=8_192,
                max_temperature=2.0,
            ),
        }

        # Snapshot mappings
        self._language_model_snapshots: Dict[tuple[ModelProvider, AVAILABLE_LANGUAGE_MODELS], list[str]] = {
            # OpenAI snapshots
            (ModelProvider.OPENAI, "gpt-4.1-nano"): ["gpt-4.1-nano-2025-04-14"],
            (ModelProvider.OPENAI, "gpt-4.1-mini"): ["gpt-4.1-mini-2025-04-14"],
            (ModelProvider.OPENAI, "gpt-4.1"): ["gpt-4.1-2025-04-14"],
            (ModelProvider.OPENAI, "gpt-4o-mini"): ["gpt-4o-mini-2024-07-18"],
            (ModelProvider.OPENAI, "gpt-4o"): [
                "gpt-4o-2024-05-13",
                "gpt-4o-2024-08-06",
                "gpt-4o-2024-11-20"
            ],
            (ModelProvider.OPENAI, "gpt-4-turbo"): ["gpt-4-turbo-2024-04-09"],
            (ModelProvider.OPENAI, "gpt-4") : [
                "gpt-4-0314",
                "gpt-4-0613",
            ],
            # Anthropic snapshots
            (ModelProvider.ANTHROPIC, "claude-opus-4-0"): ["claude-opus-4-20250514", "claude-4-opus-20250514"],
            (ModelProvider.ANTHROPIC, "claude-sonnet-4-0"): ["claude-sonnet-4-20250514", "claude-4-sonnet-20250514"],
            (ModelProvider.ANTHROPIC, "claude-3-7-sonnet-latest"): ["claude-3-7-sonnet-20250219"],
            (ModelProvider.ANTHROPIC, "claude-3-5-haiku-latest"): ["claude-3-5-haiku-20241022"],
            (ModelProvider.ANTHROPIC, "claude-3-5-sonnet-latest"): [
                "claude-3-5-sonnet-20241022",
                "claude-3-5-sonnet-20240620",
            ],
            (ModelProvider.ANTHROPIC, "claude-3-opus-latest"): ["claude-3-opus-20240229"],
            # Google (Vertex) Snapshots
            (ModelProvider.GOOGLE_VERTEX, "gemini-2.5-pro"): ["gemini-2.5-pro-preview-06-05"],
            (ModelProvider.GOOGLE_VERTEX, "gemini-2.0-flash-lite"): ["gemini-2.0-flash-lite-001"],
            (ModelProvider.GOOGLE_VERTEX, "gemini-2.0-flash"): [
                "gemini-2.0-flash-001",
                "gemini-2.0-flash-exp",
            ],
            (ModelProvider.GOOGLE_GLA, "gemini-2.5-pro") : ["gemini-2.5-pro-preview-06-05"],
            (ModelProvider.GOOGLE_GLA, "gemini-2.0-flash-lite") : ["gemini-2.0-flash-lite-001"],
            (ModelProvider.GOOGLE_GLA, "gemini-2.0-flash") : [
                "gemini-2.0-flash-001",
                "gemini-2.0-flash-exp",
            ],
        }

        #currently, no supported embedding models have snapshots.
        self._embedding_model_snapshots: Dict[tuple[ModelProvider, AVAILABLE_EMBEDDING_MODELS], list[str]] = {}

        # Create complete model collections including snapshots
        self._complete_completion_models = {
            ModelProvider.ANTHROPIC: self._create_complete_model_collection(
                self._anthropic_models,
                self._language_model_snapshots,
                ModelProvider.ANTHROPIC
            ),
            ModelProvider.OPENAI: self._create_complete_model_collection(
                self._openai_models,
                self._language_model_snapshots,
                ModelProvider.OPENAI
            ),
            ModelProvider.GOOGLE_GLA: self._create_complete_model_collection(
                self._google_developer_completion_models,
                self._language_model_snapshots,
                ModelProvider.GOOGLE_GLA
            ),
            ModelProvider.GOOGLE_VERTEX: self._create_complete_model_collection(
                self._google_vertex_completion_models,
                self._language_model_snapshots,
                ModelProvider.GOOGLE_VERTEX
            )
        }

        self._complete_embedding_models = {
            ModelProvider.OPENAI: self._create_complete_model_collection(
                self._openai_embedding_models,
                self._embedding_model_snapshots,
                ModelProvider.OPENAI
            )
        }

    # Public methods
    def get_completion_model_parameters(self, model_provider: ModelProvider,
                                        model_name: str) -> CompletionModelParameters | None:
        """Gets the parameters for a specific completion model.

        Args:
            model_provider: The provider of the model
            model_name: The name of the model

        Returns:
            Model parameters if found, None otherwise
        """
        return self._get_supported_completions_models_by_provider(model_provider).get(model_name)

    def get_embedding_model_parameters(self, model_provider: ModelProvider, model_name: str) -> EmbeddingModelParameters | None:
        """Gets the parameters for a specific embedding model.

        Args:
            model_provider: The provider of the model
            model_name: The name of the model

        Returns:
            Model parameters if found, None otherwise
        """
        return self._get_supported_embeddings_models_by_provider(model_provider).get(model_name)

    def generate_unsupported_completion_model_error_message(self, model_provider: ModelProvider,
                                                            model_name: str) -> str:
        """Generates an error message for unsupported completion models.

        Args:
            model_provider: The provider of the unsupported model
            model_name: The name of the unsupported model

        Returns:
            Error message string
        """
        return f"Model '{model_name}' is not supported for {model_provider.value}. Supported Models: {self._get_supported_completions_models_by_provider_as_string(model_provider)}"

    def generate_unsupported_embedding_model_error_message(self, model_provider: ModelProvider, model_name: str) -> str:
        """Generates an error message for unsupported embedding models.

        Args:
            model_provider: The provider of the unsupported model
            model_name: The name of the unsupported model

        Returns:
            Error message string
        """
        return f"Model '{model_name}' is not supported for {model_provider.value}. Supported Models: {self._get_supported_embeddings_models_by_provider(model_provider)}"

    def get_supported_completions_models_as_string(self) -> str:
        """Returns a comma-separated string of all supported completion models.

        Returns:
            Comma-separated string of model names in format 'provider:model'
        """
        all_models = []
        for model_provider in ModelProvider:
            for model in self._get_supported_completions_models_by_provider(model_provider).keys():
                all_models.append(f"{model_provider.value}:{model}")
        return ", ".join(sorted(all_models))

    def get_supported_embeddings_models_as_string(self) -> str:
        """Returns a comma-separated string of all supported embedding models.

        Returns:
            Comma-separated string of model names
        """
        all_models = []
        for model_provider in ModelProvider:
            for model in self._get_supported_embeddings_models_by_provider(model_provider).keys():
                all_models.append(f"{model_provider.value}:{model}")
        return ", ".join(sorted(all_models))

    def calculate_completion_model_cost(
        self,
        model_provider: ModelProvider,
        model_name: str,
        uncached_input_tokens: int,
        cached_input_tokens_read: int,
        output_tokens: int,
        cached_input_tokens_written: int = 0
    ) -> float:
        """Calculates the total cost for a completion model operation.

        Args:
            model_provider: The provider of the model
            model_name: The name of the model
            uncached_input_tokens: Number of uncached input tokens
            cached_input_tokens_read: Number of cached input tokens read
            output_tokens: Number of output tokens
            cached_input_tokens_written: Number of cached input tokens written

        Returns:
            Total cost in USD

        Raises:
            ValueError: If the model is not supported
        """
        model_parameters = self.get_completion_model_parameters(model_provider, model_name)
        if model_parameters is None:
            raise ValueError(self.generate_unsupported_completion_model_error_message(model_provider, model_name))
        input_token_cost = model_parameters.input_token_cost
        cached_input_tokens_read_cost = model_parameters.cached_input_token_read_cost
        output_token_cost = model_parameters.output_token_cost
        if model_parameters.has_tiered_input_token_costs:
            for tier_threshold in sorted(model_parameters.tiered_input_token_costs.keys()):
                if uncached_input_tokens >= tier_threshold:
                    tier_costs = model_parameters.tiered_input_token_costs[tier_threshold]
                    input_token_cost = tier_costs.input_token_cost
                    cached_input_tokens_read_cost = tier_costs.cached_input_token_read_cost
                    output_token_cost = tier_costs.output_token_cost
                    break
        return (
                uncached_input_tokens * input_token_cost
                + cached_input_tokens_read * cached_input_tokens_read_cost
                + cached_input_tokens_written * model_parameters.cached_input_token_write_cost
                + output_tokens * output_token_cost
        )

    def calculate_embedding_model_cost(self,
                                       model_provider: ModelProvider,
                                       model_name: str,
                                       billable_inputs: int
                                       ) -> float:
        """Calculates the total cost for an embedding model operation.

        Args:
            model_provider: The provider of the model
            model_name: The name of the model
            billable_inputs: Number of tokens or characters to embed (some Google models charge per character)

        Returns:
            Total cost in USD

        Raises:
            ValueError: If the model is not supported
        """
        model_costs = self.get_embedding_model_parameters(model_provider, model_name)
        if model_costs is None:
            raise ValueError(self.generate_unsupported_embedding_model_error_message(model_provider, model_name))
        return billable_inputs * model_costs.input_token_cost

    # Private methods
    def _create_complete_model_collection(
        self,
        base_models: Dict[str, Union[CompletionModelParameters, EmbeddingModelParameters]],
        snapshots: Dict[tuple[ModelProvider, str], list[str]],
        provider: ModelProvider
    ) -> Dict[str, Union[CompletionModelParameters, EmbeddingModelParameters]]:
        """Creates a complete model collection including snapshots.

        Args:
            base_models: Dictionary of base model parameters
            snapshots: Dictionary mapping (provider, base_model) to list of snapshot names
            provider: The provider to create the collection for

        Returns:
            Complete model collection including snapshots
        """
        models = base_models.copy()
        for (snapshot_provider, base_model), snapshot_names in snapshots.items():
            if snapshot_provider == provider:
                for snapshot in snapshot_names:
                    models[snapshot] = models[base_model]
        return models

    def _get_supported_completions_models_by_provider(self, model_provider: ModelProvider) -> CompletionModelCollection:
        """Returns the collection of completion models for a specific provider.

        Args:
            model_provider: The provider to get models for

        Returns:
            Collection of completion models for the specified provider, including snapshots
        """
        return self._complete_completion_models[model_provider]

    def _get_supported_embeddings_models_by_provider(self, model_provider: ModelProvider) -> EmbeddingModelCollection:
        """Returns the collection of embedding models for a specific provider.

        Args:
            model_provider: The provider to get models for

        Returns:
            Collection of embedding models for the specified provider, including snapshots
        """
        return self._complete_embedding_models[model_provider]

    def _get_supported_completions_models_by_provider_as_string(self, model_provider: ModelProvider) -> str:
        """Returns a comma-separated string of supported completion model names for a provider.

        Args:
            model_provider: The provider to get model names for

        Returns:
            Comma-separated string of model names
        """
        return ", ".join(sorted(self._get_supported_completions_models_by_provider(model_provider).keys()))

# Create a singleton instance
model_catalog = ModelCatalog()
