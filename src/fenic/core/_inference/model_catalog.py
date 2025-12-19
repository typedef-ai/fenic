import logging
import threading
from enum import Enum
from typing import Callable, Dict, Final, Literal, Optional, Set, TypeAlias, Union

from fenic.core.error import InternalError

logger = logging.getLogger(__name__)

# Type for supported thinking levels (Gemini 3+ models)
ThinkingLevelType = Literal["high", "medium", "low", "minimal"]
MediaResolutionType = Literal["low", "medium", "high", "ultra_high"]

# Thinking level sets for Gemini 3 models
GEMINI_3_PRO_THINKING_LEVELS: Final[Set[ThinkingLevelType]] = {"high", "low"}
GEMINI_3_FLASH_THINKING_LEVELS: Final[Set[ThinkingLevelType]] = {"high", "medium", "low", "minimal"}


class ModelProvider(Enum):
    """Enum representing different model providers supported by the system."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE_DEVELOPER = "google-developer"
    GOOGLE_VERTEX = "google-vertex"
    COHERE = "cohere"
    OPENROUTER = "openrouter"


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
        tiered_token_costs: Tiered token costs for the model.
        max_output_tokens: Maximum number of tokens the model can generate in a single request.
        max_temperature: Maximum temperature for the model.
        supports_profiles: Whether the model supports parameter profiles.
        supports_reasoning: Whether the model supports reasoning parameter.
        supports_minimal_reasoning: Whether the model supports minimal reasoning parameter. (Introduced with OpenAI gpt5 models)
        supports_disabled_reasoning: Whether the model supports disabling reasoning.
        supported_thinking_levels: Set of thinking levels supported by the model (Gemini 3+).
            None means the model uses thinking_budget instead. Valid values: "high", "medium", "low", "minimal".
        supports_custom_temperature: Whether the model supports custom temperature.
        supports_verbosity: Whether the model supports verbosity. (Introduced with OpenAI gpt5 models)
        supports_pdf_parsing: Whether fenic can use this model to parse PDFs.
        supports_media_resolution: Whether the model supports media_resolution setting for PDF processing. (Google Gemini 3+)
    """

    def __init__(
        self,
        input_token_cost: float,
        output_token_cost: float,
        context_window_length: int,
        max_output_tokens: int,
        max_temperature: float = 1,
        cached_input_token_write_cost: float = 0.0,
        cached_input_token_read_cost: float = 0.0,
        tiered_token_costs: Optional[Dict[int, TieredTokenCost]] = None,
        supports_profiles=True,
        supports_reasoning=False,
        supports_minimal_reasoning=False,
        supports_disabled_reasoning=True,
        supported_thinking_levels: Optional[Set[ThinkingLevelType]] = None,
        supports_custom_temperature=True,
        supports_verbosity = False,
        supports_pdf_parsing = False,
        supports_media_resolution = False,
        supported_parameters: Optional[set[str]] = None,
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
        self.supports_profiles = supports_profiles
        self.supports_reasoning = supports_reasoning
        self.supports_minimal_reasoning = supports_minimal_reasoning
        self.supports_disabled_reasoning = supports_disabled_reasoning
        self.supported_thinking_levels = supported_thinking_levels
        self.supports_custom_temperature = supports_custom_temperature
        self.supports_verbosity = supports_verbosity
        self.supports_pdf_parsing = supports_pdf_parsing
        self.supports_media_resolution = supports_media_resolution
        # Provider-specific supported request parameters (e.g., OpenRouter "supported_parameters")
        self.supported_parameters: set[str] = supported_parameters or set()

    def __str__(self):
        repr = (
            f"CompletionModelParameters(input_token_cost=${self.input_token_cost}, output_token_cost=${self.output_token_cost}, context_window_length={self.context_window_length}, max_output_tokens={self.max_output_tokens}, "
            f"max_temperature={self.max_temperature}, cached_input_token_write_cost=${self.cached_input_token_write_cost}, cached_input_token_read_cost=${self.cached_input_token_read_cost}, tiered_input_token_costs={self.tiered_input_token_costs}, "
            f"supports_profiles={self.supports_profiles}, supports_reasoning={self.supports_reasoning}, supports_minimal_reasoning={self.supports_minimal_reasoning}, supports_custom_temperature={self.supports_custom_temperature}, "
            f"supports_verbosity={self.supports_verbosity}, supported_parameters={self.supported_parameters})"
        )
        return repr


class EmbeddingModelParameters:
    """Parameters for embedding models including costs and output dimensions.

    Attributes:
        input_token_cost: Cost per input token in USD
        output_dimensions: Number of dimensions in the embedding output
        max_input_size: Maximum number of tokens in the input string
    """

    def __init__(
        self,
        input_token_cost: float,
        allowed_output_dimensions: Union[int, list[int]],
        max_input_size: int,
        default_dimensionality: Optional[int] = None,
    ):
        self.input_token_cost = input_token_cost
        self.output_dimensions = allowed_output_dimensions
        self.max_input_size = max_input_size

        # Convenience properties
        self.has_fixed_dimensions = isinstance(allowed_output_dimensions, int)
        if self.has_fixed_dimensions:
            self.fixed_dimensions = allowed_output_dimensions
            self.default_dimensions = default_dimensionality or self.fixed_dimensions
        else:
            self.fixed_dimensions = None
            self.dimension_options = allowed_output_dimensions
            if not default_dimensionality:
                default_dimensionality = allowed_output_dimensions[-1]
            elif default_dimensionality not in allowed_output_dimensions:
                raise InternalError(f"Cannot create EmbeddingModelParameters with default output dimensions: {default_dimensionality}."
                                    f" Allowed output dimensions: {allowed_output_dimensions}")
            self.default_dimensions = default_dimensionality

    def get_possible_dimensions(self) -> list[int]:
        """Get the possible dimensions for the model."""
        if self.has_fixed_dimensions:
            return [self.fixed_dimensions]
        else:
            return self.dimension_options

    def supports_dimensions(self, requested_dimensions: int) -> bool:
        """Validate if requested dimensions are supported."""
        if self.has_fixed_dimensions:
            return requested_dimensions == self.fixed_dimensions
        else:
            return requested_dimensions in self.dimension_options


CompletionModelCollection: TypeAlias = Dict[str, CompletionModelParameters]
EmbeddingModelCollection: TypeAlias = Dict[str, EmbeddingModelParameters]
OpenAILanguageModelName = Literal[
    "gpt-5.1",
    "gpt-5.1-2025-11-13",
    "gpt-5",
    "gpt-5-2025-08-07",
    "gpt-5-mini",
    "gpt-5-mini-2025-08-07",
    "gpt-5-nano",
    "gpt-5-nano-2025-08-07",
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
    "o1",
    "o3",
    "o3-mini",
    "o4-mini",
]

OpenAIEmbeddingModelName = Literal[
    "text-embedding-3-small", "text-embedding-3-large"
]

GoogleVertexEmbeddingModelName = Literal[
    "gemini-embedding-001",
    "gemini-embedding-exp-03-07",
    "text-multilingual-embedding-002",
    "text-embedding-005",
]

GoogleDeveloperEmbeddingModelName = Literal[
    "gemini-embedding-001",
    "gemini-embedding-exp-03-07",
    "text-embedding-004"
]

AnthropicLanguageModelName = Literal[
    "claude-opus-4-5",
    "claude-opus-4-5-20251101",
    "claude-sonnet-4-5",
    "claude-sonnet-4-5-20250929",
    "claude-haiku-4-5",
    "claude-haiku-4-5-20251001",
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
    "claude-opus-4-1",
    "claude-opus-4-1-20250805",
    "claude-opus-4-0",
    "claude-opus-4-20250514",
    "claude-4-opus-20250514",
    "claude-3-opus-latest",
    "claude-3-opus-20240229",
    "claude-3-haiku-20240307",
]

CohereEmbeddingModelName = Literal[
    "embed-v4.0",
    "embed-english-v3.0",
    "embed-english-light-v3.0",
    "embed-multilingual-v3.0",
    "embed-multilingual-light-v3.0",
]


GoogleDeveloperLanguageModelName = Literal[
    "gemini-3-pro-preview",
    "gemini-3-flash-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-lite-001",
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
    "gemini-2.0-flash-exp",
]
GoogleVertexLanguageModelName = GoogleDeveloperLanguageModelName


class ProviderModelCollection:
    """A collection of models for a specific provider.

    This class maintains a collection of models for a specific provider,
    including their costs, context windows, and other parameters. It provides methods
    to query model information and calculate costs for different operations.
    """


    def __init__(self, provider: ModelProvider) -> None:
        self.provider = provider
        self.completion_models: CompletionModelCollection = {}
        self.embedding_models: EmbeddingModelCollection = {}


    def add_model(
        self,
        name: str,
        parameters: Union[EmbeddingModelParameters, CompletionModelParameters],
        snapshots: Optional[list[str]] = None,
    ):
        if snapshots is None:
            snapshots = []

        if isinstance(parameters, EmbeddingModelParameters):
            if name in self.embedding_models:
                raise InternalError(
                    f"Attempted to add configuration for embedding model {name} to collection, but it already exists"
                )
            self.embedding_models[name] = parameters
            for snapshot in snapshots:
                if snapshot in self.embedding_models:
                    raise InternalError(
                        f"Attempted to add configuration for embedding model snapshot {snapshot} to collection, but it already exists"
                    )
                self.embedding_models[snapshot] = parameters
        else:
            if name in self.completion_models:
                raise InternalError(
                    f"Attempted to add configuration for completion model {name} to collection, but it already exists"
                )
            self.completion_models[name] = parameters
            for snapshot in snapshots:
                if snapshot in self.completion_models:
                    raise InternalError(
                        f"Attempted to add configuration for completion model snapshot {snapshot} to collection, but it already exists"
                    )
                self.completion_models[snapshot] = parameters


class ModelCatalog:
    """Catalog of supported models and their parameters for different providers.

    This class maintains a registry of all supported models across different providers,
    including their costs, context windows, and other parameters. It provides methods
    to query model information and calculate costs for different operations.
    """

    def __init__(self):
        self.provider_model_collections: dict[
            ModelProvider, ProviderModelCollection
        ] = {}
        # Ensure all providers have an initialized collection, even if empty (e.g., OpenRouter)
        for provider in ModelProvider:
            self.provider_model_collections[provider] = ProviderModelCollection(
                provider
            )
        # Dynamic provider loaders
        self._dynamic_loaders: dict[ModelProvider, Callable[[], None]] = {}
        self._dynamic_loaded: set[ModelProvider] = set()
        self._loader_mutex = threading.Lock()
        self._initialize_models()

    def _initialize_models(self):
        """Initialize all models in the catalog using the new API."""
        self._initialize_anthropic_models()
        self._initialize_openai_models()
        self._initialize_google_gla_models()
        self._initialize_google_vertex_models()
        self._initialize_cohere_models()

    def _initialize_anthropic_models(self):
        """Initialize Anthropic models in the catalog."""
        # Claude 4.5 models
        self._add_model_to_catalog(
            ModelProvider.ANTHROPIC,
            "claude-opus-4-5",
            CompletionModelParameters(
                input_token_cost=5.00 / 1_000_000,  # $5 per 1M tokens
                cached_input_token_write_cost=6.25 / 1_000_000,  # $6.25 per 1M tokens
                cached_input_token_read_cost=0.50 / 1_000_000,  # $0.50 per 1M tokens
                output_token_cost=25.00 / 1_000_000,  # $25 per 1M tokens
                context_window_length=200_000,
                max_output_tokens=64_000,
                supports_reasoning=True,
            ),
            snapshots=["claude-opus-4-5-20251101"],
        )

        self._add_model_to_catalog(
            ModelProvider.ANTHROPIC,
            "claude-haiku-4-5",
            CompletionModelParameters(
                input_token_cost=1.00 / 1_000_000,  # $1 per 1M tokens
                cached_input_token_write_cost=1.25 / 1_000_000,  # $1.25 per 1M tokens
                cached_input_token_read_cost=0.10 / 1_000_000,  # $0.10 per 1M tokens
                output_token_cost=5.00 / 1_000_000,  # $5 per 1M tokens
                context_window_length=200_000,
                max_output_tokens=64_000,
                supports_reasoning=True,
            ),
            snapshots=["claude-haiku-4-5-20251001"],
        )

        # Claude 4.1 models
        self._add_model_to_catalog(
            ModelProvider.ANTHROPIC,
            "claude-opus-4-1",
            CompletionModelParameters(
                input_token_cost=15.00 / 1_000_000,  # $15 per 1M tokens
                cached_input_token_write_cost=18.75 / 1_000_000,  # $18.75 per 1M tokens
                cached_input_token_read_cost=1.50 / 1_000_000,  # $1.50 per 1M tokens
                output_token_cost=75.00 / 1_000_000,  # $75 per 1M tokens
                context_window_length=200_000,
                max_output_tokens=32_000,
                supports_reasoning=True,
            ),
            snapshots=["claude-opus-4-1-20250805"],
        )

        self._add_model_to_catalog(
            ModelProvider.ANTHROPIC,
            "claude-opus-4-0",
            CompletionModelParameters(
                input_token_cost=15.00 / 1_000_000,  # $15 per 1M tokens
                cached_input_token_write_cost=18.75 / 1_000_000,  # $18.75 per 1M tokens
                cached_input_token_read_cost=1.50 / 1_000_000,  # $1.50 per 1M tokens
                output_token_cost=75.00 / 1_000_000,  # $75 per 1M tokens
                context_window_length=200_000,
                max_output_tokens=32_000,
                supports_reasoning=True,
            ),
            snapshots=["claude-opus-4-20250514", "claude-4-opus-20250514"],
        )

        self._add_model_to_catalog(
            ModelProvider.ANTHROPIC,
            "claude-sonnet-4-5",
            CompletionModelParameters(
                input_token_cost=3.00 / 1_000_000,  # $3 per 1M tokens
                cached_input_token_write_cost=3.75 / 1_000_000,  # $3.75 per 1M tokens
                cached_input_token_read_cost=0.30 / 1_000_000,  # $0.30 per 1M tokens
                output_token_cost=15.00 / 1_000_000,  # $15 per 1M tokens
                context_window_length=200_000,
                max_output_tokens=64_000,
                supports_reasoning=True,
            ),
            snapshots=["claude-sonnet-4-5-20250929"],
        )

        self._add_model_to_catalog(
            ModelProvider.ANTHROPIC,
            "claude-sonnet-4-0",
            CompletionModelParameters(
                input_token_cost=3.00 / 1_000_000,  # $3 per 1M tokens
                cached_input_token_write_cost=3.75 / 1_000_000,  # $3.75 per 1M tokens
                cached_input_token_read_cost=0.30 / 1_000_000,  # $0.30 per 1M tokens
                output_token_cost=15.00 / 1_000_000,  # $15 per 1M tokens
                context_window_length=200_000,
                max_output_tokens=64_000,
                supports_reasoning=True,
            ),
            snapshots=["claude-sonnet-4-20250514", "claude-4-sonnet-20250514"],
        )

        self._add_model_to_catalog(
            ModelProvider.ANTHROPIC,
            "claude-3-7-sonnet-latest",
            CompletionModelParameters(
                input_token_cost=3.0 / 1_000_000,  # $3 per 1M tokens
                cached_input_token_write_cost=3.75 / 1_000_000,  # $3.75 per 1M tokens
                cached_input_token_read_cost=0.30 / 1_000_000,  # $0.30 per 1M tokens
                output_token_cost=15.00 / 1_000_000,  # $15 per 1M tokens
                context_window_length=200_000,
                max_output_tokens=128_000,
                supports_reasoning=True,
            ),
            snapshots=["claude-3-7-sonnet-20250219"],
        )

        self._add_model_to_catalog(
            ModelProvider.ANTHROPIC,
            "claude-3-5-sonnet-latest",
            CompletionModelParameters(
                input_token_cost=3 / 1_000_000,  # $3 per 1M tokens
                cached_input_token_write_cost=3.75 / 1_000_000,  # $3.75 per 1M tokens
                cached_input_token_read_cost=0.30 / 1_000_000,  # $0.30 per 1M tokens
                output_token_cost=15.00 / 1_000_000,  # $15 per 1M tokens
                context_window_length=200_000,
                max_output_tokens=8_192,
                supports_profiles=False,
            ),
            snapshots=["claude-3-5-sonnet-20241022", "claude-3-5-sonnet-20240620"],
        )

        self._add_model_to_catalog(
            ModelProvider.ANTHROPIC,
            "claude-3-5-haiku-latest",
            CompletionModelParameters(
                input_token_cost=0.80 / 1_000_000,  # $0.80 per 1M tokens
                cached_input_token_write_cost=1.00 / 1_000_000,  # $1.00 per 1M tokens
                cached_input_token_read_cost=0.08 / 1_000_000,  # $0.08 per 1M tokens
                output_token_cost=4.00 / 1_000_000,  # $4 per 1M tokens
                context_window_length=200_000,
                max_output_tokens=8_000,
                supports_profiles=False,
            ),
            snapshots=["claude-3-5-haiku-20241022"],
        )

        self._add_model_to_catalog(
            ModelProvider.ANTHROPIC,
            "claude-3-opus-latest",
            CompletionModelParameters(
                input_token_cost=15.00 / 1_000_000,  # $15 per 1M tokens
                cached_input_token_write_cost=18.75 / 1_000_000,  # $18.75 per 1M tokens
                cached_input_token_read_cost=1.50 / 1_000_000,  # $1.50 per 1M tokens
                output_token_cost=75.00 / 1_000_000,  # $75 per 1M tokens
                context_window_length=200_000,
                max_output_tokens=4_096,
                supports_profiles=False,
            ),
            snapshots=["claude-3-opus-20240229"],
        )

        self._add_model_to_catalog(
            ModelProvider.ANTHROPIC,
            "claude-3-haiku-20240307",
            CompletionModelParameters(
                input_token_cost=0.25 / 1_000_000,  # $0.25 per 1M tokens
                cached_input_token_write_cost=0.30 / 1_000_000,  # $0.30 per 1M tokens
                cached_input_token_read_cost=0.03 / 1_000_000,  # $0.03 per 1M tokens
                output_token_cost=1.25 / 1_000_000,  # $1.25 per 1M tokens
                context_window_length=200_000,
                max_output_tokens=4_096,
                supports_profiles=False,
            ),
        )

    def _initialize_openai_models(self):
        """Initialize OpenAI models in the catalog."""
        self._add_model_to_catalog(
            ModelProvider.OPENAI,
            "gpt-4",
            CompletionModelParameters(
                input_token_cost=30 / 1_000_000,  # $30 per 1M tokens
                cached_input_token_read_cost=0.0,  # N/A
                output_token_cost=60 / 1_000_000,  # $60 per 1M tokens
                context_window_length=8_192,
                max_output_tokens=8_192,
                max_temperature=2,
                supports_profiles=False,
            ),
            snapshots=["gpt-4-0314", "gpt-4-0613"],
        )

        self._add_model_to_catalog(
            ModelProvider.OPENAI,
            "gpt-4-turbo",
            CompletionModelParameters(
                input_token_cost=10 / 1_000_000,  # $10 per 1M tokens
                cached_input_token_read_cost=0.0,  # N/A
                output_token_cost=30 / 1_000_000,  # $30 per 1M tokens
                context_window_length=128_000,
                max_output_tokens=4_096,
                max_temperature=2,
                supports_profiles=False,
            ),
            snapshots=["gpt-4-turbo-2024-04-09"],
        )

        self._add_model_to_catalog(
            ModelProvider.OPENAI,
            "gpt-4o-mini",
            CompletionModelParameters(
                input_token_cost=0.15 / 1_000_000,  # $0.15 per 1M tokens
                cached_input_token_read_cost=0.075 / 1_000_000,  # $0.150 per 1M tokens
                output_token_cost=0.60 / 1_000_000,  # $1.200 per 1M tokens
                context_window_length=128_000,
                max_output_tokens=16_384,
                max_temperature=2,
                supports_profiles=False,
                supports_pdf_parsing=True,
            ),
            snapshots=["gpt-4o-mini-2024-07-18"],
        )

        self._add_model_to_catalog(
            ModelProvider.OPENAI,
            "gpt-4o",
            CompletionModelParameters(
                input_token_cost=2.50 / 1_000_000,  # $2.50 per 1M tokens
                cached_input_token_read_cost=1.25 / 1_000_000,  # $1.25 per 1M tokens
                output_token_cost=10.00 / 1_000_000,  # $10.00 per 1M tokens
                context_window_length=128_000,
                max_output_tokens=16_384,
                max_temperature=2,
                supports_profiles=False,
                supports_pdf_parsing=True,
            ),
            snapshots=["gpt-4o-2024-05-13", "gpt-4o-2024-08-06", "gpt-4o-2024-11-20"],
        )

        self._add_model_to_catalog(
            ModelProvider.OPENAI,
            "gpt-4.1-nano",
            CompletionModelParameters(
                input_token_cost=0.100 / 1_000_000,  # $0.100 per 1M tokens
                cached_input_token_read_cost=0.025 / 1_000_000,  # $0.025 per 1M tokens
                output_token_cost=0.400 / 1_000_000,  # $0.400 per 1M tokens
                context_window_length=1_000_000,
                max_output_tokens=32_768,
                max_temperature=2,
                supports_profiles=False,
            ),
            snapshots=["gpt-4.1-nano-2025-04-14"],
        )

        self._add_model_to_catalog(
            ModelProvider.OPENAI,
            "gpt-4.1-mini",
            CompletionModelParameters(
                input_token_cost=0.400 / 1_000_000,  # $0.400 per 1M tokens
                cached_input_token_read_cost=0.100 / 1_000_000,  # $0.100 per 1M tokens
                output_token_cost=1.600 / 1_000_000,  # $1.600 per 1M tokens
                context_window_length=1_000_000,
                max_output_tokens=32_768,
                max_temperature=2,
                supports_profiles=False,
            ),
            snapshots=["gpt-4.1-mini-2025-04-14"],
        )

        self._add_model_to_catalog(
            ModelProvider.OPENAI,
            "gpt-4.1",
            CompletionModelParameters(
                input_token_cost=2.00 / 1_000_000,  # $2.00 per 1M tokens
                cached_input_token_read_cost=0.500 / 1_000_000,  # $0.500 per 1M tokens
                output_token_cost=8.00 / 1_000_000,  # $8.00 per 1M tokens
                context_window_length=1_000_000,
                max_output_tokens=32_768,
                max_temperature=2,
                supports_profiles=False,
            ),
            snapshots=["gpt-4.1-2025-04-14"],
        )

        self._add_model_to_catalog(
            ModelProvider.OPENAI,
            "o1",
            CompletionModelParameters(
                input_token_cost=15 / 1_000_000,  # $15 per 1M tokens
                cached_input_token_read_cost=7.50 / 1_000_000,  # $7.50 per 1M tokens
                output_token_cost=60 / 1_000_000,  # $60 per 1M tokens
                context_window_length=200_000,
                max_output_tokens=100_000,
                max_temperature=2.0,
                supports_reasoning=True,
                supports_custom_temperature=False,
            ),
        )

        self._add_model_to_catalog(
            ModelProvider.OPENAI,
            "o3",
            CompletionModelParameters(
                input_token_cost=2 / 1_000_000,  # $2 per 1M tokens
                cached_input_token_read_cost=0.50 / 1_000_000,  # $0.50 per 1M tokens
                output_token_cost=8 / 1_000_000,  # $8 per 1M tokens
                context_window_length=200_000,
                max_output_tokens=100_000,
                max_temperature=2.0,
                supports_reasoning=True,
                supports_custom_temperature=False,
                supports_pdf_parsing=True,
            ),
        )

        self._add_model_to_catalog(
            ModelProvider.OPENAI,
            "o3-mini",
            CompletionModelParameters(
                input_token_cost=1.10 / 1_000_000,  # $1.10 per 1M tokens
                cached_input_token_read_cost=0.55 / 1_000_000,  # $0.55 per 1M tokens
                output_token_cost=4.40 / 1_000_000,  # $4.40 per 1M tokens
                context_window_length=200_000,
                max_output_tokens=100_000,
                max_temperature=2.0,
                supports_reasoning=True,
                supports_custom_temperature=False,
            ),
        )

        self._add_model_to_catalog(
            ModelProvider.OPENAI,
            "o4-mini",
            CompletionModelParameters(
                input_token_cost=1.10 / 1_000_000,  # $1.10 per 1M tokens
                cached_input_token_read_cost=0.275 / 1_000_000,  # $0.275 per 1M tokens
                output_token_cost=4.40 / 1_000_000,  # $4.40 per 1M tokens
                context_window_length=200_000,
                max_output_tokens=100_000,
                max_temperature=2.0,
                supports_reasoning=True,
                supports_pdf_parsing=True,
            ),
        )

        self._add_model_to_catalog(
            ModelProvider.OPENAI,
            "gpt-5",
            CompletionModelParameters(
                input_token_cost=1.25 / 1_000_000,  # $1.25 per 1M tokens
                cached_input_token_read_cost=0.125
                / 1_000_000,  # $0.125 per 1M tokens (90% discount)
                output_token_cost=10.00 / 1_000_000,  # $10.00 per 1M tokens
                context_window_length=400_000,
                max_output_tokens=128_000,
                supports_reasoning=True,
                supports_minimal_reasoning=True,
                supports_disabled_reasoning=False,
                supports_custom_temperature=False,
                supports_verbosity=True,
                supports_pdf_parsing=True,
            ),
            snapshots=["gpt-5-2025-08-07"],
        )

        self._add_model_to_catalog(
            ModelProvider.OPENAI,
            "gpt-5-mini",
            CompletionModelParameters(
                input_token_cost=0.25 / 1_000_000,  # $0.25 per 1M tokens
                cached_input_token_read_cost=0.025 / 1_000_000,  # $0.025 per 1M tokens
                output_token_cost=2.00 / 1_000_000,  # $2.00 per 1M tokens
                context_window_length=400_000,
                max_output_tokens=128_000,
                supports_reasoning=True,
                supports_minimal_reasoning=True,
                supports_disabled_reasoning=False,
                supports_custom_temperature=False,
                supports_verbosity=True,
                supports_pdf_parsing=True,
            ),
            snapshots=["gpt-5-mini-2025-08-07"],
        )

        self._add_model_to_catalog(
            ModelProvider.OPENAI,
            "gpt-5-nano",
            CompletionModelParameters(
                input_token_cost=0.05 / 1_000_000,  # $0.05 per 1M tokens
                cached_input_token_read_cost=0.005 / 1_000_000,  # $0.005 per 1M tokens
                output_token_cost=0.40 / 1_000_000,  # $0.40 per 1M tokens
                context_window_length=400_000,
                max_output_tokens=128_000,
                supports_reasoning=True,
                supports_minimal_reasoning=True,
                supports_disabled_reasoning=False,
                supports_verbosity=True,
                supports_custom_temperature=False,
                supports_pdf_parsing=True,
            ),
            snapshots=["gpt-5-nano-2025-08-07"],
        )

        # GPT-5.1 models - support 'none' reasoning effort (disabled reasoning), default to 'none'
        self._add_model_to_catalog(
            ModelProvider.OPENAI,
            "gpt-5.1",
            CompletionModelParameters(
                input_token_cost=1.25 / 1_000_000,  # $1.25 per 1M tokens
                cached_input_token_read_cost=0.125 / 1_000_000,  # $0.125 per 1M tokens
                output_token_cost=10.00 / 1_000_000,  # $10.00 per 1M tokens
                context_window_length=400_000,
                max_output_tokens=128_000,
                supports_reasoning=True,
                supports_minimal_reasoning=False,
                supports_disabled_reasoning=True,
                supports_custom_temperature=True,  # Supports temperature when reasoning_effort='none'
                supports_verbosity=True,
                supports_pdf_parsing=True,
            ),
            snapshots=["gpt-5.1-2025-11-13"],
        )

        # OpenAI Embedding Models
        self._add_model_to_catalog(
            ModelProvider.OPENAI,
            "text-embedding-3-small",
            EmbeddingModelParameters(
                input_token_cost=0.02 / 1_000_000,  # $0.02 per 1M tokens
                allowed_output_dimensions=1536,
                max_input_size=8192,
            ),
        )

        self._add_model_to_catalog(
            ModelProvider.OPENAI,
            "text-embedding-3-large",
            EmbeddingModelParameters(
                input_token_cost=0.13 / 1_000_000,  # $0.13 per 1M tokens
                allowed_output_dimensions=3072,
                max_input_size=8192,
            ),
        )

    def _initialize_google_vertex_models(self):
        """Initialize the Google Vertex Models."""
        self._add_model_to_catalog(
            ModelProvider.GOOGLE_VERTEX,
            "gemini-3-pro-preview",
            CompletionModelParameters(
                input_token_cost=2.00 / 1_000_000,  # $2.00 per 1M tokens
                cached_input_token_read_cost=0.20 / 1_000_000,  # $0.20 per 1M tokens
                output_token_cost=12.00 / 1_000_000,  # $12.00 per 1M tokens
                context_window_length=1_048_576,
                max_output_tokens=65_535,
                max_temperature=2.0,
                supports_reasoning=True,
                supports_disabled_reasoning=False,
                supported_thinking_levels=GEMINI_3_PRO_THINKING_LEVELS,
                supports_pdf_parsing=True,
                supports_media_resolution=True,
                tiered_token_costs={
                    200_000: TieredTokenCost(
                        input_token_cost=4.00 / 1_000_000,  # $4.00 per 1M tokens
                        cached_input_token_read_cost=0.40 / 1_000_000,  # $0.40 per 1M tokens
                        output_token_cost=18.00 / 1_000_000,  # $18.00 per 1M tokens
                    )
                },
            ),
        )

        self._add_model_to_catalog(
            ModelProvider.GOOGLE_VERTEX,
            "gemini-3-flash-preview",
            CompletionModelParameters(
                input_token_cost=0.50 / 1_000_000,  # $0.50 per 1M tokens
                cached_input_token_read_cost=0.05 / 1_000_000,  # $0.05 per 1M tokens
                output_token_cost=3.00 / 1_000_000,  # $3.00 per 1M tokens
                context_window_length=1_048_576,
                max_output_tokens=65_536,
                max_temperature=2.0,
                supports_reasoning=True,
                supports_disabled_reasoning=False,
                supported_thinking_levels=GEMINI_3_FLASH_THINKING_LEVELS,
                supports_pdf_parsing=True,
                supports_media_resolution=True,
            ),
        )

        self._add_model_to_catalog(
            ModelProvider.GOOGLE_VERTEX,
            "gemini-2.5-pro",
            CompletionModelParameters(
                input_token_cost=1.25 / 1_000_000,  # $1.25 per 1M tokens
                cached_input_token_read_cost=0.03125
                / 1_000_000,  # $0.03125 per 1M tokens
                output_token_cost=10 / 1_000_000,  # $10 per 1M tokens
                context_window_length=1_048_576,
                max_output_tokens=65_536,
                supports_reasoning=True,
                supports_disabled_reasoning=False,
                max_temperature=2.0,
                tiered_token_costs={
                    200_000: TieredTokenCost(
                        input_token_cost=2.50 / 1_000_000,  # $2.50 per 1M tokens
                        cached_input_token_read_cost=0.0625
                        / 1_000_000,  # $0.0625 per 1M tokens
                        output_token_cost=15 / 1_000_000,  # $15.00 per 1M tokens
                    )
                },
                supports_pdf_parsing=True,
            ),
            snapshots=["gemini-2.5-pro-preview-06-05"],
        )

        self._add_model_to_catalog(
            ModelProvider.GOOGLE_VERTEX,
            "gemini-2.5-flash",
            CompletionModelParameters(
                input_token_cost=0.30 / 1_000_000,  # $0.30 per 1M tokens
                cached_input_token_read_cost=0.075 / 1_000_000,  # $0.075 per 1M tokens
                output_token_cost=2.50 / 1_000_000,  # $2.50 per 1M tokens
                context_window_length=1_048_576,
                max_output_tokens=65_536,
                max_temperature=2.0,
                supports_reasoning=True,
                supports_pdf_parsing=True,
            ),
        )

        self._add_model_to_catalog(
            ModelProvider.GOOGLE_VERTEX,
            "gemini-2.5-flash-lite",
            CompletionModelParameters(
                input_token_cost=0.10 / 1_000_000,  # $0.10 per 1M tokens
                output_token_cost=0.40 / 1_000_000,  # $0.40 per 1M tokens
                context_window_length=1_000_000,
                max_output_tokens=64_000,
                max_temperature=2.0,
                supports_reasoning=True,
                supports_pdf_parsing=True,
            ),
        )

        self._add_model_to_catalog(
            ModelProvider.GOOGLE_VERTEX,
            "gemini-2.0-flash-lite",
            CompletionModelParameters(
                input_token_cost=0.075 / 1_000_000,  # $0.075 per 1M tokens
                output_token_cost=0.30 / 1_000_000,  # $0.30 per 1M tokens
                context_window_length=1_048_576,
                max_output_tokens=8_192,
                max_temperature=2.0,
                supports_profiles=False,
                supports_pdf_parsing=True,
            ),
            snapshots=["gemini-2.0-flash-lite-001"],
        )

        self._add_model_to_catalog(
            ModelProvider.GOOGLE_VERTEX,
            "gemini-2.0-flash",
            CompletionModelParameters(
                input_token_cost=0.15 / 1_000_000,  # $0.15 per 1M tokens
                cached_input_token_read_cost=0.0375
                / 1_000_000,  # $0.0375 per 1M tokens
                output_token_cost=0.60 / 1_000_000,  # $0.60 per 1M tokens
                context_window_length=1_048_576,
                max_output_tokens=8_192,
                max_temperature=2.0,
                supports_profiles=False,
                supports_pdf_parsing=True,
            ),
            snapshots=["gemini-2.0-flash-001", "gemini-2.0-flash-exp"],
        )
        self._add_model_to_catalog(
            ModelProvider.GOOGLE_VERTEX,
            "gemini-embedding-001",
            EmbeddingModelParameters(
                input_token_cost=0.00015 / 1_000,  # $0.00015 per 1k tokens
                allowed_output_dimensions=[768, 1536, 3072],
                max_input_size=2048,
                default_dimensionality=3072,
            ),
            snapshots=["gemini-embedding-exp-03-07"],
        )

        self._add_model_to_catalog(
            ModelProvider.GOOGLE_VERTEX,
            "text-embedding-005",
            EmbeddingModelParameters(
                input_token_cost=0.000025
                / 250,  # $0.000025/ 1k CHARACTERS (~ 250 tokens)
                allowed_output_dimensions=[256, 512, 768],
                max_input_size=2048,
                default_dimensionality=768,
            ),
        )

        self._add_model_to_catalog(
            ModelProvider.GOOGLE_VERTEX,
            "text-multilingual-embedding-002",
            EmbeddingModelParameters(
                input_token_cost=0.000025
                / 250,  # $0.000025/ 1k CHARACTERS (~ 250 tokens)
                allowed_output_dimensions=[256, 512, 768],
                max_input_size=2048,
            ),
        )

    def _initialize_google_gla_models(self):
        """Initialize Google models in the catalog."""
        # Google GLA Models (same models, possibly different pricing)
        self._add_model_to_catalog(
            ModelProvider.GOOGLE_DEVELOPER,
            "gemini-3-pro-preview",
            CompletionModelParameters(
                input_token_cost=2.00 / 1_000_000,  # $2.00 per 1M tokens
                cached_input_token_read_cost=0.20 / 1_000_000,  # $0.20 per 1M tokens
                output_token_cost=12.00 / 1_000_000,  # $12.00 per 1M tokens
                context_window_length=1_048_576,
                max_output_tokens=65_535,
                max_temperature=2.0,
                supports_reasoning=True,
                supports_disabled_reasoning=False,
                supported_thinking_levels=GEMINI_3_PRO_THINKING_LEVELS,
                supports_pdf_parsing=True,
                supports_media_resolution=True,
                tiered_token_costs={
                    200_000: TieredTokenCost(
                        input_token_cost=4.00 / 1_000_000,  # $4.00 per 1M tokens
                        cached_input_token_read_cost=0.40 / 1_000_000,  # $0.40 per 1M tokens
                        output_token_cost=18.00 / 1_000_000,  # $18.00 per 1M tokens
                    )
                },
            ),
        )

        self._add_model_to_catalog(
            ModelProvider.GOOGLE_DEVELOPER,
            "gemini-3-flash-preview",
            CompletionModelParameters(
                input_token_cost=0.50 / 1_000_000,  # $0.50 per 1M tokens
                cached_input_token_read_cost=0.05 / 1_000_000,  # $0.05 per 1M tokens
                output_token_cost=3.00 / 1_000_000,  # $3.00 per 1M tokens
                context_window_length=1_048_576,
                max_output_tokens=65_536,
                max_temperature=2.0,
                supports_reasoning=True,
                supports_disabled_reasoning=False,
                supported_thinking_levels=GEMINI_3_FLASH_THINKING_LEVELS,
                supports_pdf_parsing=True,
                supports_media_resolution=True,
            ),
        )

        self._add_model_to_catalog(
            ModelProvider.GOOGLE_DEVELOPER,
            "gemini-2.5-pro",
            CompletionModelParameters(
                input_token_cost=1.25 / 1_000_000,  # $1.25 per 1M tokens
                cached_input_token_read_cost=0.03125
                / 1_000_000,  # $0.03125 per 1M tokens
                output_token_cost=10 / 1_000_000,  # $10 per 1M tokens
                context_window_length=1_048_576,
                max_output_tokens=65_536,
                max_temperature=2.0,
                supports_reasoning=True,
                supports_disabled_reasoning=False,
                tiered_token_costs={
                    200_000: TieredTokenCost(
                        input_token_cost=2.50 / 1_000_000,  # $2.50 per 1M tokens
                        cached_input_token_read_cost=0.0625
                        / 1_000_000,  # $0.0625 per 1M tokens
                        output_token_cost=15 / 1_000_000,  # $15.00 per 1M tokens
                    )
                },
                supports_pdf_parsing=True,
            ),
            snapshots=["gemini-2.5-pro-preview-06-05"],
        )

        self._add_model_to_catalog(
            ModelProvider.GOOGLE_DEVELOPER,
            "gemini-2.5-flash",
            CompletionModelParameters(
                input_token_cost=0.15 / 1_000_000,  # $0.15 per 1M tokens
                cached_input_token_read_cost=0.075 / 1_000_000,  # $0.075 per 1M tokens
                output_token_cost=2.50 / 1_000_000,  # $2.50 per 1M tokens
                context_window_length=1_048_576,
                max_output_tokens=65_536,
                max_temperature=2.0,
                supports_reasoning=True,
                supports_pdf_parsing=True,
            ),
        )

        self._add_model_to_catalog(
            ModelProvider.GOOGLE_DEVELOPER,
            "gemini-2.5-flash-lite",
            CompletionModelParameters(
                input_token_cost=0.10 / 1_000_000,  # $0.10 per 1M tokens
                output_token_cost=0.40 / 1_000_000,  # $0.40 per 1M tokens
                context_window_length=1_000_000,
                max_output_tokens=64_000,
                max_temperature=2.0,
                supports_reasoning=True,
                supports_pdf_parsing=True,
            ),
        )

        self._add_model_to_catalog(
            ModelProvider.GOOGLE_DEVELOPER,
            "gemini-2.0-flash-lite",
            CompletionModelParameters(
                input_token_cost=0.075 / 1_000_000,  # $0.075 per 1M tokens
                output_token_cost=0.30 / 1_000_000,  # $0.30 per 1M tokens
                context_window_length=1_048_576,
                max_output_tokens=8_192,
                max_temperature=2.0,
                supports_pdf_parsing=True,
            ),
            snapshots=["gemini-2.0-flash-lite-001"],
        )

        self._add_model_to_catalog(
            ModelProvider.GOOGLE_DEVELOPER,
            "gemini-2.0-flash",
            CompletionModelParameters(
                input_token_cost=0.10 / 1_000_000,  # $0.10 per 1M tokens
                cached_input_token_read_cost=0.0375
                / 1_000_000,  # $0.0375 per 1M tokens
                output_token_cost=0.40 / 1_000_000,  # $0.40 per 1M tokens
                context_window_length=1_048_576,
                max_output_tokens=8_192,
                max_temperature=2.0,
                supports_pdf_parsing=True,
            ),
            snapshots=["gemini-2.0-flash-001", "gemini-2.0-flash-exp"],
        )

        self._add_model_to_catalog(
            ModelProvider.GOOGLE_DEVELOPER,
            "gemini-embedding-001",
            EmbeddingModelParameters(
                input_token_cost=0.15 / 1_000_000,  # $0.15 per 1M tokens
                allowed_output_dimensions=[768, 1536, 3072],
                max_input_size=2048,
                default_dimensionality=3072,
            ),
            snapshots=["gemini-embedding-exp-03-07"],
        )

        self._add_model_to_catalog(
            ModelProvider.GOOGLE_DEVELOPER,
            "text-embedding-004",
            EmbeddingModelParameters(
                input_token_cost=0.15 / 1_000_000,  # $0.15 per 1M tokens
                allowed_output_dimensions=768,
                max_input_size=2048,
            ),
        )

    def _initialize_cohere_models(self):
        """Initialize Cohere models in the catalog."""
        # embed-v4.0 - Latest multimodal model with variable dimensions
        self._add_model_to_catalog(
            ModelProvider.COHERE,
            "embed-v4.0",
            EmbeddingModelParameters(
                input_token_cost=0.12 / 1_000_000,  # $0.12 per 1M tokens
                allowed_output_dimensions=[256, 512, 1024, 1536],
                max_input_size=128_000,  # 128k context length
                default_dimensionality=1536,
            ),
        )

        # embed-english-v3.0 - English-only model
        self._add_model_to_catalog(
            ModelProvider.COHERE,
            "embed-english-v3.0",
            EmbeddingModelParameters(
                input_token_cost=0.10 / 1_000_000,  # $0.10 per 1M tokens
                allowed_output_dimensions=1024,  # Fixed dimensions
                max_input_size=512,
            ),
        )

        # embed-english-light-v3.0 - Smaller, faster English-only model
        self._add_model_to_catalog(
            ModelProvider.COHERE,
            "embed-english-light-v3.0",
            EmbeddingModelParameters(
                input_token_cost=0.10 / 1_000_000,  # $0.10 per 1M tokens
                allowed_output_dimensions=384,  # Fixed dimensions
                max_input_size=512,
            ),
        )

        # embed-multilingual-v3.0 - Multi-language support
        self._add_model_to_catalog(
            ModelProvider.COHERE,
            "embed-multilingual-v3.0",
            EmbeddingModelParameters(
                input_token_cost=0.10 / 1_000_000,  # $0.10 per 1M tokens
                allowed_output_dimensions=1024,  # Fixed dimensions
                max_input_size=512,
            ),
        )

        # embed-multilingual-light-v3.0 - Smaller, faster multilingual model
        self._add_model_to_catalog(
            ModelProvider.COHERE,
            "embed-multilingual-light-v3.0",
            EmbeddingModelParameters(
                input_token_cost=0.10 / 1_000_000,  # $0.10 per 1M tokens
                allowed_output_dimensions=384,  # Fixed dimensions
                max_input_size=512,
            ),
        )

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
        return (f"Model '{model_name}' is not supported for {model_provider.value}. "
                f"Supported Models: {self._get_supported_completions_models_by_provider_as_string(model_provider)}")

    def generate_unsupported_embedding_model_error_message(self, model_provider: ModelProvider, model_name: str) -> str:
        """Generates an error message for unsupported embedding models.

        Args:
            model_provider: The provider of the unsupported model
            model_name: The name of the unsupported model

        Returns:
            Error message string
        """
        return (f"Model '{model_name}' is not supported for {model_provider.value}."
                f" Supported Models: {self._get_supported_embeddings_models_by_provider_as_string(model_provider)}")

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

    def get_models_for_criteria(self, criteria: Callable[[CompletionModelParameters], bool]) -> str:
        """Returns a comma-separated string of all models that match the given criteria.

        Args:
            criteria: A function that takes CompletionModelParameters and returns True if the model matches

        Returns:
            Comma-separated string of model names
        """
        all_models = []
        for model_provider in ModelProvider:
            for model, model_params in self._get_supported_completions_models_by_provider(model_provider).items():
                if criteria(model_params):
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
        cached_input_tokens_written: int = 0,
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
            + cached_input_tokens_written
            * model_parameters.cached_input_token_write_cost
            + output_tokens * output_token_cost
        )

    def calculate_embedding_model_cost(
        self, model_provider: ModelProvider, model_name: str, billable_inputs: int
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
            raise ValueError(
                self.generate_unsupported_embedding_model_error_message(
                    model_provider, model_name
                )
            )
        return billable_inputs * model_costs.input_token_cost

    def _add_model_to_catalog(
        self,
        model_provider: ModelProvider,
        name: str,
        parameters: Union[CompletionModelParameters, EmbeddingModelParameters],
        snapshots: Optional[list[str]] = None,
    ):
        """A helper method to add a model to the catalog, adding it to the appropriate provider collection.

        Args:
            model_provider: The provider of the model
            name: The name of the model
            parameters: The parameters of the model
            snapshots: The optional list of snapshot names for the model

        Raises:
            InternalError: If the model already exists in the catalog
        """
        provider_model_collection = self.provider_model_collections.get(
            model_provider, ProviderModelCollection(model_provider)
        )
        provider_model_collection.add_model(name, parameters, snapshots)
        self.provider_model_collections[model_provider] = provider_model_collection

    # Public API for providers to add models and register dynamic loaders
    def add_model(
        self,
        model_provider: ModelProvider,
        name: str,
        parameters: Union[CompletionModelParameters, EmbeddingModelParameters],
        snapshots: Optional[list[str]] = None,
    ) -> None:
        try:
            self._add_model_to_catalog(model_provider, name, parameters, snapshots)
        except InternalError:
            logger.warning(
                f"Failed to add model to catalog: {name} for provider {model_provider.value}",
                exc_info=True,
            )

    def register_dynamic_provider(
        self, model_provider: ModelProvider, loader: Callable[[], None]
    ) -> None:
        """Register a one-time loader that populates models on first request.

        The loader should synchronously call add_model() for each model to register.
        """
        self._dynamic_loaders[model_provider] = loader

    def _get_supported_completions_models_by_provider(
        self, model_provider: ModelProvider
    ) -> CompletionModelCollection:
        """Returns the collection of completion models for a specific provider.

        Args:
            model_provider: The provider to get models for

        Returns:
            Collection of completion models for the specified provider, including snapshots
        """
        completion_models = self.provider_model_collections[model_provider].completion_models
        if not completion_models:
            # Lazy load for dynamic providers
            loader = self._dynamic_loaders.get(model_provider)
            if loader and model_provider not in self._dynamic_loaded:
                with self._loader_mutex:
                    if model_provider not in self._dynamic_loaded:
                        try:
                            logger.debug(
                                f"Dynamically loading models for provider: {model_provider.value}"
                            )
                            loader()
                            self._dynamic_loaded.add(model_provider)
                            return self.provider_model_collections[model_provider].completion_models
                        except Exception as exc:
                            logger.error(
                                f"Failed dynamic load for provider {model_provider.value}: {exc}",
                                exc_info=True,
                            )
        return completion_models

    def _get_supported_embeddings_models_by_provider(
        self, model_provider: ModelProvider
    ) -> EmbeddingModelCollection:
        """Returns the collection of embedding models for a specific provider.

        Args:
            model_provider: The provider to get models for

        Returns:
            Collection of embedding models for the specified provider, including snapshots
        """
        return self.provider_model_collections[model_provider].embedding_models

    def _get_supported_completions_models_by_provider_as_string(
        self, model_provider: ModelProvider
    ) -> str:
        """Returns a comma-separated string of supported completion model names for a provider.

        Args:
            model_provider: The provider to get model names for

        Returns:
            Comma-separated string of model names
        """
        return ", ".join(
            sorted(
                self._get_supported_completions_models_by_provider(
                    model_provider
                ).keys()
            )
        )

    def _get_supported_embeddings_models_by_provider_as_string(
        self, model_provider: ModelProvider
    ) -> str:
        """Returns a comma-separated string of supported completion model names for a provider.

        Args:
            model_provider: The provider to get model names for

        Returns:
            Comma-separated string of model names
        """
        return ", ".join(
            sorted(
                self._get_supported_embeddings_models_by_provider(
                    model_provider
                ).keys()
            )
        )


# Create a singleton instance
model_catalog = ModelCatalog()
