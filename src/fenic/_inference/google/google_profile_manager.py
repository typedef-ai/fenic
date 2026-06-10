from dataclasses import dataclass, field
from typing import Optional

from google.genai.types import (
    EmbedContentConfig,
    ThinkingConfig,
    ThinkingLevel,
)

from fenic._inference.profile_manager import BaseProfileConfiguration, ProfileManager
from fenic.core._inference.model_catalog import (
    CompletionModelParameters,
    EmbeddingModelParameters,
    MediaResolutionType,
)
from fenic.core._inference.output_token_limits import (
    GOOGLE_THINKING_LEVEL_TOKEN_ESTIMATES,
)
from fenic.core._resolved_session_config import ResolvedGoogleModelProfile


@dataclass
class GoogleCompletionsProfileConfig(BaseProfileConfiguration):
    """Configuration for Google Gemini model profiles.

    Attributes:
        thinking_enabled: Whether thinking/reasoning is enabled for this profile
        thinking_token_budget: Token budget allocated for thinking/reasoning
        thinking_config: Google thinking configuration
        media_resolution: Media resolution for PDF processing (gemini-3+ models)
    """
    thinking_enabled: bool = False
    thinking_token_budget: int = 0
    thinking_config: Optional[ThinkingConfig] = None
    media_resolution: Optional[MediaResolutionType] = None

@dataclass
class GoogleEmbeddingsProfileConfig(BaseProfileConfiguration):
    """Configuration for Google Gemini embeddings model profiles."""
    embedding_config: EmbedContentConfig = field(default_factory=EmbedContentConfig)

class GoogleEmbeddingsProfileManager(ProfileManager[ResolvedGoogleModelProfile, GoogleEmbeddingsProfileConfig]):

    def __init__(
        self,
        model_parameters: EmbeddingModelParameters,
        profiles: Optional[dict[str, ResolvedGoogleModelProfile]] = None,
        default_profile_name: Optional[str] = None,
    ):
        self.model_parameters = model_parameters
        super().__init__(profiles, default_profile_name)


    def _process_profile(self, profile: ResolvedGoogleModelProfile) -> GoogleEmbeddingsProfileConfig:
        return GoogleEmbeddingsProfileConfig(
           embedding_config=EmbedContentConfig(
               output_dimensionality=profile.embedding_dimensionality,
               task_type=profile.embedding_task_type,
           ),
        )

    def get_default_profile(self) -> GoogleEmbeddingsProfileConfig:
        return GoogleEmbeddingsProfileConfig()



class GoogleCompletionsProfileManager(ProfileManager[ResolvedGoogleModelProfile, GoogleCompletionsProfileConfig]):
    """Manages Google-specific profile configurations.

    This class handles the conversion of Fenic profile configurations to
    Google Gemini-specific configurations, including thinking/reasoning settings.
    """

    def __init__(
        self,
        model_parameters: CompletionModelParameters,
        profile_configurations: Optional[dict[str, ResolvedGoogleModelProfile]] = None,
        default_profile_name: Optional[str] = None
    ):
        """Initialize the Google profile configuration manager.

        Args:
            model_parameters: Parameters for the completion model
            profile_configurations: Dictionary of profile configurations
            default_profile_name: Name of the default profile to use
        """
        self.model_parameters = model_parameters
        super().__init__(profile_configurations, default_profile_name)

    def _process_profile(self, profile: ResolvedGoogleModelProfile) -> GoogleCompletionsProfileConfig:
        """Process Google profile configuration.

        Converts a Fenic profile configuration to a Google-specific configuration,
        handling thinking/reasoning settings based on model capabilities.

        Args:
            profile: The Fenic profile configuration to process

        Returns:
            Google-specific profile configuration
        """
        thinking_config = None
        thinking_enabled = False
        expected_thinking_tokens = 0

        if self.model_parameters.supports_reasoning:
            if self.model_parameters.supported_thinking_levels:
                # Gemini 3+ models use thinking_level instead of thinking_budget
                # thinking_token_budget must be None for these models
                if profile.thinking_level is not None:
                    thinking_level = profile.thinking_level
                else:
                    # Default to low if not specified
                    thinking_level = "low"

                thinking_enabled = True

                # Map thinking level to enum and estimate token budget
                thinking_level_map = {
                    "high": ThinkingLevel.HIGH,
                    "medium": ThinkingLevel.MEDIUM,
                    "low": ThinkingLevel.LOW,
                    "minimal": ThinkingLevel.MINIMAL,
                }
                thinking_level_enum = thinking_level_map[thinking_level]
                expected_thinking_tokens = GOOGLE_THINKING_LEVEL_TOKEN_ESTIMATES[thinking_level]
                thinking_config = ThinkingConfig(thinking_level=thinking_level_enum)
            elif profile.thinking_token_budget is None or profile.thinking_token_budget == 0:
                # Thinking disabled
                thinking_enabled = False
                thinking_config = ThinkingConfig(
                    include_thoughts=False,
                    thinking_budget=0,
                )
            else:
                # Thinking enabled with budget
                thinking_enabled = True
                thinking_config = ThinkingConfig(
                    include_thoughts=False,
                    thinking_budget=profile.thinking_token_budget,
                )

                if profile.thinking_token_budget > 0:
                    expected_thinking_tokens = profile.thinking_token_budget
                else:  # profile.thinking_token_budget == -1
                    # Dynamic budget - approximate with default value
                    expected_thinking_tokens = 16384

        # Handle media_resolution for gemini-3+ models
        media_resolution = None
        if self.model_parameters.supports_media_resolution:
            media_resolution = profile.media_resolution or "low"  # Default to "low" for gemini-3+ models

        return GoogleCompletionsProfileConfig(
            thinking_enabled=thinking_enabled,
            thinking_token_budget=expected_thinking_tokens,
            thinking_config=thinking_config,
            media_resolution=media_resolution,
        )

    def get_default_profile(self) -> GoogleCompletionsProfileConfig:
        """Get default Google configuration.

        Returns:
            Default configuration with thinking disabled if supported, otherwise default to low reasoning
        """
        # Default media_resolution for gemini-3+ models
        media_resolution = "low" if self.model_parameters.supports_media_resolution else None

        if self.model_parameters.supports_reasoning:
            if self.model_parameters.supported_thinking_levels:
                # Gemini 3+ models use thinking_level - default to low
                return GoogleCompletionsProfileConfig(
                    thinking_enabled=True,
                    thinking_token_budget=8192,  # Estimated for low
                    thinking_config=ThinkingConfig(
                        thinking_level=ThinkingLevel.LOW
                    ),
                    media_resolution=media_resolution,
                )
            elif self.model_parameters.supports_disabled_reasoning:
                return GoogleCompletionsProfileConfig(
                    thinking_enabled=False,
                    thinking_token_budget=0,
                    thinking_config=ThinkingConfig(
                        include_thoughts=False,
                        thinking_budget=0,
                    ),
                    media_resolution=media_resolution,
                )
            else:
                # default to low reasoning
                return GoogleCompletionsProfileConfig(
                    thinking_enabled=True,
                    thinking_token_budget=1024,
                    thinking_config=ThinkingConfig(
                        include_thoughts=True,
                        thinking_budget=1024,
                    ),
                    media_resolution=media_resolution,
                )
        return GoogleCompletionsProfileConfig(media_resolution=media_resolution)
