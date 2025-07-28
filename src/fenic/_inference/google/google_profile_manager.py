from dataclasses import dataclass, field
from typing import Optional

from google.genai.types import (
    EmbedContentConfigDict,
    GenerateContentConfigDict,
    ThinkingConfigDict,
)

from fenic._inference.profile_manager import BaseProfileConfiguration, ProfileManager
from fenic.core._inference.model_catalog import (
    CompletionModelParameters,
    EmbeddingModelParameters,
)
from fenic.core._resolved_session_config import ResolvedGoogleModelProfile
from fenic.core.error import ConfigurationError


@dataclass
class GoogleCompletionsProfileConfiguration(BaseProfileConfiguration):
    """Configuration for Google Gemini model profiles.

    Attributes:
        thinking_enabled: Whether thinking/reasoning is enabled for this profile
        thinking_token_budget: Token budget allocated for thinking/reasoning
        additional_generation_config: Additional Google-specific generation configuration
    """
    thinking_enabled: bool = False
    thinking_token_budget: int = 0
    additional_generation_config: GenerateContentConfigDict = field(default_factory=GenerateContentConfigDict)

@dataclass
class GoogleEmbeddingsProfileConfiguration(BaseProfileConfiguration):
    """Configuration for Google Gemini embeddings model presets."""
    additional_embedding_config: EmbedContentConfigDict = field(default_factory=EmbedContentConfigDict)

class GoogleEmbeddingsProfileManager(ProfileManager[ResolvedGoogleModelProfile, GoogleEmbeddingsProfileConfiguration]):

    def __init__(
        self,
        model_parameters: EmbeddingModelParameters,
        preset_configurations: Optional[dict[str, ResolvedGoogleModelProfile]] = None,
        default_profile_name: Optional[str] = None,
    ):
        self.model_parameters = model_parameters
        super().__init__(preset_configurations, default_profile_name)


    def _process_profile(self, preset: ResolvedGoogleModelProfile) -> GoogleEmbeddingsProfileConfiguration:
        config_dict = EmbedContentConfigDict()
        if preset.embedding_dimensionality:
            if not self.model_parameters.validate_dimensions(preset.embedding_dimensionality):
                raise ConfigurationError(f"The dimensionality of the Google Embeddings model preset {preset} is invalid."
                                         f"Requested dimensionality: {preset.embedding_dimensionality}"
                                         f"Available Options: {self.model_parameters.get_possible_dimensions()}")
            config_dict["output_dimensionality"] = preset.embedding_dimensionality

        if preset.embedding_task_type:
            config_dict["task_type"] = preset.embedding_task_type

        return GoogleEmbeddingsProfileConfiguration(
           additional_embedding_config=config_dict,
        )

    def get_default_profile(self) -> GoogleEmbeddingsProfileConfiguration:
        return GoogleEmbeddingsProfileConfiguration()



class GoogleCompletionsProfileManager(ProfileManager[ResolvedGoogleModelProfile, GoogleCompletionsProfileConfiguration]):
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

    def _process_profile(self, profile: ResolvedGoogleModelProfile) -> GoogleCompletionsProfileConfiguration:
        """Process Google profile configuration.

        Converts a Fenic profile configuration to a Google-specific configuration,
        handling thinking/reasoning settings based on model capabilities.

        Args:
            profile: The Fenic profile configuration to process

        Returns:
            Google-specific profile configuration
        """
        additional_generation_config: GenerateContentConfigDict = {}
        thinking_enabled = False
        expected_thinking_tokens = 0

        if self.model_parameters.supports_reasoning:
            if profile.thinking_token_budget is None or profile.thinking_token_budget == 0:
                # Thinking disabled
                thinking_enabled = False
                thinking_config: ThinkingConfigDict = {
                    "include_thoughts": False,
                    "thinking_budget": 0
                }
                additional_generation_config.update({"thinking_config": thinking_config})
            else:
                # Thinking enabled
                thinking_enabled = True
                thinking_config: ThinkingConfigDict = {
                    "include_thoughts": False,
                    "thinking_budget": profile.thinking_token_budget
                }
                additional_generation_config.update({"thinking_config": thinking_config})

                if profile.thinking_token_budget > 0:
                    expected_thinking_tokens = profile.thinking_token_budget
                else:  # profile.thinking_token_budget == -1
                    # Dynamic budget - approximate with default value
                    expected_thinking_tokens = 16384

        return GoogleCompletionsProfileConfiguration(
            thinking_enabled=thinking_enabled,
            thinking_token_budget=expected_thinking_tokens,
            additional_generation_config=additional_generation_config
        )

    def get_default_profile(self) -> GoogleCompletionsProfileConfiguration:
        """Get default Google configuration.

        Returns:
            Default configuration with thinking disabled
        """
        if self.model_parameters.supports_reasoning:
            return GoogleCompletionsProfileConfiguration(
                thinking_enabled=False,
                thinking_token_budget=0,
                additional_generation_config=GenerateContentConfigDict(
                    thinking_config=ThinkingConfigDict(
                        include_thoughts=False,
                        thinking_budget=0
                    )
                )
            )
        return GoogleCompletionsProfileConfiguration()
