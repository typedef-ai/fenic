from dataclasses import dataclass, field
from typing import Optional

import anthropic

from fenic._inference.profile_manager import BaseProfileConfiguration, ProfileManager
from fenic.core._inference.model_catalog import CompletionModelParameters
from fenic.core._resolved_session_config import ResolvedAnthropicModelProfile


@dataclass
class AnthropicProfileConfiguration(BaseProfileConfiguration):
    """Configuration for Anthropic model profiles.

    Attributes:
        thinking_enabled: Whether thinking/reasoning is enabled for this profile
        thinking_token_budget: Token budget allocated for thinking/reasoning
        thinking_config: Anthropic-specific thinking configuration
    """
    thinking_enabled: bool = False
    thinking_token_budget: int = 0
    thinking_config: anthropic.types.ThinkingConfigParam = field(
        default_factory=lambda: anthropic.types.ThinkingConfigDisabledParam(type="disabled"))


class AnthropicCompletionsProfileManager(ProfileManager[ResolvedAnthropicModelProfile, AnthropicProfileConfiguration]):
    """Manages Anthropic-specific profile configurations.

    This class handles the conversion of Fenic profile configurations to
    Anthropic-specific configurations, including thinking/reasoning settings.
    """

    def __init__(
        self,
        model_parameters: CompletionModelParameters,
        profile_configurations: Optional[dict[str, ResolvedAnthropicModelProfile]] = None,
        default_profile_name: Optional[str] = None
    ):
        """Initialize the Anthropic profile configuration manager.

        Args:
            model_parameters: Parameters for the completion model
            profile_configurations: Dictionary of profile configurations
            default_profile_name: Name of the default profile to use
        """
        self.model_parameters = model_parameters
        super().__init__(profile_configurations, default_profile_name)

    def _process_profile(self, profile: ResolvedAnthropicModelProfile) -> AnthropicProfileConfiguration:
        """Process Anthropic profile configuration.

        Converts a Fenic profile configuration to an Anthropic-specific configuration,
        handling thinking/reasoning settings based on model capabilities.

        Args:
            profile: The Fenic profile configuration to process

        Returns:
            Anthropic-specific profile configuration
        """
        if profile.thinking_token_budget and self.model_parameters.supports_reasoning:
            return AnthropicProfileConfiguration(
                thinking_enabled=True,
                thinking_token_budget=profile.thinking_token_budget,
                thinking_config=anthropic.types.ThinkingConfigEnabledParam(
                    type="enabled",
                    budget_tokens=profile.thinking_token_budget
                )
            )
        else:
            return AnthropicProfileConfiguration()

    def get_default_profile(self) -> AnthropicProfileConfiguration:
        """Get default Anthropic configuration.

        Returns:
            Default configuration with thinking disabled
        """
        return AnthropicProfileConfiguration()
