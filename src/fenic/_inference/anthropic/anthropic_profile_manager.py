import math
from dataclasses import dataclass, field
from typing import Optional

import anthropic

from fenic._inference.profile_manager import BaseProfileConfiguration, ProfileManager
from fenic.core._inference.model_catalog import (
    AnthropicReasoningEffortType,
    CompletionModelParameters,
)
from fenic.core._inference.output_token_limits import (
    ANTHROPIC_ADAPTIVE_THINKING_EFFORT_RATIOS,
)
from fenic.core._resolved_session_config import ResolvedAnthropicModelProfile


@dataclass
class AnthropicProfileConfiguration(BaseProfileConfiguration):
    """Configuration for Anthropic model profiles.

    Attributes:
        thinking_enabled: Whether thinking/reasoning is enabled for this profile
        thinking_token_budget: Token budget allocated for thinking/reasoning
        uses_adaptive_thinking: Whether thinking_token_budget is an adaptive maximum rather than a fixed budget
        thinking_config: Anthropic-specific thinking configuration
        output_config: Anthropic output configuration for effort.
    """
    thinking_enabled: bool = False
    thinking_token_budget: int = 0
    uses_adaptive_thinking: bool = False
    effort: Optional[AnthropicReasoningEffortType] = None
    thinking_config: anthropic.types.ThinkingConfigParam = field(
        default_factory=lambda: anthropic.types.ThinkingConfigDisabledParam(type="disabled"))
    output_config: Optional[anthropic.types.OutputConfigParam] = None


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
                ),
                effort=profile.effort,
                output_config={"effort": profile.effort} if profile.effort else None,
            )
        elif self.model_parameters.requires_adaptive_thinking:
            return AnthropicProfileConfiguration(
                thinking_enabled=True,
                uses_adaptive_thinking=True,
                effort=profile.effort,
                thinking_config=anthropic.types.ThinkingConfigAdaptiveParam(
                    type="adaptive"
                ),
                output_config={"effort": profile.effort} if profile.effort else None,
            )
        elif profile.effort and self.model_parameters.uses_adaptive_thinking:
            return AnthropicProfileConfiguration(
                thinking_enabled=True,
                thinking_token_budget=math.ceil(
                    ANTHROPIC_ADAPTIVE_THINKING_EFFORT_RATIOS[profile.effort]
                    * self.model_parameters.max_output_tokens
                ),
                thinking_config=anthropic.types.ThinkingConfigAdaptiveParam(
                    type="adaptive"
                ),
                uses_adaptive_thinking=True,
                effort=profile.effort,
                output_config={"effort": profile.effort},
            )
        elif profile.effort:
            return AnthropicProfileConfiguration(
                effort=profile.effort,
                output_config={"effort": profile.effort},
            )
        else:
            return AnthropicProfileConfiguration()

    def get_default_profile(self) -> AnthropicProfileConfiguration:
        """Get default Anthropic configuration.

        Returns:
            Default configuration for the model
        """
        if self.model_parameters.requires_adaptive_thinking:
            return AnthropicProfileConfiguration(
                thinking_enabled=True,
                uses_adaptive_thinking=True,
                thinking_config=anthropic.types.ThinkingConfigAdaptiveParam(
                    type="adaptive"
                ),
            )
        return AnthropicProfileConfiguration()
