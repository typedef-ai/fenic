from dataclasses import dataclass, field
from typing import Optional

import anthropic

from fenic._inference.preset_manager import BasePresetConfiguration, PresetManager
from fenic.core._inference.model_catalog import CompletionModelParameters
from fenic.core._resolved_session_config import ResolvedAnthropicModelPreset


@dataclass
class AnthropicPresetConfiguration(BasePresetConfiguration):
    """Configuration for Anthropic model presets.

    Attributes:
        thinking_enabled: Whether thinking/reasoning is enabled for this preset
        thinking_token_budget: Token budget allocated for thinking/reasoning
        thinking_config: Anthropic-specific thinking configuration
    """
    thinking_enabled: bool = False
    thinking_token_budget: int = 0
    thinking_config: anthropic.types.ThinkingConfigParam = field(
        default_factory=lambda: anthropic.types.ThinkingConfigDisabledParam(type="disabled"))


class AnthropicCompletionsPresetManager(PresetManager[ResolvedAnthropicModelPreset, AnthropicPresetConfiguration]):
    """Manages Anthropic-specific preset configurations.

    This class handles the conversion of Fenic preset configurations to
    Anthropic-specific configurations, including thinking/reasoning settings.
    """

    def __init__(
        self,
        model_parameters: CompletionModelParameters,
        preset_configurations: Optional[dict[str, ResolvedAnthropicModelPreset]] = None,
        default_preset_name: Optional[str] = None
    ):
        """Initialize the Anthropic preset configuration manager.

        Args:
            model_parameters: Parameters for the completion model
            preset_configurations: Dictionary of preset configurations
            default_preset_name: Name of the default preset to use
        """
        self.model_parameters = model_parameters
        super().__init__(preset_configurations, default_preset_name)

    def _process_preset(self, preset: ResolvedAnthropicModelPreset) -> AnthropicPresetConfiguration:
        """Process Anthropic preset configuration.

        Converts a Fenic preset configuration to an Anthropic-specific configuration,
        handling thinking/reasoning settings based on model capabilities.

        Args:
            preset: The Fenic preset configuration to process

        Returns:
            Anthropic-specific preset configuration
        """
        if preset.thinking_token_budget and self.model_parameters.supports_reasoning:
            return AnthropicPresetConfiguration(
                thinking_enabled=True,
                thinking_token_budget=preset.thinking_token_budget,
                thinking_config=anthropic.types.ThinkingConfigEnabledParam(
                    type="enabled",
                    budget_tokens=preset.thinking_token_budget
                )
            )
        else:
            return AnthropicPresetConfiguration()

    def _get_default_configuration(self) -> AnthropicPresetConfiguration:
        """Get default Anthropic configuration.

        Returns:
            Default configuration with thinking disabled
        """
        return AnthropicPresetConfiguration()
