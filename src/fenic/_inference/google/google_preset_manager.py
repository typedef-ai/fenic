from dataclasses import dataclass, field
from typing import Optional

from google.genai.types import GenerateContentConfigDict, ThinkingConfigDict

from fenic._inference.preset_manager import BasePresetConfiguration, PresetManager
from fenic.core._inference.model_catalog import CompletionModelParameters
from fenic.core._resolved_session_config import ResolvedGoogleModelPreset


@dataclass
class GoogleCompletionsPresetConfiguration(BasePresetConfiguration):
    """Configuration for Google Gemini model presets.

    Attributes:
        thinking_enabled: Whether thinking/reasoning is enabled for this preset
        thinking_token_budget: Token budget allocated for thinking/reasoning
        additional_generation_config: Additional Google-specific generation configuration
    """
    thinking_enabled: bool = False
    thinking_token_budget: int = 0
    additional_generation_config: GenerateContentConfigDict = field(default_factory=GenerateContentConfigDict)


class GoogleCompletionsPresetManager(PresetManager[ResolvedGoogleModelPreset, GoogleCompletionsPresetConfiguration]):
    """Manages Google-specific preset configurations.

    This class handles the conversion of Fenic preset configurations to
    Google Gemini-specific configurations, including thinking/reasoning settings.
    """

    def __init__(
        self,
        model_parameters: CompletionModelParameters,
        preset_configurations: Optional[dict[str, ResolvedGoogleModelPreset]] = None,
        default_preset_name: Optional[str] = None
    ):
        """Initialize the Google preset configuration manager.

        Args:
            model_parameters: Parameters for the completion model
            preset_configurations: Dictionary of preset configurations
            default_preset_name: Name of the default preset to use
        """
        self.model_parameters = model_parameters
        super().__init__(preset_configurations, default_preset_name)

    def _process_preset(self, preset: ResolvedGoogleModelPreset) -> GoogleCompletionsPresetConfiguration:
        """Process Google preset configuration.

        Converts a Fenic preset configuration to a Google-specific configuration,
        handling thinking/reasoning settings based on model capabilities.

        Args:
            preset: The Fenic preset configuration to process

        Returns:
            Google-specific preset configuration
        """
        additional_generation_config: GenerateContentConfigDict = {}
        thinking_enabled = False
        expected_thinking_tokens = 0

        if self.model_parameters.supports_reasoning:
            if preset.thinking_token_budget is None or preset.thinking_token_budget == 0:
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
                    "thinking_budget": preset.thinking_token_budget
                }
                additional_generation_config.update({"thinking_config": thinking_config})

                if preset.thinking_token_budget > 0:
                    expected_thinking_tokens = preset.thinking_token_budget
                else:  # preset.thinking_token_budget == -1
                    # Dynamic budget - approximate with default value
                    expected_thinking_tokens = 16384

        return GoogleCompletionsPresetConfiguration(
            thinking_enabled=thinking_enabled,
            thinking_token_budget=expected_thinking_tokens,
            additional_generation_config=additional_generation_config
        )

    def _get_default_configuration(self) -> GoogleCompletionsPresetConfiguration:
        """Get default Google configuration.

        Returns:
            Default configuration with thinking disabled
        """
        if self.model_parameters.supports_reasoning:
            return GoogleCompletionsPresetConfiguration(
                thinking_enabled=False,
                thinking_token_budget=0,
                additional_generation_config=GenerateContentConfigDict(
                    thinking_config=ThinkingConfigDict(
                        include_thoughts=False,
                        thinking_budget=0
                    )
                )
            )
        return GoogleCompletionsPresetConfiguration()
