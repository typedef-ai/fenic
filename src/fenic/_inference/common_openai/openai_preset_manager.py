from dataclasses import dataclass, field
from typing import Any, Optional

from fenic._inference.preset_manager import BasePresetConfiguration, PresetManager
from fenic.core._inference.model_catalog import CompletionModelParameters
from fenic.core._resolved_session_config import ResolvedOpenAIModelPreset


@dataclass
class OpenAICompletionPresetConfiguration(BasePresetConfiguration):
    additional_parameters: dict[str, Any] = field(default_factory=dict)
    reasoning_effort: Optional[str] = None
    expected_additional_reasoning_tokens: int = 0


class OpenAICompletionsPresetManager(
    PresetManager[ResolvedOpenAIModelPreset, OpenAICompletionPresetConfiguration]):
    """Manages OpenAI-specific preset configurations."""

    def __init__(
        self,
        model_parameters: CompletionModelParameters,
        preset_configurations: Optional[dict[str, ResolvedOpenAIModelPreset]] = None,
        default_preset_name: Optional[str] = None
    ):
        self.model_parameters = model_parameters
        super().__init__(preset_configurations, default_preset_name)

    def _process_preset(self, preset: ResolvedOpenAIModelPreset) -> OpenAICompletionPresetConfiguration:
        """Process OpenAI preset configuration."""
        additional_parameters = {}
        additional_reasoning_tokens = 0
        if self.model_parameters.supports_reasoning:
            reasoning_effort = preset.reasoning_effort
            # OpenAI does not support disabling reasoning for o-series models, so we default to low
            if not reasoning_effort:
                reasoning_effort = "low"
            additional_parameters["reasoning_effort"] = reasoning_effort
            if reasoning_effort == "low":
                additional_reasoning_tokens = 4096
            elif reasoning_effort == "medium":
                additional_reasoning_tokens = 8192
            elif reasoning_effort == "high":
                additional_reasoning_tokens = 16384

        return OpenAICompletionPresetConfiguration(
            reasoning_effort=preset.reasoning_effort,
            additional_parameters=additional_parameters,
            expected_additional_reasoning_tokens=additional_reasoning_tokens
        )

    def _get_default_configuration(self) -> OpenAICompletionPresetConfiguration:
        """Get default OpenAI configuration."""
        return OpenAICompletionPresetConfiguration()
