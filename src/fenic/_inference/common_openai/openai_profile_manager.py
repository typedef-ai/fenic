from dataclasses import dataclass, field
from typing import Any, Optional

from fenic._inference.profile_manager import BaseProfileConfiguration, ProfileManager
from fenic.core._inference.model_catalog import CompletionModelParameters
from fenic.core._resolved_session_config import ResolvedOpenAIModelProfile


@dataclass
class OpenAICompletionProfileConfiguration(BaseProfileConfiguration):
    additional_parameters: dict[str, Any] = field(default_factory=dict)
    expected_additional_reasoning_tokens: int = 0


class OpenAICompletionsProfileManager(
    ProfileManager[ResolvedOpenAIModelProfile, OpenAICompletionProfileConfiguration]):
    """Manages OpenAI-specific profile configurations."""

    def __init__(
        self,
        model_parameters: CompletionModelParameters,
        profile_configurations: Optional[dict[str, ResolvedOpenAIModelProfile]] = None,
        default_profile_name: Optional[str] = None
    ):
        self.model_parameters = model_parameters
        super().__init__(profile_configurations, default_profile_name)

    def _process_profile(self, profile: ResolvedOpenAIModelProfile) -> OpenAICompletionProfileConfiguration:
        """Process OpenAI profile configuration."""
        additional_parameters = {}
        additional_reasoning_tokens = 0

        if self.model_parameters.supports_reasoning:
            # Reasoning effort behavior varies by model:
            # - o-series/gpt-5 models: do not support disabling reasoning, default to lowest effort (minimal or low)
            # - gpt-5.1 models: support 'none' to disable reasoning, default to 'none'
            reasoning_effort = profile.reasoning_effort
            if not reasoning_effort:
                if self.model_parameters.supports_disabled_reasoning:
                    reasoning_effort = "none"
                elif self.model_parameters.supports_minimal_reasoning:
                    reasoning_effort = "minimal"
                else:
                    reasoning_effort = "low"
            additional_parameters["reasoning_effort"] = reasoning_effort
            additional_reasoning_tokens = self._get_reasoning_tokens(reasoning_effort)

        if self.model_parameters.supports_verbosity and profile.verbosity:
            additional_parameters["verbosity"] = profile.verbosity

        return OpenAICompletionProfileConfiguration(
            additional_parameters=additional_parameters,
            expected_additional_reasoning_tokens=additional_reasoning_tokens
        )

    def _get_reasoning_tokens(self, reasoning_effort: str) -> int:
        """Get the expected additional reasoning tokens for a given reasoning effort level."""
        if reasoning_effort == "none":
            return 0
        elif reasoning_effort == "minimal":
            return 2048
        elif reasoning_effort == "low":
            return 4096
        elif reasoning_effort == "medium":
            return 8192
        elif reasoning_effort == "high":
            return 16384
        return 0

    def get_default_profile(self) -> OpenAICompletionProfileConfiguration:
        """Get default OpenAI configuration."""
        if self.model_parameters.supports_reasoning:
            # Reasoning effort behavior varies by model:
            # - o-series/gpt-5 models: do not support disabling reasoning, default to lowest effort (minimal or low)
            # - gpt-5.1 models: support 'none' to disable reasoning, default to 'none'
            if self.model_parameters.supports_disabled_reasoning:
                reasoning_effort = "none"
            elif self.model_parameters.supports_minimal_reasoning:
                reasoning_effort = "minimal"
            else:
                reasoning_effort = "low"
            return OpenAICompletionProfileConfiguration(
                additional_parameters={
                    "reasoning_effort": reasoning_effort
                },
                expected_additional_reasoning_tokens=self._get_reasoning_tokens(reasoning_effort)
            )
        else:
            return OpenAICompletionProfileConfiguration()
