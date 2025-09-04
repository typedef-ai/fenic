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
            # OpenAI does not support disabling reasoning for o-series or gpt5 models, so we default to the lowest effort.
            reasoning_effort = profile.reasoning_effort
            if not reasoning_effort:
                if self.model_parameters.supports_minimal_reasoning:
                    reasoning_effort = "minimal"
                else:
                    reasoning_effort = "low"
            additional_parameters["reasoning_effort"] = reasoning_effort
            if reasoning_effort == "minimal":
                additional_reasoning_tokens = 2048
            elif reasoning_effort == "low":
                additional_reasoning_tokens = 4096
            elif reasoning_effort == "medium":
                additional_reasoning_tokens = 8192
            elif reasoning_effort == "high":
                additional_reasoning_tokens = 16384
        if self.model_parameters.supports_verbosity and profile.verbosity:
            additional_parameters["verbosity"] = profile.verbosity

        return OpenAICompletionProfileConfiguration(
            additional_parameters=additional_parameters,
            expected_additional_reasoning_tokens=additional_reasoning_tokens
        )

    def get_default_profile(self) -> OpenAICompletionProfileConfiguration:
        """Get default OpenAI configuration."""
        if self.model_parameters.supports_reasoning:
            # OpenAI does not support disabling reasoning for o-series or gpt5 models, so we default to the lowest effort.
            if self.model_parameters.supports_minimal_reasoning:
                reasoning_effort = "minimal"
                additional_reasoning_tokens = 2048
            else:
                reasoning_effort = "low"
                additional_reasoning_tokens = 4096
            return OpenAICompletionProfileConfiguration(
                additional_parameters={
                    "reasoning_effort": reasoning_effort
                },
                expected_additional_reasoning_tokens=additional_reasoning_tokens
            )
        else:
            return OpenAICompletionProfileConfiguration()
