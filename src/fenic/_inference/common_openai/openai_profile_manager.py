from dataclasses import dataclass
from typing import Optional

from fenic._inference.profile_manager import BaseProfileConfiguration, ProfileManager
from fenic.core._inference.model_catalog import CompletionModelParameters
from fenic.core._inference.output_token_limits import (
    OPENAI_REASONING_TOKEN_ESTIMATES,
    resolve_openai_reasoning_effort,
)
from fenic.core._resolved_session_config import (
    ReasoningEffort,
    ResolvedOpenAIModelProfile,
    Verbosity,
)


@dataclass
class OpenAICompletionProfileConfiguration(BaseProfileConfiguration):
    reasoning_effort: Optional[ReasoningEffort] = None
    verbosity: Optional[Verbosity] = None
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
        if not self.model_parameters.supports_reasoning:
            return OpenAICompletionProfileConfiguration(
                verbosity=self._resolve_verbosity(profile),
            )

        reasoning_effort = resolve_openai_reasoning_effort(
            self.model_parameters, profile.reasoning_effort
        )
        return OpenAICompletionProfileConfiguration(
            reasoning_effort=reasoning_effort,
            verbosity=self._resolve_verbosity(profile),
            expected_additional_reasoning_tokens=self._get_reasoning_tokens(reasoning_effort)
        )

    def _resolve_verbosity(self, profile: ResolvedOpenAIModelProfile) -> Optional[Verbosity]:
        if self.model_parameters.supports_verbosity and profile.verbosity:
            return profile.verbosity
        return None

    def _get_reasoning_tokens(self, reasoning_effort: str) -> int:
        """Get the expected additional reasoning tokens for a given reasoning effort level."""
        return OPENAI_REASONING_TOKEN_ESTIMATES[reasoning_effort]

    def get_default_profile(self) -> OpenAICompletionProfileConfiguration:
        """Get default OpenAI configuration."""
        if not self.model_parameters.supports_reasoning:
            return OpenAICompletionProfileConfiguration()

        reasoning_effort = resolve_openai_reasoning_effort(self.model_parameters, None)
        return OpenAICompletionProfileConfiguration(
            reasoning_effort=reasoning_effort,
            expected_additional_reasoning_tokens=self._get_reasoning_tokens(reasoning_effort)
        )
