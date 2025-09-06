"""Profile manager for OpenRouter chat completions extra parameters.

Builds provider-specific extra_body for the OpenAI SDK request against OpenRouter.

References:
- Chat completion params: https://openrouter.ai/docs/api-reference/chat-completion
- API overview (parameters): https://openrouter.ai/docs/api-reference/overview
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from fenic._inference.profile_manager import BaseProfileConfiguration, ProfileManager
from fenic.core._inference.model_catalog import CompletionModelParameters


@dataclass
class OpenRouterCompletionProfileConfiguration(BaseProfileConfiguration):
    """Extends OpenAI profile configuration with OpenRouter extras."""

    # Only the fields we support for parity right now
    reasoning_effort: Optional[str] = None
    reasoning_max_tokens: Optional[int] = None
    models: Optional[list[str]] = None
    provider_sort: Optional[str] = None

    @property
    def extra_body(self) -> dict[str, Any]:
        params: dict[str, Any] = {}
        # Map OpenRouter params into request body
        params["provider"] = {"require_parameters": True, "sort": "throughput"}
        if self.models:
            params["models"] = list(self.models)
        if self.provider_sort:
            params["provider"]["sort"] = self.provider_sort
        reasoning_obj: dict[str, Any] = {}
        if self.reasoning_effort is not None:
            reasoning_obj["effort"] = self.reasoning_effort
            reasoning_obj["exclude"] = True
        if self.reasoning_max_tokens is not None:
            reasoning_obj["max_tokens"] = int(self.reasoning_max_tokens)
            reasoning_obj["exclude"] = True
        if reasoning_obj:
            params["reasoning"] = reasoning_obj
        return params


class OpenRouterCompletionsProfileManager(
    ProfileManager[OpenRouterCompletionProfileConfiguration, OpenRouterCompletionProfileConfiguration]
):
    """Constructs processed OpenRouter profile configurations per model/profile."""

    def __init__(
        self,
        model_parameters: CompletionModelParameters,
        profile_configurations: Optional[dict[str, OpenRouterCompletionProfileConfiguration]] = None,
        default_profile_name: Optional[str] = None,
    ):
        self._model_parameters = model_parameters
        super().__init__(
            profile_configurations=profile_configurations,
            default_profile_name=default_profile_name,
        )

    def _process_profile(
        self, profile: OpenRouterCompletionProfileConfiguration
    ) -> OpenRouterCompletionProfileConfiguration:
        # Capability-based validation: only allow reasoning params if model supports reasoning
        supports_reasoning = False
        try:
            supported = getattr(self._model_parameters, "supported_parameters", None)
            if isinstance(supported, set):
                supports_reasoning = any(
                    key in supported for key in ("reasoning", "include_reasoning", "structured_outputs")
                )
        except Exception:
            supports_reasoning = False

        if not supports_reasoning:
            if profile.reasoning_effort is not None or profile.reasoning_max_tokens is not None:
                # Drop unsupported reasoning fields to avoid invalid API parameters
                profile = OpenRouterCompletionProfileConfiguration(
                    models=profile.models,
                    provider_sort=profile.provider_sort,
                )

        # Normalize provider_sort if present
        # provider_sort is typed as a Literal via config/resolved types; no runtime validation needed here

        # Ensure models is a list of strings if provided
        if profile.models is not None:
            try:
                profile.models = [str(m) for m in profile.models]
            except Exception:
                profile.models = None

        return OpenRouterCompletionProfileConfiguration(
            models=profile.models,
            provider_sort=profile.provider_sort,
            reasoning_effort=profile.reasoning_effort,
            reasoning_max_tokens=profile.reasoning_max_tokens,
        )

    def get_default_profile(self) -> OpenRouterCompletionProfileConfiguration:
        # Empty configuration; caller may compute derived behavior (e.g., extra_body) as needed
        return OpenRouterCompletionProfileConfiguration()

