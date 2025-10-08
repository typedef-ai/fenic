"""Profile manager for OpenRouter chat completions extra parameters.

Builds provider-specific extra_body for the OpenAI SDK request against OpenRouter.

References:
- Chat completion params: https://openrouter.ai/docs/api-reference/chat-completion
- API overview (parameters): https://openrouter.ai/docs/api-reference/overview
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic.dataclasses import dataclass

from fenic._inference.profile_manager import BaseProfileConfiguration, ProfileManager
from fenic.core._inference.model_catalog import CompletionModelParameters
from fenic.core._resolved_session_config import (
    ResolvedOpenRouterModelProfile,
    ResolvedOpenRouterProviderRouting,
)
from fenic.core.types.provider_routing import StructuredOutputStrategy
from fenic.core.types.semantic import ParsingEngine


@dataclass
class OpenRouterCompletionProfileConfiguration(BaseProfileConfiguration):
    """Completion profile for OpenRouter models."""

    # Only the fields we support for parity right now
    reasoning_effort: Optional[str] = None
    reasoning_max_tokens: Optional[int] = None
    models: Optional[list[str]] = None
    provider: Optional[ResolvedOpenRouterProviderRouting] = None
    structured_output_strategy: Optional[StructuredOutputStrategy] = None
    parsing_engine: Optional[ParsingEngine] = None

    @property
    def extra_body(self) -> dict[str, Any]:
        params: dict[str, Any] = {}
        # Enable native usage accounting so we don't need a follow-up generation fetch
        params["usage"] = {"include": True}
        # Map OpenRouter params into request body
        params["provider"] = {"require_parameters": True}
        if self.models:
            params["models"] = list(self.models)
        if self.provider:
            if self.provider.order:
                params["provider"]["order"] = self.provider.order
            if self.provider.sort:
                params["provider"]["sort"] = self.provider.sort
            if self.provider.data_collection:
                params["provider"]["data_collection"] = self.provider.data_collection
            if self.provider.quantizations:
                params["provider"]["quantizations"] = self.provider.quantizations
            if self.provider.only:
                params["provider"]["only"] = self.provider.only
            if self.provider.ignore:
                params["provider"]["ignore"] = self.provider.ignore
            price_config = {}
            if self.provider.max_prompt_price is not None:
                price_config["prompt"] = self.provider.max_prompt_price
            if self.provider.max_completion_price is not None:
                price_config["completion"] = self.provider.max_completion_price
            if price_config:
                params["provider"]["max_price"] = price_config
        reasoning_obj: dict[str, Any] = {}
        if self.reasoning_effort is not None:
            reasoning_obj["effort"] = self.reasoning_effort
        if self.reasoning_max_tokens is not None:
            reasoning_obj["max_tokens"] = int(self.reasoning_max_tokens)
        if reasoning_obj:
            reasoning_obj["exclude"] = True
            params["reasoning"] = reasoning_obj
        params["plugins"] = [
            {
                "id": "file-parser",
                "pdf": {
                    "engine": self.parsing_engine or "native"
                }
            }
        ]
        return params


class OpenRouterCompletionsProfileManager(
    ProfileManager[
        ResolvedOpenRouterModelProfile, OpenRouterCompletionProfileConfiguration
    ]
):
    """Constructs processed OpenRouter profile configurations per model/profile."""

    def __init__(
        self,
        model_parameters: CompletionModelParameters,
        profile_configurations: Optional[
            dict[str, OpenRouterCompletionProfileConfiguration]
        ] = None,
        default_profile_name: Optional[str] = None,
    ):
        self._model_parameters = model_parameters
        super().__init__(
            profile_configurations=profile_configurations,
            default_profile_name=default_profile_name,
        )

    def _process_profile(
        self, profile: ResolvedOpenRouterModelProfile
    ) -> OpenRouterCompletionProfileConfiguration:
        # Capability-based validation: only allow reasoning params if model supports reasoning
        if not self._model_parameters.supports_reasoning:
            if (
                profile.reasoning_effort is not None
                or profile.reasoning_max_tokens is not None
            ):
                # Drop unsupported reasoning fields to avoid invalid API parameters
                profile = ResolvedOpenRouterModelProfile(
                    models=profile.models,
                    provider=profile.provider,
                )
        else:
            # If the model supports reasoning, but the profile doesn't have a reasoning effort or max tokens, set a default
            # reasoning effort to low, so users don't hit weird issues with max tokens being hit before any completion tokens are output.
            if (
                profile.reasoning_effort is None
                and profile.reasoning_max_tokens is None
            ):
                profile.reasoning_effort = "low"
        
        pdf_page_processing_cost = None
        if profile.parsing_engine and profile.parsing_engine == "mistral-ocr":
            pdf_page_processing_cost = 2/1000
        elif profile.parsing_engine and profile.parsing_engine == "pdf-text":
            pdf_page_processing_cost = 0

        return OpenRouterCompletionProfileConfiguration(
            models=profile.models,
            provider=profile.provider,
            reasoning_effort=profile.reasoning_effort,
            reasoning_max_tokens=profile.reasoning_max_tokens,
            structured_output_strategy=profile.structured_output_strategy,
            parsing_engine=profile.parsing_engine,
            pdf_page_processing_cost=pdf_page_processing_cost,
        )

    def get_default_profile(self) -> OpenRouterCompletionProfileConfiguration:
        return OpenRouterCompletionProfileConfiguration(
            reasoning_effort="low" if self._model_parameters.supports_reasoning else None,
            provider=ResolvedOpenRouterProviderRouting(
                sort="throughput",
            )
        )
