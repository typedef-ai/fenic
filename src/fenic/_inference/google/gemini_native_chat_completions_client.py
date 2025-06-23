import json
import os
from dataclasses import dataclass, field
from functools import cache
from typing import Optional, Union

from google import genai
from google.genai.errors import ClientError, ServerError
from google.genai.types import (
    FinishReason,
    GenerateContentConfigDict,
    GenerateContentResponse,
    ThinkingConfigDict,
)
from pydantic import BaseModel

from fenic._inference.model_client import (
    FatalException,
    FenicCompletionsRequest,
    FenicCompletionsResponse,
    ModelClient,
    ResponseUsage,
    TransientException,
)
from fenic._inference.preset_config_manager import (
    BasePresetConfiguration,
    PresetConfigurationManager,
)
from fenic._inference.rate_limit_strategy import UnifiedTokenRateLimitStrategy
from fenic._inference.request_utils import generate_completion_request_key
from fenic._inference.token_counter import TiktokenTokenCounter, Tokenizable
from fenic._inference.types import LMRequestMessages
from fenic.core._inference.model_catalog import (
    CompletionModelParameters,
    ModelProvider,
    model_catalog,
)
from fenic.core._resolved_session_config import ResolvedGoogleModelPreset
from fenic.core.error import ExecutionError
from fenic.core.metrics import LMMetrics


@dataclass
class GooglePresetConfiguration(BasePresetConfiguration):
    thinking_enabled: bool = False
    thinking_token_budget: int = 0
    additional_generation_config: GenerateContentConfigDict = field(default_factory=GenerateContentConfigDict)


class GooglePresetConfigurationManager(PresetConfigurationManager[ResolvedGoogleModelPreset, GooglePresetConfiguration]):
    """Manages Google-specific preset configurations."""

    def __init__(self,
                 model_parameters : CompletionModelParameters,
                 preset_configurations: Optional[dict[str, ResolvedGoogleModelPreset]] = None,
                 default_preset_name: Optional[str] = None):
        self.model_parameters = model_parameters
        super().__init__(preset_configurations, default_preset_name)

    def _process_preset(self, preset: ResolvedGoogleModelPreset) -> GooglePresetConfiguration:
        """Process Google preset configuration."""
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

        return GooglePresetConfiguration(
            thinking_enabled=thinking_enabled,
            thinking_token_budget=expected_thinking_tokens,
            additional_generation_config=additional_generation_config
        )

    def _get_default_configuration(self) -> GooglePresetConfiguration:
        """Get default Google configuration."""
        return GooglePresetConfiguration()


class GeminiNativeChatCompletionsClient(
    ModelClient[FenicCompletionsRequest, FenicCompletionsResponse]
):
    """Native (google-genai) Google Gemini chat-completions client.

    """

    def __init__(
        self,
        rate_limit_strategy: UnifiedTokenRateLimitStrategy,
        model_provider: ModelProvider,
        model: str = "gemini-2.0-flash-lite",
        queue_size: int = 100,
        max_backoffs: int = 10,
        preset_configurations: Optional[dict[str, ResolvedGoogleModelPreset]] = None,
        default_preset_name: Optional[str] = None,
    ):
        token_counter = TiktokenTokenCounter(
            model_name=model, fallback_encoding="o200k_base"
        )
        super().__init__(
            model=model,
            model_provider=model_provider,
            rate_limit_strategy=rate_limit_strategy,
            queue_size=queue_size,
            max_backoffs=max_backoffs,
            token_counter=token_counter,
        )

        # Native gen-ai client. Passing `vertexai=True` automatically routes traffic
        # through Vertex-AI if the environment is configured for it.
        if model_provider == ModelProvider.GOOGLE_GLA:
            if "GEMINI_API_KEY" in os.environ:
                self._base_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
            else:
                self._base_client = genai.Client()
        else:
            self._base_client = genai.Client(vertexai=True)
        self._client = self._base_client.aio
        self._metrics = LMMetrics()
        self._token_counter = token_counter  # For type checkers
        self._model_parameters = model_catalog.get_completion_model_parameters(
            model_provider, model
        )

        # Use the preset configuration manager
        self.preset_manager = GooglePresetConfigurationManager(
            model_parameters=self._model_parameters,
            preset_configurations=preset_configurations,
            default_preset_name=default_preset_name
        )


    def reset_metrics(self):
        self._metrics = LMMetrics()

    def get_metrics(self) -> LMMetrics:
        return self._metrics

    def _convert_messages(self, messages: LMRequestMessages) -> list[genai.types.ContentUnion]:
        """Convert Fenic LMRequestMessages → list of google-genai `Content` objects."""
        contents: list[genai.types.ContentUnion] = []
        # few-shot examples
        for example in messages.examples:
            contents.append(genai.types.Content(role="user", parts=[genai.types.Part(text=example.user)]))
            contents.append(genai.types.Content(role="model", parts=[genai.types.Part(text=example.assistant)]))

        # final user prompt
        contents.append(genai.types.Content(role="user", parts=[genai.types.Part(text=messages.user)]))
        return contents

    def count_tokens(self, messages: Tokenizable) -> int:  # type: ignore[override]
        # Re-expose for mypy – same implementation as parent.
        return super().count_tokens(messages)

    def _estimate_structured_output_overhead(self, response_format) -> int:
        """Use Google-specific response schema token estimation."""
        return self._estimate_response_schema_tokens(response_format)

    def _get_max_output_tokens(self, request: FenicCompletionsRequest) -> int:
        """Conservative estimate: max_completion_tokens + thinking token budget."""
        preset_config = self.preset_manager.get_preset_configuration(request.model_preset)
        return request.max_completion_tokens + int(1.5 * preset_config.thinking_token_budget)


    @cache  # noqa: B019 – builtin cache OK here.
    def _estimate_response_schema_tokens(self, response_format: type[BaseModel]) -> int:
        schema_str = json.dumps(response_format.model_json_schema(), separators=(',', ':'))
        return self._token_counter.count_tokens(schema_str)


    def get_request_key(self, request: FenicCompletionsRequest) -> str:
        return generate_completion_request_key(request)

    async def make_single_request(
        self, request: FenicCompletionsRequest
    ) -> Union[None, FenicCompletionsResponse, TransientException, FatalException]:
        
        # Get preset-specific configuration
        preset_config = self.preset_manager.get_preset_configuration(request.model_preset)
        max_output_tokens = request.max_completion_tokens + int(1.5 * preset_config.thinking_token_budget)

        generation_config: GenerateContentConfigDict = {
            "temperature": request.temperature,
            "max_output_tokens": max_output_tokens,
            "response_logprobs": request.top_logprobs is not None,
            "logprobs": request.top_logprobs,
            "system_instruction": request.messages.system,
        }
        generation_config.update(preset_config.additional_generation_config)

        if request.structured_output is not None:
            generation_config.update(
                response_mime_type="application/json",
                response_schema=request.structured_output.model_json_schema(),
            )

        # Build generation parameters
        contents = self._convert_messages(request.messages)

        try:
            response: GenerateContentResponse = await self._client.models.generate_content(
                model=self.model,
                contents=contents,
                config=generation_config,
            )
            candidate = response.candidates[0] if response.candidates else None
            completion_text = ""
            if candidate is None:
                return FatalException(ExecutionError("No candidate found in response"))
            elif candidate.finish_reason == FinishReason.MAX_TOKENS:
                return FatalException(
                    ExecutionError(
                        "Candidate generation failed due to max tokens limit. "
                        "Consider retrying the request with a higher max_completion_tokens value. "
                        "If using a reasoning model, consider increasing the thinking_token_budget. "
                        f"Current max_completions_tokens: {request.max_completion_tokens}, Current thinking_token_budget: {preset_config.thinking_token_budget}."))
            elif candidate.finish_reason != FinishReason.STOP:
                return FatalException(ExecutionError(f"Candidate generation failed due to stop reason: {candidate.finish_reason}"))
            else:
                for part in candidate.content.parts:
                    if not part.thought:
                        completion_text = part.text

            # Extract usage metrics
            usage_metadata = response.usage_metadata
            response_usage = None

            self._metrics.num_requests += 1
            if usage_metadata:
                total_prompt_tokens = usage_metadata.prompt_token_count if usage_metadata.prompt_token_count else 0
                cached_prompt_tokens = usage_metadata.cached_content_token_count if usage_metadata.cached_content_token_count else 0
                uncached_prompt_tokens = total_prompt_tokens - cached_prompt_tokens
                candidates_token_count = usage_metadata.candidates_token_count if usage_metadata.candidates_token_count else 0
                thinking_tokens_count = usage_metadata.thoughts_token_count if usage_metadata.thoughts_token_count else 0
                total_output_tokens = candidates_token_count + thinking_tokens_count

                # Create ResponseUsage object
                response_usage = ResponseUsage(
                    prompt_tokens=total_prompt_tokens,
                    completion_tokens=candidates_token_count,  # Google separates completion from thinking
                    total_tokens=total_prompt_tokens + total_output_tokens,
                    cached_tokens=cached_prompt_tokens,
                    thinking_tokens=thinking_tokens_count
                )

                # Update metrics (existing logic)
                self._metrics.num_uncached_input_tokens += uncached_prompt_tokens
                self._metrics.num_cached_input_tokens += cached_prompt_tokens
                self._metrics.num_output_tokens += total_output_tokens

                self._metrics.cost += model_catalog.calculate_completion_model_cost(
                    model_provider=self.model_provider,
                    model_name=self.model,
                    uncached_input_tokens=uncached_prompt_tokens,
                    cached_input_tokens_read=cached_prompt_tokens,
                    output_tokens=total_output_tokens,
                )

            return FenicCompletionsResponse(completion=completion_text, logprobs=None, usage=response_usage)

        except ServerError as e:
            # Treat quota/timeouts as transient (retryable)
            return TransientException(e)
        except ClientError as e:
            if e.code == 429:
                return TransientException(e)
            else:
                return FatalException(e)
        except Exception as e:  # noqa: BLE001 – catch-all mapped to Fatal
            return FatalException(e)