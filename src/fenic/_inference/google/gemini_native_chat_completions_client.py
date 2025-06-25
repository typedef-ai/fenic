import hashlib
import json
import os
from functools import cache
from typing import Optional, Union

from google import genai
from google.genai.errors import ClientError, ServerError
from google.genai.types import (
    GenerateContentConfigDict,
    GenerateContentResponse,
    ThinkingConfigDict,
)
from pydantic import BaseModel

from fenic._inference.model_catalog import (
    ModelProvider,
    model_catalog,
)
from fenic._inference.model_client import (
    FatalException,
    FenicCompletionsRequest,
    FenicCompletionsResponse,
    ModelClient,
    TokenEstimate,
    TransientException,
    UnifiedTokenRateLimitStrategy,
)
from fenic._inference.token_counter import TiktokenTokenCounter, Tokenizable
from fenic._inference.types import LMRequestMessages
from fenic.core.metrics import LMMetrics


class GeminiNativeChatCompletionsClient(
    ModelClient[FenicCompletionsRequest, FenicCompletionsResponse]
):
    """Native Google Gemini chat-completions client.

    This implementation talks directly to the `google-generativeai` SDK instead of the
    OpenAI-compatibility layer. It mirrors the behaviour/metrics semantics of
    `AnthropicBatchCompletionsClient` so that the rest of Fenic can treat all model
    providers uniformly.
    """

    def __init__(
        self,
        rate_limit_strategy: UnifiedTokenRateLimitStrategy,
        model_provider: ModelProvider,
        model: str = "gemini-2.0-flash-lite",
        queue_size: int = 100,
        max_backoffs: int = 10,
        default_thinking_budget: Optional[int] = None,
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
        self._additional_generation_config: GenerateContentConfigDict = {}
        if self._model_parameters.supports_reasoning:
            default_thinking_config: ThinkingConfigDict = {
                "include_thoughts": True,
                "thinking_budget": default_thinking_budget
            }
            self._additional_generation_config.update({"thinking_config" : default_thinking_config})


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

    def estimate_tokens_for_request(self, request: FenicCompletionsRequest) -> TokenEstimate:
        auxiliary_tokens = 0
        if request.structured_output:
            auxiliary_tokens += self._estimate_response_schema_tokens(
                request.structured_output
            )
        message_tokens = self.count_tokens(request.messages.to_message_list())
        return TokenEstimate(
            input_tokens=message_tokens + auxiliary_tokens,
            output_tokens=request.max_completion_tokens,
        )

    @cache  # noqa: B019 – builtin cache OK here.
    def _estimate_response_schema_tokens(self, response_format: type[BaseModel]) -> int:
        schema_str = json.dumps(response_format.model_json_schema(), separators=(',', ':'))
        return self._token_counter.count_tokens(schema_str)


    def get_request_key(self, request: FenicCompletionsRequest) -> str:
        return hashlib.sha256(
            json.dumps(request.messages.to_message_list(), sort_keys=True).encode()
        ).hexdigest()[:10]

    async def make_single_request(
        self, request: FenicCompletionsRequest
    ) -> Union[None, FenicCompletionsResponse, TransientException, FatalException]:
        
        generation_config: GenerateContentConfigDict = {
            "temperature": request.temperature,
            "max_output_tokens": request.max_completion_tokens,
            "response_logprobs": request.top_logprobs is not None,
            "logprobs": request.top_logprobs,
            "system_instruction" : request.messages.system,
        }
        generation_config.update(self._additional_generation_config)

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

            completion_text: str | None = response.candidates[0].content.parts[0].text if response.candidates else None
            if completion_text is None:
                completion_text = ""
            usage = response.usage_metadata
            total_prompt_tokens = usage.prompt_token_count if usage else 0
            cached_prompt_tokens = usage.cached_content_token_count if usage.cached_content_token_count else 0
            uncached_prompt_tokens = total_prompt_tokens - cached_prompt_tokens
            total_output_tokens = usage.candidates_token_count if usage else 0

            self._metrics.num_uncached_input_tokens += uncached_prompt_tokens
            self._metrics.num_cached_input_tokens += cached_prompt_tokens
            self._metrics.num_output_tokens += total_output_tokens
            self._metrics.num_requests += 1

            self._metrics.cost += model_catalog.calculate_completion_model_cost(
                model_provider=self.model_provider,
                model_name=self.model,
                uncached_input_tokens=uncached_prompt_tokens,
                cached_input_tokens_read=cached_prompt_tokens,
                output_tokens=total_output_tokens,
            )

            return FenicCompletionsResponse(completion=completion_text, logprobs=None)

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