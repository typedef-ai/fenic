import logging
from functools import cache
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from fenic._inference.cache.protocol import LLMResponseCache

from google.genai.errors import ClientError, ServerError
from google.genai.types import (
    FinishReason,
    GenerateContentConfigDict,
    GenerateContentResponse,
    MediaResolution,
)

from fenic._inference.google.gemini_token_counter import GeminiLocalTokenCounter
from fenic._inference.google.google_profile_manager import (
    GoogleCompletionsProfileConfig,
    GoogleCompletionsProfileManager,
)
from fenic._inference.google.google_provider import (
    GoogleDeveloperModelProvider,
    GoogleVertexModelProvider,
)
from fenic._inference.google.google_utils import (
    convert_messages_and_upload_files,
    delete_file,
)
from fenic._inference.model_client import (
    FatalException,
    ModelClient,
    TransientException,
)
from fenic._inference.profile_hash_mixin import ProfileHashMixin
from fenic._inference.rate_limit_strategy import (
    TokenEstimate,
    UnifiedTokenRateLimitStrategy,
)
from fenic._inference.token_counter import Tokenizable
from fenic._inference.types import (
    FenicCompletionsRequest,
    FenicCompletionsResponse,
    ResponseUsage,
)
from fenic.core._inference.model_catalog import (
    ModelProvider,
    model_catalog,
)
from fenic.core._logical_plan.resolved_types import ResolvedResponseFormat
from fenic.core._resolved_session_config import ResolvedGoogleModelProfile
from fenic.core.error import ExecutionError
from fenic.core.metrics import LMMetrics

logger = logging.getLogger(__name__)


class GeminiNativeChatCompletionsClient(
    ProfileHashMixin, ModelClient[FenicCompletionsRequest, FenicCompletionsResponse]
):
    """Native (google-genai) Google Gemini chat-completions client.

    This client handles communication with Google's Gemini models using the native
    google-genai library. It supports both standard and Vertex AI environments,
    thinking/reasoning capabilities, structured output, and comprehensive token
    tracking.

    """

    def __init__(
        self,
        rate_limit_strategy: UnifiedTokenRateLimitStrategy,
        model_provider: ModelProvider,
        model: str,
        queue_size: int = 100,
        max_backoffs: int = 10,
        profiles: Optional[dict[str, ResolvedGoogleModelProfile]] = None,
        default_profile_name: Optional[str] = None,
        cache: Optional["LLMResponseCache"] = None,
    ):
        """Initialize the Gemini native chat completions client.

        Args:
            rate_limit_strategy: Strategy for rate limiting requests
            model_provider: Google model provider (Developer or Vertex AI)
            model: Gemini model name to use
            queue_size: Maximum size of the request queue
            max_backoffs: Maximum number of retry backoffs
            profiles: Dictionary of profile configurations
            default_profile_name: Name of the default profile to use
            cache: Optional LLM response cache
        """
        token_counter = GeminiLocalTokenCounter(model_name=model)
        super().__init__(
            model=model,
            model_provider=model_provider,
            model_provider_class=GoogleDeveloperModelProvider()
            if model_provider == ModelProvider.GOOGLE_DEVELOPER
            else GoogleVertexModelProvider(),
            rate_limit_strategy=rate_limit_strategy,
            queue_size=queue_size,
            max_backoffs=max_backoffs,
            token_counter=token_counter,
            cache=cache,
        )

        self._client = self.model_provider_class.create_aio_client()
        self._metrics = LMMetrics()
        self._token_counter = token_counter  # For type checkers
        self._model_parameters = model_catalog.get_completion_model_parameters(
            model_provider, model
        )

        self._profile_manager = GoogleCompletionsProfileManager(
            model_parameters=self._model_parameters,
            profile_configurations=profiles,
            default_profile_name=default_profile_name,
        )


    def reset_metrics(self):
        """Reset metrics to initial state."""
        self._metrics = LMMetrics()

    def get_metrics(self) -> LMMetrics:
        """Get current metrics.

        Returns:
            Current language model metrics
        """
        return self._metrics

    def count_tokens(self, messages: Tokenizable, ignore_file: bool = False) -> int:  # type: ignore[override]
        """Count tokens in messages.

        Re-exposes the parent implementation for type checking.

        Args:
            messages: Messages to count tokens for

        Returns:
            Token count
        """
        # Re-expose for mypy – same implementation as parent.
        return super().count_tokens(messages, ignore_file=ignore_file)

    def estimate_tokens_for_request(self, request: FenicCompletionsRequest):
        """Estimate the number of tokens for a request.

        If the request provides a max_completion_tokens value, use that.  Otherwise, estimate the output tokens based on the file size.

        Args:
            request: The request to estimate tokens for

        Returns:
            TokenEstimate: The estimated token usage
        """
        # Count text tokens (excluding file) so we can handle file tokens separately with media_resolution
        input_tokens = self.count_tokens(request.messages, ignore_file=True)

        # Add file tokens with media_resolution from profile if there's a file
        if request.messages.user_file:
            profile_config = self._profile_manager.get_profile_by_name(request.model_profile)
            input_tokens += self._token_counter.count_file_input_tokens(
                request.messages, media_resolution=profile_config.media_resolution
            )

        input_tokens += self._count_auxiliary_input_tokens(request)
        output_tokens = self._estimate_output_tokens(request)
        return TokenEstimate(input_tokens=input_tokens, output_tokens=output_tokens)

    async def make_single_request(
        self, request: FenicCompletionsRequest
    ) -> Union[None, FenicCompletionsResponse, TransientException, FatalException]:
        """Make a single completion request to Google Gemini.

        Handles both text and structured output requests, with support for
        thinking/reasoning when enabled. Processes responses and extracts
        comprehensive usage metrics including thinking tokens.

        Args:
            request: The completion request to process

        Returns:
            Completion response, transient exception, or fatal exception
        """

        profile_config = self._profile_manager.get_profile_by_name(
            request.model_profile
        )
        generation_config: GenerateContentConfigDict = {
            "temperature": request.temperature,
            "response_logprobs": request.top_logprobs is not None,
            "logprobs": request.top_logprobs,
            "system_instruction": request.messages.system,
        }

        max_output_tokens = self._get_max_output_token_request_limit(request)
        if max_output_tokens is not None:
            generation_config["max_output_tokens"] = max_output_tokens

        generation_config.update(profile_config.additional_generation_config)

        # Add media_resolution from profile if specified (for PDF processing)
        if profile_config.media_resolution is not None:
            media_resolution_map = {
                "low": MediaResolution.MEDIA_RESOLUTION_LOW,
                "medium": MediaResolution.MEDIA_RESOLUTION_MEDIUM,
                "high": MediaResolution.MEDIA_RESOLUTION_HIGH,
            }
            generation_config["media_resolution"] = media_resolution_map[profile_config.media_resolution]

        if request.structured_output is not None:
            generation_config.update(
                response_mime_type="application/json",
                response_schema=request.structured_output.pydantic_model,
            )

        file_obj = None
        try:
            # Build generation parameters and upload any files to Google's file store
            contents, file_obj = await convert_messages_and_upload_files(
                self._client, request.messages
            )

            response: GenerateContentResponse = (
                await self._client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=generation_config,
                )
            )
            candidate = response.candidates[0] if response.candidates else None
            completion_text = ""
            if candidate is None:
                return FatalException(ExecutionError("No candidate found in response"))
            if candidate.content is None or candidate.content.parts is None:
                if candidate.finish_reason == FinishReason.MAX_TOKENS:
                    return FatalException(
                        ExecutionError(
                            "Candidate generation failed due to max tokens limit. "
                            "Consider retrying the request with a higher max_completion_tokens value. "
                            "If using a reasoning model, consider increasing the thinking_token_budget. "
                            f"Current max_completions_tokens: {request.max_completion_tokens}, Current thinking_token_budget: {profile_config.thinking_token_budget}."
                        )
                    )
                else:
                    return FatalException(
                        ExecutionError(
                            f"Candidate generation failed due to stop reason: {candidate.finish_reason}"
                        )
                    )
            else:
                for part in candidate.content.parts:
                    if not part.thought:
                        completion_text = part.text

            if candidate.finish_reason != FinishReason.STOP:
                logger.warning(
                    f"Candidate generation for request {self.get_request_key(request)} was truncated for stop reason {candidate.finish_reason}"
                )

            # Extract usage metrics
            usage_metadata = response.usage_metadata
            response_usage = None

            self._metrics.num_requests += 1
            if usage_metadata:
                total_prompt_tokens = (
                    usage_metadata.prompt_token_count
                    if usage_metadata.prompt_token_count
                    else 0
                )
                cached_prompt_tokens = (
                    usage_metadata.cached_content_token_count
                    if usage_metadata.cached_content_token_count
                    else 0
                )
                uncached_prompt_tokens = total_prompt_tokens - cached_prompt_tokens
                candidates_token_count = (
                    usage_metadata.candidates_token_count
                    if usage_metadata.candidates_token_count
                    else 0
                )
                thinking_tokens_count = (
                    usage_metadata.thoughts_token_count
                    if usage_metadata.thoughts_token_count
                    else 0
                )
                total_output_tokens = candidates_token_count + thinking_tokens_count

                # Create ResponseUsage object
                response_usage = ResponseUsage(
                    prompt_tokens=total_prompt_tokens,
                    completion_tokens=candidates_token_count,  # Google separates completion from thinking
                    total_tokens=total_prompt_tokens + total_output_tokens,
                    cached_tokens=cached_prompt_tokens,
                    thinking_tokens=thinking_tokens_count,
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

            return FenicCompletionsResponse(
                completion=completion_text, logprobs=None, usage=response_usage
            )

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
        finally:
            if file_obj:
                await delete_file(self._client, file_obj.name)

    @cache  # noqa: B019 – builtin cache OK here.
    def _estimate_response_schema_tokens(
        self, response_format: ResolvedResponseFormat
    ) -> int:
        """Estimate token count for a response format schema.

        Uses Google's tokenizer to count tokens in a JSON schema representation
        of the response format. Results are cached for performance.

        Args:
            response_format: Pydantic model class defining the response format

        Returns:
            Estimated token count for the response format
        """
        schema_str = response_format.schema_fingerprint
        return self._token_counter.count_tokens(schema_str)

    def _estimate_structured_output_overhead(
        self, response_format: ResolvedResponseFormat
    ) -> int:
        """Use Google-specific response schema token estimation.

        Args:
            response_format: Pydantic model class defining the response format

        Returns:
            Estimated token overhead for structured output
        """
        return self._estimate_response_schema_tokens(response_format)

    def _estimate_output_tokens(self, request: FenicCompletionsRequest) -> int:
        """Estimate the number of output tokens for a request."""
        estimated_output_tokens = request.max_completion_tokens or 0
        if request.max_completion_tokens is None and request.messages.user_file:
            # TODO(DY): the semantic operator should dictate how the file affects the token estimate
            estimated_output_tokens = self.token_counter.count_file_output_tokens(
                request.messages
            )
        return estimated_output_tokens + self._get_expected_additional_reasoning_tokens(
            request
        )

    def _get_max_output_token_request_limit(
        self, request: FenicCompletionsRequest
    ) -> Optional[int]:
        """Get the upper limit of output tokens for a request.

        Returns None if max_completion_tokens is not provided (no limit should be set).
        If max_completion_tokens is provided, includes the thinking token budget with a safety margin."""
        if request.max_completion_tokens is None:
            return None
        return (
            request.max_completion_tokens
            + self._get_expected_additional_reasoning_tokens(request)
        )

    def _get_expected_additional_reasoning_tokens(
        self, request: FenicCompletionsRequest
    ) -> int:
        """Get the expected additional reasoning tokens for a request.  Include a safety margin."""
        profile_config = self._profile_manager.get_profile_by_name(
            request.model_profile
        )
        return int(1.5 * profile_config.thinking_token_budget)

    def _resolve_profile_for_hash(self, profile_name: Optional[str]) -> GoogleCompletionsProfileConfig:
        return self._profile_manager.get_profile_by_name(profile_name)