"""Client for making batch requests to OpenAI's chat completions API."""

from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from fenic._inference.cache.protocol import LLMResponseCache

from fenic._inference.common_openai.openai_chat_completions_core import (
    OpenAIChatCompletionsCore,
)
from fenic._inference.common_openai.openai_profile_manager import (
    OpenAICompletionProfileConfiguration,
    OpenAICompletionsProfileManager,
)
from fenic._inference.model_client import (
    FatalException,
    ModelClient,
    TransientException,
)
from fenic._inference.openai.openai_provider import OpenAIModelProvider
from fenic._inference.profile_hash_mixin import ProfileHashMixin
from fenic._inference.rate_limit_strategy import (
    RateLimitStrategy,
    TokenEstimate,
)
from fenic._inference.token_counter import TiktokenTokenCounter
from fenic._inference.types import FenicCompletionsRequest, FenicCompletionsResponse
from fenic.core._inference.model_catalog import ModelProvider, model_catalog
from fenic.core._resolved_session_config import ResolvedOpenAIModelProfile
from fenic.core.metrics import LMMetrics


class OpenAIBatchChatCompletionsClient(
    ProfileHashMixin, ModelClient[FenicCompletionsRequest, FenicCompletionsResponse]
):
    """Client for making batch requests to OpenAI's chat completions API."""

    def __init__(
        self,
        rate_limit_strategy: RateLimitStrategy,
        model: str,
        queue_size: int = 100,
        max_backoffs: int = 10,
        profiles: Optional[dict[str, ResolvedOpenAIModelProfile]] = None,
        default_profile_name: Optional[str] = None,
        cache: Optional["LLMResponseCache"] = None,
    ):
        """Initialize the OpenAI batch chat completions client.

        Args:
            rate_limit_strategy: Strategy for handling rate limits
            queue_size: Size of the request queue
            model: The model to use
            max_backoffs: Maximum number of backoff attempts
            profiles: Dictionary of profile configurations
            default_profile_name: Default profile to use when none specified
            cache: Optional LLM response cache
        """
        token_counter = TiktokenTokenCounter(
            model_name=model, fallback_encoding="o200k_base"
        )
        super().__init__(
            model=model,
            model_provider=ModelProvider.OPENAI,
            model_provider_class=OpenAIModelProvider(),
            rate_limit_strategy=rate_limit_strategy,
            queue_size=queue_size,
            max_backoffs=max_backoffs,
            token_counter=token_counter,
            cache=cache,
        )
        self._model_parameters = model_catalog.get_completion_model_parameters(
            ModelProvider.OPENAI, model
        )
        self._profile_manager = OpenAICompletionsProfileManager(
            model_parameters=self._model_parameters,
            profile_configurations=profiles,
            default_profile_name=default_profile_name,
        )

        self._core = OpenAIChatCompletionsCore(
            model=model,
            model_provider=ModelProvider.OPENAI,
            token_counter=token_counter,
            client=self.model_provider_class.create_aio_client(),
        )



    async def make_single_request(
        self, request: FenicCompletionsRequest
    ) -> Union[None, FenicCompletionsResponse, TransientException, FatalException]:
        """Make a single request to the OpenAI API.

        Args:
            request: The request to make

        Returns:
            The response from the API or an exception
        """
        profile = self._profile_manager.get_profile_by_name(request.model_profile)
        return await self._core.make_single_request(request, profile)

    def estimate_tokens_for_request(
        self, request: FenicCompletionsRequest
    ) -> TokenEstimate:
        """Estimate the number of tokens for a request.

        Args:
            request: The request to estimate tokens for

        Returns:
            TokenEstimate: The estimated token usage
        """
        return TokenEstimate(
            input_tokens=self.token_counter.count_tokens(request.messages),
            output_tokens=self._estimate_output_tokens(request),
        )

    def reset_metrics(self):
        """Reset all metrics to their initial values."""
        self._core.reset_metrics()

    def get_metrics(self) -> LMMetrics:
        """Get the current metrics.

        Returns:
            The current metrics
        """
        return self._core.get_metrics()

    def _estimate_output_tokens(self, request: FenicCompletionsRequest) -> int:
        """Estimate the number of output tokens for a request."""
        base_tokens = request.max_completion_tokens or 0
        if request.max_completion_tokens is None and request.messages.user_file:
            # TODO(DY): the semantic operator should dictate how the file affects the token estimate
            base_tokens += self.token_counter.count_file_output_tokens(
                messages=request.messages
            )
        profile_config = self._profile_manager.get_profile_by_name(
            request.model_profile
        )
        return base_tokens + profile_config.expected_additional_reasoning_tokens

    def _get_max_output_token_request_limit(
        self, request: FenicCompletionsRequest
    ) -> int:
        """Return the maximum output token limit for a request.

        For file parsing requests, use a guardrail limit of 8192 tokens (the lowest output limit of a VLM model we support).

        Include the thinking token budget with a safety margin.
        """
        profile_config = self._profile_manager.get_profile_by_name(
            request.model_profile
        )
        return self._core.get_max_output_token_request_limit(request, profile_config)

    def _resolve_profile_for_hash(self, profile_name: Optional[str]) -> OpenAICompletionProfileConfiguration:
        return self._profile_manager.get_profile_by_name(profile_name)