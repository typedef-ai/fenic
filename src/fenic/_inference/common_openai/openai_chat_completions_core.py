"""Core functionality for OpenAI chat completions clients."""

import logging
from typing import Any, Optional, Union

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    NotFoundError,
    OpenAIError,
    RateLimitError,
)
from openai.types import CompletionUsage

from fenic._inference.common_openai.openai_profile_manager import (
    OpenAICompletionProfileConfiguration,
)
from fenic._inference.common_openai.openai_utils import convert_messages
from fenic._inference.common_openai.utils import handle_openai_compatible_response
from fenic._inference.model_client import (
    FatalException,
    TransientException,
)
from fenic._inference.request_utils import generate_completion_request_key
from fenic._inference.token_counter import TokenCounter
from fenic._inference.types import (
    FenicCompletionsRequest,
    FenicCompletionsResponse,
    ResponseUsage,
)
from fenic.core._inference.model_catalog import (
    ModelProvider,
    model_catalog,
)
from fenic.core.metrics import LMMetrics

logger = logging.getLogger(__name__)


class OpenAIChatCompletionsCore:
    """Core functionality for OpenAI chat completions clients."""

    def __init__(
        self,
        model: str,
        model_provider: ModelProvider,
        token_counter: TokenCounter,
        client: AsyncOpenAI,
    ):
        """Initialize the OpenAI chat completions client core.

        Args:
            model: The model to use
            model_provider: The provider of the model
            token_counter: Counter for estimating token usage
            client: The OpenAI client
            additional_params: Additional parameters to pass to the API, e.g. {"reasoning_effort": "none"} for thinking models.
        """
        self._model = model
        self._model_provider = model_provider
        self._token_counter = token_counter
        self._client = client
        self._metrics = LMMetrics()
        self._model_parameters = model_catalog.get_completion_model_parameters(self._model_provider, self._model)
        self._model_identifier = f"{model_provider.value}:{model}"

    def reset_metrics(self) -> None:
        """Reset the metrics."""
        self._metrics = LMMetrics()

    def get_metrics(self) -> LMMetrics:
        """Get the metrics."""
        return self._metrics

    async def make_single_request(
        self,
        request: FenicCompletionsRequest,
        profile_configuration: Optional[OpenAICompletionProfileConfiguration] = None
    ) -> Union[None, FenicCompletionsResponse, TransientException, FatalException]:
        """Make a single request to the OpenAI API.

        Args:
            request: The messages to send
            profile_configuration: The optional profile configuration for the request (for passing reasoning_effort and verbosity)
        Returns:
            The response text or an exception
        """
        try:
            common_params: dict[str, Any] = {
                "model": self._model,
                "messages": convert_messages(request.messages),
                "n": 1,
            }

            max_completion_tokens = self.get_max_output_token_request_limit(request, profile_configuration)
            if max_completion_tokens is not None:
                common_params["max_completion_tokens"] = max_completion_tokens

            reasoning_effort = profile_configuration.additional_parameters.get("reasoning_effort") if profile_configuration else None
            # Temperature is only allowed when reasoning_effort is 'none' for models that support it
            if request.temperature:
                if self._model_parameters.supports_reasoning and reasoning_effort != "none":
                    logger.warning(
                        f"Model {self._model} does not support custom temperature when reasoning is enabled.  Ignoring temperature parameter."
                    )
                else:
                    common_params.update({"temperature": request.temperature})

            # Determine if we need logprobs
            if request.top_logprobs:
                if self._model_parameters.supports_reasoning and reasoning_effort != "none":
                    logger.warning(
                        f"Model {self._model} does not support logprobs when reasoning is enabled.  Ignoring logprobs parameter."
                    )
                else:
                    common_params.update(
                        {
                            "logprobs": True,
                            "top_logprobs": request.top_logprobs,
                        }
                    )
            if profile_configuration:
                common_params.update(profile_configuration.additional_parameters)

            # Choose between parse and create based on structured_output
            if request.structured_output:
                common_params["response_format"] = request.structured_output.pydantic_model
                response = await self._client.beta.chat.completions.parse(**common_params)
            else:
                response = await self._client.chat.completions.create(**common_params)

            completion_choice, err = handle_openai_compatible_response(
                model_provider=self._model_provider,
                model_name=self._model,
                request=request,
                response=response,
                request_key_generator=generate_completion_request_key,
            )
            if err is not None:
                return err

            # Extract usage metrics
            usage: CompletionUsage = response.usage

            cached_input_tokens = (
                usage.prompt_tokens_details.cached_tokens
                if usage.prompt_tokens_details
                else 0
            )
            uncached_input_tokens = usage.prompt_tokens - cached_input_tokens
            total_prompt_tokens = usage.prompt_tokens

            # Extract reasoning (thinking) tokens if available
            reasoning_tokens = (
                usage.completion_tokens_details.reasoning_tokens
                if usage.completion_tokens_details
                else 0
            )

            # Separate completion tokens from reasoning tokens
            total_output_tokens = usage.completion_tokens
            completion_tokens = total_output_tokens - reasoning_tokens

            # Create ResponseUsage object
            response_usage = ResponseUsage(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=completion_tokens,  # Actual completion tokens (excluding reasoning)
                total_tokens=total_prompt_tokens + total_output_tokens,
                cached_tokens=cached_input_tokens,
                thinking_tokens=reasoning_tokens,  # OpenAI's reasoning tokens
            )

            # Update metrics (existing logic)
            self._metrics.num_cached_input_tokens += cached_input_tokens
            self._metrics.num_uncached_input_tokens += uncached_input_tokens
            self._metrics.num_output_tokens += total_output_tokens
            self._metrics.num_requests += 1

            self._metrics.cost += model_catalog.calculate_completion_model_cost(
                model_provider=self._model_provider,
                model_name=self._model,
                uncached_input_tokens=uncached_input_tokens,
                cached_input_tokens_read=cached_input_tokens,
                output_tokens=total_output_tokens,
            )
            completion = completion_choice.message.content
            if completion is None:
                logger.warning(
                    f"[{self._model_provider.value}:{self._model}] returned None for completion for {self.get_request_key(request)}: {response}"
                )
            return FenicCompletionsResponse(
                completion=completion_choice.message.content,
                logprobs=completion_choice.logprobs,
                usage=response_usage,
            )

        except (APITimeoutError, APIConnectionError) as e:
            return TransientException(e)

        except RateLimitError as e:
            if e.response and e.response.json()["error"]["type"] == "insufficient_quota":
                logger.error(f"Insufficient quota on {self._model_provider.value} provider: {e}")
                return FatalException(e)
            return TransientException(e)

        except NotFoundError as e:
            # During our CI tests, where we run a larger set of tests, we've seen an intermittent 404 error
            # that is not coming from OpenAI.
            # Usually 404 errors should be considered as fatal, as there is no use in retrying them,
            # in this case some of the requests have succeeded, so we're marking this as transient and as such
            # the request will be retried.
            if e.response.headers.get("openai-processing-ms") == "0":
                logger.error("404 with zero processing time → likely routing glitch - Marking this as transient.")
                return TransientException(e)
            else:
                return FatalException(e)

        except OpenAIError as e:
            return FatalException(e)

    def get_request_key(self, request: FenicCompletionsRequest) -> str:
        """Generate a unique key for request deduplication.

        Args:
            request: The request to generate a key for

        Returns:
            A unique key for the request
        """
        return generate_completion_request_key(request)

    def get_max_output_token_request_limit(self, request: FenicCompletionsRequest, profile_config:OpenAICompletionProfileConfiguration) -> Optional[int]:
        """Return the maximum output token limit for a request.

        Returns None if max_completion_tokens is not provided (no limit should be set).
        If max_completion_tokens is provided, includes the thinking token budget with a safety margin.
        """
        if request.max_completion_tokens is None:
            return None
        return request.max_completion_tokens + profile_config.expected_additional_reasoning_tokens
