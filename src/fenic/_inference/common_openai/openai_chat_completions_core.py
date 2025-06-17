"""Core functionality for OpenAI chat completions clients."""

import hashlib
import json
import logging
from typing import Union

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    OpenAIError,
    RateLimitError,
)
from openai.types import CompletionUsage

from fenic._inference.model_catalog import (
    ModelProvider,
    model_catalog,
)
from fenic._inference.model_client import (
    FatalException,
    FenicCompletionsRequest,
    FenicCompletionsResponse,
    TokenEstimate,
    TransientException,
)
from fenic._inference.token_counter import TokenCounter
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
            additional_params: dict = None,
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
        if additional_params is None:
            self._additional_params = {}
        else:
            self._additional_params = additional_params

    def reset_metrics(self) -> None:
        """Reset the metrics."""
        self._metrics = LMMetrics()

    def get_metrics(self) -> LMMetrics:
        """Get the metrics."""
        return self._metrics

    async def make_single_request(
            self, request: FenicCompletionsRequest
    ) -> Union[None, FenicCompletionsResponse, TransientException, FatalException]:
        """Make a single request to the OpenAI API.

        Args:
            request: The messages to send

        Returns:
            The response text or an exception
        """
        try:
            common_params = {
                "model": self._model,
                "messages": request.messages.to_message_list(),
                "max_tokens": request.max_completion_tokens,
                "temperature": request.temperature,
                "n": 1,
            }

            # Determine if we need logprobs
            if request.top_logprobs:
                common_params.update(
                    {
                        "logprobs": True,
                        "top_logprobs": request.top_logprobs,
                    }
                )
            common_params.update(self._additional_params)

            # Choose between parse and create based on structured_output
            if request.structured_output:
                common_params["response_format"] = request.structured_output
                response = await self._client.beta.chat.completions.parse(
                    **common_params
                )
                if response.choices[0].message.refusal:
                    return None
            else:
                response = await self._client.chat.completions.create(**common_params)

            usage: CompletionUsage = response.usage

            input_tokens = (
                usage.prompt_tokens_details.cached_tokens
                if usage.prompt_tokens_details
                else 0
            )
            uncached_input_tokens = usage.prompt_tokens - input_tokens
            output_tokens = usage.completion_tokens

            self._metrics.num_cached_input_tokens += input_tokens
            self._metrics.num_uncached_input_tokens += uncached_input_tokens
            self._metrics.num_output_tokens += output_tokens
            self._metrics.num_requests += 1

            self._metrics.cost += model_catalog.calculate_completion_model_cost(
                model_provider=self._model_provider,
                model_name=self._model,
                uncached_input_tokens=uncached_input_tokens,
                cached_input_tokens_read=input_tokens,
                output_tokens=output_tokens,
            )
            completion = response.choices[0].message.content
            if completion is None:
                logger.warning(
                    f"[{self._model_provider.value}:{self._model}] returned None for completion for {self.get_request_key(request)}: {response}")
            return FenicCompletionsResponse(
                completion=response.choices[0].message.content,
                logprobs=response.choices[0].logprobs,
            )

        except (RateLimitError, APITimeoutError, APIConnectionError) as e:
            return TransientException(e)

        except OpenAIError as e:
            return FatalException(e)

    def get_request_key(self, request: FenicCompletionsRequest) -> str:
        """Generate a unique key for request deduplication.

        Args:
            request: The request to generate a key for

        Returns:
            A unique key for the request
        """
        return hashlib.sha256(json.dumps(request.messages.to_message_list()).encode()).hexdigest()[:10]

    def estimate_tokens_for_request(self, request: FenicCompletionsRequest) -> TokenEstimate:
        """Estimate the number of tokens for a request.

        Args:
            request: The request to estimate tokens for

        Returns:
            TokenEstimate with input token count
        """
        return TokenEstimate(
            input_tokens=self._token_counter.count_tokens(request.messages.to_message_list()),
            output_tokens=request.max_completion_tokens
        )
