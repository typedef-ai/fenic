"""Client for making batch requests to Ollama's chat completions API.

This client is optimized for Ollama's native batching and parallel processing capabilities.
Key optimizations:
- Leverages Ollama's automatic request batching for the same model
- Uses concurrent connections aligned with OLLAMA_NUM_PARALLEL
- Handles Ollama's built-in queuing and memory management
- Optimized error handling for local model constraints
"""

import json
import logging
import os
from typing import Optional, Union

import ollama

from fenic._inference.ollama.ollama_provider import OllamaModelProvider
from fenic._inference.ollama.ollama_model_manager import OllamaModelManager, OllamaModelInfo
from fenic._inference.model_client import (
    FatalException,
    ModelClient,
    TransientException,
)
from fenic._inference.rate_limit_strategy import (
    RateLimitStrategy,
    TokenEstimate,
)
from fenic._inference.request_utils import generate_completion_request_key
from fenic._inference.token_counter import TiktokenTokenCounter
from fenic._inference.types import (
    FenicCompletionsRequest,
    FenicCompletionsResponse,
    ResponseUsage,
)
from fenic.core._inference.model_catalog import ModelProvider, model_catalog
from fenic.core.metrics import LMMetrics

logger = logging.getLogger(__name__)


class OllamaBatchChatCompletionsClient(
    ModelClient[FenicCompletionsRequest, FenicCompletionsResponse]
):
    """Client for making batch requests to Ollama's chat completions API.

    This client enables communication with local models via Ollama's
    native Python library. It supports standard chat completions and structured output
    where available.
    """

    def __init__(
        self,
        rate_limit_strategy: RateLimitStrategy,
        model: str,
        queue_size: int = 100,
        max_backoffs: int = 10,
        host: Optional[str] = None,
    ):
        """Initialize the Ollama batch chat completions client.

        Args:
            rate_limit_strategy: Strategy for handling rate limits
            model: The model to use (e.g., "qwen3:4b")
            queue_size: Size of the request queue (note: Ollama has its own queue)
            max_backoffs: Maximum number of backoff attempts
            host: Host URL for Ollama server
        """
        # Use tiktoken for token counting as a reasonable approximation for local models
        token_counter = TiktokenTokenCounter(model_name="gpt-3.5-turbo", fallback_encoding="cl100k_base")

        super().__init__(
            model=model,
            model_provider=ModelProvider.OLLAMA,
            model_provider_class=OllamaModelProvider(host=host),
            rate_limit_strategy=rate_limit_strategy,
            queue_size=queue_size,
            max_backoffs=max_backoffs,
            token_counter=token_counter,
        )

        self._model_parameters = model_catalog.get_completion_model_parameters(
            ModelProvider.OLLAMA, model
        )
        self._host = host or "http://localhost:11434"
        self._metrics = LMMetrics()

        # Ollama-specific optimizations
        self._ollama_parallel = int(os.getenv("OLLAMA_NUM_PARALLEL", "4"))
        self._ollama_max_queue = int(os.getenv("OLLAMA_MAX_QUEUE", "512"))
        self._model_manager = OllamaModelManager(host=self._host)
        self._model_info: Optional[OllamaModelInfo] = None

        logger.debug(f"Ollama client initialized for model '{model}' with parallelism={self._ollama_parallel}")

    async def _ensure_model_info(self) -> Optional[OllamaModelInfo]:
        """Lazy load model information for dynamic configuration."""
        if self._model_info is None:
            self._model_info = await self._model_manager.get_model_info(self.model)
            if self._model_info:
                logger.debug(f"Loaded model info for '{self.model}': "
                           f"context_length={self._model_info.context_length}, "
                           f"params={self._model_info.parameter_count}, "
                           f"arch={self._model_info.architecture}")
        return self._model_info

    async def make_single_request(
        self, request: FenicCompletionsRequest
    ) -> Union[None, FenicCompletionsResponse, TransientException, FatalException]:
        """Make a single request to the Ollama API.

        Args:
            request: The request to make

        Returns:
            The response from the API or an exception
        """
        try:
            # Ensure model info is loaded for dynamic configuration
            model_info = await self._ensure_model_info()

            # Check if model is available, auto-pull if needed
            if not await self._model_manager.ensure_model_available(self.model):
                return FatalException(Exception(f"Model '{self.model}' could not be loaded or pulled"))

            # Create async client for this request
            client = ollama.AsyncClient(host=self._host)

            # Build the request parameters
            common_params = {
                "model": self.model,
                "messages": request.messages.to_message_list(),
            }

            # Add options with dynamic configuration based on model metadata
            options = {}

            # Disable thinking tokens for cleaner output
            options['think'] = False

            # Use model-specific max output tokens if available
            if request.max_completion_tokens is not None:
                options["num_predict"] = request.max_completion_tokens
            elif model_info:
                # Use model-specific recommended max output tokens
                options["num_predict"] = model_info.get_max_output_tokens()

            if request.temperature is not None:
                options["temperature"] = request.temperature

            # Add context length optimization
            if model_info and model_info.context_length:
                options["num_ctx"] = model_info.context_length

            if options:
                common_params["options"] = options

            # Handle structured output using Ollama's JSON mode
            if request.structured_output:
                # Use Ollama's format parameter for JSON mode
                common_params["format"] = "json"

                # Add schema to the system message
                messages = common_params["messages"]
                if messages and messages[0].get("role") == "system":
                    system_msg_content = messages[0]["content"]
                    schema_prompt = f"\n\nPlease respond with valid JSON that matches this schema:\n{json.dumps(request.structured_output.json_schema, indent=2)}"
                    messages[0]["content"] = system_msg_content + schema_prompt
                else:
                    # Add a system message if none exists
                    schema_prompt = f"Please respond with valid JSON that matches this schema:\n{json.dumps(request.structured_output.json_schema, indent=2)}"
                    messages.insert(0, {"role": "system", "content": schema_prompt})

            # Also detect if the prompt is asking for JSON and enable JSON mode
            elif "json" in str(request.messages.system).lower() or "json" in str(request.messages.user).lower():
                # Enable JSON mode for prompts that explicitly ask for JSON
                common_params["format"] = "json"

            # Make the API call
            response = await client.chat(**common_params)

            # Extract the completion text safely
            completion_text = ""
            if "message" in response and "content" in response["message"]:
                completion_text = response["message"]["content"] or ""

            # Handle usage information - Ollama typically doesn't provide detailed token counts
            response_usage = None
            if "eval_count" in response or "prompt_eval_count" in response:
                prompt_tokens = response.get("prompt_eval_count", 0)
                completion_tokens = response.get("eval_count", 0)
                total_tokens = prompt_tokens + completion_tokens

                response_usage = ResponseUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    cached_tokens=0,  # Local models don't typically have caching
                    thinking_tokens=0,  # Local models don't have separate thinking tokens
                )

                # Update metrics
                self._metrics.num_uncached_input_tokens += response_usage.prompt_tokens
                self._metrics.num_output_tokens += response_usage.completion_tokens
            else:
                # Fallback: estimate tokens using our counter
                prompt_tokens = self.token_counter.count_tokens(request.messages)
                completion_tokens = len(completion_text.split()) * 1.3  # Rough approximation

                response_usage = ResponseUsage(
                    prompt_tokens=int(prompt_tokens),
                    completion_tokens=int(completion_tokens),
                    total_tokens=int(prompt_tokens + completion_tokens),
                    cached_tokens=0,
                    thinking_tokens=0,
                )

            self._metrics.num_requests += 1
            # Local models are typically free, so cost is 0
            self._metrics.cost += 0.0

            # Debug logging for structured output
            if request.structured_output and completion_text:
                logger.debug(f"Ollama structured response: {completion_text[:200]}...")
                # Validate JSON format
                try:
                    json.loads(completion_text)
                    logger.debug("JSON validation successful")
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON from Ollama: {e}")
                    # Try to clean up common JSON issues
                    completion_text = self._clean_json_response(completion_text)

            return FenicCompletionsResponse(
                completion=completion_text,
                logprobs=None,  # Ollama doesn't typically support logprobs
                usage=response_usage,
            )

        except ollama.ResponseError as e:
            # Handle Ollama-specific errors optimized for local model behavior
            if e.status_code == 404:
                # Model not found - attempt auto-pull if configured, otherwise fatal
                logger.error(f"Model '{self.model}' not found. Try running: ollama pull {self.model}")
                # Could add auto-pull logic here in the future
                return FatalException(e)
            elif e.status_code == 503:
                # Server overloaded - Ollama's queue is full (OLLAMA_MAX_QUEUE exceeded)
                logger.warning(f"Ollama server overloaded (queue full: {self._ollama_max_queue}): {e}")
                return TransientException(e)
            elif e.status_code == 429:
                # Rate limited - respect Ollama's internal rate limiting
                logger.warning(f"Ollama rate limit exceeded (parallel limit: {self._ollama_parallel}): {e}")
                return TransientException(e)
            elif e.status_code == 500:
                # Internal server error - often memory issues with local models
                logger.warning(f"Ollama internal error (possibly memory constrained): {e}")
                return TransientException(e)
            else:
                logger.error(f"Ollama API error: {e}")
                return FatalException(e)
        except Exception as e:
            # Handle other errors optimized for local model scenarios
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in ["connection", "timeout", "network"]):
                logger.warning(f"Connection/timeout error with Ollama: {e}")
                return TransientException(e)
            elif any(keyword in error_str for keyword in ["memory", "cuda", "out of memory"]):
                logger.warning(f"Memory error with Ollama (model may need unloading): {e}")
                return TransientException(e)
            else:
                logger.error(f"Fatal error with Ollama request: {e}")
                return FatalException(e)

    def get_request_key(self, request: FenicCompletionsRequest) -> str:
        """Generate a unique key for request deduplication.

        Args:
            request: The request to generate a key for

        Returns:
            A unique key for the request
        """
        return generate_completion_request_key(request)

    def estimate_tokens_for_request(self, request: FenicCompletionsRequest) -> TokenEstimate:
        """Estimate the number of tokens for a request.

        Args:
            request: The request to estimate tokens for

        Returns:
            TokenEstimate: The estimated token usage
        """
        input_tokens = self.token_counter.count_tokens(request.messages)
        output_tokens = request.max_completion_tokens

        return TokenEstimate(
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )

    def reset_metrics(self):
        """Reset all metrics to their initial values."""
        self._metrics = LMMetrics()

    def get_metrics(self) -> LMMetrics:
        """Get the current metrics.

        Returns:
            The current metrics
        """
        return self._metrics

    def _get_max_output_tokens(self, request: FenicCompletionsRequest) -> int:
        """Get the maximum output tokens for a request."""
        return request.max_completion_tokens

    def _clean_json_response(self, response_text: str) -> str:
        """Clean up common JSON formatting issues from local LLM responses."""
        import re

        # Remove markdown code blocks
        response_text = re.sub(r'```json\s*', '', response_text)
        response_text = re.sub(r'```\s*$', '', response_text)

        # Remove any text before the first {
        if '{' in response_text:
            start = response_text.find('{')
            response_text = response_text[start:]

        # Remove any text after the last }
        if '}' in response_text:
            end = response_text.rfind('}') + 1
            response_text = response_text[:end]

        # Fix common JSON issues
        response_text = response_text.strip()

        # Try to validate and return
        try:
            json.loads(response_text)
            return response_text
        except json.JSONDecodeError:
            logger.warning(f"Could not clean JSON response: {response_text[:100]}...")
            return response_text