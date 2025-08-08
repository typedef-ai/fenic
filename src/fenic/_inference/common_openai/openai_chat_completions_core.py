"""Core functionality for OpenAI chat completions clients."""

import logging
from typing import Any, Optional, Union

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    OpenAIError,
    RateLimitError,
)
from openai.types import CompletionUsage

from fenic._inference.common_openai.openai_profile_manager import (
    OpenAICompletionProfileConfiguration,
)
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
from fenic.core._utils.json_schema_utils import (
    deep_copy_json,
    strip_defaults_in_place,
    strip_schema_metadata_in_place,
    make_nullable as util_make_nullable,
)

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

    @staticmethod
    def _strictify_schema_for_openai(schema: dict[str, Any]) -> dict[str, Any]:
        """Produce a strict OpenAI-compatible schema:
        - additionalProperties: false on all objects
        - required = all property keys
        - optional properties allow null
        """
        def make_nullable(node: dict[str, Any]) -> dict[str, Any]:
            return util_make_nullable(node)

        def walk(node: Any) -> Any:
            if not isinstance(node, dict):
                return node
            t = node.get("type")
            if t == "object":
                node.setdefault("additionalProperties", False)
                props = node.get("properties", {})
                if isinstance(props, dict):
                    original_required = set(node.get("required", []))
                    for k, v in list(props.items()):
                        props[k] = walk(v)
                    keys = list(props.keys())
                    node["required"] = keys
                    for k in keys:
                        if k not in original_required:
                            props[k] = make_nullable(props[k])
            elif t == "array":
                items = node.get("items")
                if isinstance(items, dict):
                    node["items"] = walk(items)
            for key in ("allOf", "anyOf", "oneOf"):
                if key in node and isinstance(node[key], list):
                    node[key] = [walk(s) for s in node[key]]
            for defs_key in ("$defs", "definitions"):
                defs = node.get(defs_key)
                if isinstance(defs, dict):
                    for dk, dv in list(defs.items()):
                        defs[dk] = walk(dv)
            return node

        # Start from a deep copy once, then mutate in place
        base = deep_copy_json(schema)
        strip_schema_metadata_in_place(base)
        base = walk(base)
        strip_defaults_in_place(base)
        return base

    async def make_single_request(
        self,
        request: FenicCompletionsRequest,
        profile_configuration: Optional[OpenAICompletionProfileConfiguration] = None
    ) -> Union[None, FenicCompletionsResponse, TransientException, FatalException]:
        """Make a single request to the OpenAI API.

        Args:
            request: The messages to send
            profile_configuration: The optional profile configuration for the request (for passing reasoning_effort)
        Returns:
            The response text or an exception
        """
        try:
            common_params: dict[str, Any] = {
                "model": self._model,
                "messages": request.messages.to_message_list(),
                "max_completion_tokens": request.max_completion_tokens + profile_configuration.expected_additional_reasoning_tokens,
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
            if profile_configuration:
                common_params.update(profile_configuration.additional_parameters)

            # Choose between parse and create based on structured_output
            if request.structured_output:
                # Build strict schema for OpenAI parse from the provided schema
                common_params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "fenic_response",
                        "schema": request.structured_output.strict_schema,
                        "strict": True,
                    },
                }
                response = await self._client.beta.chat.completions.parse(
                    **common_params
                )
                if response.choices[0].message.refusal:
                    return None
            else:
                response = await self._client.chat.completions.create(**common_params)

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
                thinking_tokens=reasoning_tokens  # OpenAI's reasoning tokens
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
            completion = response.choices[0].message.content
            if completion is None:
                logger.warning(
                    f"[{self._model_provider.value}:{self._model}] returned None for completion for {self.get_request_key(request)}: {response}")
            return FenicCompletionsResponse(
                completion=response.choices[0].message.content,
                logprobs=response.choices[0].logprobs,
                usage=response_usage,
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
        return generate_completion_request_key(request)
