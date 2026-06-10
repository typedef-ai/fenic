from typing import Callable, Optional, Union

from openai import RateLimitError
from openai.types.chat import ChatCompletion, ParsedChatCompletion, ParsedChoice
from openai.types.chat.chat_completion import Choice

from fenic._inference.model_client import FatalException, TransientException
from fenic._inference.types import FenicCompletionsRequest
from fenic.core._inference.model_catalog import ModelProvider
from fenic.core.error import ExecutionError


def is_insufficient_quota_error(error: RateLimitError) -> bool:
    """Return True only for a 429 that indicates an exhausted account quota.

    An ``insufficient_quota`` 429 cannot be resolved by retrying, so callers treat it
    as fatal. This inspection is hardened against malformed or non-OpenAI 429 bodies
    (proxies, gateways, unread streaming bodies): any failure to read or parse the
    body returns False so the caller falls back to the safe default of treating the
    429 as a transient (retryable) rate limit instead of letting the inspection raise.
    """
    try:
        response = getattr(error, "response", None)
        if response is None:
            return False
        body = response.json()
    except Exception:
        return False
    error_obj = body.get("error") if isinstance(body, dict) else None
    return isinstance(error_obj, dict) and error_obj.get("type") == "insufficient_quota"


def handle_openai_compatible_response(
    model_provider: ModelProvider,
    model_name: str,
    request: FenicCompletionsRequest,
    response: Optional[Union[ChatCompletion, ParsedChatCompletion]],
    request_key_generator: Callable[[FenicCompletionsRequest], str],
) -> tuple[
        Optional[Union[ParsedChoice, Choice]],
        Optional[Union[FatalException, TransientException]]
    ]:
    if not response:
        return None, TransientException(ExecutionError("No response from OpenAI"))
    if not response.choices:
        return None, TransientException(
            ExecutionError(
                f"The completion model {model_provider}/{model_name} encountered an error while processing request {request_key_generator(request)}: {response.error}"
            )
        )

    completion_choice = response.choices[0]
    if completion_choice.message.refusal:
        return None, TransientException(
            ExecutionError(
                f"The completion model {model_provider}/{model_name} refused to generate a response for request {request_key_generator(request)}: {completion_choice.message.refusal}"
            )
        )
    if completion_choice.finish_reason == "error":
        return None,TransientException(
            ExecutionError(
                f"The completion model {model_provider}/{model_name} encountered an error while generating content for request {request_key_generator(request)}: {completion_choice.error}"
            )
        )
    return completion_choice, None
