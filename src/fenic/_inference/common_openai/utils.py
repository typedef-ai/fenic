from typing import Callable, Optional, Union

from openai.types.chat import ChatCompletion, ParsedChatCompletion, ParsedChoice
from openai.types.chat.chat_completion import Choice

from fenic._inference.model_client import FatalException, TransientException
from fenic._inference.types import FenicCompletionsRequest
from fenic.core._inference.model_catalog import ModelProvider
from fenic.core.error import ExecutionError


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
