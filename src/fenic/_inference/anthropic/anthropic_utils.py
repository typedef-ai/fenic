from anthropic.types import (
        CacheControlEphemeralParam,
        MessageParam,
        TextBlockParam,
)

from fenic._inference.types import LMRequestMessages

TEXT_DELTA = "text_delta"

MESSAGE_STOP = "message_stop"

INPUT_JSON_DELTA = "input_json_delta"

CONTENT_BLOCK_DELTA = "content_block_delta"

EPHEMERAL_CACHE_CONTROL = CacheControlEphemeralParam(type="ephemeral")

def convert_messages(messages: LMRequestMessages) -> tuple[TextBlockParam, list[MessageParam]]:
        """Convert Fenic messages to Anthropic format.

        Converts Fenic LMRequestMessages to Anthropic's TextBlockParam and
        MessageParam format, including system prompt and conversation history.

        Args:
            messages: Fenic message format

        Returns:
            Tuple of (system_prompt, message_params)
        """
        system_prompt = TextBlockParam(
            text=messages.system,
            type="text",
            cache_control=EPHEMERAL_CACHE_CONTROL
        )
        message_params: list[MessageParam] = []
        for example in messages.examples:
            message_params.append(MessageParam(content=example.user, role="user"))
            message_params.append(MessageParam(content=example.assistant, role="assistant"))
        user_prompt = TextBlockParam(
            text=messages.user,
            type="text",
            cache_control=EPHEMERAL_CACHE_CONTROL
        )
        message_params.append(MessageParam(content=[user_prompt], role="user"))
        return system_prompt, message_params