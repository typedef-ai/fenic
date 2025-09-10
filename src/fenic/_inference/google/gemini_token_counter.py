import logging

import tiktoken
from google.genai.local_tokenizer import LocalTokenizer
from google.genai.types import (
    CountTokensResult,
)

from fenic._constants import PREFIX_TOKENS_PER_MESSAGE, TOKENS_PER_NAME
from fenic._inference.google.google_utils import convert_messages
from fenic._inference.token_counter import TokenCounter, Tokenizable
from fenic._inference.types import LMRequestMessages

logger = logging.getLogger(__name__)


class GeminiLocalTokenCounter(TokenCounter):
    """Token counter for Google Gemini models using native local tokenization.

    This counter prefers the Google `LocalTokenizer` for accurate counts that
    match the Gemini backend. If the Google tokenizer cannot be constructed for
    the given model (e.g., unsupported model name), it falls back to `tiktoken`.

    Note:
        This module assumes `google-genai` is installed. Tests that depend on
        the Google tokenizer should be skipped when the package is unavailable.

    Args:
        model_name: The target model to tokenize for (e.g., "gemini-1.5-pro").
        fallback_encoding: The tiktoken encoding to use if `encoding_for_model`
            does not recognize `model_name`.
    """

    def __init__(self, model_name: str, fallback_encoding: str = "o200k_base") -> None:
        # Always build a tiktoken tokenizer as a reliable fallback path
        try:
            self.tiktoken_tokenizer = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.tiktoken_tokenizer = tiktoken.get_encoding(fallback_encoding)

        try:
            self.use_fallback_tokenizer: bool = False
            self.google_tokenizer: LocalTokenizer = LocalTokenizer(model_name=model_name)
        except ValueError:
            # If LocalTokenizer cannot be constructed (e.g., unsupported model),
            # use tiktoken as a best-effort fallback for estimation.
            self.use_fallback_tokenizer = True

    def count_tokens(self, messages: Tokenizable) -> int:
        """Count tokens for a string, message list, or `LMRequestMessages`.

        Args:
            messages: Either a raw string, a list of role/content dicts, or an
                `LMRequestMessages` instance.

        Returns:
            Total token count as an integer.
        """
        if isinstance(messages, str):
            return self._count_text_tokens(messages)
        elif isinstance(messages, LMRequestMessages):
            return self._count_request_tokens(messages)
        else:
            return self._count_message_tokens(messages)

    def _count_request_tokens(self, messages: LMRequestMessages) -> int:
        """Count tokens for an `LMRequestMessages` object."""
        if not self.use_fallback_tokenizer:
            google_messages = convert_messages(messages)
            return self.google_tokenizer.count_tokens(google_messages).total_tokens
        else:
            return self._count_message_tokens(messages.to_message_list())

    def _count_message_tokens(self, messages: list[dict[str, str]]) -> int:
        """Count tokens for a list of message dicts with role/content keys."""
        num_tokens = 0
        for message in messages:
            # Every message starts with <im_start>{role/name}\n{content}<im_end>\n
            num_tokens += PREFIX_TOKENS_PER_MESSAGE
            for key, value in message.items():
                num_tokens += self._count_text_tokens(value)
                if key == "name":
                    # Subtract one token if the 'name' field is present
                    num_tokens -= TOKENS_PER_NAME
        # Every assistant reply is primed with <im_start>assistant
        num_tokens += 2

        return num_tokens

    def _count_text_tokens(self, text: str) -> int:
        """Count tokens for a raw text string using the best available tokenizer.

        Prefers the Google local tokenizer when available, falling back to
        `tiktoken` if not.
        """
        if self.use_fallback_tokenizer:
            return len(self.tiktoken_tokenizer.encode(text))
        token_count_result: CountTokensResult = self.google_tokenizer.count_tokens(text)
        total_tokens = token_count_result.total_tokens
        if total_tokens:
            return total_tokens
        logger.warning(
            "Gemini Native Tokenizer did not return any tokens, falling back to tiktoken",
        )
        return len(self.tiktoken_tokenizer.encode(text))