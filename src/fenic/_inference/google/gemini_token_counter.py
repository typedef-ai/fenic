import logging

from google.genai.local_tokenizer import LocalTokenizer
from google.genai.types import (
    CountTokensConfig,
)

from fenic._inference.google.google_utils import convert_messages
from fenic._inference.token_counter import (
    TokenCounter,
    Tokenizable,
)
from fenic._inference.types import LMRequestMessages

logger = logging.getLogger(__name__)


class GeminiLocalTokenCounter(TokenCounter):
    """Token counter for Google Gemini models using native local tokenization.

    This counter prefers the Google `LocalTokenizer` for accurate counts that
    match the Gemini backend. If the Google tokenizer cannot be constructed for
    the given model (e.g., unsupported model name), it falls back to the encoding mapped
    for the fallback model (typically `gemini-2.5-flash` -> `gemma3`).

    Note:
        This module assumes `google-genai` is installed. Tests that depend on
        the Google tokenizer should be skipped when the package is unavailable.

    Args:
        model_name: The target model to tokenize for (e.g., "gemini-1.5-pro").
        fallback_encoding: The target model to use as a fallback if `LocalTokenizer`
            does not recognize `model_name`.
    """

    def __init__(self, model_name: str, fallback_encoding: str = "gemini-2.5-flash") -> None:
        try:
            self.google_tokenizer: LocalTokenizer = LocalTokenizer(model_name=model_name)
        except ValueError:
            self.google_tokenizer = LocalTokenizer(model_name=fallback_encoding)

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

    def _count_request_tokens(self, messages: LMRequestMessages) -> int:
        """Count tokens for an `LMRequestMessages` object."""
        return self.google_tokenizer.count_tokens(
            convert_messages(messages),
            config=CountTokensConfig(system_instruction=messages.system)
        ).total_tokens

    def _count_text_tokens(self, text: str) -> int:
        """Count tokens for a raw text string"""
        return self.google_tokenizer.count_tokens(text).total_tokens
