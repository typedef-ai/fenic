import logging

from google.genai.local_tokenizer import LocalTokenizer
from google.genai.types import (
    CountTokensConfig,
)

from fenic._inference.google.google_utils import convert_text_messages
from fenic._inference.request_utils import get_pdf_page_count, get_pdf_text
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

    def count_tokens(self, messages: Tokenizable, ignore_file: bool = False) -> int:
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
            return self._count_request_tokens(messages, ignore_file)

    def count_file_input_tokens(self, messages: LMRequestMessages) -> int:
        # Gemini 2.0 charges 258 tokens per page for all PDF inputs.  For more detail, see https://gemini-api.apidog.io/doc-965859#technical-details
        page_count = get_pdf_page_count(messages.user_file)
        tokens = page_count * 258
        return tokens

    def count_file_output_tokens(self, messages: LMRequestMessages) -> int:
        # TODO: we do this twice, once for estimating input and once for estimating output.  We can cache the text in the LMFile object.
        text = get_pdf_text(messages.user_file)
        # Note: we currently aren't counting any text tokens for describing images, since that defaults to False.
        # In our estimates we add buffer, both for markdown structure and in case we ask the model to describe images.
        return self.google_tokenizer.count_tokens(text).total_tokens

    def _count_request_tokens(self, messages: LMRequestMessages, ignore_file: bool = False) -> int:
        """Count tokens for an `LMRequestMessages` object."""
        contents = convert_text_messages(messages)
        tokens = 0
        if len(contents) > 0:
            count_tokens = self.google_tokenizer.count_tokens(
                convert_text_messages(messages),
                config=CountTokensConfig(system_instruction=messages.system)
            ).total_tokens
            tokens += count_tokens

        if messages.user_file and not ignore_file:
            tokens += self.count_file_input_tokens(messages)
        return tokens


    def _count_text_tokens(self, text: str) -> int:
        """Count tokens for a raw text string"""
        return self.google_tokenizer.count_tokens(text).total_tokens
