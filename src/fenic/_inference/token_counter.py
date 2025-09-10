from typing import Protocol, Union

import tiktoken
from image_token import get_token as get_image_tokens
from image_token.config import openai_config as image_token_openai_config
from image_token.core import calculate_image_tokens

from fenic._constants import PREFIX_TOKENS_PER_MESSAGE
from fenic._inference.request_utils import get_pdf_text_and_image_sizes
from fenic._inference.types import LMRequestMessages

Tokenizable = Union[str | LMRequestMessages]

class TokenCounter(Protocol):
    def count_tokens(self, messages: Tokenizable) -> int: ...
    def count_tokens_pdf(self, messages: LMRequestMessages, for_input: bool = False) -> int: ...

class TiktokenTokenCounter(TokenCounter):

    def __init__(self, model_name: str, fallback_encoding: str = "o200k_base"):
        self.model_name = model_name
        try:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.tokenizer = tiktoken.get_encoding(fallback_encoding)

    def count_tokens(self, messages: Tokenizable) -> int:
        if isinstance(messages, str):
            return len(self.tokenizer.encode(messages))
        elif isinstance(messages, LMRequestMessages):
            return self._count_message_tokens(messages)
        else:
            raise TypeError(f"Expected str or LMRequestMessages, got {type(messages)}")

    def _count_message_tokens(self, messages: LMRequestMessages) -> int:
        num_tokens = 0
        message_count = 2 # system message and user parent message
        num_tokens += self.count_tokens(messages.system)
        if messages.user:
            num_tokens += self.count_tokens(messages.user)
            message_count += 1
        for example in messages.examples:
            num_tokens += self.count_tokens(example.user)
            num_tokens += self.count_tokens(example.assistant)
            message_count += 2
        if messages.user_file_path:
            num_tokens += self.count_tokens_pdf(messages, for_input=True)
            message_count += 1
        num_tokens += message_count * PREFIX_TOKENS_PER_MESSAGE
        num_tokens += 2  # Every assistant reply is primed with <im_start>assistant
        return num_tokens

    def count_tokens_pdf(self, messages: LMRequestMessages, for_input: bool = False) -> int:
        text, image_sizes = get_pdf_text_and_image_sizes(messages.user_file_path)
        text_tokens = self.count_tokens(text)
        closest_model_name = self._closest_model_by_image_token_strategy()
        if not for_input or closest_model_name is None:
            # Note: we currently aren't counting any text tokens for describing images, since that defaults to False.
            # TODO: figure out how to tell whether we're prompting the model to describe images
            return text_tokens

        # if the model supports image tokenization, count the image tokens
        image_tokens = 0
        image_config = image_token_openai_config[closest_model_name]
        for image_size_tuple in image_sizes:
            image_tokens += calculate_image_tokens(
                model_name=closest_model_name, width=image_size_tuple[0], height=image_size_tuple[1], max_tokens=image_config["max_tokens"], model_config=image_config)
        return text_tokens + image_tokens

    def _closest_model_by_image_token_strategy(self) -> str:
        """For openai models, return the closest model image_token library supports.

        image_token library supports a subset of openai models.
        For openai models, map unsupported models to the closest supported model.
        If the model uses path based image tokenization, use a model that uses patch based image tokenization.  If the model uses tile based image tokenization, use a model that uses tile based image tokenization.

        If the model is not an openai model, return None.
        """
        if self.model_name in image_token_openai_config:
            return self.model_name
        elif self.model_name in ["o1", "o1-mini", "o3", "o3-mini"]:
            return "o4-mini"
        elif self.model_name.startswith("gpt-5-mini"):
            return "gpt-5-mini"
        elif self.model_name.startswith("gpt-5-nano"):
            return "gpt-5-nano"
        elif self.model_name.startswith("gpt-5"):
            return "gpt-5"
        elif self.model_name.startswith("gpt-4.1-nano"):
            return "gpt-4.1-nano"
        elif self.model_name.startswith("gpt-4.1-mini"):
            return "gpt-4.1-mini"
        elif self.model_name.startswith("gpt-4.1"):
            return "gpt-4.1"
        elif self.model_name.startswith("gpt-4o-mini"):
            return "gpt-4o-mini"
        elif self.model_name.startswith("gpt-4") or self.model_name.startswith("gpt-4-turbo"):
            return "gpt-4o"
        else:
            return None

    def _count_tokens_image(self, image_path: str) -> int:
        return get_image_tokens(self._closest_model_by_image_token_strategy(), image_path=image_path)
