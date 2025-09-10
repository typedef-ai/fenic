import logging
from typing import Protocol, Union

import tiktoken
from image_token.config import openai_config as image_token_openai_config
from image_token.core import calculate_image_tokens

from fenic._constants import PREFIX_TOKENS_PER_MESSAGE
from fenic._inference.request_utils import get_pdf_text_and_image_sizes
from fenic._inference.types import LMRequestMessages
from fenic.core.error import InternalError

logger = logging.getLogger(__name__)

Tokenizable = Union[str | LMRequestMessages]

class TokenCounter(Protocol):
    def count_tokens(self, messages: Tokenizable) -> int: ...
    def count_file_input_tokens(self, messages: LMRequestMessages) -> int: ...
    def count_file_output_tokens(self, messages: LMRequestMessages) -> int: ...

class TiktokenTokenCounter(TokenCounter):

    def __init__(self, model_name: str, fallback_encoding: str = "o200k_base"):
        self.model_name = model_name
        self._setup_image_token_config()
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

    def count_file_input_tokens(self, messages: LMRequestMessages) -> int:
        # get file type from file extension
        file_type = messages.user_file_path.split(".")[-1]
        if file_type == "pdf":
            text, image_sizes = get_pdf_text_and_image_sizes(messages.user_file_path)
            text_tokens = self.count_tokens(text)
            image_tokens = 0
            for image_size_tuple in image_sizes:
                image_tokens += calculate_image_tokens(
                    model_name=self.model_image_token_strategy, width=image_size_tuple[0], height=image_size_tuple[1], max_tokens=self.image_token_config["max_tokens"], model_config=self.image_token_config)
            return text_tokens + image_tokens
        else:
            raise InternalError(f"File{messages.user_file_path}'s extension is not supported for llm completions.")

    def count_file_output_tokens(self, messages: LMRequestMessages) -> int:
        file_type = messages.user_file_path.split(".")[-1]
        if file_type == "pdf":
            text, _ = get_pdf_text_and_image_sizes(messages.user_file_path)
            # Note: we currently aren't counting any text tokens for describing images, since that defaults to False.
            # TODO: figure out how to tell whether we're prompting the model to describe images
            return self.count_tokens(text)
        else:
            raise InternalError(f"File{messages.user_file_path}'s extension is not supported for llm completions.")

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
            num_tokens += self.count_file_input_tokens(messages)
            message_count += 1
        num_tokens += message_count * PREFIX_TOKENS_PER_MESSAGE
        num_tokens += 2  # Every assistant reply is primed with <im_start>assistant
        return num_tokens

    def _setup_image_token_config(self) -> str:
        """Choose the image token strategy for the model."""
        # First check if the model is directly supported
        if self.model_name in image_token_openai_config:
            self.model_image_token_strategy = self.model_name
        else:
            # Lookup for models with similar image token strategies.
            self.model_image_token_strategy = self._lookup_model_with_similar_image_token_strategies(self.model_name)
            logger.warning(f"Model {self.model_name} is not supported by image_token library. Using closest supported model {self.model_image_token_strategy}.")
        self.image_token_config = image_token_openai_config[self.model_image_token_strategy]

    def _lookup_model_with_similar_image_token_strategies(self, model_name: str) -> str:
        # Lookup for models with similar image token strategies.
        # image_token library supports a subset of openai models, so we map unsupported models to the closest supported model.
        # If the model uses path based image tokenization, use a model that uses patch based image tokenization.  If the model uses tile based image tokenization, use a model that uses tile based image tokenization.
        # Warning: the token count accuracy has not been tested for these models.
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
        elif self.model_name.startswith("gpt-4o"):
            return "gpt-4o"
        else:
            return "gpt-4o" # The industry standard for image tokenization is through patching, so use gpt-4o as a fallback.