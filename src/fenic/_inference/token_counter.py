from typing import Protocol, Union

import tiktoken

from fenic._constants import PREFIX_TOKENS_PER_MESSAGE
from fenic._inference.request_utils import get_pdf_page_count, get_pdf_text
from fenic._inference.types import LMRequestMessages
from fenic.core.error import InternalError

Tokenizable = Union[str | LMRequestMessages]

class TokenCounter(Protocol):
    def count_tokens(self, messages: Tokenizable, ignore_file: bool = False) -> int: ...
    def count_file_input_tokens(self, messages: LMRequestMessages) -> int: ...
    def count_file_output_tokens(self, messages: LMRequestMessages) -> int: ...

class TiktokenTokenCounter(TokenCounter):

    def __init__(self, model_name: str, fallback_encoding: str = "o200k_base"):
        try:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.tokenizer = tiktoken.get_encoding(fallback_encoding)

    def count_tokens(self, messages: Tokenizable, ignore_file: bool = False) -> int:
        if isinstance(messages, str):
            return len(self.tokenizer.encode(messages))
        elif isinstance(messages, LMRequestMessages):
            return self._count_message_tokens(messages, ignore_file)
        else:
            raise TypeError(f"Expected str or LMRequestMessages, got {type(messages)}")

    def count_file_input_tokens(self, messages: LMRequestMessages) -> int:
        # get file type from file extension
        file_type = messages.user_file.path.split(".")[-1]
        if file_type == "pdf":
            text = get_pdf_text(messages.user_file)
            page_count = get_pdf_page_count(messages.user_file)
            text_tokens = self.count_tokens(text)
            # OpenAI documentation states that they convert PDF pages into images and ingest both text and image into their VLM. 
            # Based on experimentation, OpenAI seems to count no more than 1024 tokens per page.
            image_tokens = page_count * 1024 
            return text_tokens + image_tokens
        else:
            raise InternalError(f"File{messages.user_file.path}'s extension is not supported for llm completions.")

    def count_file_output_tokens(self, messages: LMRequestMessages) -> int:
        file_type = messages.user_file.path.split(".")[-1]
        if file_type == "pdf":
            # TODO: we do this twice, once for estimating input and once for estimating output.  We can cache the text in the LMFile object.
            text = get_pdf_text(messages.user_file)
            # Note: we currently aren't counting any text tokens for describing images, since that defaults to False.
            # In our estimates we add buffer, both for markdown structure and in case we ask the model to describe images.
            return self.count_tokens(text)
        else:
            raise InternalError(f"File{messages.user_file.path}'s extension is not supported for llm completions.")

    def _count_message_tokens(self, messages: LMRequestMessages, ignore_file: bool = False) -> int:
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
        if messages.user_file and not ignore_file:
            num_tokens += self.count_file_input_tokens(messages)
            message_count += 1
        num_tokens += message_count * PREFIX_TOKENS_PER_MESSAGE
        num_tokens += 2  # Every assistant reply is primed with <im_start>assistant
        
        return num_tokens
