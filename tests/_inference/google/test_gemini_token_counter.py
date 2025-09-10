import tiktoken

from fenic._constants import PREFIX_TOKENS_PER_MESSAGE, TOKENS_PER_NAME
from fenic._inference.google.gemini_token_counter import GeminiLocalTokenCounter


def test_fallback_tokenizer_counts_string():
    # Use a clearly unknown model name to ensure fallback is used
    text = "hello"
    counter = GeminiLocalTokenCounter(model_name="__unknown_model_for_test__")
    expected = len(tiktoken.get_encoding("o200k_base").encode(text))
    assert counter.count_tokens(text) == expected


def test_fallback_tokenizer_counts_messages():
    # Use a clearly unknown model name to ensure fallback is used
    counter = GeminiLocalTokenCounter(model_name="__unknown_model_for_test__")
    enc = tiktoken.get_encoding("o200k_base")

    messages = [
        {"role": "system", "content": "abc"},
        {"role": "user", "content": "xyz", "name": "tool"},
    ]

    # Manually compute expected using fallback encoding, mirroring implementation
    expected = 0
    for msg in messages:
        expected += PREFIX_TOKENS_PER_MESSAGE
        for k, v in msg.items():
            expected += len(enc.encode(v))
            if k == "name":
                expected -= TOKENS_PER_NAME
    expected += 2

    assert counter.count_tokens(messages) == expected


