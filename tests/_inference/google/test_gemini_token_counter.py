
from fenic._inference.google.gemini_token_counter import GeminiLocalTokenCounter
from fenic._inference.token_counter import TiktokenTokenCounter
from fenic._inference.types import FewShotExample, LMRequestMessages


def test_fallback_tokenizer_counts_string():
    # Use a clearly unknown model name to ensure fallback is used
    text = "hello"
    counter = GeminiLocalTokenCounter(model_name="__unknown_model_for_test__")
    expected = TiktokenTokenCounter(model_name="__unknown_model_for_test__").count_tokens(text)
    assert counter.count_tokens(text) == expected

def test_fallback_tokenizer_counts_messages():
    # Use a clearly unknown model name to ensure fallback is used
    counter = GeminiLocalTokenCounter(model_name="__unknown_model_for_test__")
    tiktoken_counter = TiktokenTokenCounter(model_name="__unknown_model_for_test__")

    messages = [
        {"role": "system", "content": "abc"},
        {"role": "user", "content": "xyz", "name": "tool"},
    ]
    assert counter.count_tokens(messages) == tiktoken_counter.count_tokens(messages)


def test_local_token_counter_falls_back_to_tiktoken():
    model = "gemini-2.812341-flash"
    counter = GeminiLocalTokenCounter(model_name=model)
    assert counter.use_fallback_tokenizer is True
    assert counter.google_tokenizer is None
    assert counter.tiktoken_tokenizer

def test_local_token_counter_initializes():
    model = "gemini-2.0-flash"
    counter = GeminiLocalTokenCounter(model_name=model)
    assert counter.use_fallback_tokenizer is False
    assert counter.google_tokenizer

def test_local_token_counter_counts_tokens():
    model = "gemini-2.0-flash"
    counter = GeminiLocalTokenCounter(model_name=model)
    assert counter.use_fallback_tokenizer is False
    assert counter.google_tokenizer
    assert counter.count_tokens("Hello, Gemini!") == 4


def test_google_tokenizer_counts_lm_request_messages_matches_convert_messages():
    model = "gemini-2.5-flash"

    counter = GeminiLocalTokenCounter(model_name=model)
    assert counter.use_fallback_tokenizer is False

    messages = LMRequestMessages(
        system="You are a helpful assistant.",
        examples=[FewShotExample(user="ping", assistant="pong")],
        user="Summarize: The quick brown fox jumps over the lazy dog.",
    )

    expected = counter.count_tokens(messages)
    assert counter.count_tokens(messages.to_message_list()) == expected

