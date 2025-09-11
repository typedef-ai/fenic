
from fenic._inference.google.gemini_token_counter import GeminiLocalTokenCounter
from fenic._inference.types import FewShotExample, LMRequestMessages


def test_local_token_counter_counts_tokens():
    model = "gemini-2.0-flash" #gemma3
    counter = GeminiLocalTokenCounter(model_name=model)
    assert counter.count_tokens("This is a longer string of text with characters: 那只敏捷的棕色狐狸跳过了懒惰的狗") == 25

    model = "gemini-1.5-pro" #gemma2
    pro_counter = GeminiLocalTokenCounter(model_name=model)
    assert pro_counter.count_tokens("This is a longer string of text with characters: 那只敏捷的棕色狐狸跳过了懒惰的狗") == 23

def test_local_token_counter_falls_back_to_gemma3():
    model = "gemini-242342" #non-existent model
    counter = GeminiLocalTokenCounter(model_name=model)
    assert counter.count_tokens("This is a longer string of text with characters: 那只敏捷的棕色狐狸跳过了懒惰的狗") == 25

def test_google_tokenizer_counts_tokens_for_message_list():
    model = "gemini-2.5-flash"

    counter = GeminiLocalTokenCounter(model_name=model)
    messages = LMRequestMessages(
        system="You are a helpful assistant.",
        examples=[FewShotExample(user="ping", assistant="pong")],
        user="Summarize: The quick brown fox jumps over the lazy dog.",
    )
    assert counter.count_tokens(messages) == 21
