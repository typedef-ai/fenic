import os

import fitz
import pytest

pytest.importorskip("google.genai")

from fenic._inference.google.gemini_token_counter import GeminiLocalTokenCounter
from fenic._inference.types import FewShotExample, LMRequestFile, LMRequestMessages
from tests.conftest import _save_pdf_file


def test_local_token_counter_counts_tokens():
    model = "gemini-2.0-flash" #gemma3
    counter = GeminiLocalTokenCounter(model_name=model)
    assert counter.count_tokens("This is a longer string of text with characters: 那只敏捷的棕色狐狸跳过了懒惰的狗") == 25

    model = "gemini-1.5-pro" #gemma2
    pro_counter = GeminiLocalTokenCounter(model_name=model)
    assert pro_counter.count_tokens("This is a longer string of text with characters: 那只敏捷的棕色狐狸跳过了懒惰的狗") == 25

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

def test_google_tokenizer_counts_tokens_for_pdfs(temp_dir_just_one_file):
    model = "gemini-2.0-flash"
    pdf_path1 = os.path.join(temp_dir_just_one_file, "test_pdf_one_page.pdf")
    pdf_path2 = os.path.join(temp_dir_just_one_file, "test_pdf_three_pages.pdf")
    _save_pdf_file(pdf_path1, page_count=1, text_content="The quick brown fox jumps over the lazy dog.")
    _save_pdf_file(pdf_path2, page_count=3, text_content="The quick brown fox jumps over the lazy dog.")
    counter = GeminiLocalTokenCounter(model_name=model)
    messages = LMRequestMessages(
        system="You are a helpful assistant.",
        examples=[],
        user_file=LMRequestFile(path=pdf_path1, page_range=(0, 0))
    )
    # 258 tokens per page.  System message is counted separately.
    assert counter.count_tokens(messages) == 258

    messages = LMRequestMessages(
        system="You are a helpful assistant.",
        examples=[],
        user_file=LMRequestFile(path=pdf_path2, page_range=(0, 2)),
    )
    # 258 tokens per page.  System message is counted separately.
    assert counter.count_tokens(messages) == 258 * 3
    
    # Test chunk sized 1 page
    pdf_2 = fitz.open(pdf_path2)
    pdf_2_chunk = fitz.open()
    pdf_2_chunk.insert_pdf(pdf_2, from_page=1, to_page=1)
    messages = LMRequestMessages(
        system="You are a helpful assistant.",
        examples=[],
        user_file=LMRequestFile(path=pdf_path2, pdf_chunk_bytes=pdf_2_chunk.tobytes(), page_range=(1, 1)),
    )
    # 258 tokens per page.  System message is counted separately.
    assert counter.count_tokens(messages) == 258