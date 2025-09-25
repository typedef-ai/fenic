import os

import fitz

from fenic._inference.token_counter import TiktokenTokenCounter
from fenic._inference.types import FewShotExample, LMRequestFile, LMRequestMessages
from tests.conftest import _save_pdf_file


def test_local_token_counter_counts_tokens():
    model = "gpt-4o-mini"
    counter = TiktokenTokenCounter(model_name=model)
    assert counter.count_tokens("This is a longer string of text with characters: 那只敏捷的棕色狐狸跳过了懒惰的狗") == 28

    model = "gpt-4o"
    pro_counter = TiktokenTokenCounter(model_name=model)
    assert pro_counter.count_tokens("This is a longer string of text with characters: 那只敏捷的棕色狐狸跳过了懒惰的狗") == 28

def test_local_token_counter_falls_back_to_o200k_base():
    model = "gpt-242342"  # non-existent model
    counter = TiktokenTokenCounter(model_name=model)
    assert counter.count_tokens("This is a longer string of text with characters: 那只敏捷的棕色狐狸跳过了懒惰的狗") == 28

def test_openai_tokenizer_counts_tokens_for_message_list():
    model = "gpt-4o-mini"

    counter = TiktokenTokenCounter(model_name=model)
    messages = LMRequestMessages(
        system="You are a helpful assistant.",
        examples=[FewShotExample(user="ping", assistant="pong")],
        user="Summarize: The quick brown fox jumps over the lazy dog.",
    )
    # Note: The exact token count may differ from Gemini due to different tokenization
    # This test verifies the method works and returns a reasonable value
    token_count = counter.count_tokens(messages)
    assert token_count == 44  # Actual token count for OpenAI tokenizer

def test_openai_tokenizer_counts_tokens_for_pdfs(temp_dir_just_one_file):
    model = "gpt-4o-mini"
    pdf_path1 = os.path.join(temp_dir_just_one_file, "test_pdf_one_page.pdf")
    pdf_path2 = os.path.join(temp_dir_just_one_file, "test_pdf_three_pages.pdf")
    _save_pdf_file(pdf_path1, page_count=1, text_content="The quick brown fox jumps over the lazy dog.")
    _save_pdf_file(pdf_path2, page_count=3, text_content="The quick brown fox jumps over the lazy dog.")
    counter = TiktokenTokenCounter(model_name=model)
    messages = LMRequestMessages(
        system="You are a helpful assistant.",
        examples=[],
        user_file=LMRequestFile(path=pdf_path1, page_range=(0, 1)),
    )
    # OpenAI uses different tokenization for PDFs, including image tokens
    # Just verify it returns a positive number and processes the PDF
    token_count = counter.count_tokens(messages)
    assert token_count > 0

    messages = LMRequestMessages(
        system="You are a helpful assistant.",
        examples=[],
        user_file=LMRequestFile(path=pdf_path2, page_range=(0, 3)),
    )
    # Verify that 3-page PDF has more tokens than 1-page PDF
    token_count_3pages = counter.count_tokens(messages)
    assert token_count_3pages > token_count

    # verify token estimation for chunked PDF works
    pdf_2 = fitz.open(pdf_path2)
    pdf_2_chunk = fitz.open()
    pdf_2_chunk.insert_pdf(pdf_2, from_page=1, to_page=1)
    messages = LMRequestMessages(
        system="You are a helpful assistant.",
        examples=[],
        user_file=LMRequestFile(path=pdf_path2, pdf_chunk_bytes=pdf_2_chunk.tobytes(), page_range=(1, 2)),
    )
    # Verify that a 1-page chunk has the same number of tokens as a 1-page PDF
    token_count_1_page_chunk = counter.count_tokens(messages)
    assert token_count_1_page_chunk == token_count