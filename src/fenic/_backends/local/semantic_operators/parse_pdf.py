import logging
from textwrap import dedent
from typing import List, Optional, Tuple

import fitz
import jinja2
import polars as pl

from fenic._backends.local.semantic_operators.base import (
    BaseSingleColumnFilePathOperator,
    CompletionOnlyRequestSender,
)
from fenic._backends.local.utils.doc_loader import DocFolderLoader
from fenic._inference.language_model import InferenceConfiguration, LanguageModel
from fenic._inference.types import LMRequestFile, LMRequestMessages
from fenic.core._inference.model_catalog import ModelProvider
from fenic.core._logical_plan.resolved_types import ResolvedModelAlias

logger = logging.getLogger(__name__)

PDF_MARKDOWN_OUTPUT_TOKEN_MULTIPLIER = 1.5 # Add buffer, account for markdown and table formatting, image descriptions, and other overheads
PDF_MAX_PAGES_CHUNK = 3 # Fenic can't handle large outputs until we implement streamed responses, so limit chunk size

class ParsePDF(BaseSingleColumnFilePathOperator[str, str]):
    """Operator for parsing PDF files using language models with PDF parsing capabilities."""
    SYSTEM_PROMPT = jinja2.Template(dedent("""\
        Transcribe the main content of this PDF document to clean, well-formatted markdown.
         - Output should be raw markdown, don't surround the whole output in code fences or backticks.
         - For each topic, create a markdown heading. For key terms, use bold text.
         - Preserve the structure, formatting, headings, lists, table of contents, and any tables using markdown syntax.
         - Format tables as github markdown tables, however:
             - for table headings, immediately add ' |' after the table heading
        {% if multiple_pages %}
        {% if page_separator and '{page}' in page_separator %}
         - Insert the page separator '{{ page_separator }}' as a markdown line for each page break, replacing the '{{ '{page}' }}' pattern with the current page number. If the document contains page numbers, do not include them in the output, instead replace them with this page separator.
        {% elif page_separator %}
         - Don't include the page numbers in the output, instead insert the page separator '{{ page_separator }}' as a markdown line for each page break
        {% endif %}
        {% endif %}
        {% if describe_images %}
         - For each image, describe them briefly in a markdown section with 'Image' in the title, preserving the output order.
        {% else %}
         - Ignore any images that aren't tables or charts that can be converted to markdown.
        {% endif %}""").strip()
    )
    def __init__(
        self,
        input: pl.Series,
        model: LanguageModel,
        page_separator: Optional[str] = None,
        describe_images: bool = False,
        model_alias: Optional[ResolvedModelAlias] = None,
        max_output_tokens: Optional[int] = None,
        request_timeout: Optional[float] = None,
    ):
        self.page_separator = page_separator
        self.describe_images = describe_images
        self.model = model
        self.model_alias = model_alias
        self.max_output_tokens = max_output_tokens

        DocFolderLoader.check_file_extensions(input.to_list(), "pdf")

        temperature = 0.0
        if model.provider == ModelProvider.GOOGLE_DEVELOPER or model.provider == ModelProvider.GOOGLE_VERTEX or (model.provider == ModelProvider.OPENROUTER and model.model.split("/")[0] == "google"):
            temperature = 1.0  # Use a higher temperature so gemini flash models can handle complex table formatting.  For more info see the conversation here: https://discuss.ai.google.dev/t/gemini-2-0-flash-has-a-weird-bug/65119/26

        super().__init__(
            input=input,
            request_sender=CompletionOnlyRequestSender(
                model=model,
                operator_name="semantic.parse_pdf",
                inference_config=InferenceConfiguration(
                    max_output_tokens=max_output_tokens,
                    model_profile=model_alias.profile if model_alias else None,
                    temperature=temperature,
                    request_timeout=request_timeout,
                ),
            ),
            examples=None,  # PDF parsing doesn't use examples
        )


    def build_system_message(self, multiple_pages: bool = False) -> str:
        """Build system message for PDF parsing."""
        return self.SYSTEM_PROMPT.render(
            page_separator=self.page_separator,
            describe_images=self.describe_images,
            multiple_pages=multiple_pages
        )
    def execute(self) -> pl.Series:
        """ Executes the PDF parsing operator.

        Flow:
            - build messages may chunk the PDF based on internal limits and model output token limit,
                - A request is created for each PDF chunk, holding an in-memory PDF file with that range of pages
                - We keep a list of chunk sizes (i.e. page ranges) for each row
            - send the requests to the LLM and wait for the responses.
            - Postprocess concats the markdown responses for PDF chunks in each row.
            - NOTE: this depends on the behavior of ModelClient returning responses in the same order they are queued.

        Returns:
            A Polars Series containing the markdown translation of each PDF file
        """
        prompts, page_counts_per_chunk_per_row = self.build_request_messages_batch()
        responses = self.request_sender.send_requests(prompts)
        postprocessed_responses = self.postprocess(responses, page_counts_per_chunk_per_row)
        return pl.Series(postprocessed_responses)

    def build_request_messages_batch(self) -> Tuple[List[Optional[LMRequestMessages]], List[List[int]]]:
        """ Create the messages for each PDF in this column, chunking large PDFs into multiple messages.

        Returns:
            Messages for this column of PDFs
            List of the each chunk size (page count) per PDF (page_counts_per_chunk_per_row)"""
        messages_batch = []
        page_counts_per_chunk_per_row = []
        for path in self.input:
            if not path:
                messages_batch.append(None)
                page_counts_per_chunk_per_row.append([1])
            else:
                file_chunks = self._get_file_chunks(path)
                page_counts_per_chunk = []
                for file in file_chunks:
                    messages_batch.append(
                        self.build_request_messages(file, multiple_pages=len(file_chunks) > 1)
                    )
                    page_counts_per_chunk.append(file.page_range[1] - file.page_range[0] + 1)
                page_counts_per_chunk_per_row.append(page_counts_per_chunk)
        return messages_batch, page_counts_per_chunk_per_row


    def _get_file_chunks(self, file_path: str) -> List[LMRequestFile]:
        """Get the page chunks for the PDF file.

        Limit the pages based on the model's output token limit and internal max pages per chunk.

        Args:
            file_path: Path to the PDF file

        Returns:
            List of LMRequestFile objects
            List of (start_page, end_page) tuples (inclusive, 0-indexed)
        """
        chunks = []
        range_start_page = 0
        range_tokens = 0
        range_page_count = 0

        with fitz.open(file_path) as doc:
            total_pages = doc.page_count
            for page_num in range(total_pages):
                text = doc[page_num].get_text("text")
                page_tokens = self.model.count_tokens(text)
                # Check if we need to start a new range, either by reaching the token limit or the requested page range size
                would_exceed_tokens = range_tokens > 0 and (range_tokens + page_tokens) * PDF_MARKDOWN_OUTPUT_TOKEN_MULTIPLIER > self.model.model_parameters.max_output_tokens
                would_exceed_page_limit = range_page_count >= PDF_MAX_PAGES_CHUNK

                if would_exceed_tokens or would_exceed_page_limit:
                    # Save current batch
                    last_page = page_num - 1
                    page_range = (range_start_page, last_page)
                    with fitz.open() as doc_chunk:
                        doc_chunk.insert_pdf(doc, from_page=range_start_page, to_page=last_page)
                        chunks.append(LMRequestFile(path=file_path, pdf_chunk_bytes=doc_chunk.tobytes(), page_range=page_range))
                    range_start_page = page_num
                    range_tokens = page_tokens
                    range_page_count = 1
                else:
                    range_tokens += page_tokens
                    range_page_count += 1

            # Add the last batch if there are remaining pages
            if range_start_page < total_pages:
                if range_start_page == 0:
                    # whole pdf fits in one chunk, no need to keep data in memory
                    chunks.append(LMRequestFile(path=file_path, pdf_chunk_bytes=None, page_range=(0, total_pages - 1)))
                else:
                    # multi-page chunk
                    with fitz.open() as doc_chunk:
                        doc_chunk.insert_pdf(doc, from_page=range_start_page, to_page=total_pages - 1)
                        chunks.append(LMRequestFile(path=file_path, pdf_chunk_bytes=doc_chunk.tobytes(), page_range=(range_start_page, total_pages - 1)))

            return chunks

    def build_request_messages(self, file: LMRequestFile, multiple_pages: bool = False) -> LMRequestMessages:
        """Construct a request for PDF parsing with optional page range.

        Args:
            file_path: Path to the PDF file
            page_range: Optional tuple of (start_page, end_page) inclusive, 0-indexed
        """
        # Determine if we're processing multiple pages for system message
        return LMRequestMessages(
            system=self.build_system_message(multiple_pages=multiple_pages),
            user_file=file,
            examples=self.build_examples() if self.examples else [],
        )

    def postprocess(self, responses: List[Optional[str]], page_counts_per_chunk_per_row: List[List[int]]) -> List[Optional[str]]:
        """Combine responses from multiple page ranges if needed.

        Uses the list of chunk sizes per PDF to keep track of the page count and add correct page delimination"""
        combined_responses = []
        path_first_chunk_idx = 0
        chunk_separator = "\n"
        if self.page_separator:
            chunk_separator += self.page_separator + "\n"

        for page_counts_per_chunk in page_counts_per_chunk_per_row:
            # Combine the responses/chunks for this path
            combined_response = ""
            last_page_number = 0
            for chunk_idx, page_count in enumerate(page_counts_per_chunk):
                combined_response += responses[path_first_chunk_idx + chunk_idx]
                if chunk_idx < len(page_counts_per_chunk) - 1:
                    # Add page separator between chunks, skip the last chunk
                    last_page_number = last_page_number + page_count
                    combined_response += chunk_separator.replace("{page}", str(last_page_number))
            combined_responses.append(combined_response)
            path_first_chunk_idx += len(page_counts_per_chunk)

        return combined_responses