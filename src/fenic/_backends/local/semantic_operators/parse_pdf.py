import logging
from textwrap import dedent
from typing import List, Optional

import jinja2
import polars as pl

from fenic._backends.local.semantic_operators.base import (
    BaseSingleColumnFilePathOperator,
    CompletionOnlyRequestSender,
)
from fenic._backends.local.utils.doc_loader import DocFolderLoader
from fenic._inference.language_model import InferenceConfiguration, LanguageModel
from fenic.core._logical_plan.resolved_types import ResolvedModelAlias

logger = logging.getLogger(__name__)


class ParsePDF(BaseSingleColumnFilePathOperator[str, str]):
    """Operator for parsing PDF files using language models with PDF parsing capabilities."""
    SYSTEM_PROMPT = jinja2.Template(dedent("""\
        Transcribe the main content of this PDF document to clean, well-formatted markdown.
         - Output should be raw markdown, don't surround in code fences or backticks.
         - Preserve the structure, formatting, headings, lists, and any tables to the best of your ability
         - Format tables as github markdown tables, however:
             - for table headings, immediately add ' |' after the table heading
        {% if page_separator and '{page}' in page_separator %}
         - Insert the page separator '{{ page_separator }}' as a markdown line for each page break, replacing the '{{ '{page}' }}' pattern with the current page number. If the document contains page numbers, do not include them in the output, instead replace them with this page separator.
        {% elif page_separator %}
         - Don't include the page numbers in the output, instead insert the page separator '{{ page_separator }}' as a markdown line for each page break
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
    ):
        self.page_separator = page_separator
        self.describe_images = describe_images
        self.model = model
        self.model_alias = model_alias

        DocFolderLoader.check_file_extensions(input.to_list(), "pdf")

        super().__init__(
            input=input,
            request_sender=CompletionOnlyRequestSender(
                model=model,
                operator_name="semantic.parse_pdf",
                inference_config=InferenceConfiguration(
                    max_output_tokens=None,
                    temperature=1.0,  # Use a higher temperature so gemini flash models can handle complex table formatting.  For more info see the conversation here: https://discuss.ai.google.dev/t/gemini-2-0-flash-has-a-weird-bug/65119/26
                    model_profile=model_alias.profile if model_alias else None,
                ),
            ),
            examples=None,  # PDF parsing doesn't use examples
        )


    def build_system_message(self) -> str:
        """Build system message for PDF parsing."""
        return self.SYSTEM_PROMPT.render(
            page_separator=self.page_separator,
            describe_images=self.describe_images
        )

    def postprocess(self, responses: List[Optional[str]]) -> List[Optional[str]]:
        """Return parsed PDF content as-is."""
        return responses