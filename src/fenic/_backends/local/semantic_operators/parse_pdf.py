import logging
from typing import List, Optional

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
                    temperature=0.0,  # Use deterministic parsing
                    model_profile=model_alias.profile if model_alias else None,
                ),
            ),
            examples=None,  # PDF parsing doesn't use examples
        )

    def build_system_message(self) -> str:
        """Build system message for PDF parsing."""
        parts = []
        parts.append("Convert the main content of this PDF document to clean, well-formatted markdown. Output should be raw markdown, don't surround in code fences or backticks. ")
        parts.append("Preserve the structure, formatting, headings, lists, and any tables.")
        parts.append("Never skip any text from the body of the document, even if it's repetitive.")
        if self.page_separator:
            if "{page}" in self.page_separator:
                parts.append(f"Insert the page separator '{self.page_separator}' as a markdown line for each page break, replacing the '{'{page}'}' pattern with the current page number.  If the document contains page numbers, do not include them in the output, instead replace them with this page separator.")
            else:
                parts.append(f"Don't include the page numbers in the output, instead insert the page separator '{self.page_separator}' as a markdown line for each page break.")


        if self.describe_images:
            parts.append("For each image, describe them briefly in a markdown section with 'Image' in the title, preserving the output order.")
        else:
            parts.append("Ignore any images that aren't tables or charts that can be converted to markdown.")
        
        return "\n".join(parts)

    def postprocess(self, responses: List[Optional[str]]) -> List[Optional[str]]:
        """Return parsed PDF content as-is."""
        return responses