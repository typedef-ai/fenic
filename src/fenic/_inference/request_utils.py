"""Utilities for request processing and deduplication."""

import base64
import hashlib
import logging
from typing import Annotated, Optional

import fitz  # PyMuPDF
from pydantic import BeforeValidator

from fenic._constants import MAX_MODEL_CLIENT_TIMEOUT
from fenic._inference.types import FenicCompletionsRequest, LMRequestFile
from fenic.core.error import ValidationError

logger = logging.getLogger(__name__)

def validate_timeout(value: Optional[float]) -> Optional[float]:
    """Validate timeout value using Pydantic constraints."""
    if value is not None:
        if value <= 0:
            raise ValidationError("The `request_timeout` argument must be a positive number.")
        if value > MAX_MODEL_CLIENT_TIMEOUT:
            raise ValidationError(f"The `request_timeout` argument can't be greater than the system's max timeout of {MAX_MODEL_CLIENT_TIMEOUT} seconds.")
    return value


# Type alias for validated timeout parameter
TimeoutParam = Annotated[
    Optional[float],
    BeforeValidator(validate_timeout),
]

def parse_openrouter_rate_limit_headers(
    headers: dict | None,
) -> tuple[int | None, float | None]:
    """Parse OpenRouter rate limit headers into (rpm_hint, retry_at_epoch_seconds).

    Assumptions for OpenRouter:
      - "x-ratelimit-limit": integer RPM limit
      - "x-ratelimit-reset": epoch in milliseconds (absolute time)

    Returns (rpm_hint, retry_at_epoch_seconds). Missing/invalid values yield None.
    """
    if not headers:
        return None, None
    try:
        norm = {str(k).lower(): v for k, v in headers.items()}
        rpm_hint: int | None = None
        retry_at_s: float | None = None
        if "x-ratelimit-limit" in norm and norm["x-ratelimit-limit"] is not None:
            rpm_hint = (
                int(norm["x-ratelimit-limit"])
                if str(norm["x-ratelimit-limit"]).isdigit()
                else None
            )
        reset_ms_val = norm.get("x-ratelimit-reset")
        if reset_ms_val is not None:
            reset_ms_f = float(reset_ms_val)
            retry_at_s = reset_ms_f / 1000.0
        return rpm_hint, retry_at_s
    except Exception:
        return None, None


def generate_completion_request_key(request: FenicCompletionsRequest) -> str:
    """Generate a standard SHA256-based key for completion request deduplication.

    Args:
        request: Completion request to generate key for

    Returns:
        10-character SHA256 hash of the messages
    """
    return hashlib.sha256(request.messages.encode()).hexdigest()[:10]


def pdf_to_base64(file: LMRequestFile) -> bytes:
    """Encode PDF file content to base64.

    Args:
        file: LMRequestFile object

    Returns:
        Base64 encoded string of the PDF content (full file from disk or chunk in memory)

    Raises:
        FileNotFoundError: If the PDF file doesn't exist
        IOError: If there's an error reading the file
    """
    if file.pdf_chunk_bytes is None:
        # Return full PDF as before
        with open(file.path, 'rb') as pdf_file:
            pdf_content = pdf_file.read()
            return base64.b64encode(pdf_content).decode('utf-8')
    else:
        pdf_chunk = fitz.open(stream=file.pdf_chunk_bytes, filetype="pdf")
        pdf_bytes = pdf_chunk.tobytes()
        pdf_chunk.close()
        return base64.b64encode(pdf_bytes).decode('utf-8')

def get_pdf_page_count(file: LMRequestFile) -> int:
    """Get the page count of a PDF file."""
    return file.page_range[1] - file.page_range[0] + 1

def get_pdf_text(file: LMRequestFile) -> str:
    """Extract text content from a PDF file."""
    text_content = []
    # Open the PDF
    if file.pdf_chunk_bytes is None:
        pdf_document = fitz.open(file.path)
    else:
        pdf_document = fitz.open(stream=file.pdf_chunk_bytes, filetype="pdf")
    
    for page_num in range(pdf_document.page_count):
        # Extract text from the page
        text_content.append(pdf_document[page_num].get_text())

    # Close the PDF
    pdf_document.close()

    # Combine all text content
    full_text = "\n".join(text_content)

    return full_text
