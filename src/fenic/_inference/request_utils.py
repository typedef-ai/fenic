"""Utilities for request processing and deduplication."""

import base64
import hashlib
from typing import List, Tuple

import fitz  # PyMuPDF

from fenic._inference.types import FenicCompletionsRequest


def generate_completion_request_key(request: FenicCompletionsRequest) -> str:
    """Generate a standard SHA256-based key for completion request deduplication.

    Args:
        request: Completion request to generate key for

    Returns:
        10-character SHA256 hash of the messages
    """
    return hashlib.sha256(request.messages.encode()).hexdigest()[:10]


def pdf_to_base64(pdf_path: str) -> str:
    """Encode PDF file content to base64.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Base64 encoded string of the PDF content

    Raises:
        FileNotFoundError: If the PDF file doesn't exist
        IOError: If there's an error reading the file
    """
    with open(pdf_path, 'rb') as pdf_file:
        pdf_content = pdf_file.read()
        return base64.b64encode(pdf_content).decode('utf-8')


def get_pdf_text_and_image_sizes(pdf_path: str) -> Tuple[str, List[Tuple[int, int]]]:
    """Extract text content and image dimensions from a PDF file.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Tuple containing:
            - Complete text content from all pages
            - List of (width, height) tuples for all images in the document

    Raises:
        FileNotFoundError: If the PDF file doesn't exist
        RuntimeError: If there's an error processing the PDF
    """
    text_content = []
    image_sizes = []

    try:
        # Open the PDF
        pdf_document = fitz.open(pdf_path)

        # Process each page
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]

            # Extract text from the page
            text_content.append(page.get_text())

            # Extract image information
            image_list = page.get_images(full=True)

            for _img_index, img in enumerate(image_list):
                # img[0] is the xref (cross-reference number)
                xref = img[0]

                # Get the image object
                pix = fitz.Pixmap(pdf_document, xref)

                # Get image dimensions
                width = pix.width
                height = pix.height

                image_sizes.append((width, height))

                # Clean up the pixmap
                pix = None


        # Close the PDF
        pdf_document.close()

        # Combine all text content
        full_text = "\n".join(text_content)

        return full_text, image_sizes

    except Exception as e:
        raise RuntimeError("Error processing PDF") from e
