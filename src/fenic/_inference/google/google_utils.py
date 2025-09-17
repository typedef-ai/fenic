import io
import logging
from typing import List, Optional, Tuple

from google.genai import Client
from google.genai.types import Content, ContentUnion, File, Part

from fenic._inference.types import LMRequestFile, LMRequestMessages

logger = logging.getLogger(__name__)

async def _upload_file(client: Client, file: LMRequestFile) -> File:
    """Upload a file to Google's file store.

    Args:
        client: Google genai client
        file_path: Path to the file to upload
        page_range: Optional page range for PDF files (start, end) inclusive, 0-indexed

    Returns:
        Uploaded file object
    """
    if file.pdf_chunk_bytes:
        # We keep pdf chunks in memory
        buffer = io.BytesIO(file.pdf_chunk_bytes)
        logger.debug(f"Uploading file {file.path} size {len(file.pdf_chunk_bytes)} pages {file.page_range[0]}-{file.page_range[1]+1} to Google file store")
        file_obj = await client.files.upload(file=buffer, config={"mime_type": "application/pdf"})
        logger.debug(f"Uploaded file {file.path} pages {file.page_range[0]}-{file.page_range[1]+1} to Google file store: {file_obj.name}")
    else:
        logger.debug(f"Uploading file {file.path} to Google file store: {file.path}")
        file_obj = await client.files.upload(file=file.path, config={"mime_type": "application/pdf"})
        logger.debug(f"Uploaded file {file.path} to Google file store: {file_obj.name}")
    return file_obj

async def delete_file(client: Client, file_name: str) -> None:
    """Delete a file from Googl e's file store."""
    await client.files.delete(name=file_name)
    logger.info(f"Deleted file {file_name} from Google file store.")

async def convert_messages_and_upload_files(client: Client, messages: LMRequestMessages) -> Tuple[List[ContentUnion], File]:
    """Convert Fenic LMRequestMessages → list of google-genai `Content` objects.

    Converts Fenic message format to Google's Content format, including
    few-shot examples and the final user prompt.

    If a file is included in the messages, upload it to Google's file store and add the file object to the end of the contents list.

    Args:
        messages: Fenic message format

    Returns:
        List of Google Content objects
    """
    contents = convert_text_messages(messages)
    file_obj = None
    if messages.user_file:
        file_obj = await _upload_file(client, messages.user_file)
        contents.append(file_obj)
    return contents, file_obj


def convert_text_messages(messages: LMRequestMessages) -> Tuple[List[ContentUnion], Optional[File]]:
    """Convert Fenic LMRequestMessages → list of google-genai `Content` objects.

    Converts Fenic message format to Google's Content format, including
    few-shot examples and the final user prompt.

    Only returns the messages with text content.  Ignores any files in the messages.

    Args:
        messages: Fenic message format

    Returns:
        List of Google Content objects with text content
    """
    contents: List[ContentUnion] = []
    # few-shot examples
    for example in messages.examples:
        contents.append(
            Content(
                role="user", parts=[Part(text=example.user)]
            )
        )
        contents.append(
            Content(
                role="model", parts=[Part(text=example.assistant)]
            )
        )

    # final user prompt
    if messages.user:
        contents.append(
            Content(
                role="user", parts=[Part(text=messages.user)]
            )
        )
    return contents