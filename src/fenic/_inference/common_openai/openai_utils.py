from typing import Dict, List

from fenic._inference.request_utils import pdf_to_base64
from fenic._inference.types import LMRequestMessages


def convert_messages(lm_request_messages: LMRequestMessages, supports_system_messages: bool) -> List[Dict[str, str]]:
    """Convert Fenic messages to OpenAI format.

    Any files are converted to base64 encoded strings."""
    messages = []
    if supports_system_messages:
        messages.append({"role": "system", "content": lm_request_messages.system})
    else:
        messages.append({"role": "user", "content": lm_request_messages.system})

    for example in lm_request_messages.examples:
        messages.append({"role": "user", "content": example.user})
        messages.append({"role": "assistant", "content": example.assistant})

    # Handle user message based on type of content
    if lm_request_messages.user and lm_request_messages.user_file_path:
        # Both text and file - use structured content
        user_message = {"role": "user", "content": []}
        user_message["content"].append({
            "type": "input_text",
            "content": lm_request_messages.user
        })
        user_message["content"].append({
            "type": "file",
            "file": {
                "filename": lm_request_messages.user_file_path,
                "file_data": f"data:application/pdf;base64,{pdf_to_base64(lm_request_messages.user_file_path)}",
            }
        })
        messages.append(user_message)
    elif lm_request_messages.user:
        # Just text - use simple string content
        messages.append({"role": "user", "content": lm_request_messages.user})
    elif lm_request_messages.user_file_path:
        # Just file - use structured content with file only
        user_message = {"role": "user", "content": [
            {
                "type": "file",
                "file": {
                    "filename": lm_request_messages.user_file_path,
                    "file_data": f"data:application/pdf;base64,{pdf_to_base64(lm_request_messages.user_file_path)}",
                }
            }
        ]}
        messages.append(user_message)
    return messages
