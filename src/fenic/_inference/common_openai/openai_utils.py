from typing import Dict, List

from fenic._inference.request_utils import pdf_to_base64
from fenic._inference.types import LMRequestMessages


def convert_messages(lm_request_messages: LMRequestMessages) -> List[Dict[str, str]]:
    messages = [{"role": "system", "content": lm_request_messages.system}]

    for example in lm_request_messages.examples:
        messages.append({"role": "user", "content": example.user})
        messages.append({"role": "assistant", "content": example.assistant})

    # Handle user message based on type of content
    if lm_request_messages.user:
        # text - use simple string content
        messages.append({"role": "user", "content": lm_request_messages.user})
    if lm_request_messages.user_file:
        # file - use structured content with file
        user_message = {"role": "user", "content": [
            {
                "type": "file",
                "file": {
                    "filename": lm_request_messages.user_file.path,
                    "file_data": f"data:application/pdf;base64,{pdf_to_base64(lm_request_messages.user_file)}",
                }
            }
        ]}
        messages.append(user_message)
    return messages