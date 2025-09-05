from typing import Dict, List

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
    if lm_request_messages.user_file_path:
        # file - use structured content with file
        user_message = {"role": "user", "content": [
            {
                "type": "file",
                "content": {
                    "filename": lm_request_messages.user_file_path,
                    "file_data": lm_request_messages.user_file_path
                }
            }
        ]}
        messages.append(user_message)
    return messages