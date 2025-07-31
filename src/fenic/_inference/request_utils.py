"""Utilities for request processing and deduplication."""

import hashlib
import json

from fenic._inference.types import FenicCompletionsRequest


def generate_completion_request_key(request: FenicCompletionsRequest) -> str:
    """Generate a standard SHA256-based key for completion request deduplication.

    Args:
        request: Completion request to generate key for
        
    Returns:
        10-character SHA256 hash of the messages
    """
    messages_json = json.dumps(request.messages.to_message_list(), sort_keys=True)
    return hashlib.sha256(messages_json.encode()).hexdigest()[:10]
