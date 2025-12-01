"""Shared helpers for building deterministic request/cache keys."""

from __future__ import annotations

import hashlib
import json
from typing import Optional, Union

from fenic._inference.types import (
    FenicCompletionsRequest,
    FenicEmbeddingsRequest,
)


def compute_request_fingerprint(
    request: Union[FenicCompletionsRequest, FenicEmbeddingsRequest],
    model: str,
    profile_hash: Optional[str] = None,
) -> str:
    """Build a deterministic SHA-256 key for the given request.

    Args:
        request: The request object to fingerprint.
        model: The model name for the request.
        profile_hash: Optional hash of the resolved profile configuration.

    Returns:
        A 64-character hexadecimal string representing the request.

    Raises:
        NotImplementedError: If the request type is not yet supported.
        ValueError: If the request type is unrecognized.
    """
    if isinstance(request, FenicCompletionsRequest):
        key_data = {
            "model": model,
            "messages": request.messages.encode().hex(),
            "max_tokens": request.max_completion_tokens,
            "temperature": request.temperature,
            "model_profile": request.model_profile,
            "profile_hash": profile_hash,
            "top_logprobs": request.top_logprobs,
        }

        if request.structured_output:
            key_data["structured_output"] = request.structured_output.schema_fingerprint
    elif isinstance(request, FenicEmbeddingsRequest):
        key_data = {
            "model": model,
            "doc_hash": hashlib.sha256(request.doc.encode("utf-8")).hexdigest(),
            "model_profile": request.model_profile,
            "profile_hash": profile_hash,
        }
    else:
        raise ValueError(f"Unsupported request type for caching: {type(request)}")

    serialized = json.dumps(key_data, sort_keys=True).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()

