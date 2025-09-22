"""Utilities for request processing and deduplication."""

import hashlib
import json

from fenic._inference.types import FenicCompletionsRequest


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
    messages_json = json.dumps(request.messages.to_message_list(), sort_keys=True)
    return hashlib.sha256(messages_json.encode()).hexdigest()[:10]
