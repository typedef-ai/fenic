"""Cache key generation for LLM requests."""

import hashlib
import json

from fenic._inference.types import FenicCompletionsRequest


class CacheKeyGenerator:
    """Generates deterministic cache keys from LLM requests.

    Cache keys are computed using SHA-256 hashing of request parameters to ensure
    deterministic and collision-resistant identification of unique requests.

    Example:
        Computing a cache key:

        ```python
        from fenic._inference.types import FenicCompletionsRequest, LMRequestMessages

        request = FenicCompletionsRequest(
            messages=LMRequestMessages(
                system="You are helpful", examples=[], user="Hi"
            ),
            max_completion_tokens=100,
            top_logprobs=None,
            structured_output=None,
            temperature=0.7,
        )

        key = CacheKeyGenerator.compute_key(request, "gpt-4o-mini")
        print(f"Cache key: {key}")  # 64-character hex string
        ```
    """

    @staticmethod
    def compute_key(request: FenicCompletionsRequest, model: str) -> str:
        """Compute SHA-256 hash of request parameters.

        Includes all parameters that affect the response:
        - model name
        - messages (system, examples, user text/file)
        - max_completion_tokens
        - temperature
        - structured_output schema (if present)
        - model_profile (if present)
        - top_logprobs (if present)

        Args:
            request: The completion request to hash.
            model: The model name.

        Returns:
            64-character hexadecimal SHA-256 hash string.

        Example:
            ```python
            key1 = CacheKeyGenerator.compute_key(request1, "gpt-4o-mini")
            key2 = CacheKeyGenerator.compute_key(request2, "gpt-4o-mini")

            if key1 == key2:
                print("Identical requests - cache hit!")
            else:
                print("Different requests - cache miss")
            ```
        """
        # Build key data with all relevant parameters
        key_data = {
            "model": model,
            "messages": request.messages.encode().hex(),
            "max_tokens": request.max_completion_tokens,
            "temperature": request.temperature,
            "model_profile": request.model_profile,
            "top_logprobs": request.top_logprobs,
        }

        # Include structured output schema if present
        if request.structured_output:
            key_data["structured_output"] = json.dumps(
                request.structured_output.json_schema,
                sort_keys=True,
                separators=(",", ":"),
            )

        # Serialize to JSON with deterministic ordering
        serialized = json.dumps(key_data, sort_keys=True).encode("utf-8")

        # Compute SHA-256 hash
        return hashlib.sha256(serialized).hexdigest()
