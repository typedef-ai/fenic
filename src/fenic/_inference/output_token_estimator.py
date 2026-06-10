"""Adaptive output-token reservation estimator.

Learns the distribution of ACTUAL output tokens (completion + thinking) per
(profile_hash, max_completion_tokens) key and returns a high-quantile reservation
for the rate limiter, always clamped to the caller-supplied static ceiling (which
equals the provider's max_tokens cap). Thread-safe: written on the asyncio event
loop, read on the producer thread.
"""

import math
import threading
from collections import defaultdict, deque
from typing import Hashable

_DEFAULT_QUANTILE = 0.95
_REASONING_QUANTILE = 0.99


def _quantile(sorted_vals: list[int], q: float) -> float:
    """Linear-interpolation quantile of an already-sorted, non-empty list."""
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = q * (len(sorted_vals) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(sorted_vals[int(pos)])
    return sorted_vals[lo] * (hi - pos) + sorted_vals[hi] * (pos - lo)


class OutputTokenEstimator:
    """Per-client learned output-token reservation."""

    def __init__(
        self,
        *,
        enabled: bool,
        safety_margin: float,
        min_samples: int = 30,
        window: int = 256,
    ):
        self._enabled = enabled
        self._safety_margin = safety_margin
        self._min_samples = min_samples
        self._samples: dict[Hashable, deque[int]] = defaultdict(
            lambda: deque(maxlen=window)
        )
        self._lock = threading.Lock()

    def reserve(self, key: Hashable, *, static_ceiling: int, reasoning: bool) -> int:
        """Output-token reservation in [1, static_ceiling] (or the ceiling itself when <= 0).

        A non-positive ceiling (e.g. max_completion_tokens=None with no file
        estimate) is preserved as-is so behavior matches the static path exactly.
        """
        if static_ceiling <= 0:
            return static_ceiling
        if not self._enabled:
            return static_ceiling
        with self._lock:
            samples = self._samples.get(key)
            if samples is None or len(samples) < self._min_samples:
                return static_ceiling
            ordered = sorted(samples)
        q = _REASONING_QUANTILE if reasoning else _DEFAULT_QUANTILE
        modeled = _quantile(ordered, q) * self._safety_margin
        return max(1, min(static_ceiling, math.ceil(modeled)))

    def observe(self, key: Hashable, actual_output_tokens: int) -> None:
        """Record an actual output-token count for `key`."""
        if not self._enabled:
            return
        with self._lock:
            self._samples[key].append(actual_output_tokens)
