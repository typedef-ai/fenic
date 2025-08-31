"""Provider routing shared types and constants.

Defines the allowed provider sorting strategies for OpenRouter provider routing.
"""
from __future__ import annotations

from typing import Literal

# Sorting strategies supported by OpenRouter provider routing
ProviderSort = Literal["price", "throughput", "latency"]

__all__ = ["ProviderSort"]


