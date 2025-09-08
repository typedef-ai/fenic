"""Provider routing shared types and constants.

Defines the allowed provider sorting strategies for OpenRouter provider routing.
"""
from __future__ import annotations

from typing import Literal

# Sorting strategies supported by OpenRouter provider routing
ProviderSort = Literal["price", "throughput", "latency"]
DataCollection = Literal["allow", "deny"]
ModelQuantization = Literal[
    "int4",
    "int8",
    "fp4",
    "fp6",
    "fp8",
    "fp16",
    "bf16",
    "fp32",
    "unknown",
]


