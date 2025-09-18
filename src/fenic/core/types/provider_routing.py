"""Provider routing shared types and constants.

Defines the allowed provider sorting strategies for OpenRouter provider routing.
"""

from __future__ import annotations

from typing import Literal

# Sorting strategies supported by OpenRouter provider routing
ProviderSort = Literal["price", "throughput", "latency"]
"""
Type alias representing provider sorting strategies used by OpenRouter routing.

Valid values:

- "price": Prefer providers with the lowest recent price.
- "throughput": Prefer providers with the highest recent throughput.
- "latency": Prefer providers with the lowest recent latency.
"""
DataCollection = Literal["allow", "deny"]
"""
Type alias representing provider data collection policies.

Valid values:

- "allow": Permit providers that may retain or train on prompts non-transiently.
- "deny": Restrict to providers that do not collect/store user data.
"""
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
"""
Type alias representing supported quantization formats for provider models.

Common values:

- "int4", "int8": Integer quantization for smaller, faster models.
- "fp4", "fp6", "fp8": Low-precision floating point formats.
- "fp16", "bf16": Half-precision formats commonly used on GPUs/TPUs.
- "fp32": Full precision floating point.
- "unknown": Provider did not specify a quantization.
"""

StructuredOutputStrategy = Literal["prefer_tools", "prefer_response_format"]
"""
Type alias representing the strategy to use when a model supports both
tool-calling and response-format-based structured outputs.

Valid values:

- "prefer_tools": Prefer tool/function calling with a JSON schema.
- "prefer_response_format": Prefer response_format structured outputs.
"""
