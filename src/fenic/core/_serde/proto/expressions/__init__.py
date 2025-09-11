"""Expression serialization modules.

This package contains modular expression serialization organized by expression type,
mirroring the structure of core._logical_plan.expressions.
"""

from fenic.core._serde.proto.expressions import (  # noqa: F401
    aggregate,
    basic,
    binary,
    case,
    dt,
    embedding,
    json,
    markdown,
    semantic,
    text,
)

__all__ = [
    "aggregate",
    "basic",
    "binary",
    "case",
    "dt",
    "embedding",
    "json",
    "markdown",
    "semantic",
    "text",
]