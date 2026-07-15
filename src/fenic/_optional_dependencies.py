"""Helpers for loading optional dependencies with actionable errors."""

from __future__ import annotations

import importlib
from types import ModuleType


def import_optional_dependency(
    module_name: str,
    *,
    extra: str,
    feature: str,
) -> ModuleType:
    """Import an optional dependency or raise an install hint for the feature."""
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"To use {feature}, install the '{extra}' extra: "
            f"pip install \"fenic[{extra}]\""
        ) from exc
