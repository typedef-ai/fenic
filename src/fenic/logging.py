"""Logging configuration utilities for Fenic."""

import logging
import sys
from typing import Optional, TextIO

NOISY_LOGGER_NAMES = ("openai", "httpx", "google_genai", "cohere", "anthropic")


def configure_logging(
    log_level: int = logging.INFO,
    log_format: str = "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    log_stream: Optional[TextIO] = None,
) -> None:
    """Configure logging for the library and root logger in interactive environments.

    This function ensures that logs from the library's modules appear in output by
    setting up a default handler on the root logger *only if* one does not already
    exist. This is especially useful in notebooks, scripts, or REPLs where logging
    is often unset. It configures the root logger and sets the library's top-level
    logger to propagate logs to the root.

    If the root logger has no handlers, this function sets up a default configuration
    and silences noisy dependencies like 'openai' and 'httpx'.

    In more complex applications or when integrating with existing logging
    configurations, you might prefer to manage logging setup externally. In such
    cases, you may not need to call this function.
    """
    stream = log_stream or sys.stderr
    formatter = logging.Formatter(log_format)
    handler = logging.StreamHandler(stream)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    if not root_logger.hasHandlers():
        # Set up root logger only if not already configured
        root_logger.setLevel(log_level)
        root_logger.addHandler(handler)

        # Silence noisy dependencies
        for noisy_logger_name in NOISY_LOGGER_NAMES:
            noisy_logger = logging.getLogger(noisy_logger_name)
            noisy_logger.setLevel(logging.ERROR)

    # Set the library logger level and enable propagation
    library_root_name = __name__.split(".")[0]
    library_logger = logging.getLogger(library_root_name)
    library_logger.setLevel(log_level)
    library_logger.propagate = True
