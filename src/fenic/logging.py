"""Logging configuration utilities for Fenic."""

import logging
import sys
from typing import Optional, TextIO
from rich.console import Console
from rich.logging import RichHandler

_shared_console = Console()

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

    # Only configure if root logger has no handlers
    root_logger = logging.getLogger()
    if not root_logger.hasHandlers():
        handler = RichHandler(
            console=_shared_console,
            rich_tracebacks=True,
            markup=True,
            show_time=False  # or True if you want timestamps in Rich
        )

        # Use user-supplied or default format (Rich usually handles format itself)
        formatter = logging.Formatter(log_format or "%(message)s")
        handler.setFormatter(formatter)

        root_logger.setLevel(log_level)
        root_logger.addHandler(handler)

        # Silence noisy dependencies
        for noisy in ("openai", "httpx"):
            logging.getLogger(noisy).setLevel(logging.ERROR)

    # Make sure your own loggers propagate to root
    library_root_name = __name__.split(".")[0]
    logging.getLogger(library_root_name).setLevel(log_level)
    logging.getLogger(library_root_name).propagate = True
