"""Shared Modal image, application, volume, and logging configuration."""

import logging
import os
import sys

import modal
import structlog

from fenic_mcp.release import normalize_fenic_version


def _fenic_requirement() -> str:
    """Pin release images while retaining a convenient local development default."""
    version = os.environ.get("FENIC_VERSION")
    if version is None:
        return "fenic[mcp]>=0.10.0"
    return f"fenic[mcp]=={normalize_fenic_version(version)}"

image = (
    modal.Image.debian_slim()
    .pip_install(
        "griffe>=0.42.0",
        _fenic_requirement(),
        "structlog>=24.1.0",
        "modal>=1.1.1",
    )
    .env(
        {
            "FENIC_DATA_DIR": "/root/data",
            "FENIC_VERSION": os.environ.get("FENIC_VERSION", ""),
            "FENIC_SOURCE_SHA": os.environ.get("FENIC_SOURCE_SHA", ""),
        }
    )
)
data_prep = modal.App(image=image)
volume = modal.Volume.from_name("fenic-mcp-data")


def configure_logging() -> structlog.BoundLogger:
    """Configure structured logging for Modal entrypoints."""
    structlog.configure(
        processors=[
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    # Configure the standard library logging to output to stdout
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )
    logger = structlog.get_logger(__name__)
    return logger
