import logging
import sys

import modal
import structlog

image = (
    modal.Image.debian_slim()
    .pip_install(
        "griffe>=0.42.0",
        "fastmcp>=0.1.0",
        "fenic[google]>=0.10.0",
        "structlog>=24.1.0",
        "modal>=1.1.1",
    )
    .env({"FENIC_DATA_DIR": "/root/data"})
)
data_prep = modal.App(image=image)
volume = modal.Volume.from_name("fenic-mcp-data")


def configure_logging() -> structlog.BoundLogger:
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
