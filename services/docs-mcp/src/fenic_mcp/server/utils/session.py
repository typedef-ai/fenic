import os
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version

import fenic as fc
import structlog

logger = structlog.get_logger(__name__)


def create_session():
    # Determine data directory: prefer FENIC_DATA_DIR, otherwise default to ./data
    work_dir = os.environ.get("FENIC_DATA_DIR") or "./data"
    logger.info(f"Using Fenic data directory: {work_dir}")
    os.makedirs(work_dir, exist_ok=True)
    os.chdir(work_dir)

    # Set DuckDB temp directory
    os.environ["DUCKDB_TMPDIR"] = work_dir

    return fc.Session.get_or_create(fc.SessionConfig(app_name="docs"))


def log_fenic_version() -> None:
    """Log the version of the installed fenic package."""
    try:
        version = getattr(fc, "__version__", None) or pkg_version("fenic")
    except PackageNotFoundError:
        version = "unknown"
    logger.info(f"Using fenic version: {version}")
