"""Modal deployment for the hosted Fenic documentation MCP server."""

import os
import shutil
import tempfile
import uuid

import modal

import fenic as fc
from fenic_mcp.modal_setup import configure_logging, image, volume
from fenic_mcp.server.native import create_native_server

logger = configure_logging()

app = modal.App(name="fenic-docs", image=image)


@app.function(
    volumes={"/root/data": volume},
    max_containers=5,
    enable_memory_snapshot=True,
    scaledown_window=300,
    timeout=1800,
)
@modal.concurrent(max_inputs=64)
@modal.asgi_app(custom_domains=["mcp.fenic.ai"])
def mcp_server():
    """Create the stateless ASGI application used by the Modal deployment."""
    logger.info("Starting MCP server...")
    # Copy shared DB/data from the mounted volume into a per-container tmp dir
    # to avoid concurrent writes to the same DuckDB file across containers.
    try:
        src_dir = "/root/data"
        if os.path.isdir(src_dir):
            # Create a unique, private directory under the system temp dir
            temp_base = tempfile.gettempdir()
            dst_dir = os.path.join(temp_base, f"fenic-data-{uuid.uuid4().hex}")
            shutil.copytree(src_dir, dst_dir)
            os.chmod(dst_dir, 0o700)
            os.environ["FENIC_DATA_DIR"] = dst_dir
            logger.info("Using per-container data directory", src=src_dir, dst=dst_dir)
    except Exception as e:
        # If copy fails, continue with the mounted volume to avoid downtime
        logger.warning("Falling back to mounted data directory", error=str(e))
    server = create_native_server()
    return fc.run_mcp_server_asgi(server, path="/", stateless_http=True)
