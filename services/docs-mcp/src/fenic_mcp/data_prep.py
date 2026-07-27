"""Modal entrypoint for generating the documentation catalog."""

import modal

from fenic_mcp.modal_setup import configure_logging, data_prep, volume
from fenic_mcp.setup.populate_tables import populate_tables

logger = configure_logging()


@data_prep.function(
    volumes={"/root/data": volume}, secrets=[modal.Secret.from_name("llm_api_keys")]
)
def perform_data_preparation():
    """Generate the production tables in the shared Modal volume."""
    populate_tables("/root/data")
