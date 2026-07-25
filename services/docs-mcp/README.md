# Fenic MCP

This is the hosted Fenic documentation MCP server. It uses Fenic's native
parameterized-tool system for all documentation tools:

- `search_fenic_api` performs ranked, case-insensitive multi-keyword search.
- `get_entity` returns the full record for an exact qualified API name.
- `get_entities` returns up to ten full entity records in one call.
- `get_project_overview` returns a precomputed project summary and API tree.
- `get_api_tree` returns the precomputed compact API tree.

Serving does not require model-provider secrets. Data preparation uses a
language model to generate the project summary, precomputes all zero-argument
tool results, and writes the documentation catalog to the Modal volume.

## Local verification

```bash
cd services/docs-mcp
uv sync
uv run pytest
uv run fenic check src/fenic_mcp/server/native.py
```

After preparing data, run the server at `http://127.0.0.1:8000/mcp`:

```bash
FENIC_DATA_DIR=/path/to/prepared/data uv run fenic-mcp
```

## Bumping the fenic version for data prep

To update which version of `fenic` the documentation/data-prep uses, you must change it in two places so local runs and Modal runs stay in sync:

- **Update local dependency**: edit `services/docs-mcp/pyproject.toml` and bump the `fenic[...]` spec under `[project].dependencies`.
- **Update Modal image dependency**: edit `services/docs-mcp/src/fenic_mcp/modal_setup.py` and bump the `fenic[...]` spec in the `.pip_install(...)` list.

After updating both:

- **Verify locally**:

```bash
just fenic-mcp-data-prep
```

- **Verify on Modal** (builds the image with the new version):

```bash
just fenic-mcp-data-prep-modal
```

The data prep logs include the `fenic` version ("Using fenic version: X.Y.Z"). If local shows the new version but Modal does not, ensure `modal_setup.py` was updated and re-run the Modal command so the image rebuilds.
