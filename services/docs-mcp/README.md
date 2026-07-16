# Fenic MCP

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
