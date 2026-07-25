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

## Release deployment

Every tagged Fenic release invokes `.github/workflows/deploy_docs_mcp.yaml` after
its exact wheel has been published to PyPI. The protected `production`
environment gates the workflow. It exports and verifies the versioned Hugging
Face dataset, tests the native MCP contract, prepares the Modal volume, deploys
an image pinned to `fenic[mcp]==$FENIC_VERSION`, and probes the public endpoint.

The workflow is also manually dispatchable for safe reruns. Select the release
tag in GitHub's **Run workflow** branch selector; the version is derived from
that tag, and existing Hugging Face artifacts are verified and reused.

Configure these secrets on the GitHub `production` environment:

- `HF_TOKEN`: write access to `typedef-ai/fenic-codebase`
- `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`: deployment access to the Modal app
- `OPENAI_API_KEY`: project-summary generation during Hugging Face export

Modal must separately contain a secret named `llm_api_keys` with
`OPENAI_API_KEY`; the remote data-preparation function consumes that secret.
PyPI publishing uses GitHub OIDC and requires no repository secret.

For local data preparation, the project dependency remains a lower bound:

```bash
just fenic-mcp-data-prep
```

To exercise an exact release manually:

```bash
FENIC_VERSION=0.10.0 just fenic-mcp-modal-deploy
```
