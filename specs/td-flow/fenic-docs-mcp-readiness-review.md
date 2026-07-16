# Fenic documentation MCP readiness review

## Outcome

`services/docs-mcp` is the authoritative implementation for the hosted
`mcp.fenic.ai` service. It is the code currently deployed, owns Modal and
Hugging Face integration, and exposes the live four-tool contract.

`examples/mcp_server/docs-server` remains a self-hosting example. It must not
claim to define the hosted service's exact tool contract or release behavior.
After the production service is modernized, the example should either be
reduced to a small native-fenic demonstration or deliberately share a stable,
public library seam with the service.

Do not deploy the imported lockfile unchanged. The source is suitable for a
public repository, but dependency and least-privilege work listed below must
land before production automation is enabled.

## Evidence

- The live `https://mcp.fenic.ai` health route returned HTTP 200 during this
  review.
- A live MCP client listed `get_api_tree`, `get_entity`,
  `get_project_overview`, and `search`, matching `services/docs-mcp`.
- The example instead advertises `search_fenic_api` and `search_by_type` and
  uses fenic's catalog-generated MCP server.
- `services/docs-mcp` contains Modal deployment, per-container DuckDB copying,
  Hugging Face version export, release-tag discovery, regex validation, and the
  tests for the deployed search behavior. The example contains none of the
  deployment or Hugging Face lifecycle.

## Security posture

The focused daily audit found no reportable vulnerability at the 8/10
confidence threshold.

### Public-repository review

- No credential value was found in the imported source or baseline commit.
- Modal, Hugging Face, OpenAI, and Google credentials are referenced by secret
  or environment-variable name only.
- The exported dataset is derived from the already-public fenic source and API
  documentation.
- Publishing the Modal app name, volume name, custom domain, and deployment
  structure does not cross a security boundary.
- The runtime tools expose read-only searches over public documentation and do
  not accept paths, URLs, commands, arbitrary SQL, or executable code.

### Independently filtered candidates

The lockfile resolves FastMCP 2.11.3 and MCP 1.12.4. A dependency audit found
advisories in FastMCP's OAuth, OpenAPI-provider, client-callback, and local
installer features. This service uses none of those features: it constructs an
unauthenticated server with fixed read-only tools. The MCP SDK DNS-rebinding
advisory concerns localhost trust boundaries, while this service intentionally
publishes the same documentation on a public domain. Independent verification
therefore scored current exploitability 3/10, below the reporting threshold.

The Modal serving function also receives the `llm_api_keys` secret even though
only the separate data-preparation function performs LLM work. No serving code
reads or returns those variables, and no arbitrary execution path was found, so
independent verification also scored current exploitability 3/10. The secret
attachment still violates least privilege and should be removed before the next
deployment.

## Pre-deployment blockers

1. Replace `fastmcp>=0.1.0` with the supported FastMCP 3.x range already used by
   fenic, regenerate `uv.lock`, and rerun the dependency audit. FastMCP 2.x is
   unsupported upstream even though the reviewed advisories are not reachable
   through this server.
2. Remove `llm_api_keys` from the Modal serving function. Retain it only on the
   data-preparation function.
3. Replace the duplicated `fenic[google]>=0.10.0` declarations in
   `pyproject.toml` and the Modal image with one exact release input.
4. Decide whether production keeps the live `search` contract or deliberately
   introduces the example's differently named tools. Treat any rename as a
   versioned user-facing change, not incidental consolidation.
5. Add a repository secret-scanning policy such as `.gitleaks.toml` before
   deployment workflows begin receiving Modal and Hugging Face credentials.

## Implementation comparison

| Concern            | `services/docs-mcp`                                      | Example docs server                                          | Decision                                                           |
| ------------------ | -------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------ |
| Live contract      | Four tools including `search`                            | Five tools including `search_fenic_api` and `search_by_type` | Preserve the live service contract until an explicit API migration |
| MCP construction   | Direct FastMCP server                                    | Fenic catalog tools and `create_mcp_server`                  | Move production toward fenic-native MCP after contract tests exist |
| Search behavior    | Ranked regex search with validation and result limits    | Catalog regex filters without the same validation/ranking    | Production behavior remains authoritative                          |
| Deployment         | Modal custom domain, volume, snapshot, concurrency       | Local server only                                            | Service owns deployment                                            |
| Release data       | Modal data preparation and versioned Hugging Face export | Local database population                                    | Service owns release data                                          |
| Dependencies       | Independent locked application, currently stale          | Broad example requirements aligned with current fenic        | Refresh service lock; do not deploy the example                    |
| Documentation role | Operator and release documentation                       | User-facing self-hosting tutorial                            | Keep both roles separate                                           |

## Consolidation direction

The long-term target is not two implementations. Keep deployment and release
code in `services/docs-mcp`, then replace its hand-built FastMCP wiring with
fenic-native MCP construction where the live behavior can be preserved. Lock
the four live tool schemas and representative results with contract tests before
that refactor. Once production uses the native path, rewrite the example as a
small consumer of the same supported public APIs and remove its claims about the
hosted server's exact implementation.

## Next implementation slice

The next slice should address the five pre-deployment blockers without adding a
release workflow yet. Its acceptance gate is:

- a supported, audited dependency lock;
- no LLM credentials attached to the serving container;
- one exact fenic version input for local and Modal environments;
- contract tests for all four live tools; and
- unchanged live-compatible tool names and response behavior.

This is an AI-assisted first-pass security review, not a substitute for a
professional penetration test. Production systems should still receive an
independent security assessment appropriate to their data and operational risk.
