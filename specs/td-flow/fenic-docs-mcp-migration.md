# Fenic documentation MCP migration

## Status

Proposed migration with the baseline source import included in this repository.

## Context

The production documentation MCP server at `https://mcp.fenic.ai` is maintained in
the archived `typedef-ai/typedef-mono-archive` repository under
`services/fenic-mcp`. Updating it currently requires a manual fenic dependency
bump in both the service project and its Modal image definition, followed by
manual data preparation and deployment.

The fenic repository already owns the MCP runtime, the public API being indexed,
and a self-hosting example. The hosted documentation server therefore belongs to
fenic's release lifecycle, while remaining separate from the published `fenic`
wheel.

## Decision

Host the production service source in `services/docs-mcp` in the public fenic
repository. Keep it as an independent uv project so Modal, Griffe, and deployment
dependencies do not become runtime dependencies of the fenic package.

`mcp.fenic.ai` will track the latest stable fenic release. Deployments will be
triggered by the fenic release workflow only after the matching package is
available from PyPI.

Every stable fenic release will also export that version's API documentation to
Hugging Face. This is part of the release pipeline, not a separately maintained
or developer-local publishing process.

## Migration boundaries

The baseline import includes:

- the `fenic_mcp` Python package;
- Modal application and image configuration;
- documentation data preparation and Hugging Face export code;
- unit tests, project metadata, lockfile, and service-local commands; and
- a provenance record identifying the archived source revision.

The baseline import intentionally excludes the archived Dagger wrapper. It
depends on relative monorepo modules outside `services/fenic-mcp`, is not used by
the Modal deployment path, and cannot run after a direct copy. If container
publishing is still required, it should be reintroduced later using fenic-owned
CI rather than preserved as broken vendored infrastructure.

The import is a behavior-preserving baseline. Security fixes, dependency
upgrades, API cleanup, and consolidation with `examples/mcp_server/docs-server`
are separate reviewable changes.

## Target layout

```text
services/docs-mcp/
├── README.md
├── VENDORED_FROM.md
├── justfile
├── pyproject.toml
├── src/fenic_mcp/
├── tests/
└── uv.lock
```

The existing example remains in `examples/mcp_server/docs-server` until the
production implementation has been inspected. Afterward, reduce the example to
a small self-hosting example or make it consume shared, explicitly public code.

## Delivery phases

### 1. Establish the imported baseline

- Copy the migration boundary without functional rewrites.
- Adapt archived monorepo paths to `services/docs-mcp`.
- Record source repository, path, and commit SHA.
- Run the imported unit tests and local command-line smoke checks.

### 2. Inspect before enabling deployment

- Search the imported history and files for credentials, internal endpoints,
  private datasets, account identifiers, and sensitive log fields.
- Review every Modal secret, volume, image, timeout, concurrency, and network
  setting.
- Confirm generated documentation contains only public fenic material.
- Review licenses for copied code and service-only dependencies.
- Compare the production service with the existing docs-server example and
  decide which implementation is authoritative.
- Produce an explicit security review with blocking findings resolved before
  adding production credentials to repository workflows.

### 3. Make the release version a single input

- Remove duplicated hard-coded fenic versions from project and Modal image
  configuration.
- Accept an exact `FENIC_VERSION` from the release workflow.
- Test pull requests against the current fenic checkout.
- Build release deployments against the exact published PyPI version.
- Store the fenic version and source SHA with the generated documentation data.

### 4. Automate deployment

- Add a manually dispatchable deployment workflow first.
- Require a protected GitHub environment for production and store credentials
  only in GitHub Actions or Modal secret storage.
- After the exact fenic version is available from PyPI, generate versioned
  documentation data once for the release.
- Export that exact version's documentation Parquet artifacts to Hugging Face
  and verify the expected version is readable there.
- Run MCP contract tests before deployment.
- Deploy an immutable Modal revision and probe the public endpoint.
- Promote only after `list_tools` and a representative documentation lookup
  succeed; retain the previous revision for rollback.
- Invoke this workflow after the fenic package publication job succeeds.

The release sequence is:

```text
publish fenic to PyPI
        ↓
generate documentation for the exact released version
        ↓
publish and verify versioned Hugging Face artifacts
        ↓
run MCP contract tests against those artifacts
        ↓
deploy and verify the matching Modal revision
```

Hugging Face publication is a release gate. A failed or unverifiable export must
prevent deployment of a new MCP revision, while leaving the currently deployed
revision untouched. Rerunning the workflow for the same fenic version must be
idempotent and must not create a second logical version.

### 5. Retire the archived service

- Verify `mcp.fenic.ai` is served by the fenic-owned deployment.
- Mark the archived directory as migrated with a pointer to this repository.
- Remove any remaining manual release instructions that reference the archive.
- Keep endpoint health monitoring separate from rebuild and deployment jobs.

## Required CI checks

Pull requests changing fenic's public API or `services/docs-mcp` must run:

1. service unit tests against the fenic checkout;
2. documentation data-generation smoke tests without production credentials;
3. an in-process MCP client test that lists tools; and
4. representative successful and invalid tool calls.

Release deployment additionally requires a smoke test against the exact PyPI
version, verification of the matching Hugging Face dataset version, and a
post-deploy probe of `https://mcp.fenic.ai`.

## Security constraints

- No credential values, generated secret files, or production databases are
  committed.
- Workflows may reference secret names but must not print secret-bearing
  environment variables or URLs.
- The server receives only the secrets it needs; data preparation and serving
  use separate secret sets where practical.
- Public request handling has explicit payload, result, timeout, rate, and
  concurrency limits.
- Generated data is versioned and treated as an artifact, not mutable shared
  state.
- Deployment is allowed only from protected branches or tags and a protected
  production environment.

## Acceptance criteria

The migration is complete when:

- production source and tests have an authoritative home in fenic;
- a fenic release supplies one exact version to data generation and deployment;
- every stable release publishes and verifies versioned Hugging Face
  documentation artifacts before MCP deployment;
- CI observes the MCP server serving documentation for that version;
- `mcp.fenic.ai` is updated without editing the archived repository or running a
  developer-local release command;
- rollback to the previous deployed revision is documented and tested; and
- the archived service is clearly tombstoned.

## Non-goals

- Shipping the hosted server inside the `fenic` wheel.
- Changing the public MCP tool contract during the baseline import.
- Replacing Modal as part of this migration.
- Publishing deployment secrets or generated production data.
- Preserving obsolete monorepo-specific build infrastructure unchanged.
