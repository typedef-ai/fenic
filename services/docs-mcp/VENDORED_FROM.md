# Import provenance

This directory was imported from:

- Repository: `https://github.com/typedef-ai/typedef-mono-archive`
- Path: `services/fenic-mcp`
- Commit: `211e4342f453bb8c3e6fa2cae401ba05211d6158`

The data extraction, Hugging Face export, deployment support, project metadata,
and lockfile originated in that revision. References to the service directory
were changed from `services/fenic-mcp` to `services/docs-mcp`.

The archived FastMCP server, regex search/validation helpers, bundled resource,
and their dedicated tests were removed after the service moved to Fenic's
native parameterized-tool server in `fenic_mcp.server.native`.

The archived `.dagger` directory, `dagger.json`, and container-publishing recipe
were not copied because they depend on monorepo-relative Dagger modules outside
the archived service. See `specs/td-flow/fenic-docs-mcp-migration.md` for the
migration boundary and follow-up work.
