# Import provenance

This directory was imported from:

- Repository: `https://github.com/typedef-ai/typedef-mono-archive`
- Path: `services/fenic-mcp`
- Commit: `211e4342f453bb8c3e6fa2cae401ba05211d6158`

The Python package, tests, project metadata, and lockfile are preserved from that
revision. References to the service directory were changed from
`services/fenic-mcp` to `services/docs-mcp`. The broken console-script target
`fenic_mcp.server:main` was corrected to `fenic_mcp.server.mcp:main` so the
declared executable resolves to the archived server entry point.

The archived `.dagger` directory, `dagger.json`, and container-publishing recipe
were not copied because they depend on monorepo-relative Dagger modules outside
the archived service. See `docs/plans/fenic-docs-mcp-migration.md` for the
migration boundary and follow-up work.
