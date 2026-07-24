# specs/

Internal engineering specs and planning artifacts. **Not part of the published docs site.**

Anything in this directory is written for maintainers and agents working in this repo — design notes, td-flow workflow artifacts, and similar. It is deliberately outside `docs/` so it is never built, indexed, sitemapped, or included in the LLM-facing docs assets (`llms.txt`, `llms-full.txt`, per-page Markdown alternatives).

## Layout

| Path                           | Contents                                                                        |
| ------------------------------ | ------------------------------------------------------------------------------- |
| `specs/*.md`                   | Standalone design documents (e.g. `llm_cache_design.md`)                        |
| `specs/td-flow/<workflow_id>/` | td-flow phase artifacts — `research.md`, `design.md`, `structure.md`, `plan.md` |

## td-flow artifacts live here, not in `docs/`

**`specs/td-flow/` is the canonical location for td-flow artifacts in this repo.** They were previously written to `docs/td-flow/` and `docs/plans/`, which meant internal planning documents were built into the public documentation site and had to be individually excluded from search, the sitemap, and the LLM assets.

If you are running a td-flow phase in this repo, write the artifact to `specs/td-flow/<workflow_id>/<phase>.md`. Do not create `docs/td-flow/` or `docs/plans/`.

## Why not just `noindex` them under `docs/`?

Because it is a per-directory exclusion that has to be remembered and re-applied every time someone adds an internal doc, and a missed one is silently published. Keeping internal material out of `docs/` entirely makes the rule structural: **everything under `docs/` is public; everything here is not.**
