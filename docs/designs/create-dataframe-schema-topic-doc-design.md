# Design: `create_dataframe` schema — Topics documentation page

**Date:** 2026-07-13
**Status:** approved (brainstorming)
**Feature under documentation:** `Session.create_dataframe(data, schema=...)` (PR #329)
**Related artifacts:** `docs/plans/create-dataframe-schema-plan.md`, `docs/designs/create-dataframe-schema-design.md`

## Problem

The `schema=` argument on `Session.create_dataframe` is implemented and its API
reference is auto-generated from the (already-updated) docstring via
`mkdocstrings`. What is missing is the **conceptual layer**: a page that explains
_why_ and _when_ to make a `Schema` authoritative at ingestion, and the three
capabilities it uniquely enables (logical string-backed types, embedding
preservation, typed empty frames). The reference documents the signature; it does
not teach the concept.

## Decision

Add one hand-written **Topics** page, `docs/topics/create_dataframe_schema.md`,
wired into the `mkdocs.yml` nav under `Overview → Topics`. Model its structure on
the existing `docs/topics/timestamps_and_dates.md` (a type-behavior +
ingestion-normalization concept page — the closest existing analog).

### Why this form (not the alternatives)

- **Not a new runnable example** (`examples/create_dataframe_schema/`): would
  duplicate the existing `json_processing` / `markdown_processing` examples, and
  the embedding angle would drag in an embedding-model config. Redundant.
- **Not edits to existing example pages**: `json_processing.md` /
  `markdown_processing.md` deliberately teach `.cast(fc.JsonType/MarkdownType)`
  on columns produced mid-pipeline, which remains the correct technique when no
  ingestion schema exists. Rewriting them would remove pedagogy, not add it.
- **Not more docstring work**: the reference is already covered by the PR.

## Page outline

1. **Overview** — what an explicit `schema=` makes authoritative (field names,
   column order, physical dtypes, exact logical schema); schema-free vs
   schema-backed decision guidance.
2. **The contract** — authoritative names; reorder to `schema.column_names()`;
   physical coercion via the schema's Polars representation; typed errors
   (`ValidationError` for unsupported input / column-shape mismatch, `PlanError`
   for coercion failure / invalid schema / duplicate column names); schema-backed
   empty inputs (`[]`, `{}`, `pl.DataFrame()`, `pd.DataFrame()`, `pa.table({})`)
   → zero-row typed frames.
3. **Logical string-backed types (JSON / Markdown)** — declare `JsonType` /
   `MarkdownType` at ingestion instead of a post-hoc `.cast()`; the logical type
   stays visible in `df.schema` while stored as a physical string; explicit note
   that there is **no content validation**.
4. **Embedding preservation** — `EmbeddingType` fields keep their fixed-size
   `pl.Array(pl.Float32, dims)` representation through local execution to
   `to_polars()`, whereas schema-free arrays normalize to `pl.List`. Runnable
   snippet using literal float lists (no provider call).
5. **Relationship to schema-free ingestion** — the no-schema path is unchanged
   (still rejects `[]`, still infers types, still coerces arrays to lists);
   cross-link to `timestamps_and_dates.md` for related ingestion-normalization
   behavior.

## Constraints & success criteria

- All code snippets are real and runnable **without any API key** (verified
  against the installed fenic before commit; `EmbeddingType` frames are
  constructed from float literals + the schema, no embedding call).
- Signatures/type names verified against the source (`fc.Schema`,
  `fc.ColumnField`, `fc.JsonType`, `fc.MarkdownType`, `fc.EmbeddingType`) — no
  invented API. Lint the finished snippets conceptually with `fenic check`
  patterns where practical.
- Length ~120–180 lines, matching the lighter topic pages.
- Wired into `mkdocs.yml` nav; page renders in a local `just preview-docs` build
  without warnings.

## Scope guard — NOT doing

Mirrors the feature's non-goals: no partial-schema docs, no Pydantic-shortcut
docs, no example-script rewrites, no changes to the auto-generated reference.

## Placement & rollout

Commit the page + the one-line nav change onto the current PR #329 branch (safe
pre-teammate-review). After committing, regenerate and refresh the PR walkthrough
comment so the diff (now +1 doc file, +1 nav line, +this design doc) stays
accurately described.
