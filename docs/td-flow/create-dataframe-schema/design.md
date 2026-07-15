---
workflow_id: create-dataframe-schema
phase: design
track: engineering
size_class: high-risk
status: approved
portability_level: 2
source_inputs:
  - docs/td-flow/create-dataframe-schema/research.md
last_updated: 2026-06-26
---

# Design: create_dataframe Schema Handling

`Session.create_dataframe` needs a public schema contract that is stronger than inference but still fits the existing lazy plan and backend serialization model. The key decision is where to enforce the schema: at the session ingestion boundary, in the logical source, or later in physical execution.

## Chosen Approach

Add `schema: Schema | None = None` to `Session.create_dataframe` and `Session.createDataFrame`, and treat a provided schema as the authoritative top-level logical schema for the resulting `DataFrame`.

When `schema` is omitted, preserve today's behavior: normalize supported `DataLike` inputs through Polars, reject empty row lists, infer the logical schema from the resulting Polars schema, and let local source execution apply its current ingestion coercions.

When `schema` is provided, normalize input into a Polars DataFrame, enforce that the top-level data columns match the schema columns, coerce the physical data to the schema's Polars representation, then build `InMemorySource.from_schema(pl_df, schema)`. The schema order becomes the DataFrame column order. Empty `[]`, `{}`, and zero-column/zero-row frame or table inputs are allowed with a schema and produce a zero-row DataFrame with the requested schema; the current empty-list rejection remains for schema-free calls.

The local physical source must become target-aware for logical types whose physical representation can be erased by generic ingestion coercion. In particular, `EmbeddingType` maps to fixed-size `pl.Array`, while current in-memory ingestion converts arrays to `pl.List`. The plan converter should pass the logical source schema into `InMemorySourceExec`, but source execution should only use it to protect embedding-typed schema paths, not to broadly recast every column. This keeps `df.schema` and `df.to_polars().schema` consistent for explicit embeddings without changing no-schema array behavior or deserialized inferred-schema sources.

This wins because it uses contracts the codebase already has: public `Schema`/`ColumnField` exports, fenic-to-Polars schema conversion, `InMemorySource.from_schema`, wrapper-level logical-plan schema serde, and local ingestion coercion. It avoids introducing a new source node, a new schema representation, or a post-source logical projection solely to repair input types.

The grounded v0 is: full top-level schemas only, no partial schemas, no new Pydantic model schema shortcut, no new content validation pass for tagged string logical types, and no cloud protocol change. Logical types such as `JsonType`, `MarkdownType`, `HtmlType`, `TranscriptType`, and `DocumentPathType` are preserved in the fenic schema while their physical values follow the existing string-backed representation. JSON/markdown-specific parsing remains the responsibility of existing expression operators.

## Alternatives Considered

- **Pass `schema` directly to Polars constructors only** - This is small, but Polars' row-dict construction silently fills missing schema columns and drops extra row keys in some cases, which is too loose for a public fenic schema contract. It also does not address local ingestion converting explicit embedding arrays back to lists.
- **Create normally, then append a projection of `col(...).cast(...)` expressions** - This reuses the public cast path, including logical JSON and embedding casts, but it cannot handle schema-only empty inputs cleanly because the source schema must already be inferable. It also turns source creation into a transform plan, complicating source schema serde and making the logical source itself still carry the wrong schema.
- **Require physical input to already match the schema exactly** - This is simpler and stricter, but it does not deliver the main value of "force schema": common convertible values, all-null columns, empty list values, and empty DataFrames would still require users to pre-build typed Polars frames.
- **Add a new schema-enforced source plan node** - This makes the contract explicit, but it duplicates `InMemorySource` behavior and plan serde for a case the existing source can already represent with a preserved schema plus target-aware local execution.

## Contracts & Seams

Public API:

```python
def create_dataframe(
    self,
    data: DataLike,
    schema: Schema | None = None,
) -> DataFrame:
    ...

Session.createDataFrame = Session.create_dataframe
```

Schema-free calls keep the existing input contract and error behavior unless a current Polars exception message is already wrapped by `PlanError`.

Schema-provided calls use these contracts:

- `schema` must be a complete top-level `Schema`; partial schemas are not supported.
- Top-level schema names must be unique, using the existing logical-plan duplicate-name validation.
- For column-oriented inputs (`dict`, pandas, Polars, Arrow), data with any columns must have exactly the same top-level column-name set as `schema`. Zero-column/zero-row inputs are treated as schema-only empties. The result is ordered by `schema.column_fields`.
- For row-oriented `list[dict]`, the union of row keys must not contain keys outside the schema. Missing schema keys in individual rows are allowed and become nulls. Empty row lists are allowed only when `schema` is provided.
- Lists whose first item is not a dict remain unsupported, even with `schema`.
- Values are physically coerced to `convert_custom_schema_to_polars_schema(schema)`. Conversion failures are surfaced as fenic `PlanError` with the original exception as the cause.
- Unsupported top-level data types and top-level schema column-name mismatches are surfaced as fenic `ValidationError`. Lower-level construction failures, such as inconsistent column lengths or uncastable values, remain `PlanError`.

Logical source contract:

- `schema is None` continues to construct `InMemorySource.from_session_state(pl_df, session_state)`.
- `schema is not None` constructs `InMemorySource.from_schema(pl_df, schema)`.
- `InMemorySource.with_children` and plan serde continue to preserve the stored logical schema; no protobuf shape change is required because wrapper-level schema serialization already exists.

Local execution seam:

- `PlanConverter` passes `logical.schema()` to `InMemorySourceExec`.
- `InMemorySourceExec` continues to run ingestion normalization, but it must preserve schema paths whose logical type is `EmbeddingType` by avoiding or repairing array-to-list coercion for those paths.
- The schema-aware physical seam is intentionally narrow: do not recast all columns from `logical.schema()` during source execution, because `InMemorySource.from_schema` is also used by clones and plan deserialization for inferred schemas.
- No-schema `pl.Array` inputs keep today's behavior and are still normalized to `pl.List`.
- Explicit `EmbeddingType(dimensions=N, ...)` inputs materialize as `pl.Array(pl.Float32, N)` after `to_polars()`.

Test seams:

- API tests cover each `DataLike` form with `schema`, schema-driven column order, convertible primitive values, empty `[]`/`{}` and zero-column/zero-row frames with schema, no-schema empty-list behavior unchanged, unsupported list values, missing/extra top-level columns, and failed casts.
- Source/logical tests assert explicit schemas are stored on `InMemorySource`, participate in equality, and survive plan serde round trips with the same data.
- Local execution tests assert explicit `EmbeddingType` survives source execution as fixed-size `pl.Array`, while no-schema arrays still coerce to lists.
- Logical-type tests assert `JsonType`/`MarkdownType` schemas are visible to plan-time function validation and execute on string-backed data.
- Cloud-facing tests can stay at plan serialization/request-boundary level unless an existing cloud execution fixture can run in-memory sources.

## Open Questions

None.

## Decision Log

- **[applied]** Chose session-boundary schema enforcement plus `InMemorySource.from_schema` because it is the smallest change that makes the public schema authoritative before downstream plan validation.
- **[applied]** Required complete top-level schemas, not partial schemas, to match the existing CSV explicit-schema contract and avoid silently dropping data.
- **[applied]** Allowed empty `[]`, `{}`, and zero-column/zero-row frame or table inputs only when schema is provided because explicit schema supplies the otherwise-missing field names and data types.
- **[applied]** Added a narrow local physical execution seam for embedding schemas because generic in-memory ingestion currently converts fixed-size arrays to lists; the seam must not broadly recast inferred-schema sources.
- **[applied]** plan-ceo-review-lite selected HOLD scope: the direct public schema contract is worth implementing, but partial schemas, Pydantic schema shortcuts, and content validation would expand beyond the core pain.
- **[applied]** ce-doc-review-lite tightened the local execution contract so `logical.schema()` is not misread as permission to recast every deserialized `InMemorySource`.
- **[rejected]** Direct Polars constructor schema pass-through because it has input-shape behavior that is too permissive for fenic's public schema contract.
- **[rejected]** Post-source projection casting because it cannot cleanly support empty or otherwise uninferrable source data.
- **[deferred]** A separate logical-content validation pass for tagged string types, such as rejecting invalid JSON during `create_dataframe(..., schema=Schema([... JsonType ...]))`, is out of v0. Add it only if users need schema ingestion to be a content validator rather than a logical annotation plus physical coercion.

## Handoff

**Next step (paste into a fresh tab):**

> Use the `td-structure` skill. The design is approved at
> `docs/td-flow/create-dataframe-schema/design.md` (track: engineering, size: high-risk).
> Build the vertically-sliced structure outline from the chosen approach. Use the
> design + the research findings — do not read the research Questions section as a
> spec.
> If this is Codex: I explicitly permit optional subagent use for this phase
> where the skill allows it.

**Approved decisions:** Add `schema: Schema | None = None` to `Session.create_dataframe`; enforce complete top-level schema contracts at session ingestion; allow schema-backed empty inputs; preserve explicit logical schemas with `InMemorySource.from_schema`; pass source schema into local in-memory execution only to protect embedding-typed physical paths from array-to-list coercion.
**Open questions (carried forward):** None.
**Non-goals / out of scope:** Partial schemas, Pydantic schema shortcuts, new cloud protocol changes, a new source node, broad recasting of all source columns during execution, and content validation for tagged string logical types such as `JsonType`/`MarkdownType`.
**Evidence summary:** Research showed `create_dataframe` currently has no schema parameter and normalizes all supported inputs through Polars before `InMemorySource`; `InMemorySource` already supports preserved schemas and plan serde stores wrapper-level schemas; fenic-to-Polars conversion already maps all fenic data types to physical dtypes; local in-memory execution currently applies generic ingestion coercion that converts fixed-size arrays to lists.
**Known weak assumptions:** Structure should pressure-test the exact implementation of embedding-path preservation in `InMemorySourceExec`, and confirm whether strict physical coercion should use existing fenic cast semantics or Polars casts for each supported schema type.
**Next artifact:** `docs/td-flow/create-dataframe-schema/structure.md`
**Rollback if:** building the structure exposes the chosen approach is infeasible — ROLLBACK to design.
