---
workflow_id: create-dataframe-schema
phase: structure
track: engineering
size_class: high-risk
status: approved
portability_level: 3
source_inputs:
  - docs/brainstorms/create-dataframe-schema-research.md
  - docs/designs/create-dataframe-schema-design.md
last_updated: 2026-06-26
---

# Structure: create_dataframe Schema Handling

Add an explicit schema path to `Session.create_dataframe` that makes a provided `Schema` authoritative at the ingestion boundary while preserving current inference behavior when no schema is supplied. The work is sliced from public API behavior through downstream logical and physical execution so each phase can be verified before the next phase begins.

## Desired End State

- `Session.create_dataframe(data, schema=...)` and `Session.createDataFrame(data, schema=...)` accept a complete top-level `Schema`.
- Schema-free calls keep today's normalization and error behavior, including rejecting empty row lists.
- Schema-backed calls validate top-level column shape, reorder to `schema.column_fields`, coerce physical data through `convert_custom_schema_to_polars_schema`, and create `InMemorySource.from_schema`.
- Empty `[]`, `{}`, and zero-column/zero-row frame/table inputs produce zero-row DataFrames with the requested schema only when a schema is provided.
- Logical string-backed types such as `JsonType` and `MarkdownType` remain visible in `df.schema` and downstream plan validation while storing physical strings.
- Explicit `EmbeddingType` columns materialize locally as fixed-size `pl.Array(pl.Float32, dimensions)` after `to_polars()`, while no-schema arrays still normalize to `pl.List`.
- No new source node, public schema representation, cloud protocol shape, partial-schema mode, Pydantic schema shortcut, or tagged-string content validation is introduced.

## Implementation Overview

- [x] Phase 1: Basic schema-backed creation end to end
- [x] Phase 2: Complete input-shape and error contract
- [ ] Phase 3: Logical schema propagation through planning, serde, and cloud request boundaries
- [ ] Phase 4: Local embedding physical preservation

---

## ✅ Phase 1: Basic schema-backed creation end to end

This phase makes the smallest useful public behavior work: a provided primitive schema can create a DataFrame from common in-memory Python inputs, preserve the logical schema, order columns by the schema, and keep no-schema behavior unchanged.

### Phase 1 File Changes

- **`src/fenic/api/session/session.py`**: Change the public signature to:
  `def create_dataframe(self, data: DataLike, schema: Schema | None = None) -> DataFrame`.
- **`src/fenic/api/session/session.py`**: Import `Schema` and `convert_custom_schema_to_polars_schema`.
- **`src/fenic/api/session/session.py`**: Preserve the current `schema is None` path: normalize `DataLike` exactly as today and return `InMemorySource.from_session_state(pl_df, self._session_state)`.
- **`src/fenic/api/session/session.py`**: Add private helpers local to the session module:
  - `_normalize_data_like_to_polars(data: DataLike, *, allow_empty_list: bool) -> tuple[pl.DataFrame, bool]`, returning whether input was row-oriented.
  - `_coerce_to_schema(pl_df: pl.DataFrame, schema: Schema, *, row_oriented: bool) -> pl.DataFrame`, which selects fields in `schema.column_fields` order and casts with `convert_custom_schema_to_polars_schema(schema)`.
  - `_schema_only_empty_frame(schema: Schema) -> pl.DataFrame`, producing a zero-row Polars frame with the schema's physical dtypes.
- **`src/fenic/api/session/session.py`**: For `schema is not None`, support `dict`, `list[dict]`, and `pl.DataFrame` first; build the logical plan with `InMemorySource.from_schema(coerced_pl_df, schema)`.
- **`src/fenic/api/session/session.py`**: Before converting the provided schema to a Polars schema, trigger the existing logical-plan duplicate-name validation with `InMemorySource.from_schema(pl.DataFrame(), schema)` or equivalent reuse of that validation path. This prevents duplicate names from being hidden by the dict-based `convert_custom_schema_to_polars_schema` implementation.
- **`src/fenic/api/session/session.py`**: Keep `Session.createDataFrame = Session.create_dataframe`; no separate alias wrapper is needed.
- **`tests/api/test_session.py`**: Add tests:
  - `test_create_dataframe_with_schema_from_dict_coerces_and_orders_columns`
  - `test_create_dataframe_with_schema_from_polars_uses_provided_schema`
  - `test_create_dataframe_with_schema_from_list_of_dicts_allows_missing_row_keys`
  - `test_create_dataframe_with_schema_allows_empty_list`
  - `test_create_dataframe_empty_list_without_schema_still_fails`

### Phase 1 Validation

#### Phase 1 Automated Verification

- [x] `uv run pytest tests/api/test_session.py -k "create_dataframe_with_schema or create_dataframe_empty_list" -q` (5 passed, 34 deselected)
- [x] `uv run pytest --collect-only tests/api/test_session.py -q` (39 tests collected)

#### Phase 1 Manual Verification

None.

---

## ✅ Phase 2: Complete input-shape and error contract

This phase fills out the public contract for every supported `DataLike` shape and locks the error boundaries so the schema parameter is strict without relying on Polars' permissive constructor behavior.

### Phase 2 File Changes

- **`src/fenic/api/session/session.py`**: Extend schema-backed normalization to `pd.DataFrame` via `pl.from_pandas` and `pa.Table` via `pl.from_arrow`.
- **`src/fenic/api/session/session.py`**: Enforce column-oriented shape rules:
  - If the normalized input has any columns, its top-level column-name set must exactly equal `set(schema.column_names())`.
  - If the normalized input has zero columns and zero rows, treat it as schema-only empty input.
  - Missing or extra top-level columns raise fenic `ValidationError` with both missing and extra names when present.
- **`src/fenic/api/session/session.py`**: Enforce row-oriented shape rules:
  - Empty lists are allowed only when `schema` is provided.
  - Every non-empty list item must be a `dict`; lists whose first item or later items are non-dicts raise fenic `ValidationError`.
  - The union of row keys must not contain keys outside the schema.
  - Schema fields absent from some or all rows are added as nulls before casting.
- **`src/fenic/api/session/session.py`**: Wrap physical construction/cast failures in fenic `PlanError` with the original exception as `__cause__`; keep top-level contract violations as fenic `ValidationError`.
- **`src/fenic/api/session/session.py`**: Keep duplicate schema-name handling on the existing logical-plan validation path from Phase 1; do not introduce a parallel public validator.
- **`tests/api/test_session.py`**: Add tests for all supported input forms with `schema`: Polars, pandas, dict, list-of-dicts, and PyArrow.
- **`tests/api/test_session.py`**: Add mismatch/error tests:
  - column-oriented missing schema field raises `FenicValidationError`
  - column-oriented extra data field raises `FenicValidationError`
  - row-oriented extra key raises `FenicValidationError`
  - list containing a later non-dict raises `FenicValidationError`
  - unsupported top-level input with schema still raises `FenicValidationError`
  - uncastable value raises `PlanError`
  - duplicate schema names surface the existing duplicate-column `PlanError`
- **`tests/api/test_session.py`**: Add schema-only empty tests for `[]`, `{}`, `pl.DataFrame()`, `pd.DataFrame()`, and `pa.table({})`.

### Phase 2 Validation

#### Phase 2 Automated Verification

- [x] `uv run --env-file .env uv run pytest tests/api/test_session.py -k "create_dataframe_with_schema or create_dataframe_empty" -q` (17 passed, 34 deselected)
- [x] `uv run --env-file .env uv run pytest tests/api/test_session.py -k "create_dataframe" -q` (23 passed, 28 deselected)

#### Phase 2 Manual Verification

None.

---

## Phase 3: Logical schema propagation through planning, serde, and cloud request boundaries

This phase proves that explicit logical schemas are not just accepted at the API boundary; they are the schema seen by downstream validators and serializers. It deliberately avoids embedding physical preservation, which is handled in Phase 4.

### Phase 3 File Changes

- **`src/fenic/api/session/session.py`**: If Phase 2 helper logic is already sufficient, make no behavior change here; keep this phase focused on downstream verification.
- **`tests/api/test_session.py`**: Add logical string-backed schema tests:
  - `test_create_dataframe_with_json_schema_exposes_logical_type`
  - `test_create_dataframe_with_markdown_schema_exposes_logical_type`
  - Assert `df.schema.column_fields` uses `JsonType`/`MarkdownType` while `df.to_polars()` stores physical strings.
- **`tests/_backends/local/functions/test_jq.py`**: Add a test creating a JSON-typed DataFrame directly with `schema=Schema([ColumnField("json_col", JsonType)])`, then call `json.jq(col("json_col"), ".user.name")` without an intermediate `.cast(JsonType)`.
- **`tests/_backends/local/functions/test_markdown.py`**: Add a test creating a Markdown-typed DataFrame directly with `schema=Schema([ColumnField("md_col", MarkdownType)])`, then call an existing markdown function such as `markdown.generate_toc(col("md_col"))` without an intermediate `.cast(MarkdownType)`.
- **`tests/_logical_plan/serde/test_plan_serde.py`**: Add `test_inmemory_source_with_explicit_schema_round_trips`, parameterized over `ProtoSerde` and `CloudPickleSerde`, that creates a schema-backed DataFrame, serializes its plan, deserializes it, and asserts plan equality plus exact schema equality.
- **`tests/_logical_plan/test_plan_equality.py`**: Add an explicit-schema source equality case where two `InMemorySource` nodes with identical Polars data but different logical schema types are not equal, using a schema-backed source path if useful.
- **`tests/_backends/cloud/test_cloud_execution.py`**: Add a request-boundary test for a cloud `count()` or `show()` on a schema-backed DataFrame. The mock `StartExecution` should deserialize the serialized plan and assert the embedded `InMemorySource` schema matches the explicit schema. Do not add or expect a new cloud protobuf field.

### Phase 3 Validation

#### Phase 3 Automated Verification

- [ ] `uv run pytest tests/api/test_session.py tests/_backends/local/functions/test_jq.py tests/_backends/local/functions/test_markdown.py -k "schema" -q`
- [ ] `uv run pytest tests/_logical_plan/serde/test_plan_serde.py -k "explicit_schema or basic_plan" -q`
- [ ] `uv run pytest tests/_logical_plan/test_plan_equality.py -k "schema_mismatch or inmemory_source" -q`
- [ ] `just sync-cloud`
- [ ] `uv run pytest tests/_backends/cloud/test_cloud_execution.py -k "schema" -q`

#### Phase 3 Manual Verification

None.

---

## Phase 4: Local embedding physical preservation

This phase addresses the known local execution mismatch: explicit `EmbeddingType` schemas cast input to fixed-size `pl.Array`, but current in-memory source execution normalizes every array back to `pl.List`. The fix is intentionally narrow and should not recast all source columns from the logical schema.

### Phase 4 File Changes

- **`src/fenic/_backends/local/transpiler/plan_converter.py`**: When converting `InMemorySource`, pass `schema=logical.schema()` into `InMemorySourceExec`.
- **`src/fenic/_backends/local/physical_plan/source.py`**: Change `InMemorySourceExec.__init__` to accept and store `schema: Schema`; update `execute_node` to call `apply_ingestion_coercions(self.df, coerce_array=True, logical_schema=self.schema)`.
- **`src/fenic/_backends/local/physical_plan/source.py`**: Update `InMemorySourceExec.with_children` to preserve the stored schema.
- **`src/fenic/_backends/local/physical_plan/utils.py`**: Extend `apply_ingestion_coercions` to:
  - accept `logical_schema: Schema | None = None`;
  - keep existing behavior when `logical_schema is None`;
  - use logical schema only to preserve paths whose logical type is `EmbeddingType`;
  - continue normal datetime UTC normalization and non-embedding array-to-list coercion.
- **`src/fenic/_backends/local/physical_plan/utils.py`**: Update `_build_target_dtype` to recurse through matching `StructType` and `ArrayType` logical types so nested embeddings are protected, while unrelated `pl.Array` values still coerce to `pl.List`.
- **`tests/_backends/test_ingestion_coercion.py`**: Add utility tests proving:
  - no-schema `pl.Array` still coerces to `pl.List`;
  - an explicit top-level `EmbeddingType` path preserves `pl.Array(pl.Float32, dimensions)`;
  - a nested embedding inside a struct or list path is preserved;
  - unrelated arrays in the same DataFrame still coerce to lists.
- **`tests/_backends/local/test_transpiler.py`**: Extend `test_convert_source_plan` to assert the physical `InMemorySourceExec` stores the logical schema passed by `PlanConverter`.
- **`tests/api/test_session.py`**: Add `test_create_dataframe_with_embedding_schema_preserves_polars_array`, asserting both `df.schema` uses `EmbeddingType` and `df.to_polars().schema["embedding"] == pl.Array(pl.Float32, dimensions)`.
- **`tests/_logical_plan/serde/test_plan_serde.py`**: Add or extend an explicit `EmbeddingType` schema round-trip test so serialized/deserialized in-memory sources keep the logical embedding schema.

### Phase 4 Validation

#### Phase 4 Automated Verification

- [ ] `uv run pytest tests/_backends/test_ingestion_coercion.py -q`
- [ ] `uv run pytest tests/_backends/local/test_transpiler.py -k "source_plan" -q`
- [ ] `uv run pytest tests/api/test_session.py -k "embedding_schema or create_dataframe_with_schema" -q`
- [ ] `uv run pytest tests/_logical_plan/serde/test_plan_serde.py -k "embedding or explicit_schema" -q`
- [ ] `just sync=false test-local`

#### Phase 4 Manual Verification

None.

## Non-Goals / Out of Scope

- Partial schemas: schema-backed creation requires a complete top-level result schema.
- Pydantic model schema shortcuts: only public `Schema` / `ColumnField` inputs are in scope.
- New cloud protobuf fields or execution protocol changes: cloud coverage stays at serialized logical-plan request boundaries.
- A new logical source node: reuse `InMemorySource.from_schema`.
- Broad source recasting during local execution: schema-aware physical execution may protect embedding paths only.
- Tagged-string content validation: `JsonType` and `MarkdownType` annotate logical string-backed data but do not validate JSON or markdown content at ingestion.
- Public documentation updates: this structure covers implementation and tests only; user docs can follow once the API behavior lands.

## Existing Reuse

- `Session.create_dataframe` already centralizes all supported `DataLike` normalization before creating an `InMemorySource`.
- `InMemorySource.from_schema` already stores an explicit `Schema`, participates in plan equality, and survives serde wrapper round trips.
- `LogicalPlan._validate_schema` already rejects duplicate top-level column names.
- `convert_custom_schema_to_polars_schema` already maps fenic data types, including logical string-backed types and `EmbeddingType`, to physical Polars dtypes.
- `apply_ingestion_coercions` already owns local source dtype normalization; Phase 4 extends it narrowly instead of adding a second physical coercion path.
- Existing API, serde, equality, local function, cloud execution, transpiler, and ingestion-coercion test files already cover the affected seams.

## Failure Modes

- Schema-backed column-oriented input silently drops or creates columns if Polars constructor behavior is trusted directly. Covered by Phase 2 exact-set `ValidationError` tests.
- Duplicate schema names are masked by dict-based Polars schema conversion before logical validation runs. Covered by Phase 1 validation ordering and Phase 2 duplicate-name test.
- A row-oriented list has a non-dict after the first row and slips through the existing first-item-only check. Covered by Phase 2 all-row validation.
- Cast failures surface as raw Polars/PyArrow exceptions instead of fenic `PlanError`. Covered by Phase 2 uncastable-value test.
- Logical `JsonType` / `MarkdownType` schemas are visible on `df.schema` but not accepted by downstream function validation. Covered by Phase 3 jq/markdown function tests.
- Explicit embedding inputs cast to `pl.Array` at creation but return as `pl.List` after local execution. Covered by Phase 4 API and utility tests.
- Schema-aware source execution accidentally preserves every array or recasts non-embedding columns. Covered by Phase 4 mixed-column and no-schema regression tests.

## Open Questions

None.

## Decision Log

- **[applied]** Sliced the work vertically from public primitive schema creation through complete input coverage, downstream logical behavior, and finally local physical embedding preservation.
- **[applied]** Kept schema enforcement at the session ingestion boundary and reused `InMemorySource.from_schema`, matching the approved design instead of adding a new source node.
- **[applied]** Deferred partial schemas, Pydantic schema shortcuts, cloud protocol changes, and tagged-string content validation as explicit non-goals from the design.
- **[applied]** Verification uses direct `uv run pytest ...` commands for phase-targeted checks because the repo's `just test-local` recipe has no file or `-k` passthrough; the final phase includes `just sync=false test-local` as the full local-suite recipe. This direct pytest use has recipe-drift risk if project-wide pytest flags change.
- **[applied]** No dedicated typecheck command exists in `just --list`, `pyproject.toml`, or CI; automated validation therefore uses pytest collection, targeted tests, and the full local pytest recipe.
- **[applied]** plan-eng-review found no scope reduction: the structure touches fewer than eight source/test areas per phase, adds no new classes/services, and reuses existing schema/source/coercion seams.
- **[applied]** plan-eng-review added explicit non-goals, existing reuse, and failure-mode coverage so implementation has the required architecture/test review outputs.
- **[applied]** ce-doc-review found and fixed a feasibility ambiguity: duplicate schema validation must run before dict-based custom-to-Polars conversion can hide duplicate column names.
- **[applied]** ce-doc-review found and fixed a verification ambiguity: targeted cloud tests require `just sync-cloud` before direct `uv run pytest ...` because the `test-cloud` recipe has no file or `-k` passthrough.
- **[deferred]** User-facing docs are out of this implementation structure. Capture a follow-up documentation update after the API behavior lands.

## Review Notes

- `plan-eng-review`: completed inline from the installed skill. No blocking architecture, code-quality, test, or performance findings remain after the edits above.
- `ce-doc-review`: completed inline from the installed skill after subagent dispatch failed with child-model resolution errors. Coherence, feasibility, scope, and adversarial checks are reflected in the Decision Log.

## Handoff

**Next step (paste into a fresh tab):**

> Use the `td-plan` skill. The structure is approved at
> `docs/plans/create-dataframe-schema-structure.md` (track: engineering, size: high-risk).
> Deepen it into a full implementation plan with per-file code examples and dual
> success criteria.
> If this is Codex: I explicitly permit optional subagent use for this phase
> where the skill allows it.

**Approved decisions:** Add `schema: Schema | None = None` to `Session.create_dataframe` / `createDataFrame`; enforce complete top-level schema contracts at session ingestion; allow schema-backed empty inputs; reuse `InMemorySource.from_schema`; preserve logical string-backed types without content validation; pass logical source schema into local in-memory execution only to protect `EmbeddingType` paths from array-to-list coercion.
**Open questions (carried forward):** None.
**Non-goals / out of scope:** Partial schemas, Pydantic schema shortcuts, new cloud protocol fields, a new source node, broad source recasting, tagged-string content validation, and user-facing docs in this implementation pass.
**Evidence summary:** `Session.create_dataframe` currently normalizes supported inputs before `InMemorySource.from_session_state`; `InMemorySource.from_schema` already stores explicit schemas and participates in equality/serde; wrapper-level plan serde already carries logical schema; `convert_custom_schema_to_polars_schema` maps fenic schemas to Polars dtypes including `EmbeddingType`; local `InMemorySourceExec` currently applies generic ingestion coercions that turn fixed-size arrays into lists.
**Known weak assumptions:** The Plan should pressure-test duplicate schema validation before Polars schema conversion, cloud test setup through `just sync-cloud`, and the exact recursive mechanism for preserving only embedding-typed physical paths without changing no-schema array behavior.
**Next artifact:** `docs/plans/create-dataframe-schema-plan.md`
**Rollback if:** deepening exposes that the outline is infeasible — ROLLBACK to research/design.
