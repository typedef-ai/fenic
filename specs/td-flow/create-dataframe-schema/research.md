---
workflow_id: create-dataframe-schema
phase: research
research_stage: findings_ready
track: engineering
size_class: high-risk
status: approved
portability_level: 3
recommended_track: engineering
source_inputs:
  - specs/td-flow/create-dataframe-schema/research.md (Research Questions section)
  - src/fenic/api/session/session.py
  - src/fenic/core/types/query_result.py
  - src/fenic/core/_logical_plan/plans/source.py
  - src/fenic/core/_logical_plan/plans/base.py
  - src/fenic/core/_serde/proto/plans/source.py
  - src/fenic/core/_serde/proto/plan_serde.py
  - src/fenic/core/types/schema.py
  - src/fenic/core/types/datatypes.py
  - src/fenic/core/_utils/schema.py
  - src/fenic/core/_utils/type_inference.py
  - src/fenic/_backends/local/transpiler/plan_converter.py
  - src/fenic/_backends/local/physical_plan/source.py
  - src/fenic/_backends/local/physical_plan/utils.py
  - src/fenic/api/io/reader.py
  - src/fenic/_backends/local/utils/io_utils.py
  - src/fenic/_backends/local/execution.py
  - src/fenic/_backends/local/catalog.py
  - src/fenic/_backends/local/system_table_client.py
  - src/fenic/_backends/schema_serde.py
  - src/fenic/_backends/cloud/execution.py
  - src/fenic/_backends/cloud/catalog.py
  - src/fenic/_backends/cloud/cloud_catalog_utils.py
  - tests/api/test_session.py
  - tests/_backends/test_ingestion_coercion.py
  - tests/_logical_plan/serde/test_plan_serde.py
  - tests/_logical_plan/test_plan_equality.py
  - tests/_backends/local/test_transpiler.py
  - tests/_backends/local/io/test_reader.py
  - tests/_backends/local/io/test_writer.py
  - tests/_backends/local/catalog/test_catalog.py
  - tests/_backends/test_schema_serde.py
  - tests/_backends/cloud/test_cloud_execution.py
last_updated: 2026-06-26
---

# Research: create_dataframe Schema Handling

**Date:** 2026-06-26 · **Status:** approved · **Stage:** findings_ready · **Branch/commit:** brandon/td-3666-create_dataframe-should-accept-an-optional-schema-to-force@85d4394a01324f5f932b3821078f3211dd256456

**Inputs used:** This doc's Research Questions; the read files listed in `source_inputs`; derived scope from named files into `LogicalPlan`, type inference, plan serde wrapper, local/cloud execution, local/cloud catalog schema persistence, system-table schema storage, and tests directly covering source/schema/read/write behavior.
**Inputs deliberately excluded:** The originating ticket, per ticket-minimization; no product/design artifacts outside this research doc were read.

## Research Questions

A _query plan_ for the Research findings below (written next, by `td-research`).
You do not need to answer these — they steer the research toward how the system
works today. (After R fills the Findings, this section stays as provenance — it is
**not** a spec for downstream phases.)

1. In `src/fenic/api/session/session.py`, how does `Session.create_dataframe` normalize each supported `DataLike` input into a Polars DataFrame, and where are unsupported or invalid input shapes converted into fenic errors?
2. In `src/fenic/core/_logical_plan/plans/source.py` and `src/fenic/core/_serde/proto/plans/source.py`, how does `InMemorySource` derive, store, clone, compare, and serialize its `Schema` today?
3. In `src/fenic/core/types/schema.py`, `src/fenic/core/types/datatypes.py`, and `src/fenic/core/_utils/schema.py`, how are fenic `Schema`/`ColumnField` definitions mapped to and from Polars schemas, including nested, semantic, timestamp, and embedding-related data types?
4. In `src/fenic/_backends/local/physical_plan/source.py` and `src/fenic/_backends/local/physical_plan/utils.py`, how is an in-memory source materialized during local execution, and what ingestion coercions can change the physical Polars schema relative to the logical schema?
5. In `src/fenic/api/io/reader.py`, `src/fenic/_backends/local/utils/io_utils.py`, and catalog-related schema paths, how do existing APIs accept explicit `Schema` values and surface schema mismatch or conversion errors?
6. In `tests/api/test_session.py`, `tests/_backends/test_ingestion_coercion.py`, plan serde/equality tests, and cloud/local execution tests, what coverage already exists for dataframe creation input forms, schema inference, source schemas, and ingestion-time type coercion?

## Findings

### Public `create_dataframe` Input Flow

`DataLike` is the shared input/output union for Polars DataFrames, pandas DataFrames, column-oriented Python dictionaries, row-oriented lists of dictionaries, and PyArrow tables (`src/fenic/core/types/query_result.py:27-34`). `Session.create_dataframe` currently accepts only `data: DataLike`; there is no public schema parameter on this method (`src/fenic/api/session/session.py:132-135`).

`Session.create_dataframe` normalizes supported inputs into a Polars DataFrame before creating the logical plan: an incoming Polars DataFrame is reused as-is, pandas is converted with `pl.from_pandas`, dict input is passed to `pl.DataFrame`, list input is first checked for non-empty row dictionaries and then passed to `pl.DataFrame`, and PyArrow tables are converted with `pl.from_arrow` (`src/fenic/api/session/session.py:186-205`). Empty lists and lists whose first item is not a dict raise fenic `ValidationError`, while unsupported top-level input types raise fenic `ValidationError` with the runtime type in the message (`src/fenic/api/session/session.py:193-210`). Other exceptions raised during Polars/PyArrow construction are wrapped in fenic `PlanError` with the original exception as cause (`src/fenic/api/session/session.py:212-215`).

After normalization, `Session.create_dataframe` constructs a `DataFrame` from an `InMemorySource` built with the Polars DataFrame and current session state (`src/fenic/api/session/session.py:217-220`). The public alias `Session.createDataFrame` points to the same method (`src/fenic/api/session/session.py:347`).

### Logical Source Schema Behavior

`InMemorySource` stores the Polars DataFrame on `_source`, then delegates schema initialization to `LogicalPlan.__init__` with either a session state or an explicit schema (`src/fenic/core/_logical_plan/plans/source.py:21-28`). `InMemorySource.from_session_state` creates a source that derives its schema from the session state path, while `InMemorySource.from_schema` creates a source that preserves a supplied `Schema` (`src/fenic/core/_logical_plan/plans/source.py:30-36`).

`LogicalPlan.__init__` requires either a session state or a schema, stores a provided schema directly, otherwise calls `_build_schema(session_state)`, and validates the resulting schema before the node is usable (`src/fenic/core/_logical_plan/plans/base.py:20-34`). Logical-plan validation rejects duplicate column names by raising `PlanError` with an aliasing hint (`src/fenic/core/_logical_plan/plans/base.py:35-49`).

`InMemorySource._build_schema` derives the logical fenic schema from the Polars DataFrame's schema via `convert_polars_schema_to_custom_schema` (`src/fenic/core/_logical_plan/plans/source.py:44-45`). Cloning through `with_children` enforces that the source has no children and returns `from_schema(self._source, self._schema)`, preserving the stored logical schema and source data (`src/fenic/core/_logical_plan/plans/source.py:55-62`). Source-specific equality compares only the underlying Polars DataFrame contents; overall logical-plan equality also compares type, schema, source-specific fields, and children before returning equality (`src/fenic/core/_logical_plan/plans/source.py:64-65`; `src/fenic/core/_logical_plan/plans/base.py:134-150`).

Protobuf plan serialization stores the common logical schema at the wrapper level for every plan, including sources (`src/fenic/core/_serde/proto/plan_serde.py:38-50`). The `InMemorySource` plan-specific serializer stores the Polars DataFrame bytes using Polars binary serialization, and deserialization reconstructs the Polars DataFrame and calls `InMemorySource.from_schema` with the wrapper-level schema supplied by the plan deserializer (`src/fenic/core/_serde/proto/plans/source.py:34-56`; `src/fenic/core/_serde/proto/plan_serde.py:85-90`).

### Schema and Data-Type Conversion

`ColumnField` is a frozen pydantic dataclass with `name` and `data_type`, and `Schema` is a frozen pydantic dataclass with an ordered `column_fields` list (`src/fenic/core/types/schema.py:17-30`; `src/fenic/core/types/schema.py:77-90`). `Schema.column_names()` returns the ordered field names from `column_fields` (`src/fenic/core/types/schema.py:117-123`).

The fenic type system defines singleton primitive instances for string, integer, float, double, boolean, date, and timestamp; composite `ArrayType`, `StructField`, and `StructType`; logical `EmbeddingType`; tagged string logical types for markdown, HTML, and JSON; and parameterized logical types for transcripts and document paths (`src/fenic/core/types/datatypes.py:82-238`; `src/fenic/core/types/datatypes.py:242-398`; `src/fenic/core/types/datatypes.py:401-528`; `src/fenic/core/types/datatypes.py:546-575`). Logical-type detection treats direct logical types, structs containing logical fields, and arrays containing logical elements as logical types (`src/fenic/core/types/datatypes.py:535-543`).

Polars-to-fenic schema conversion maps each Polars field to a `ColumnField` by calling `infer_dtype_from_polars` on the Polars dtype (`src/fenic/core/_utils/schema.py:31-53`). Polars booleans, signed/unsigned integers, floats/decimals, UTF-8 strings, dates, datetimes/times, categoricals, lists/arrays, and structs are mapped to fenic boolean, integer, float/double, string, date, timestamp, string, `ArrayType`, and `StructType` respectively; unsupported Polars dtypes raise `TypeInferenceError` (`src/fenic/core/_utils/type_inference.py:142-181`).

Fenic-to-Polars schema conversion builds a `pl.Schema` from the ordered `Schema.column_fields`, converting each fenic data type with `convert_custom_dtype_to_polars` (`src/fenic/core/_utils/schema.py:56-75`). Primitive fenic timestamps map to `pl.Datetime(time_unit="us", time_zone="UTC")`; `ArrayType` maps recursively to `pl.List`; `StructType` maps recursively to `pl.Struct`; `EmbeddingType` maps to fixed-size `pl.Array(pl.Float32, dimensions)`; JSON, markdown, HTML, transcript, and document path logical types map physically to `pl.String` (`src/fenic/core/_utils/schema.py:136-190`).

Pydantic models are converted to fenic `StructType` by walking model fields, unwrapping `Optional[T]`, recursively converting nested models and lists, and mapping supported Python scalar types to fenic primitive types (`src/fenic/core/_utils/schema.py:78-134`; `src/fenic/core/_utils/schema.py:193-229`). Unsupported non-optional unions and unsupported Python types raise `TypeError` or `ValueError` from this conversion path (`src/fenic/core/_utils/schema.py:207-226`).

### Local Materialization and Ingestion Coercion

The local plan converter lowers an `InMemorySource` logical node to `InMemorySourceExec` with the source Polars DataFrame and session state, without passing the logical schema into the physical node (`src/fenic/_backends/local/transpiler/plan_converter.py:155-166`). `InMemorySourceExec.execute_node` expects no child DataFrames and returns `apply_ingestion_coercions(self.df, coerce_array=True)` (`src/fenic/_backends/local/physical_plan/source.py:27-36`).

`apply_ingestion_coercions` iterates over every physical Polars column, computes a target dtype, casts only columns whose target dtype differs, and returns a selected DataFrame with the resulting expressions (`src/fenic/_backends/local/physical_plan/utils.py:6-36`). The target-dtype builder converts fixed-size `pl.Array` to `pl.List` when `coerce_array=True`, recursively normalizes `pl.List` inner types, rewrites all `pl.Datetime` dtypes to microsecond UTC datetimes, recursively normalizes struct fields, and otherwise leaves dtypes unchanged (`src/fenic/_backends/local/physical_plan/utils.py:39-53`).

The same ingestion coercion path is used by local file sources and doc sources after loading data, while table sources read directly from the local catalog table without applying this helper in `DuckDBTableSourceExec` (`src/fenic/_backends/local/physical_plan/source.py:66-72`; `src/fenic/_backends/local/physical_plan/source.py:95-103`; `src/fenic/_backends/local/physical_plan/source.py:139-158`).

### Existing Explicit-Schema APIs and Error Surfaces

The CSV reader accepts an optional `Schema` and `merge_schemas` flag; it rejects using both at once with fenic `ValidationError`, and when a schema is provided it validates that every CSV schema field is a primitive fenic type (`src/fenic/api/io/reader.py:70-152`). The reader then passes the schema and merge flag through `FileSource.options` into the logical file source (`src/fenic/api/io/reader.py:205-248`).

File-source schema inference delegates to the session execution backend; CSV failures with an explicit schema are reported as `PlanError("Schema mismatch: ...")`, CSV failures with schema merging are reported as inconsistent CSV schemas, and other CSV/parquet inference failures are wrapped as file-specific `PlanError`s (`src/fenic/core/_logical_plan/plans/source.py:89-126`). The local execution backend infers file schemas by reading through `query_files`, taking the resulting Polars schema, and converting it to fenic `Schema` (`src/fenic/_backends/local/execution.py:124-140`).

Local CSV query construction converts an explicit fenic `Schema` to DuckDB `columns = {...}` by mapping string, integer, float, double, boolean, date, and timestamp to DuckDB `VARCHAR`, `BIGINT`, `FLOAT`, `DOUBLE`, `BOOLEAN`, `DATE`, and `TIMESTAMPTZ`; unsupported CSV schema types raise `InternalError` from this lower-level conversion (`src/fenic/_backends/local/utils/io_utils.py:252-263`; `src/fenic/_backends/local/utils/io_utils.py:286-309`). `query_files` executes the generated DuckDB SQL and wraps DuckDB read errors as `FileLoaderError`, with specific HTTP messages for S3/Hugging Face failures (`src/fenic/_backends/local/utils/io_utils.py:72-91`; `src/fenic/_backends/local/utils/io_utils.py:312-345`).

Local catalog table creation accepts a `Schema`, converts it to a Polars schema, creates an empty DuckDB table from an empty Polars DataFrame with that schema, and saves the original fenic schema metadata in the system table (`src/fenic/_backends/local/catalog.py:416-459`). Local table metadata retrieval reads schema metadata from the system table and returns it as `DatasetMetadata`; missing metadata for an existing table is surfaced as table-not-found or internal catalog errors depending on the caller path (`src/fenic/_backends/local/catalog.py:335-346`; `src/fenic/_backends/local/system_table_client.py:107-145`).

The local system table stores schema metadata as serialized JSON in a `schema_blob` column keyed by database and table name (`src/fenic/_backends/local/system_table_client.py:68-99`; `src/fenic/_backends/local/system_table_client.py:684-708`). The backend schema serializer recursively preserves primitive, array, struct, embedding, markdown, HTML, JSON, transcript, and document path types, and deserialization rebuilds `Schema(column_fields=...)` from that JSON representation (`src/fenic/_backends/schema_serde.py:29-46`; `src/fenic/_backends/schema_serde.py:53-96`; `src/fenic/_backends/schema_serde.py:99-176`).

Table writes use the logical plan schema as the table schema input to execution, compare append-mode writes against the existing catalog schema, and raise `PlanError` on append schema mismatch (`src/fenic/_backends/local/execution.py:74-91`; `src/fenic/_backends/local/execution.py:142-176`). Local catalog insert paths also compare existing table metadata schemas to incoming schemas and raise `CatalogError` on mismatch (`src/fenic/_backends/local/catalog.py:591-624`).

Cloud schema inference accepts only cloud-supported paths, sends CSV/parquet infer-schema requests to the engine service, serializes an explicit CSV schema into the request when present, and deserializes the response schema back into fenic `Schema` (`src/fenic/_backends/cloud/execution.py:236-291`). Cloud catalog table creation turns each `ColumnField` into a catalog `NestedFieldInput` by protobuf-serializing the fenic data type, base64-encoding that proto payload, and also storing an Arrow type string; cloud table schema reads reverse the base64/protobuf path into fenic `ColumnField`s (`src/fenic/_backends/cloud/catalog.py:621-678`; `src/fenic/_backends/cloud/catalog.py:718-755`).

### Current Test Coverage

`tests/api/test_session.py` covers `create_dataframe` from Polars, pandas, dict, list-of-dicts, and PyArrow inputs, including expected schema fields and collected Polars data for each supported input form (`tests/api/test_session.py:48-131`). The same file covers empty-list `ValidationError`, unsupported-top-level-type `ValidationError`, and the fact that a Polars-backed created DataFrame uses an `InMemorySource` logical plan (`tests/api/test_session.py:134-145`; `tests/api/test_session.py:207-229`).

`tests/_logical_plan/serde/test_plan_serde.py` exercises plan serialization with `InMemorySource`, `FileSource`, `TableSource`, and `DocSource` examples and asserts that deserialized plans have the same type, compare equal to the original plan, and rebuild the same schema (`tests/_logical_plan/serde/test_plan_serde.py:60-104`; `tests/_logical_plan/serde/test_plan_serde.py:204-225`). It also round-trips basic created DataFrames, transform plans, file-source plans, and table-source plans through the configured serde implementations and compares output data or schema (`tests/_logical_plan/serde/test_plan_serde.py:247-309`; `tests/_logical_plan/serde/test_plan_serde.py:385-456`).

`tests/_logical_plan/test_plan_equality.py` covers schema-sensitive logical-plan equality, `InMemorySource` equality by Polars DataFrame contents, and source equality fields for file, table, and doc sources (`tests/_logical_plan/test_plan_equality.py:123-155`; `tests/_logical_plan/test_plan_equality.py:157-219`). `tests/_backends/local/test_transpiler.py` covers lowering an `InMemorySource` logical source to `InMemorySourceExec` (`tests/_backends/local/test_transpiler.py:83-91`).

`tests/_backends/test_ingestion_coercion.py` covers fixed-size array-to-list coercion, non-coercion when `coerce_array=False`, nested array/list/struct coercion, no-op behavior when no dtype changes are needed, and datetime UTC normalization through `create_dataframe(...).to_polars()` (`tests/_backends/test_ingestion_coercion.py:10-120`; `tests/_backends/test_ingestion_coercion.py:129-145`). Local reader tests also cover datetime behavior for parquet, in-memory DataFrames, CSV, CSV timestamp/date schema overrides, table reads, array ingestion from parquet and in-memory sources, and persisted embedding table schemas (`tests/_backends/local/io/test_reader.py:995-1131`; `tests/_backends/local/io/test_reader.py:1134-1168`; `tests/_backends/local/io/test_reader.py:1476-1490`).

`tests/_backends/local/io/test_reader.py` covers explicit CSV schemas, primitive type validation, schema mismatch on incompatible or incomplete CSV schemas, and invalid `schema` plus `merge_schemas=True` combinations (`tests/_backends/local/io/test_reader.py:296-365`; `tests/_backends/local/io/test_reader.py:670-732`). `tests/_backends/local/io/test_writer.py` covers table overwrite, append with the same schema, and append schema mismatch errors (`tests/_backends/local/io/test_writer.py:149-209`).

Catalog schema persistence coverage includes local struct-schema describe-table behavior, local create-table behavior, table metadata/description reads, and schema serde round trips for nested structs, arrays, embeddings, and logical string/path/transcript types (`tests/_backends/local/catalog/test_catalog.py:301-313`; `tests/_backends/local/catalog/test_catalog.py:360-410`; `tests/_backends/local/catalog/test_catalog.py:416-432`; `tests/_backends/test_schema_serde.py:34-154`). Cloud execution tests cover cloud CSV/parquet schema inference and explicit CSV schema pass-through in infer-schema requests (`tests/_backends/cloud/test_cloud_execution.py:357-385`).

No test found in the scoped files exercises a public `Session.create_dataframe(..., schema=...)` call, because the current public method signature has only the `data` parameter (`src/fenic/api/session/session.py:132-135`; `tests/api/test_session.py:48-145`).

## Open Questions

None.

## Decision Log

- **[applied]** Scoped the questions to the public dataframe creation entry point, logical source schema flow, schema conversion utilities, execution-time ingestion behavior, existing explicit-schema APIs, and current tests because those are the areas touched by dataframe creation semantics.
- **[applied]** Omitted a design-system question because the task is backend/API-only and no UI or visual work is plausibly in scope.
- **[applied]** Used derived scope for logical-plan base schema validation, type inference, plan serde wrapper behavior, local/cloud execution, local/cloud catalog schema persistence, system-table schema storage, and directly matching tests because the named files call into those paths.
- **[applied]** ce-doc-review tier-3 pass completed inline after subagent dispatch was unavailable; fixed source-input wording so the deliberately excluded ticket is not listed as an input.

## Handoff

**Next step (paste into a fresh tab after research approval):**

> Use the `td-design` skill. Research is approved at
> `specs/td-flow/create-dataframe-schema/research.md` (track: engineering, size: high-risk).
> Design-stage trigger(s) held: new or changed public API/schema contract. Choose
> the approach, weigh alternatives, and fix the cross-cutting contracts before the structure.
> Use this research doc's Findings — the `## Research Questions` section is provenance, not a spec.
> Repo command note: run `mise use just` first; after that, `just` commands are available.
> If this is Codex: I explicitly permit optional subagent use for this phase
> where the skill allows it.

**Approved decisions:** Current `Session.create_dataframe` accepts only `data`; all current supported inputs normalize through Polars before `InMemorySource`; `InMemorySource` derives or preserves a fenic `Schema`; local execution applies ingestion coercions that can change physical Polars dtypes; explicit schemas currently exist on CSV reader and catalog/table paths.
**Open questions (carried forward):** None.
**Non-goals / out of scope:** The originating ticket was not read; this document does not prescribe implementation changes.
**Evidence summary:** Public input and normalization flow: `src/fenic/api/session/session.py:132-220`; logical schema and serde flow: `src/fenic/core/_logical_plan/plans/source.py:21-65`, `src/fenic/core/_serde/proto/plan_serde.py:38-90`, `src/fenic/core/_serde/proto/plans/source.py:34-56`; conversion/coercion flow: `src/fenic/core/_utils/schema.py:31-190`, `src/fenic/_backends/local/physical_plan/utils.py:6-53`; explicit-schema APIs: `src/fenic/api/io/reader.py:70-152`, `src/fenic/_backends/local/catalog.py:416-459`.
**Known weak assumptions:** None.
**Next artifact:** `specs/td-flow/create-dataframe-schema/design.md`
**Rollback if:** the research questions are found to be too narrow for the intended API/schema change.
