---
workflow_id: td-5871-polars-2-impact
phase: research
research_stage: findings_ready
track: planning
recommended_track: planning
size_class: full
status: needs_review
portability_level: 1
last_updated: 2026-10-08
---

# Research: Polars 2.0 impact on fenic

**Inspected revision:** `71609ab6a2916fd36c7db80557aa614ebdbe5880`, including merged #393. **Date:** 2026-10-08.

**Inputs used:** the questions below; production source, tests, dependency manifests, primary Polars sources, and local comparison results. The scope includes the full DuckDB call inventory and undeclared Polars constructors.

**Inputs deliberately excluded:** private datasets, real inference requests, and implementation of a replacement storage or SQL backend.

## Summary

Fenic currently executes eager Polars frames, not a Polars lazy query graph. Polars 2.0's streaming default therefore does not automatically change fenic's join/group execution engine. The change applies to `LazyFrame.collect`, including Polars SQL collection, while eager frame operations retain the in-memory engine [P1].

Direct upgrade exposure includes empty-list explode behavior, nested flattening, mixed signed/unsigned arithmetic, Arrow datatype import, and stricter casts. Existing undeclared output constructors remain exposed to empty/all-null inputs. These are not all new 2.0 defects.

DuckDB is also fenic's persistent catalog and intermediate-data store. Better Polars SQL does not replace transactions, metadata storage, or existing `.duckdb` files. Local SQL probes find several successful equivalents and several differences. The upgrade and DuckDB replacement are separate decisions.

## Research Questions

1. How do fenic's physical plans materialize Polars frames, preserve row order, and handle empty frames and nested nulls?
2. How do semantic operators turn rows into requests and derive their cache keys?
3. Where does DuckDB execute queries, read files, store catalogs, and persist intermediate or system data?
4. How do fenic's Python requirements and Rust extension constrain its dataframe dependencies?
5. What execution, datatype, interoperability, and API contracts do the current Polars primary sources document?
6. How do the local test fixtures select providers, handle missing credentials, and exercise dataframe operations?

## Decision Log

The scope is non-UI research and planning. Findings describe inspected behavior; recommendations belong in [breakdown.md](breakdown.md). The inspected base advanced from `0944a2c3` to merged #393 before empirical runs. Proposed #403 changes are not part of this base.

## Findings

### 1. Execution, row order, and semantic cache identity

Fenic recursively materializes child frames and passes eager `pl.DataFrame` objects to physical operators (`src/fenic/_backends/local/physical_plan/base.py:65-80`). Eager joins and aggregation execute on those frames (`physical_plan/join.py:55-70`, `physical_plan/aggregate.py:33-44`, both under `src/fenic/_backends/local/`). Searches found no explicit production `LazyFrame`, `scan_*`, or Polars lazy `collect` pipeline.

This distinction limits the streaming-default risk [P1]. It does not establish a row-order guarantee. Some existing join tests compare unsorted frames (`tests/_backends/local/dataframe/test_standard_join.py:10-64,102-128`). Future adoption of Polars SQL or lazy execution introduces the documented streaming-order risk unless ordering is explicit [P1, P4].

Independent semantic requests use content-based SHA-256 fingerprints, not row positions or Polars hashes (`src/fenic/_inference/cache/key_builder.py:40-71`). The client retains input order while collecting futures (`src/fenic/_inference/model_client.py:753-848`). Reordering unchanged independent prompts changes their sequence, not their content keys. #393 adds structured judgment state to the fingerprint and restores partitioned answers by owner index (`src/fenic/_inference/language_model.py:110-169`, `src/fenic/_inference/types.py:49-56,77-86`).

Reduction is different. It packs documents left-to-right and preserves that order through its hierarchy (`src/fenic/_backends/local/semantic_operators/reduce.py:23-45,124-191,236-303`). Reordering documents within a group changes prompt bytes, document numbering, token-boundary batches, cache keys, and potentially answers. Changing only the order of groups need not change each group's content key.

Materialized DataFrame caching uses a generated UUID name, not a content hash (`src/fenic/api/dataframe/dataframe.py:417-421`, `src/fenic/_backends/local/physical_plan/base.py:90-94`). LLM response caching uses SQLite, not DuckDB (`src/fenic/_backends/local/session_state.py:82-97`).

### 2. Breaking-change mapping

Each row distinguishes a documented change from a demonstrated fenic failure. Paths abbreviated as `local/` begin at `src/fenic/_backends/local/`; `core/` begins at `src/fenic/core/`. Unless another source is named, the primary authority is the full 2.0 upgrade guide [P1], retrieved 2026-10-08.

| Change                                                                                                                         | Fenic call site                                                                                                                                    | Exposure and silent behavior                                                                                                                                                                                                                    |
| ------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Lazy `collect` and SQL default to streaming                                                                                    | `local/physical_plan/base.py:65-80`; `local/physical_plan/join.py:55-70`; `local/physical_plan/aggregate.py:33-44`                                 | Fenic uses eager frames here. No automatic engine switch. Future lazy/SQL joins or group-by can reorder rows, affecting reduction inputs.                                                                                                       |
| `read_csv` now uses scan/collect, removes four tuning arguments, changes partial overrides and selected-column order           | `local/physical_plan/source.py:72-78`; `local/utils/io_utils.py:72-90`                                                                             | Production CSV reads use DuckDB, not `pl.read_csv`. No direct removed-argument exposure. A replacement reader would inherit these contracts.                                                                                                    |
| IPC reader dispatch and removed `memory_map`/`rechunk`                                                                         | `core/_serde/proto/expressions/basic.py:103-131`                                                                                                   | Actual `pl.read_ipc` use passes neither removed argument. It reads a fresh buffer.                                                                                                                                                              |
| Empty lists explode to zero rows; null lists still yield a null row                                                            | `local/physical_plan/transform.py:228-237,251-259,300-312`                                                                                         | **Direct silent outer-explode regression.** `keep_null_and_empty=True` does not currently pass `empty_as_null=True`. Indexed outer explode also loses the empty-list row.                                                                       |
| Expression explode also drops empty inner lists                                                                                | `local/transpiler/expr_converter.py:1164-1168`                                                                                                     | **Observed:** flattening `[[1], [], [2]]` changes from `[1, None, 2]` to `[1, 2]`.                                                                                                                                                              |
| Other explode consumers                                                                                                        | `local/physical_plan/aggregate.py:77-81`; `local/semantic_operators/sim_join.py:135-156`                                                           | Empty lineage/match lists can produce fewer intermediate rows. Similarity joins later inner-join IDs; a final-result regression is not established.                                                                                             |
| Signed integer plus `UInt64` now promotes to `Int128`, not `Float64`                                                           | `core/_utils/type_inference.py:154-157`; `local/transpiler/expr_converter.py:273-286`; `local/physical_plan/utils.py:75-116`                       | **Observed silent dtype/value change.** Logical `IntegerType` hides physical widths; ingestion does not normalize integer widths.                                                                                                               |
| New `Int128` reaches plugin scalar conversion                                                                                  | `rust/src/arrow_scalar_extractor.rs:214-276`; `rust/src/jinja/render.rs:45-63`; `rust/src/dtypes/json.rs:25-32`                                    | The Rust converter handles integer widths through 64 bits, not Int128. Arithmetic can succeed and later template/JSON conversion can fail.                                                                                                      |
| `is_in` requires lossless coercion; decimal/float and naive/aware membership become stricter; temporal-unit comparison changes | `local/transpiler/expr_converter.py:908-912,1245-1249`; `core/_logical_plan/expressions/basic.py:499-513`                                          | Logical type checks prevent many mismatches, but Decimal and Float64 both map to `DoubleType`. Series literals can retain differing physical time units. Conditional errors or corrected comparisons remain possible.                           |
| Arrow maps load as `Map`; Python values become dictionaries                                                                    | `src/fenic/api/session/session.py:400-401`; `src/fenic/_backends/cloud/execution.py:375-376`; `core/_utils/type_inference.py:171-181`              | **Observed imported schema/value change.** The probe's null map changes from an empty entries list to `None`. Fenic has no Map inference branch. Python dictionary inference still creates Struct [P1].                                         |
| Unknown Arrow extensions retain Extension dtype                                                                                | Same Arrow boundaries; `local/physical_plan/utils.py:75-116`                                                                                       | **Observed schema change.** Fenic has no Extension inference branch or normalization rule. The local inference boundary rejects unsupported types.                                                                                              |
| String-to-temporal native casts removed                                                                                        | `src/fenic/api/session/session.py:438`; `local/transpiler/expr_converter.py:1201-1209,1559-1573`; `rust/src/dtypes/primitives.rs:9-16`             | Explicit-schema ingestion uses native frame casts and is exposed. Existing parsing functions use `str.strptime`. Public `Column.cast` uses the compiled Rust plugin, not the target Python runtime's native cast alone.                         |
| Scalar-to-List native casts removed                                                                                            | `src/fenic/api/session/session.py:438`; `core/_logical_plan/expressions/basic.py:592-616`                                                          | **Observed native cast error.** Explicit-schema scalar-to-array ingestion is exposed; ordinary public primitive-to-array casts are rejected earlier.                                                                                            |
| Invalid strict Struct casts now raise                                                                                          | `src/fenic/api/session/session.py:438`; `rust/src/dtypes/collections.rs:7-63`                                                                      | The guide's Series example raises in the target runtime. A frame/expression cast selecting a subset still succeeds in the local probe. Do not treat these paths as equivalent. Public fenic struct casts reconstruct fields through its plugin. |
| Integer/Categorical casts disabled                                                                                             | `core/_utils/type_inference.py:168-170`; `src/fenic/api/session/session.py:438`; `rust/src/dtypes/primitives.rs:9-16`                              | Retained physical Categorical input mapped to logical String can expose a native ingestion cast. Fenic has no categorical target type.                                                                                                          |
| Zero-width frames preserve height; empty frames cannot adopt a longer Series through `with_columns`                            | `src/fenic/api/dataframe/dataframe.py:488-489,1028-1033`; `src/fenic/api/session/session.py:425-449`; `local/physical_plan/base.py:287-290`        | **Observed:** dropping all columns preserves two rows; adding a two-row Series to `DataFrame()` raises. Fenic's empty select is a no-op and dropping all columns is prohibited. Direct zero-width input remains a boundary case.                |
| Stream-only Arrow imports return Series; interchange protocol removed                                                          | `src/fenic/api/session/session.py:400-401`; `src/fenic/_backends/cloud/execution.py:375-376`; `local/transpiler/expr_converter.py:350-351,879-882` | Identified inputs are PyArrow Tables or arrays, not arbitrary stream-only producers. No interchange-protocol use found.                                                                                                                         |
| File-like scans no longer rewind                                                                                               | `core/_serde/proto/expressions/basic.py:110-128`; `core/_serde/proto/plans/source.py:51-52`                                                        | Identified reads construct fresh buffers at position zero. No same-buffer write/read hazard found.                                                                                                                                              |
| Deterministic plugins permit common-subexpression/subplan elimination                                                          | `local/polars_plugins/jinja.py:60-66`; `local/polars_plugins/dtypes.py:39-45`; other registrations in that package                                 | 2.0 adds `is_deterministic=True` by default [P2, P5]. Existing registrations omit it. These compiled parsers/renderers differ from Python model-backed `map_batches`; do not conflate their side effects.                                       |
| SQL window evaluation, exact decimal literals, and `%`/`DIV` change                                                            | `core/_logical_plan/plans/transform.py:671-707`; `local/physical_plan/transform.py:518-524`                                                        | Fenic SQL currently runs in DuckDB, so these are replacement-design concerns, not direct engine regressions [P1, P2].                                                                                                                           |
| Order-insensitive windows need not retain input order across engines                                                           | `local/transpiler/expr_converter.py:1049-1055`                                                                                                     | Release notes also change window execution [P2]. Fenic uses `.over` for dynamic string replacement. Existing string-function tests are relevant; this does not establish a reduction-order regression.                                          |

#### Guide changes with no affected production usage found

The source search covered the remaining guide categories [P1]: horizontal concat padding; selector/column set operators; `pl.datetime`/`pl.repeat` naming; list/array `to_struct` outer nulls; empty transpose; null shift amounts; Struct arithmetic; Decimal rounding/sign/list-sum precision; Duration statistics; flat list arguments; Enum membership; nonnumeric logarithms; list-to-struct fields; business-day/Decimal argument changes; hash API changes; list `search_sorted`; struct field renaming; Categorical ordering constructors; and Polars-specific multi-file/headerless CSV rules.

It also covered removed/deprecated APIs: lazy profiling, graph defaults, `melt`, `with_row_count`, old join-null/outer names, expression `flatten`/`rechunk`, optimization booleans, legacy streaming arguments, old reader/writer arguments, validity helpers, old NumPy/equality/replacement/map keywords, interchange/type-alias modules, parametric strategies, Avro instability, and `cut`/`qcut`. No concrete removed-call blocker was found. Fenic's own `where`, `collect`, or group count methods are not those removed Polars APIs.

Polars Parquet ENUM changes do not directly describe fenic's DuckDB file reader [P1, P2]. The guide's Map, Extension, and UInt64 changes do reach actual Arrow/physical-type boundaries, as listed above.

### 3. Undeclared-type inventory

An AST scan of production `src/` at the inspected revision found **34 constructors without explicit dtype/schema**: 19 Series and 15 DataFrame calls. It found **zero** `map_batches` or `map_elements` calls without `return_dtype` (16 and four calls respectively).

The count includes typed-Arrow/typed-Series wrappers and deliberate no-column sentinels. It is not a count of 34 equivalent bugs. Callback metadata also does not guarantee that the callback's inner Series has the declared physical dtype.

| Call sites                                            | Purpose and classification                                                                    | Existing or proposed coverage                                                                         |
| ----------------------------------------------------- | --------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `src/fenic/_backends/cloud/execution.py:376`          | Arrow result wrapper; upstream physical schema exists, but no logical-schema enforcement      | #403 adds logical selection/casting and cloud-result tests                                            |
| `local/physical_plan/base.py:289`                     | Python UUID output; empty list lacks String dtype                                             | #403 adds empty UUID coverage                                                                         |
| `local/physical_plan/sink.py:55,58,123,137`           | Four intentional `(0,0)` sink sentinels, not inferred result columns                          | #403 makes the empty schema explicit                                                                  |
| `local/physical_plan/transform.py:192,196`            | Union-lineage UUID outputs from empty sides                                                   | #403 adds declared String and empty-side tests                                                        |
| `local/semantic_operators/base.py:127`                | Generic model-output fallback without `output_type`; empty/all-null output can infer Null     | Deferred from #403; TypeSafe routes added in #393 use the typed branch                                |
| `local/semantic_operators/cluster.py:79`              | Cluster labels from Python/sklearn output                                                     | Existing valid/mixed-null test; #403 adds empty/all-invalid tests                                     |
| `local/semantic_operators/join.py:99`                 | Wrapper around an already constructed result Series                                           | Upstream Predicate dtype matters; #393 tests empty/all-null decision joins                            |
| `local/semantic_operators/join.py:124-126`            | Empty frame assembled from individually typed Series                                          | Metadata wrapper, not fresh scalar inference; deferred from #403                                      |
| `local/semantic_operators/parse_pdf.py:109`           | Markdown output Series from model results                                                     | #403 declares String and tests failed/null/empty paths                                                |
| `local/semantic_operators/reduce.py:93,97-100,102`    | Three empty/null/group summary constructions without String dtype                             | Existing tests assert values; #403 adds dtype coverage                                                |
| `local/semantic_operators/sim_join.py:185-187`        | Empty frame assembled from typed Series and a typed distance field                            | #403 makes the frame schema explicit                                                                  |
| `local/transpiler/expr_converter.py:1364,1376`        | Normalize callback null/NumPy output branches lack embedding Array dtype                      | Existing valid/mixed-null tests; deferred from #403                                                   |
| `local/transpiler/expr_converter.py:1409,1417,1425`   | Pairwise similarity null/NumPy/Python output branches lack Float32 declaration                | Existing mixed-null tests do not cover an entirely null batch; deferred                               |
| `local/transpiler/expr_converter.py:1440,1447,1455`   | Query-vector similarity null/NumPy/Python output branches lack Float32 declaration            | Empty/all-null dtype assertions not found; deferred                                                   |
| `src/fenic/api/mcp/_tool_generation_utils.py:367`     | Known dataset-name/count result without complete schema                                       | #403 adds declared schema                                                                             |
| `src/fenic/api/mcp/_tool_generation_utils.py:552-557` | Known nested dataset-schema descriptions without schema                                       | #403 tests empty nested schema                                                                        |
| `src/fenic/api/mcp/_tool_generation_utils.py:829-836` | Profile frame declares three statistics overrides, not the full result schema                 | #403 adds full schema and empty/all-null profiling tests                                              |
| `src/fenic/api/session/session.py:380,398,399`        | Three public user-ingestion boundaries infer physical types before logical conversion/casting | #403 retains and documents inference; explicit schema does not eliminate every staging inference step |
| `src/fenic/api/session/session.py:384`                | Empty staging placeholder, subsequently replaced by a schema-only frame                       | Not an inferred data column; #403 makes empty schema explicit                                         |
| `src/fenic/api/session/session.py:436`                | Backfilled all-null column has known `target_schema` but undeclared constructor               | #403 declares target dtype                                                                            |
| `src/fenic/core/types/semantic_examples.py:197`       | Populated example export infers user-value precision/timezone/types                           | #403 retains populated inference and tests value preservation                                         |

The inspected #403 patch addresses 19 current sites and leaves 15 syntactically undeclared. Remaining sites comprise the generic semantic fallback, two semantic-join wrappers, eight embedding branches, three user-ingestion boundaries, and populated example export. Its head was `55408d48fd1839cf52a3cef9fdaf86c0ec33b256`; it was open when inspected.

#393 explicitly sets Boolean/String output types for TypeSafe predicate, classification, and sentiment (`local/semantic_operators/predicate.py:69-73`, `classify.py:70-74`, `analyze_sentiment.py:157-166`). Its decision sender constructs no Polars object (`local/semantic_operators/decision.py:67-116`). Empty/all-null decision regression tests assert those dtypes (`tests/_inference/test_decision_operators.py:368-428`).

Local constructor probes show empty and all-null untyped Series infer `Null` in **both** versions. A declared empty String Series remains String. This supports treating the typing sweep as existing-contract hardening, not claiming Polars 2 newly introduced Null inference [P1, P6].

### 4. DuckDB-use inventory and coverage verdicts

This inventory covers production execution and its public wrappers. Tests, documentation, and independent example/service dependency declarations are separated below.

| Use                                              | File:line and purpose                                                                                                                                 | Polars 2.0 coverage verdict                                                                                                                                                                                                                               |
| ------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Persistent session connection                    | `local/session_state.py:50-63,150-153`; `local/manager.py:55-56`; `local/utils/io_utils.py:114-117,366-369`                                           | **Not equivalent.** Opens application `.duckdb` and temporary `.duckdb` files; sets Arrow large-buffer export. Polars SQLContext is a frame registry, not this file-backed database [P4].                                                                 |
| Local CSV/Parquet reads                          | `local/utils/io_utils.py:72-90,230-273`; `local/physical_plan/source.py:72-78`                                                                        | Native Polars readers exist [P1, P7]. **Not established as equivalent:** schema inference, first-file behavior, union-by-name widening, errors, timezones, and map imports differ. Local ordinary Parquet probe matches; default multi-file CSV does not. |
| Plan-time file schema discovery                  | `local/execution.py:124-140`; `core/_logical_plan/plans/source.py:89-126`                                                                             | Polars `collect_schema` provides lazy schema resolution [P3], but fenic's exact reader contracts still need comparison. The current calls forward options to `query_files`; they do not themselves set `schema_inference=True`.                           |
| Remote S3/Hugging Face reads and credentials     | `local/utils/io_utils.py:140-187,217-227,312-364`                                                                                                     | Polars documents cloud credentials and `hf://` token support [P7]. **Not equivalent without an adapter:** boto credentials, mixed paths, auth/status errors, and extension initialization are part of fenic's contract.                                   |
| CSV/Parquet writes                               | `local/utils/io_utils.py:92-111,191-215,276-283`; `local/physical_plan/sink.py:40-58`                                                                 | Polars has writers/sinks [P1, P7]. Local format writing is a candidate, but overwrite/ignore checks, timezone/type round trips, and S3 credential/error behavior are not proven equal.                                                                    |
| SQL planning/schema validation                   | `core/_logical_plan/plans/transform.py:657-707,874-895`                                                                                               | **Partial.** DuckDB resolves SQL on typed empty frames; sqlglot parses the DuckDB dialect and rejects DDL/DML. Polars SQL schema resolution is supported [P1, P4], but query syntax and result schema parity are not complete.                            |
| SQL execution                                    | `local/physical_plan/transform.py:518-531`                                                                                                            | **Partial.** Registers child frames, executes DuckDB SQL, converts results, drops views. Polars executes representative joins/CTEs/windows [P3, P4]; local probes find output-column and dtype differences.                                               |
| Public SQL/MCP contract                          | `src/fenic/api/session/session.py:279-280`; `src/fenic/api/mcp/_tool_generation_utils.py:577-606`; `src/fenic/api/mcp/tools.py:6`                     | Explicitly promises the DuckDB SQL dialect. Routing existing queries to another frontend changes a public contract, even when common queries work.                                                                                                        |
| Catalog schemas and object discovery             | `local/catalog.py:155-230,263-299,693-713`; `src/fenic/_backends/utils/catalog_utils.py:154`                                                          | **Not equivalent.** Uses schemas, `duckdb_schemas`, `information_schema.tables`, and DuckDB naming rules. Target probes reject schema creation and information-schema access.                                                                             |
| Catalog transactions and persistent table CRUD   | `local/catalog.py:56-81,364-489,555-678`; `local/physical_plan/source.py:101-109`; `local/physical_plan/sink.py:104-137`                              | **Not equivalent.** Needs transaction boundaries, atomic table/schema metadata updates, inserts, replacement, persistence, and reopen behavior. Target probes reject BEGIN, INSERT, and UPDATE.                                                           |
| Logical schemas, views, descriptions, MCP tools  | `local/system_table_client.py:63-566,684-714,780-835`                                                                                                 | **Not equivalent.** Stores serialized logical metadata with bound parameters and upserts. SQLContext table registration does not reproduce this durable store [P4].                                                                                       |
| Query metrics and session totals                 | `local/system_table_client.py:568-682,716-773`; `local/catalog.py:685-691`                                                                            | **Not equivalent as storage.** Polars can aggregate frames, but append-only persisted rows, parameter binding, and reopening must be implemented separately.                                                                                              |
| Materialized DataFrame cache and lineage storage | `local/temp_df_db_client.py:15-53`; `local/lineage.py:90-112,157-183`; `local/physical_plan/source.py:189-197`                                        | Polars IPC/Parquet can hold frames, but **not a drop-in replacement** for table discovery, lifecycle, type round trips, concurrency, and catalog queries.                                                                                                 |
| Result normalization at database boundaries      | `local/physical_plan/utils.py:12-116`; catalog/temp/source/SQL call sites above                                                                       | Array/List/Struct normalization and UTC-microsecond timestamp casts must remain equivalent regardless of reader/query engine.                                                                                                                             |
| DuckDB-fusion scaffolding                        | `local/physical_plan/optimizer/merge_duckdb_nodes.py:24-114`; `local/physical_plan/transform.py:623-658`; optimizer and physical-plan package exports | The merge rule exists, but `MergedDuckDBExec.execute_node` is `pass`. No production invocation of the rule was found. Do not count a working DuckDB join-fusion engine as an existing use.                                                                |
| Transpiler routing                               | `local/transpiler/plan_converter.py:169,455`                                                                                                          | Routes catalog source/sink nodes to the DuckDB wrappers. Any storage change must update these routes; this is not another database connection.                                                                                                            |
| Lance similarity index                           | `local/semantic_operators/sim_join.py:86-104`                                                                                                         | Uses the **lancedb Python package**, not a DuckDB Lance extension. Polars 2 adds unstable `scan_lance` [P2]; a scan is not the existing vector-search/index contract.                                                                                     |

Only `httpfs` installation/loading was found in production SQL (`local/utils/io_utils.py:154,203`). No DuckDB Lance/vector extension load was found. Standard fenic joins already use Polars; DuckDB executes joins inside user SQL.

Other references do not define additional query engines. The independent docs-server directly caps DuckDB (`examples/mcp_server/docs-server/pyproject.toml:11`). The hosted docs-MCP service opens a persisted fenic catalog and sets `DUCKDB_TMPDIR` (`services/docs-mcp/src/fenic_mcp/server/utils/session.py:14-25`); its independent lock resolves DuckDB 1.4.5 and fenic 0.11.0 (`services/docs-mcp/uv.lock:858-859,988-990`). Tests, SQL/date documentation, changelog, and contributor documentation also refer to DuckDB. A dependency-removal proposal must inspect these independent consumers rather than assume the root pin controls them.

#### Local SQL and reader parity observations

The probes use DuckDB 1.4.5 and Polars 2.0.0 in the same Python environment. All cases use local synthetic frames/files.

| Case                                         | Result                                                                                                  |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| Explicit left-join projection                | Rows and physical schema match                                                                          |
| CTE plus `UNION ALL` with ordering           | Rows and physical schema match                                                                          |
| `ROW_NUMBER` ordered within groups           | Rows and Int64 rank match in 2.0; Polars 1.43.2 rank is UInt32                                          |
| Exact decimal literal and negative remainder | 2.0 matches the DuckDB example; 1.43.2 differs [P1, P2]                                                 |
| `SELECT * ... JOIN ... USING (id)`           | Polars retains `id:r`; DuckDB returns one join-key column                                               |
| `DATE_PART` after timestamp-minus-interval   | Values match; Polars day is Int8, DuckDB day is Int64                                                   |
| `duckdb_settings()`                          | Polars rejects this table function                                                                      |
| Bound `?` parameter                          | DuckDB accepts a bound value; Polars rejects the placeholder                                            |
| Catalog operations                           | Polars rejects BEGIN, CREATE SCHEMA, INSERT, UPDATE, and information-schema lookup; DELETE is supported |
| Ordinary local Parquet round trip            | Rows and physical schema match                                                                          |
| Two CSV files with disjoint columns          | DuckDB `union_by_name=true` merges them; default Polars 2 CSV reading raises on differing names         |

Eight representative SELECT cases are not an exhaustive DuckDB compatibility suite. Successful examples do not establish timezone, null, overflow, nested-type, or arbitrary-query equivalence.

### 5. Dependency and Rust findings

| Dependency           | Fenic declaration                                                                                                         | Effect of the target distribution                                                              |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Python               | `>=3.10`, classifiers through 3.14 (`pyproject.toml:11-19`)                                                               | Polars 2.0.0 declares `>=3.10`; no additional floor is required [P6]                           |
| NumPy                | `>=2.1.0`, or `>=2.3.2` on Python 3.14 (`pyproject.toml:29-30`)                                                           | Polars' optional NumPy extra requires `>=1.16.0`; no target-driven increase [P6]               |
| PyArrow              | `>=23.0.1,<23.0.2`, also cloud extra (`pyproject.toml:32,82`)                                                             | Polars' optional Arrow extra requires `>=7.0.0`; fenic's tighter pin is independent [P6]       |
| pandas               | `>=2.3.3` (`pyproject.toml:36`)                                                                                           | Polars' pandas extra declares pandas plus its Arrow extra, without a pandas version floor [P6] |
| DuckDB               | `>=1.1.3,<1.5`; Python 3.14 requires `>=1.4.2,<1.5` (`pyproject.toml:27-28`)                                              | #398 proposes `>=1.5.6,<1.6`. This is a separate change; it is not used in these comparisons   |
| Python Polars        | `>=1.43.2,<1.44.0` (`pyproject.toml:31`)                                                                                  | Target 2.0 is intentionally installed only in the overlay; the tracked pin remains unchanged   |
| Rust Polars / Arrow  | Semver requirements `0.55.1` (`rust/Cargo.toml:13,20-24`); lock resolves `0.55.2` (`rust/Cargo.lock:1983-1987,2007-2011`) | A Python package upgrade does not update these crates                                          |
| pyo3-polars / derive | `0.28.0`; derive lock `0.22.0` (`rust/Cargo.toml:15`, `rust/Cargo.lock:2776-2810`)                                        | Expression-plugin FFI compatibility must be tested independently [P8]                          |
| PyO3                 | Requirement `0.29`, lock `0.29.2`, `abi3-py310` (`rust/Cargo.toml:14`, `rust/Cargo.lock:2719-2731`)                       | CPython stable ABI support is not a promise of Polars engine/data-layout compatibility         |

The compiled extension registers dtype, Jinja, JSON, markdown, regex, tokenization, chunking, fuzz, and transcript functions (`local/polars_plugins/__init__.py:1-28`). Four plugin modules import private `polars._typing.IntoExpr`: fuzz, regex, tokenization, and chunking. No removed `polars.type_aliases` import was found.

Version-specific installed crate sources show plugin FFI protocol major 0/minor 1 and explicit version checking [P8]. This is distinct from PyO3's wrapper stability. The empirical run uses a rebuilt **unchanged** Rust dependency set, then swaps only Python Polars. It does not test a future Rust rebase.

Python 3.14 is not tested locally; this Mac run uses 3.11.11. Compatible package metadata does not prove fenic's complete 3.14 dependency/build matrix.

### 6. Packaging, install cost, and Hugging Face cap

The locked macOS arm64 CPython 3.11 DuckDB wheel is **13,681,675 bytes**, about 13.05 MiB (`uv.lock:719`). The locked Polars runtime wheel is **47,540,529 bytes**, about 45.34 MiB (`uv.lock:2842`), plus an 847,150-byte Python wrapper (`uv.lock:2832`). Removing some query/read uses saves **zero dependency-wheel bytes** while the catalog still requires DuckDB. Complete removal would eliminate the DuckDB download, not shrink fenic's independently compiled extension by that amount.

The local sync reused cached packages: 169 installed in 1.38 seconds. The explicit Rust development build took 1 minute 16 seconds. Target Polars installation reported a 45.7 MiB runtime download, 1.94 seconds preparation, and 3 milliseconds installation. These are warm-cache/local observations, not cold-install benchmarks or general speedup claims.

#324 caps DuckDB below 1.5 because private `hf://` error behavior changed. Fenic currently distinguishes HTTP status codes and whether an HF token exists (`local/utils/io_utils.py:335-345`). Its regression accepts 401/403/429 error codes (`tests/_backends/local/io/test_reader.py:786-827`). The cap therefore protects observable errors, not merely support for the URI scheme.

Polars documents `hf://` and `HF_TOKEN`/token options [P7]. That does not prove equality of authentication errors, private/gated datasets, glob/revision syntax, or mixed-source reads. A native-reader adapter would need to retain fenic's error contract. #398 proposed a separate DuckDB update and closed without merge at final refresh. Combining any future DuckDB update with this experiment would obscure attribution.

The release post reports TPC-H/TPC-DS results, with machine, hot-run, best-of-five, and exclusion conditions [P3]. They do not establish performance for fenic's eager semantic pipelines or persistent catalog workload. A local warm 100,000-row join/sum probe is included only as a sanity check. Median times were 3.860 ms for DuckDB and 0.521 ms for Polars SQL over five runs, with the same numeric total. DuckDB exports Decimal(38,0), while Polars exports Int64. Background test activity and warm caches were not isolated, so this is not a replacement-performance verdict.

### 7. Empirical procedure

Both environments use the same source revision and unchanged Rust crates. The target changes only imported Python Polars and its runtime package through `PYTHONPATH`.

The repository commands are `just sync` and `just sync-rust`. `UV_LOCKED=true` prevents lockfile updates; `UV_PROJECT_ENVIRONMENT` and `CARGO_TARGET_DIR` place this run's environment and build output under `.context/`.

Every test command launches with these variables removed:

```sh
env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY -u GOOGLE_API_KEY \
  -u GEMINI_API_KEY -u TYPESAFE_API_KEY -u COHERE_API_KEY \
  -u OPENROUTER_API_KEY -u HF_TOKEN
```

The test fixture itself adds known placeholder OpenAI/Gemini/OpenRouter keys for SDK construction (`tests/conftest.py:42-59`). Thus keys are absent **at command launch**, but those dummy fixture values are not absent throughout pytest. No real credentials are restored. The experiment blocks real HTTPX transports, including HTTPX2, and requests except two allowlisted public GET hosts for tokenizer/example assets. Mock transports remain available.

The suite command, used for each version, is:

```sh
just sync=false test-local openai gpt-6-luna openai \
  text-embedding-3-small 'not cloud and not requires_provider_key'
```

`PYTEST_ADDOPTS` supplies `-q --tb=short -ra -p offline_guard`, a per-run JUnit file, and a `.context/` pytest cache. `UV_NO_SYNC=true` prevents `uv run` from replacing the overlay. Plugin autoload is disabled. The baseline sets `PYTHONPATH=$R/.context`; the target prepends `$R/.context/polars2-overlay`, where `$R` is this worktree.

With the retained scratch scripts, the exact launcher form is:

```sh
R="$(git rev-parse --show-toplevel)"
env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY -u GOOGLE_API_KEY \
  -u GEMINI_API_KEY -u TYPESAFE_API_KEY -u COHERE_API_KEY \
  -u OPENROUTER_API_KEY -u HF_TOKEN \
  "$R/.context/baseline-venv/bin/python" "$R/.context/run_empirical.py" baseline
env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY -u GOOGLE_API_KEY \
  -u GEMINI_API_KEY -u TYPESAFE_API_KEY -u COHERE_API_KEY \
  -u OPENROUTER_API_KEY -u HF_TOKEN \
  "$R/.context/baseline-venv/bin/python" "$R/.context/run_empirical.py" polars2
```

The scratch build used `UV_PROJECT_ENVIRONMENT=$R/.context/baseline-venv`, `CARGO_TARGET_DIR=$R/.context/cargo-target`, `UV_LOCKED=true`, and `UV_NO_ENV_FILE=true` with each repository build command. The overlay install was `uv pip install --target "$R/.context/polars2-overlay" 'polars==2.0.0'`. Every build/install/probe launch also removed the same eight variables.

Exact sanitized commands and environment settings are recorded in `.context/baseline-receipt.json` and `.context/polars2-receipt.json`. Raw suite logs/JUnit files, package metadata, build logs, and probes remain in `.context/`.

The selection deliberately excludes marked real-provider tests and cloud tests. It is **not** a claim that the credentialed full suite passes. Both runs have a 1,800-second timeout; neither reached it, so the Polars-heavy fallback was not invoked.

An initial recipe invocation incorrectly treated `markerExpr` as a named recipe argument. It executed zero tests and exited 4. Its separate invocation-error log/receipt is retained; the corrected recipe passes the marker expression positionally.

#### Contract probes

The probe script runs on each runtime with the same eight provider variables unset. It exercises the matrix examples and eight SQL cases without provider requests. Both same-version binary round trips pass. A simple 1.43.2 binary frame also deserializes under 2.0; this does not prove every persisted plan/type/plugin payload is compatible.

#### Suite results and failure attribution

The environment is macOS arm64, CPython 3.11.11, DuckDB 1.4.5, NumPy 2.4.6, pandas 3.0.5, and PyArrow 23.0.1. The unchanged Rust build resolves Polars/Arrow 0.55.2.

| Run                  | Passed | Failed | Skipped | Deselected | Pytest duration | Total runner duration |
| -------------------- | -----: | -----: | ------: | ---------: | --------------: | --------------------: |
| Pinned Polars 1.43.2 |  1,918 |      4 |       5 |        109 |        100.81 s |             103.506 s |
| Polars 2.0.0 overlay |  1,912 |     10 |       5 |        109 |         85.58 s |              88.652 s |

There were no collection/test errors. Both commands exited 1 because of the reported assertion failures. All six new failures belong to the same cause: outer explode removes empty-list rows under the new default [P1].

The new failures are in `tests/_backends/local/dataframe/test_explode_with_index.py`:

- `test_explode_with_index_index_name_only_outer`
- `test_case_explode_with_index_both_names_outer`
- `test_explode_outer_basic`
- `test_posexplode_outer_basic`
- `test_posexplode_outer_null_elements_keep_positions`
- `test_posexplode_outer_vs_posexplode`

Four failures are shared by both versions. They are `test_session_config_with_invalid_api_keys`, `test_session_config_with_invalid_gemini_api_key`, `test_session_config_with_invalid_cohere_api_key`, and `test_session_config_with_invalid_anthropic_api_key` in `tests/api/test_session.py`. These tests expect real provider authentication-error text. The transport guard instead returns a blocked-transport or connection error. They are **not Polars regressions**.

The five skips are identical: cloud catalog lacks `fenic_cloud`; cloud execution lacks `grpc`; two public Hugging Face reader tests require an opt-in flag; and `test_embedding_with_no_profile` requires Google Vertex selection. The marker expression deselects 109 tests rather than calling their providers.

The exercised unchanged Rust-plugin paths produced no new failure group. This is empirical support for those exercised paths, not a guarantee covering every plugin/type/chunk layout. The test logs also contain post-summary pending-async-task notices in both runs.

The complete attempt took minutes, not the allowed 30-minute ceiling per suite. A repeat from cached dependencies can budget about five minutes for both guarded suites plus probes. A cold Rust/environment setup needs separate allowance; this machine's explicit Rust build took 76 seconds. Remote/cloud, live-provider, Python 3.14, large-memory, and cold-install validation remain unclaimed.

### 8. Review status

A scoped in-thread review checked coherence, source grounding, result attribution, and plan scope. It clarified the Map null-value change, native Series versus expression casts, and independent docs-service dependencies. This is not a completed multi-persona or cross-model review. The report remains a reviewable research artifact, not approval to upgrade.

## Open Questions

1. Does a future Rust-crate rebase change public plugin casts or scalar conversion beyond the Python-only upgrade?
2. How should fenic represent or normalize Map, Extension, and Int128 at user/Arrow boundaries?
3. Which DuckDB-dialect subset could an explicitly selected Polars SQL backend preserve?
4. What storage design could preserve existing catalog transaction, metadata, reopen, and migration contracts without DuckDB?
5. What cold-install, large-memory, Python 3.14, and remote-reader results hold outside this Mac's bounded experiment?

## Primary sources

All sources below were retrieved or inspected on **2026-10-08**. Moving stable API pages are corroborated with the installed 2.0.0 distribution where version-specific behavior matters.

- **P1:** [Polars 2.0 upgrade guide](https://docs.pola.rs/releases/upgrade/2/). The full guide was read, including its removal/deprecation tables.
- **P2:** [Polars Python 2.0.0 release/changelog](https://github.com/pola-rs/polars/releases/tag/py-2.0.0), published 2026-10-06T11:52:15Z. The full release body was retrieved.
- **P3:** [Release of Polars 2.0](https://pola.rs/posts/release-polars-2/), published 2026-10-06. Includes streaming/out-of-core scope and benchmark conditions.
- **P4:** [SQLContext execution API](https://docs.pola.rs/api/python/stable/reference/sql/api/polars.SQLContext.execute.html) and [2.0.0 SQLContext source](https://github.com/pola-rs/polars/blob/py-2.0.0/py-polars/src/polars/sql/context.py). Installed source documents frame registration, lazy query execution, and eager collection.
- **P5:** [2.0.0 plugin registration source](https://github.com/pola-rs/polars/blob/py-2.0.0/py-polars/src/polars/plugins.py). Installed source declares the deterministic flag and its default.
- **P6:** [Polars 2.0.0 distribution](https://pypi.org/project/polars/2.0.0/) and its installed wheel `METADATA`. Requirements were read directly from the downloaded distribution, not inferred from moving docs.
- **P7:** [Parquet scan API](https://docs.pola.rs/api/python/stable/reference/api/polars.scan_parquet.html), [cloud storage guide](https://docs.pola.rs/user-guide/io/cloud-storage/), and [2.0.0 reader source](https://github.com/pola-rs/polars/blob/py-2.0.0/py-polars/src/polars/io/parquet/functions.py). Installed docstrings document `hf://`, token options, cloud credential providers, and missing-column behavior.
- **P8:** [Expression plugins guide](https://docs.pola.rs/user-guide/plugins/expr_plugins/), [pyo3-polars 0.28.0 source](https://docs.rs/crate/pyo3-polars/0.28.0/source/), and [polars-ffi 0.55.2 source](https://docs.rs/crate/polars-ffi/0.55.2/source/src/lib.rs). Concrete dependency/protocol facts were checked against the version-specific local crate sources.

## Related public work

- [#393](https://github.com/typedef-ai/fenic/pull/393), merged `71609ab6`, supplies decision-operator source and offline tests in this inspected base.
- [#403](https://github.com/typedef-ai/fenic/pull/403), open at inspection, addresses most of the undeclared-constructor sweep.
- [#398](https://github.com/typedef-ai/fenic/pull/398) proposed independent DuckDB pin changes. It closed without merge at the final refresh (updated 2026-10-08T22:23:11Z).
- [#381](https://github.com/typedef-ai/fenic/pull/381), open at inspection, adds OpenAI-compatible providers. A future offline compatibility run must keep its transports guarded.
- [#324](https://github.com/typedef-ai/fenic/pull/324) records the DuckDB Hugging Face cap.
