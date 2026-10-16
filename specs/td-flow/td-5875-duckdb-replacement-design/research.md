---
workflow_id: td-5875-duckdb-replacement-design
phase: research
research_stage: findings_ready
track: engineering
recommended_track: engineering
size_class: full
status: needs_review
portability_level: 1
source_inputs:
  - https://github.com/typedef-ai/fenic/pull/410
last_updated: 2026-10-09
---

# Research: DuckDB compatibility and storage boundaries

**Inspected revision:** `71609ab6a2916fd36c7db80557aa614ebdbe5880`.
**Research date:** 2026-10-08 local, 2026-10-09 UTC.

**Current scope, 2026-10-09:** SQL in the middle of a pipeline without the DuckDB conversion/materialization boundary. The broad inventory below remains historical evidence. Catalog replacement, durable storage, reader migration, and dependency removal are out of scope. The current evidence is [SQL materialization measurements](sql-materialization-measurements.md); the selected proposal is [design.md](design.md). The earlier [split proposal](breakdown.md) is superseded, not approved or filed.

**Inputs used:** the questions below, [TD-5871's research](https://github.com/typedef-ai/fenic/blob/ff87d5f9179f39e27f885792a8f2fcdcd00f334c/specs/td-flow/td-5871-polars-2-impact/research.md) and [breakdown](https://github.com/typedef-ai/fenic/blob/ff87d5f9179f39e27f885792a8f2fcdcd00f334c/specs/td-flow/td-5871-polars-2-impact/breakdown.md), the cited production and test files, dependency metadata, primary documentation, and synthetic probes. Derived scope includes SQL result-type inference, object-name normalization, and independently locked documentation services.

**Inputs deliberately excluded:** provider execution, private datasets, live remote IO, product implementation, and unrelated semantic operators.

## Summary

The historical inspection found three distinct boundaries: SQL compatibility, durable storage, and file/remote IO. Their replacements have different failure modes. It initially proposed the now-superseded [breakdown](breakdown.md). The current commissioned unit is SQL only; [design.md](design.md) selects its bounded SQL-only Polars route using the newer measurements.

Polars SQL handles several tested SELECT shapes. It does not preserve all existing SQL examples. In particular, wildcard `USING` joins add a column, integer `SUM` changes physical and logical types, and MCP's `REGEXP_MATCHES()` example fails in both tested Polars versions.

DuckDB provides the current catalog's file persistence and data/metadata transactions. A frame registry does not provide those contracts. Baseline engine probes verified rollback, snapshot reads, reopen, and same-version export/import. They did not test a replacement catalog.

Native reader calls were faster in the bounded local comparison, including empty and null-heavy inputs. Reader-only migration removes no required dependency. Complete-removal performance, cold installation, and net package savings after adding replacement dependencies remain unmeasured.

## Research Questions

1. How do `Session.sql`, SQL logical and physical plans, and MCP Analyze parse queries, register frames, derive schemas, and report errors?
2. How do the catalog and system-table client persist data, schemas, views, tools, metrics, and query history across transactions and reopen?
3. How do CSV/Parquet and S3/Hugging Face readers and writers handle schemas, credentials, status codes, paths, and concurrent operations?
4. How do the materialized frame cache and lineage store preserve types, discover objects, and manage their lifetimes?
5. Which SQL, storage, and IO behavior do the current DuckDB and Polars primary sources document, and which cases differ in local probes?
6. What package-size, process-memory, and reader-throughput results do isolated synthetic measurements produce on this host?

## Findings

Paths abbreviated as `local/` start at `src/fenic/_backends/local/`. Paths abbreviated as `core/` start at `src/fenic/core/`. Code citations refer to the inspected revision.

### 1. SQL entry points and the existing public contract

`Session.sql` accepts query text and named DataFrame placeholders. It promises DuckDB syntax, validates missing placeholders, and requires inputs from the same session (`src/fenic/api/session/session.py:273-339`). Its signature has no bound-value parameter interface. A DuckDB driver's successful `SELECT ?` probe therefore does not demonstrate a supported public `Session.sql` feature.

Planning replaces placeholders with generated view names. It registers typed empty frames, parses with sqlglot's DuckDB dialect, rejects multiple statements and listed DDL/DML nodes, and derives the result schema through DuckDB (`core/_logical_plan/plans/transform.py:657-707,874-895`). The current validator is a node rejection list, not proof that an alternate parser accepts exactly the same language.

Execution registers materialized child frames on a cursor in the intermediate database. It runs the query, applies ingestion coercions, and drops the temporary views (`local/physical_plan/transform.py:518-531`). Query execution errors pass through `ExecutionError`; planning failures use `PlanError` (`local/execution.py:197-209`; `core/_logical_plan/plans/transform.py:678-704`).

MCP Analyze delegates to `Session.sql`. Its argument and tool descriptions explicitly name DuckDB. Its examples use `REGEXP_MATCHES()`, grouped counts, and ordered pagination (`src/fenic/api/mcp/_tool_generation_utils.py:577-624`). This tool description is part of the compatibility boundary, not incidental documentation.

Polars documents SQL execution against registered frames, with lazy execution even when an eager result is requested [P1]. Polars 2 changes lazy/SQL collection to the streaming default. It also documents deferred query resolution through `collect` or `collect_schema` [P2]. The local probe returned a lazy frame for an unknown column and failed at `collect_schema`. Malformed SQL failed earlier. Both stages are relevant to error adaptation.

#### Differential case list

The same typed inputs contain duplicate keys, a null key, unmatched rows, and a null integer value. DuckDB is 1.4.5. All comparisons use explicit ordering where row order matters. “Equal” below means equal values, column names/order, and physical schema for that case only.

| Case                                           | Polars 1.43.2                         | Polars 2.0.0                            | Compatibility fact                                          |
| ---------------------------------------------- | ------------------------------------- | --------------------------------------- | ----------------------------------------------------------- |
| Projection, filter, explicit `NULLS LAST`      | Equal                                 | Equal                                   | Common frame query demonstrated                             |
| Explicit-projection left `JOIN ... ON`         | Equal                                 | Equal                                   | Duplicate multiplicity and unmatched/null keys demonstrated |
| `JOIN ... USING`, explicit left key projection | Equal                                 | Equal                                   | This projection avoids the duplicate output key             |
| `SELECT * ... JOIN ... USING`                  | Extra `id:r`                          | Extra `id:r`                            | Existing public join example is not preserved               |
| CTE with `UNION ALL` and ordering              | Equal                                 | Equal                                   | Tested scalar schema only                                   |
| `ROW_NUMBER` partition/order                   | `UInt32` versus DuckDB `Int64`        | Equal                                   | Physical width depends on the runtime                       |
| Integer `SUM`, grouped `COUNT(*)`              | SUM `Int64`; count `UInt32`           | SUM `Int64`; count `Int64`              | DuckDB SUM exports `Decimal(38,0)`                          |
| `REGEXP_MATCHES(g,'a')`                        | Unsupported function                  | Unsupported function                    | Existing MCP search example is not preserved                |
| Ordered `LIMIT`/`OFFSET`                       | Equal                                 | Equal                                   | Stable paging still requires a total ordering for ties      |
| `0.1`, negative `%`                            | Float literal and different remainder | Same values; decimal precision differs  | 2.0 fixes values, not exact physical schema parity          |
| `DATE_PART` after timestamp minus interval     | Syntax error                          | Same value; `Int8` versus `Int64`       | Runtime-specific syntax and width difference                |
| `duckdb_settings()`                            | Unsupported table function            | Unsupported table function              | An existing timezone test uses this function                |
| Driver-bound `?`                               | Unsupported placeholder               | Unsupported placeholder                 | Used by storage APIs, not exposed as public SQL binding     |
| Empty filtered `SUM`/`COUNT`                   | Same null/zero values; types differ   | Same null/zero values; SUM type differs | Empty output does not resolve aggregate-type differences    |
| Null arithmetic and null predicate             | Equal                                 | Equal                                   | Declared `Int64` survives an all-null derived column        |

These are 15 engine cases, not 15 passing fenic integration tests. Six cases have complete row/schema parity on 1.43.2; seven do on 2.0.0. Raw results compare numeric Decimal and integer values before JSON serialization. Decimal payloads serialize as strings in the scratch JSON.

The SUM difference also changes fenic's logical schema. Its type converter maps Decimal to `DoubleType` and integer widths to `IntegerType` (`core/_utils/type_inference.py:154-161`). Equal small numbers therefore do not establish API parity or equivalent overflow behavior. The DATE_PART width difference maps to the same logical integer type, but remains visible through Polars results.

Existing tests assert the coalesced `USING` output, CTE/union behavior, windows, and planning errors (`tests/_backends/local/test_sql.py:25-155,191-274`). Temporal/nested tests cover arrays, date casts, timezone coercion, and cached results. A timezone test queries `duckdb_settings()` (`tests/_backends/local/test_sql.py:157-189,276-361`). These cases prevent treating the demonstrated common subset as a drop-in backend.

The observed subset consists of explicit scalar projections, tested predicates, explicit-key joins, CTE/union, and ordered pagination. Ordered row numbering joins that subset only with an explicit dtype rule. Aggregate widening, regex syntax, `USING` wildcard behavior, system table functions, and parameters remain distinct contract decisions. No opt-in backend or fallback is implemented by this research.

### 2. Durable catalog and migration boundary

The session opens `<app_name>.duckdb` under the configured database directory. It opens a separate temporary frame database (`local/session_state.py:50-63`). The catalog supports one fixed catalog name. Its “databases” are DuckDB schemas, with case-folded qualified names and a protected system schema (`local/catalog.py:106-239`; `src/fenic/_backends/utils/catalog_utils.py:12-155`).

| Current contract                 | Implementation and limits                                                                                                                                          | Verification boundary                                                              |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------- |
| File reopen                      | Persistent DuckDB path, separate from temporary frames (`local/session_state.py:50-63`)                                                                            | Reopen data and metadata in a new process/session                                  |
| Schemas and discovery            | Schema creation/drop, `duckdb_schemas`, `information_schema.tables`; serialized views use their own registry (`local/catalog.py:155-299,693-713`)                  | Qualified/quoted/case-folded names; visible versus hidden schemas; empty discovery |
| Atomic data and logical metadata | Explicit BEGIN/COMMIT/ROLLBACK; create, overwrite, and drop modify data and metadata on the same cursor (`local/catalog.py:56-81,364-385,435-447,571-578,647-654`) | Fail between data and metadata updates, then reopen                                |
| Appends                          | Schema equality check and INSERT, separate from create/overwrite (`local/catalog.py:591-624`)                                                                      | Matching and mismatched schemas; concurrent append/read                            |
| Metadata upserts                 | Bound `INSERT OR REPLACE`; preserve an existing description when no description is supplied (`local/system_table_client.py:82-105,263-293,469-488`)                | Preserve schema/plan/tool payloads and descriptions exactly                        |
| Views and tools                  | Base64/protobuf logical plans and tool definitions, not SQL views alone (`local/system_table_client.py:263-343,469-528,780-840`)                                   | Deserialize and resolve after reopen, without calling a model                      |
| Metrics/history                  | Append execution IDs, session IDs, timing, row count, token/request/cost totals; query by session (`local/system_table_client.py:568-677,716-773`)                 | Persist rows and totals; protect `fenic_system.query_metrics`                      |
| Concurrency                      | Per-thread cursors; catalog RLock protects many check-then-act operations, not every write (`local/catalog.py:89-104,558-568,591-602,633-644`)                     | Reader snapshot, conflicting writer, object creation races                         |

The persisted “query history” is an execution-metrics history. The table does not store SQL text or the complete physical/operator plan (`local/system_table_client.py:583-605,730-745`). Full query-text history would be a new feature, not a preservation requirement.

Metrics insertion occurs after physical execution, outside the data/metadata transaction (`local/execution.py:197-209`). The inspected code does not establish one transaction containing table writes and their metrics row. Preserving catalog atomicity must not be described as an existing all-query atomicity guarantee.

DuckDB documents transactions, snapshot isolation, and rollback [D1]. Its in-process read/write model permits multiple threads in one writer process. Conflicting edits can fail; this is not arbitrary multi-process writing to the current file path [D2].

The engine-level baseline probe replaced a table and metadata within one transaction, then injected a failure. Rollback restored both. A reader cursor retained its snapshot while another cursor committed an append. Closing and reopening retained two rows, metadata, and two synthetic metrics records.

The second probe exported that synthetic database as Parquet and imported it into a fresh DuckDB 1.4.5 file. Data, metadata, and metrics matched. The original file remained. DuckDB documents export/import of schemas, tables, views, and sequences [D3]. This is a baseline export fixture, not migration to a replacement backend or proof of fenic plan/tool serialization.

DuckDB's storage documentation distinguishes backward compatibility from best-effort forward compatibility [D4]. Keeping an original file supports rollback to that snapshot. It does **not** preserve writes made after switching to another store. Reversible post-cutover migration therefore has a separate requirement: reverse-export all supported new state, or reject rollback when newer writes cannot be represented. No such path exists in these probes.

Polars SQLContext documents a frame registry [P1]. No inspected API establishes catalog files, durable schema discovery, metadata upserts, or a shared transaction across external data files and metadata. A Parquet-file-plus-metadata proposal must specify publication and crash recovery before it can claim equivalence. Selecting its storage engine, commit protocol, and migration interface belongs to a separate design unit.

#### Catalog fixture gaps

Existing tests exercise database/table/view/tool operations and descriptions (`tests/_backends/local/catalog/test_catalog.py:61-171,203-271,316-440,491-536`). Metrics tests cover public schema, read-only protection, execution contents, and session separation (`tests/_backends/local/catalog/test_metrics_table.py:16-133`). The targeted catalog-directory search found no tests named for injected rollback, process reopen, or concurrent writers. This is a scoped search result, not a whole-repository absence claim.

A replacement needs fixtures for data plus metadata rollback, restart after each publication boundary, append/overwrite/drop races, protobuf round trips, and the original/new-store/reverse-export sequence. Successful SELECT or same-version DuckDB import cannot stand in for those fixtures.

### 3. Local IO, remote IO, and temporary storage

#### Local readers and writers

The reader opens a fresh DuckDB connection, builds the scan query, and converts the result to Polars. CSV explicit schemas map to primitive DuckDB types. `merge_schemas=True` selects `union_by_name` for CSV/Parquet (`local/utils/io_utils.py:72-90,230-307`). The physical source then recursively normalizes arrays and timestamps (`local/physical_plan/source.py:72-78`).

Plan-time schema inference calls this same read function. The inspected callers do not request the helper's `schema_inference=True` option (`local/execution.py:124-140`; `core/_logical_plan/plans/source.py:89-126`). A measured schema-only benefit cannot be credited to a reader swap without separately testing that planning path.

Existing CSV tests require explicit-schema conversion, union column order, null fill, and numeric/string widening. Default multi-file CSV derives columns from the first file but can widen types across files (`tests/_backends/local/io/test_reader.py:296-533`). Default multi-file Parquet uses the first file's schema and casts later files when possible. Merge mode unions names and widens types (`tests/_backends/local/io/test_reader.py:536-662`).

The fresh two-file CSV probe confirmed that DuckDB unions disjoint columns, while default Polars 2 reading rejects differing column names. Polars 1.43.2 does not accept the same list-source `read_csv` call. The 2.0 upgrade guide documents list-source CSV reading and changed schema override behavior [P2]. Its Parquet scan API offers explicit missing/extra-column and cast options, some marked unstable [P3]. These options do not establish fenic's exact widening or first-file contract.

The writer converts the frame to Arrow and uses DuckDB COPY with CSV headers or Parquet (`local/utils/io_utils.py:92-111,276-283`). The file sink checks existence for `error`, `ignore`, and `overwrite`; ignored and completed writes return a zero-column sentinel (`local/physical_plan/sink.py:40-58`). Local existence checks and write calls are separate. The current path has no demonstrated atomic no-clobber guarantee against a competing writer.

Datetime normalization casts to UTC microseconds recursively inside List/Struct. File and SQL ingestion normalize fixed arrays to lists unless a logical embedding schema is supplied. Catalog and temporary reads call coercion with `coerce_array=False` (`local/physical_plan/utils.py:12-116`; `local/catalog.py:667-678`; `local/temp_df_db_client.py:25-28`).

The additional Parquet probe preserved list nulls, empty lists, null structs, and equal timestamp instants. Raw DuckDB export returned the embedding column as List(Float32) and the timestamp in the host timezone. Native Polars returned Array(Float32,2) and UTC. Equal row values did not mean equal physical schema. An all-null scalar Parquet frame retained its declared Int64/Float64/String schema in both engines.

#### Remote credentials and errors

Remote reads install/load `httpfs`. Missing S3 credentials permit an anonymous read attempt. S3 credential extraction requires access key, secret key, and region; it includes a session token when present. Hugging Face credentials come from `HF_TOKEN` (`local/utils/io_utils.py:140-178,217-227,348-364`).

S3 writes require credentials. S3 existence checks issue HEAD and treat only 404 as absent; other errors propagate (`local/utils/io_utils.py:51-70,191-215`). The public sink does not establish Hugging Face write support: its existence helper rejects that scheme. The generic writer's “hf” docstring is not sufficient evidence of a supported HF write route.

HTTP formatting distinguishes 404, 401/403 with credentials, and 401/403 without credentials. Other statuses include the underlying error. Mixed S3/HF requests give S3 formatting precedence (`local/utils/io_utils.py:312-345`). Twenty source-extracted helper cases checked that matrix with fake statuses. They did not exercise native Polars remote exceptions or network calls.

The HF regression requires a `PlanError` caused by `FileLoaderError`, accepting 401, 403, or 429 for absent/invalid tokens (`tests/_backends/local/io/test_reader.py:786-805`). S3 tests assert the no-credential message and cause chain (`tests/_backends/local/io/test_reader.py:739-777`). Public HF tests require an explicit opt-in flag (`tests/_backends/local/io/test_reader.py:830-847`). #324 introduced the DuckDB cap to protect this error behavior [F1].

Polars documents S3 credentials through `storage_options` and credential providers [P4]. Its Parquet API documents HF tokens and `hf://` [P3]. URI/token support alone does not prove private/gated, revision, glob, mixed-source, or error-status equivalence. No live remote result is claimed here.

#### Materialized frames and lineage

The temporary client deletes its prior database at initialization and cleanup. It creates named tables, checks presence through `sqlite_master`, and reads through per-operation cursors (`local/temp_df_db_client.py:15-53`). These files are disposable session state, unlike the catalog. The probes never instantiated this cleanup path.

Physical execution writes a materialized frame after a cache miss (`local/physical_plan/base.py:90-94`). Cache reads use the same temporary client (`local/physical_plan/source.py:189-197`). Lineage reads materialized rows and UUID mapping tables, then filters them in Polars (`local/lineage.py:90-112,157-183`). Existing lineage tests cover forwards/backwards, joins, unions, and transformations (`tests/_backends/local/test_lineage.py:43-237`).

SQL execution currently shares the intermediate database's connection and named objects. Replacing frame storage without moving SQL must account for that coupling (`local/physical_plan/transform.py:518-523`). It cannot be treated as replacing only `read_df` and `write_df`.

The LLM response cache already uses SQLite and is separate (`local/session_state.py:80-97`). Lance similarity search is separate from the DuckDB inventory in the prior report. Neither belongs in claimed DuckDB removal savings.

#### Adapter contract checklist

This table records the contract evidence and missing verification. It does not approve new adapters.

| Boundary                       | Error and type evidence to preserve                                                                                                                        | Concurrency/lifetime verification                                                                               |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| CSV/Parquet read plus planning | Validation/PlanError/FileLoaderError layering; explicit types; merge versus first-file behavior; empty/all-null results; recursive UTC/array normalization | Independent reader state; files changed between planning and collection; deterministic file expansion           |
| Local write                    | Existing mode errors and zero-column result; CSV headers/null/quoting; Parquet nested/timezone/logical-type round trips                                    | Competing writes; partial write; existing target retained on failure; state whether atomicity is newly provided |
| S3 read/write                  | Existing boto selection and temporary credentials; anonymous read, authenticated write; 401/403/404/429; causes and sanitized messages                     | Credential expiry/refresh during a scan; independent sessions; HEAD/write races                                 |
| HF read                        | Missing/invalid token, private/gated/public data, revisions/globs, status preservation; no inferred HF write support                                       | No global token leakage between sessions; bounded retry; mixed-source attribution                               |
| Materialized cache             | Declared output schema, nested arrays/embeddings, missing/duplicate keys, discovery, exactly associated rows                                               | Same-key concurrent materialization; cache miss versus published result; stop/restart and scoped cleanup        |
| Lineage store                  | UUID String and mapping schemas; empty/duplicate/null mapping cases; forward/backward results                                                              | Publish materialized frame and mapping before traversal; failed builds; lifetime bound to its session           |

### 4. Measured gains and their limits

#### Procedure

Both environments use CPython 3.11.11, PyArrow 23.0.1, and DuckDB 1.4.5 on macOS arm64. One environment has pinned Polars 1.43.2; the other has target 2.0.0. Both engines use four threads. Heavy steps ran sequentially. Recorded five-minute load averages at probe start were 5.01 and 5.05, below 16.

Each file has three declared columns: Int64 `id`, Float64 `value`, and String `text`. Populated cases have 100,000 rows. IDs ascend from zero, value is `id/10`, and text repeats `row-0` through `row-99`. The null-heavy case replaces every second value/text with null. Empty files retain the declared schema. CSV columns are declared explicitly in **both** arms.

Each DuckDB arm includes fresh connection setup, the reader, Polars export, and explicit connection close. The native arm calls `pl.read_csv` or `pl.read_parquet`. One warmup precedes seven paired samples with alternating engine order. Both arms materialize all rows. Each result passes `assert_frame_equal`, including schema.

This measures warm local reader calls, not an implemented fenic adapter. It excludes schema inference, remote IO, metadata/catalog work, and provider time. Explicit DuckDB close also differs from the source helper's implicit connection lifetime. No predicate/projection pushdown or out-of-core gain is credited.

#### Pinned runtime results

Times below are milliseconds, median with minimum–maximum across seven samples.

| Format/input                 | DuckDB unchanged reader shape | Native Polars 1.43.2 reader |
| ---------------------------- | ----------------------------- | --------------------------- |
| CSV, ordinary                | 25.141 (24.922–26.074)        | 1.221 (1.133–1.319)         |
| Parquet, ordinary            | 6.711 (6.382–6.993)           | 1.989 (1.875–2.229)         |
| CSV, 50% null value/text     | 20.832 (20.343–21.544)        | 1.125 (1.044–1.435)         |
| Parquet, 50% null value/text | 6.318 (6.090–6.678)           | 1.601 (1.431–1.727)         |
| CSV, empty                   | 4.492 (4.236–4.562)           | 0.050 (0.045–0.150)         |
| Parquet, empty               | 3.698 (3.296–3.979)           | 0.300 (0.181–0.388)         |

#### Target runtime results

| Format/input                 | DuckDB reader shape in same process | Native Polars 2.0.0 reader |
| ---------------------------- | ----------------------------------- | -------------------------- |
| CSV, ordinary                | 24.755 (23.684–25.774)              | 1.737 (1.544–1.888)        |
| Parquet, ordinary            | 6.539 (6.408–6.762)                 | 1.908 (1.705–2.360)        |
| CSV, 50% null value/text     | 20.447 (20.145–20.929)              | 2.093 (1.869–2.254)        |
| Parquet, 50% null value/text | 6.206 (5.780–6.435)                 | 1.598 (1.458–1.654)        |
| CSV, empty                   | 4.073 (3.916–4.578)                 | 0.377 (0.193–0.448)        |
| Parquet, empty               | 3.474 (3.308–3.630)                 | 0.218 (0.161–0.324)        |

The two runtime experiments are sequential and not a controlled Polars-version speed comparison. Each native-versus-DuckDB pair is interleaved on one runtime. Background activity and warm OS caches limit generalization.

#### Packaging, installation, and memory

| Dimension                                       | Leave unchanged                                                               | Reader-only move                                                   | Complete removal                                                                           |
| ----------------------------------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------ |
| Required DuckDB download                        | Locked macOS CPython 3.11 wheel: 13,681,675 bytes, 13.048 MiB (`uv.lock:719`) | Zero wheel bytes removed while SQL/catalog depend on DuckDB        | Gross removable wheel is 13.048 MiB; net savings after replacement dependencies unmeasured |
| Installed DuckDB files                          | Probe distribution: 38,980,491 bytes, 37.175 MiB across 54 recorded files     | No dependency removal                                              | Gross package footprint only; not a built replacement environment                          |
| Full cold fenic install                         | Unmeasured                                                                    | Unmeasured                                                         | Unmeasured                                                                                 |
| Small warm reader latency                       | Table above                                                                   | Native-call opportunity demonstrated, adapter overhead unmeasured  | No complete backend exists; throughput unmeasured                                          |
| Process peak memory                             | Three fresh-process micro-probes import Polars/PyArrow and open/use DuckDB    | Catalog/SQL still import DuckDB; production memory gain unmeasured | Omission micro-probe only, not production memory gain                                      |
| Large-memory, remote, write, catalog throughput | Unmeasured                                                                    | Unmeasured                                                         | Unmeasured                                                                                 |

For pinned Polars, median peak resident memory was 90,734,592 bytes with a small DuckDB connection/query, versus 68,370,432 bytes without DuckDB. For Polars 2 it was 93,847,552 versus 71,614,464 bytes. These roughly 21 MiB differences include import and connection activity. They are not a measured saving for reader-only migration or a complete replacement.

The isolated first environment preparation reported 5.49 seconds and installation 8 milliseconds for four packages. The pinned setup reused DuckDB/PyArrow and prepared its two Polars packages in 68 seconds. Network/cache conditions differed. Neither is a cold-install comparison across the three product strategies.

Root dependencies still require DuckDB, pinned Polars, and PyArrow (`pyproject.toml:27-32`). The prior report also identifies independently locked documentation consumers. Reader migration does not remove those requirements. Polars release SQL benchmarks are not evidence for these fenic workloads.

#### Exact retained commands and result locations

`R` below is the absolute worktree root. All scratch paths remain under its `.context/`. These commands use direct engine imports, not fenic sessions.

```sh
R="$(git rev-parse --show-toplevel)"
env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY -u GOOGLE_API_KEY \
  -u GEMINI_API_KEY -u TYPESAFE_API_KEY -u COHERE_API_KEY \
  -u OPENROUTER_API_KEY -u HF_TOKEN \
  UV_NO_ENV_FILE=true UV_CACHE_DIR="$R/.context/uv-cache" \
  uv venv --python 3.11 "$R/.context/probe-venv"
env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY -u GOOGLE_API_KEY \
  -u GEMINI_API_KEY -u TYPESAFE_API_KEY -u COHERE_API_KEY \
  -u OPENROUTER_API_KEY -u HF_TOKEN \
  UV_NO_ENV_FILE=true UV_CACHE_DIR="$R/.context/uv-cache" \
  uv pip install --python "$R/.context/probe-venv/bin/python" \
  duckdb==1.4.5 polars==2.0.0 pyarrow==23.0.1
```

The pinned environment uses identical commands with `pinned-venv` and `polars==1.43.2`. The actual setup wrapper checked the five-minute load before each heavy step and imposed a 1,800-second subprocess timeout. Each probe also asserts the load gate and absence of all eight variables. Synthetic probes disable Python socket connections.

```sh
env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY -u GOOGLE_API_KEY \
  -u GEMINI_API_KEY -u TYPESAFE_API_KEY -u COHERE_API_KEY \
  -u OPENROUTER_API_KEY -u HF_TOKEN POLARS_MAX_THREADS=4 \
  TD5875_PROBE_OUTPUT="$R/.context/pinned-results" \
  "$R/.context/pinned-venv/bin/python" "$R/.context/probe.py"
env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY -u GOOGLE_API_KEY \
  -u GEMINI_API_KEY -u TYPESAFE_API_KEY -u COHERE_API_KEY \
  -u OPENROUTER_API_KEY -u HF_TOKEN POLARS_MAX_THREADS=4 \
  TD5875_PROBE_OUTPUT="$R/.context/polars2-results" \
  "$R/.context/probe-venv/bin/python" "$R/.context/probe.py"
env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY -u GOOGLE_API_KEY \
  -u GEMINI_API_KEY -u TYPESAFE_API_KEY -u COHERE_API_KEY \
  -u OPENROUTER_API_KEY -u HF_TOKEN POLARS_MAX_THREADS=4 \
  "$R/.context/probe-venv/bin/python" "$R/.context/boundary_probe.py"
```

Output directories must be newly created before a rerun. The scripts refuse to overwrite results. Memory cases run fresh child Python processes with `probe.py memory duckdb` and `probe.py memory native`.

The retained `probe.py` SHA-256 is `28d5e9e8c44d424b70591403364bb4584cefcd45958d341b609e54db9abeec73`. The retained `boundary_probe.py` SHA-256 is `d88c7dfe2d4e3ea75d696d60803facdba093ee5cec33e590d801451460c3ec4a`.

Final compared outputs are `.context/pinned-results/` and `.context/polars2-results/`: `probe-receipt.json`, `sql-results.json`, `storage-results.json`, `reader-results.json`, `memory-results.json`, `package-results.json`, and `polars-primary-docstrings.json`. Extra boundary results are `.context/boundary-results.json`. An earlier target run remains at `.context/` with its own results. No scratch was deleted. Scratch is not committed; this report contains the decision-bearing measurements.

### 5. Research outcome and remaining verification

The historical split separated SQL language compatibility, storage durability/migration, and IO inference/credential/error contracts. The temporary frame store also differs from the durable catalog despite shared SQL infrastructure. These facts explain the earlier [breakdown](breakdown.md), not a current request to pursue it. Command narrowed the work to the SQL materialization boundary; the newer measurements and design govern that unit.

No replacement backend, new SQL option, or remote adapter exists from this work. No whole fenic suite ran. Passing engine probes establish only the tested shapes. The integration, crash, remote, overflow, and package fixtures remain open.

## Open Questions

1. Which opt-in SQL grammar, physical/logical dtype rules, regex behavior, and error phases will be accepted?
2. Which store and commit protocol will preserve data plus metadata through process crashes, concurrent writes, and migration?
3. Which native remote exceptions expose reliable status codes without string parsing, including HF 401/403/429?
4. Which exact CSV inference and multi-file widening results can a native adapter preserve on the selected runtime?
5. What end-to-end reader memory/latency and full-removal installation/throughput results hold after approved adapters exist?

## Decision Log

- **Applied:** Keep the existing inventory as the baseline; verify its SQL, catalog, IO, and cache anchors against the identical source revision.
- **Applied:** Compare native readers on both pinned and target Polars. Do not attribute their gains exclusively to Polars 2.
- **Applied:** Distinguish engine fixtures, source-extracted error helpers, and product integration tests.
- **Applied:** Keep physical dtype comparisons separate from logical type mapping and numeric-value equality.
- **Historical checkpoint:** Research originally stopped at a proposed split. Command superseded that split with one SQL-only unit; its architecture and contracts now live in `design.md`. Other units remain uncommissioned.
- **Review scope:** Opus 5.5 reviewed the four-page candidate at `64e1bc3b` and returned Not ready. The authorized fix round adds unfused evidence and routing precedence. Delta review is pending; this is not design approval.

## Primary sources

All sources below were retrieved on **2026-10-09 UTC** (2026-10-08 local). Stable documentation can change; runtime observations above identify exact tested versions.

- **P1:** [Polars SQLContext.execute API](https://docs.pola.rs/api/python/stable/reference/sql/api/polars.SQLContext.execute.html). Registered frames, lazy execution, and eager collection.
- **P2:** [Polars 2 upgrade guide](https://docs.pola.rs/releases/upgrade/2/). Streaming collection, SQL resolution, CSV list sources, decimal and remainder changes. The installed 2.0.0 runtime corroborates the tested behavior; not every error is deferred.
- **P3:** [Polars scan_parquet API](https://docs.pola.rs/api/python/stable/reference/api/polars.scan_parquet.html). HF token options, missing/extra columns, cast options, credential-provider stability warnings.
- **P4:** [Polars cloud storage guide](https://docs.pola.rs/user-guide/io/cloud-storage/). S3 reads/writes, storage options, custom credentials, and retry configuration.
- **P5:** [Polars scan_csv API](https://docs.pola.rs/api/python/stable/reference/api/polars.scan_csv.html). CSV inference and schema options. Version-specific local docstrings also remain in the result directories.
- **D1:** [DuckDB transaction management](https://duckdb.org/docs/current/sql/statements/transactions.html). Transactions, rollback, and snapshot isolation.
- **D2:** [DuckDB concurrency](https://duckdb.org/docs/current/connect/concurrency). In-process writer threads and conflict behavior. Newer remote protocols are not used by this inspected fenic version.
- **D3:** [DuckDB EXPORT and IMPORT](https://duckdb.org/docs/current/sql/statements/export.html). Database contents, Parquet export, and import into a fresh database.
- **D4:** [DuckDB storage versions and format](https://duckdb.org/docs/current/internals/storage). Backward versus forward compatibility and export-based conversion.
- **F1:** [fenic #324](https://github.com/typedef-ai/fenic/pull/324). Maintainer record of the HF error-contract cap; this is project provenance, not proof of native Polars remote parity.

The stable DuckDB transaction/export URLs initially returned redirect pages. Research followed their explicit current-documentation links; no redirect body was treated as substantive evidence.

## Related public work

- [#410](https://github.com/typedef-ai/fenic/pull/410), TD-5871 research, remained open at inspection.
- [#403](https://github.com/typedef-ai/fenic/pull/403), dtype hardening, remained open at head `55408d48fd1839cf52a3cef9fdaf86c0ec33b256`.
- [#381](https://github.com/typedef-ai/fenic/pull/381), compatible-provider work, remained open at head `ab1a3a92df470cc078bb59db0402993356008704`. These probes do not use provider transports.
- [#411](https://github.com/typedef-ai/fenic/pull/411), DuckDB pin work, remained open at head `be4d2708fee4d9a11d185322a02ee89e6966b31c`. No dependency pin or lockfile changed in this research.

## Handoff

The current handoff is review of [the focused SQL design](design.md), grounded in [the materialization comparison](sql-materialization-measurements.md). The broad split proposal is superseded. No implementation, ticket filing, or publication follows automatically.
