---
workflow_id: td-5875-duckdb-replacement-design
phase: breakdown
track: planning
size_class: full
status: needs_review
portability_level: 1
source_inputs:
  - research.md
last_updated: 2026-10-09
---

# Proposed split: DuckDB replacement design boundaries

Research found three separate design problems. This proposal stops before architecture selection and before `design.md`. Each unit below is a draft design ticket, not an implementation authorization or filed issue.

**Grounding:** [research.md](research.md), source revision `71609ab6a2916fd36c7db80557aa614ebdbe5880`. Its primary-source register supplies URLs and retrieval dates for dependency claims.

## Recommendation at this checkpoint

**Leave DuckDB unchanged for now.** Make reader-only adoption the first candidate design because the local measurements show a bounded benefit without changing public SQL or catalog durability. Do not recommend full removal before its storage/migration and SQL decisions exist.

A reader-only move saves zero dependency bytes while DuckDB still serves the catalog and SQL. The measured native-call latency does not prove an integrated adapter has equal errors, inference, memory use, or remote behavior.

## Draft design tickets

### D1. Define native-reader and IO adapter compatibility

**Goal:** Decide whether a bounded native-reader move is worthwhile while retaining the existing SQL engine and durable catalog.

**Evidence:** Fresh native reader calls were faster on six scalar workloads under both pinned and target Polars. Default multi-file CSV union differs. Nested Parquet arrays and timezone metadata differ before normalization. Remote status parity is untested.

**Scope:** Local CSV/Parquet planning and reads; write-mode and round-trip contracts; S3/HF credential/error adapters; resource lifetime and bounded retry. Separate reader adoption from writer/remote adoption if the latter do not meet their own fixtures.

**Decisions required:**

- Exact first-file, multi-file widening, union column order, and explicit schema rules.
- Declared physical types for empty/all-null output, nested fields, UTC microseconds, and embeddings.
- Public exception class/cause/status behavior, including #324.
- Credential selection/refresh and per-session isolation, without global secrets or logged credentials.
- Existing versus newly promised concurrent-write atomicity.

**Acceptance:** Name every preserved or intentionally excluded case. Run differential IO fixtures on the selected runtime with equal logical and physical schemas. Measure the integrated planning-plus-read route against the unchanged route. Report memory, remote and cold-install results as unmeasured until they exist. Do not remove DuckDB's dependency.

**Fixtures:** Existing reader merge/first-file/type tests and writer-mode tests; typed 0-row, all-null, nested/embedding/timezone inputs; CSV quoting/null/encoding; missing/corrupt files; no-token/invalid-token/401/403/404/429/500 mocks; S3 HEAD/write races; mixed-source paths; HF revision/glob/gated/public cases.

**Boundary:** One focused design report, with local-only adoption separable from remote/writer adoption. No SQL or catalog implementation. No live credentialed validation without separate permission.

**Dependency:** None on D2 or D3. Refresh the selected Polars and DuckDB pins first. A Polars 2-specific option cannot silently become a requirement while main remains pinned to 1.43.2.

### D2. Define an explicit Polars SQL opt-in contract

**Goal:** Decide whether a versioned, explicitly selected alternate SQL language is useful without silently changing DuckDB's default.

**Evidence:** Seven of 15 tested cases have complete physical row/schema parity on Polars 2. The `USING` wildcard, integer SUM logical type, regex example, system table functions, and parameters do not match.

**Scope:** Public SQL selection, query validation/planning/execution, output schemas and ordering, serialized plans, and MCP Analyze's advertised syntax.

**Candidate admission boundary, not an approved API:**

- Admit the tested explicit scalar projection/filter, explicit `ON` join, CTE/UNION ALL, and ordered paging shapes only after integration fixtures pass.
- Require explicit projection for `USING`; exclude wildcard `USING` and untested outer/coalesced-key cases rather than drop right-hand columns heuristically.
- Define each admitted result's logical and physical dtype at planning, including empty/all-null results. Do not use the first observed value to choose it.
- Keep SUM/overflow and decimal/date-part differences out until a declared normalization or explicit non-compatibility rule exists.
- Exclude `duckdb_settings()` and other DuckDB-only table functions. Do not silently route rejected queries back to DuckDB.
- Exclude driver-bound parameters from the alternate public query contract. Preserve bound parameters independently in whichever metadata store D3 selects.
- Exclude `REGEXP_MATCHES()` until a tested alternate spelling/semantics is chosen. Generate alternate MCP examples from its actual contract, not the current DuckDB examples.
- Validate both parser construction and `collect_schema` errors before execution. Require explicit stable ordering for paging and order-sensitive downstream processing.

**Acceptance:** A supported grammar/function list with a refusal case for everything excluded; no changed default; engine/dialect identity in plans, persisted views/tools, and MCP descriptions; per-result logical/physical dtype fixtures; explicit error phases; full results for duplicates, null keys, empty aggregates, overflow, timestamps, and arrays.

**Fixtures:** Existing SQL and MCP examples plus all 15 differential cases. Add multi-join/star/outer-USING refusal or equivalence fixtures, stable pagination ties, aggregate overflow and nulls, plan serde, and planning/execution error classification. Keep DuckDB tests green independently.

**Boundary:** One public-language decision. It can produce “do not offer an alternate backend.” It does not remove DuckDB or select persistent storage.

**Dependency:** None on D1. D3's eventual complete-removal decision needs D2's accepted compatibility limits.

### D3. Define durable storage, temporary frames, and reversible migration

**Goal:** Decide whether any replacement store can preserve catalog contracts at acceptable migration and operational cost.

**Evidence:** Current create/overwrite/drop transactions couple data and logical metadata. Reopen, snapshots, and rollback pass engine probes. Temporary frames are disposable but share SQL's database today. Full replacement is unimplemented.

**Scope:** Catalog data plus schema/view/tool metadata; object discovery and case-folded names; append and upserts; metrics/history; read-only system tables; temporary cache and lineage; crash/concurrency protocol; conversion from existing `.duckdb` catalogs.

**Decisions required:**

- Select the store and serialization formats, including how physical types and protobuf logical metadata survive reopen.
- If immutable frame files use a separate metadata database, define one durable publication point. Cover orphan files, failed commits, overwrite readers, and restart recovery.
- Specify the writer model, reader snapshots, conflict errors, object uniqueness, and cache same-key publication. Do not claim arbitrary multi-process compatibility from current DuckDB thread behavior.
- Keep metrics persistence distinct from table data/metadata atomicity. Existing metrics insertion follows execution; any stronger guarantee must be named as new.
- Choose a distinct new format identity. Do not overwrite an existing `.duckdb` file or label another format as DuckDB.
- Define migration discovery, read-only extraction, validation, opt-in cutover, and repeat-run behavior.
- Preserve the original file for pre-cutover rollback. For post-cutover writes, define reverse-export or explicit rollback refusal. An old snapshot alone is not reversible migration.
- Decide how migration reads old DuckDB files after a core dependency removal, such as a separately installed migration tool. Do not claim that Polars opens them directly.
- Define temporary frame/lineage lifetime independently of durable catalog lifetime. Account for SQL's current connection coupling.

**Acceptance:** Cover every Q2 contract and each cache/lineage Q3 rule; data/metadata crash consistency; no loss on rejected conversion; old-file/new-format/reverse-export reopen; supported serde/type set; explicit unsupported states and recovery; package and workload measurements versus unchanged DuckDB.

**Fixtures:** Inject failure before/after data staging, metadata change, commit, and cutover. Restart between each boundary. Check descriptions, schema blobs, serialized views/tools, metrics totals, protected system tables, quoted/case-folded discovery, schema-mismatch append, conflicting writer, reader snapshots, concurrent materialization, and complete lineage traversal.

**Boundary:** One storage feasibility/design report, not a replacement backend. Treat permanent and disposable data as separate lifetime contracts. Do not introduce a new vector store or replace the SQLite LLM cache.

**Dependency:** D2 is a decision dependency for claiming complete removal, not for studying storage alternatives. Final dependency retirement and net install/memory/throughput measurements follow accepted D1/D2/D3 contracts.

## Sequence and stopping rule

1. Review this split.
2. If approved, start D1 from fresh main and its actual dependency pins.
3. Investigate D2 and D3 independently when their decisions are commissioned. Neither blocks a local-reader-only evaluation.
4. Make the final move-readers/remove-fully/leave-unchanged recommendation only after the selected contracts and integrated measurements exist.

All three draft tickets remain unfiled. No `design.md` exists. No product file, dependency pin, or lockfile changed.

## Verification state

The research contains complete bounded measurements, code anchors, primary-source dates, and explicit unmeasured items. It does not meet the final-design acceptance criteria. In particular, a replacement storage protocol and migration proof are absent, not implicitly accepted by the baseline probes.

## Decision Log

- **Proposed:** Split by public language, durable state, and IO rather than by dependency import locations.
- **Proposed:** Study reader-only value first; keep default SQL and catalog unchanged.
- **Rejected as unsupported:** Full removal justified by SQL benchmark claims or the DuckDB wheel size alone.
- **Held:** Final architecture, approved adapter/SQL/storage contracts, implementation sizing, ticket filing, and publication.
