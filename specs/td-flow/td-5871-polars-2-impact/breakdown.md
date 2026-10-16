---
workflow_id: td-5871-polars-2-impact
phase: breakdown
track: planning
size_class: full
status: needs_review
portability_level: 1
source_inputs:
  - research.md
last_updated: 2026-10-08
---

# Breakdown: upgrade Polars without changing fenic contracts

The Python-only 2.0 comparison finds six new outer-explode failures and several additional boundary changes in local probes. Upgrade work must preserve row/null/type contracts before changing the dependency pin. Replacing DuckDB is separate design work, not a prerequisite or an automatic benefit.

**Grounding:** [research.md](research.md), inspected at `71609ab6a2916fd36c7db80557aa614ebdbe5880`. Its primary-source register supplies URLs and retrieval dates for Polars claims. Recommendations below are proposals, not implementation or filed tickets.

## Project resolution

Use a small dependency-upgrade workstream, not a new initiative. Reuse the existing typing sweep and its deferred follow-up rather than create duplicate work. DuckDB replacement remains a held design proposal with its own contracts and decision gate.

## Coverage map

| Work unit                                      | Verified state on 2026-10-08                                                                                | Disposition                                         |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| Most declared-output typing                    | #403 open; 19 of 34 current constructor sites addressed by its inspected patch                              | REUSE, do not refile                                |
| Generic semantic and embedding callback typing | Existing deferred typing follow-up covers these files; #393 fixes its TypeSafe routes, not generic fallback | REUSE, do not refile                                |
| Outer explode/flatten compatibility            | Six existing tests pass on 1.43.2 and fail on 2.0 for outer explode; empty-inner flatten changes in a probe | NEW N1                                              |
| Python pin plus boundary fixtures              | Target currently exists only in the isolated overlay                                                        | NEW N2                                              |
| DuckDB SQL/storage/reader replacement          | Public dialect and storage contracts differ; no replacement design approved                                 | HELD N3                                             |
| DuckDB version update                          | #398 closed without merge at final refresh; `hf://` regression history remains in #324                      | Historical proposal, do not duplicate automatically |

## Keep, move, or drop DuckDB uses

Every use below maps to the file:line inventory in the research report.

| Use                                                     | Recommendation                                                | Gains, losses, and required proof                                                                                                                                                                                                                  |
| ------------------------------------------------------- | ------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Local CSV/Parquet reads and file-schema discovery       | **Keep now; candidate to move after N3**                      | Could avoid Arrow/database setup and improve projection/schema discovery. Must preserve explicit types, merge/first-file behavior, nested nulls, timestamp UTC coercion, and exception classes. Ordinary Parquet parity does not prove CSV parity. |
| Local CSV/Parquet writes                                | **Keep now; candidate to move after N3**                      | Native writing is feasible. Preserve error/overwrite/ignore behavior, local path handling, and nested/timezone round trips. No dependency-size saving while the catalog remains.                                                                   |
| Remote S3/Hugging Face IO and `httpfs`                  | **Keep until adapter parity exists**                          | Moving could remove extension installation for moved routes. It loses DuckDB-specific credential/status behavior unless recreated. Test no-token, invalid-token, private/gated, public, revision, glob, mixed-source, and S3 permission cases.     |
| `session.sql` planning/execution, including MCP Analyze | **Keep DuckDB as the current default**                        | Polars SQL covers many representative queries but adds a join-key column in the local `USING` probe and rejects DuckDB table functions. Any alternate SQL frontend needs an explicit dialect/engine contract, not silent routing.                  |
| Persistent catalog tables and schema/object discovery   | **Must stay for the bounded upgrade**                         | Polars SQLContext does not replace `.duckdb` files, schemas, object metadata, transactions, or reopening. Removal requires storage design and migration.                                                                                           |
| Schema/view/tool metadata and persisted metrics         | **Must stay for the bounded upgrade**                         | Needs upserts, bound parameters, atomic metadata/data consistency, and durable query history. Frame aggregation alone is insufficient.                                                                                                             |
| Materialized frame cache and lineage store              | **Keep now; candidate storage replacement only after design** | IPC/Parquet may store frames, but need lifecycle, discovery, concurrency, row/type round trips, and cache invalidation decisions. This does not move the SQLite LLM cache.                                                                         |
| Result normalization and source/sink routing            | **Keep the contract; adapt only with a moved backend**        | UTC-microsecond timestamps and recursive arrays/structs must remain stable. Removing normalization is not justified by a reader swap.                                                                                                              |
| Dormant DuckDB merge scaffolding                        | **Leave unchanged in the upgrade; possible later drop**       | Its executor is not implemented. Removal can reduce misleading scaffolding, but it is not a runtime performance or wheel-size gain.                                                                                                                |
| Standard DataFrame joins                                | **No DuckDB move needed**                                     | Already eager Polars. Upgrading does not itself make them out-of-core.                                                                                                                                                                             |
| Lance similarity index                                  | **Keep lancedb**                                              | It is not a DuckDB extension. Polars' unstable Lance scan does not establish vector-index/search parity.                                                                                                                                           |
| Independent example/service requirements                | **Keep while those consumers use DuckDB**                     | Root dependency removal alone would not remove independent lock requirements or dialect assumptions.                                                                                                                                               |

The complete DuckDB wheel saving on this Mac would be about **13.05 MiB downloaded**, based on the locked arm64 wheel. Partial migration saves zero dependency-wheel bytes. Cold install time and complete-removal performance remain unmeasured. Python 3.14 support is a matrix check, not a reason to assume a storage replacement is necessary.

## Issues

### REUSE: finish declared-output typing

- **Outcome / Goal:** Land the existing #403 sweep, then complete the already tracked deferred files on their current merged bases.
- **Why:** Empty/all-null output must keep its logical type independently of values and provider.
- **Scope:** Reuse existing work. Refresh the constructor inventory after each relevant stack lands. Do not duplicate edits in overlapping semantic/transpiler files.
- **Acceptance criteria:** Known output constructors declare schema/dtype; empty/all-null regressions assert it; typed-Series wrappers and intentional `(0,0)` sinks are distinguished from inference defects.
- **Verification:** Existing #403 regressions; deferred-file tests for empty/all-null legacy and decision routes, normalization, pairwise similarity, and vector similarity. Run the guarded non-live suite.
- **Grounding:** Research section 3; `src/fenic/_backends/local/semantic_operators/base.py:127`; `src/fenic/_backends/local/transpiler/expr_converter.py:1364-1455`.
- **Size / review surface:** Preserve the existing two PR boundaries: broad already-authored sweep, then one bounded deferred-file follow-up. The follow-up's current derived-output core is one generic fallback plus eight embedding branches.
- **Routing:** Maintenance / Fenic; preserve existing state, priority, ownership, and placement. Existing work, no tracker mutation proposed here.
- **Dependencies:** Existing overlapping stacks control availability. N2 requires relevant output-typing contracts to be resolved.

### NEW N1: preserve explode and flatten null/empty behavior

- **Outcome / Goal:** One behavior-preserving PR that passes under pinned 1.43.2 and target 2.0.
- **Why:** Polars 2 silently drops rows that fenic's outer-explode API promises to retain.
- **Scope:** Explicit empty-list behavior in ordinary and indexed outer explode. Preserve the existing flatten contract for empty inner lists. Test lineage/similarity intermediate explosions without changing unrelated semantics.
- **Out of scope:** Lazy-engine adoption, standard join redesign, SQL migration, dependency pins, or Rust-crate changes.
- **Acceptance criteria:** All six new outer-explode failures pass on both runtimes; ordinary explode still drops null/empty rows as before; null elements retain indexed outer positions; nested flatten has an explicit empty-inner-list regression.
- **Verification:** `tests/_backends/local/dataframe/test_explode_with_index.py`, `test_misc.py`, `tests/_backends/local/functions/test_array_functions.py`, lineage tests, and fake/local similarity-join cases. Test 0 rows, null list, empty list, null element, duplicate values, and ordinary populated lists.
- **Grounding:** `src/fenic/_backends/local/physical_plan/transform.py:228-259,300-312`; `src/fenic/_backends/local/transpiler/expr_converter.py:1164-1168`; research failure-attribution section.
- **Size / review surface:** Two production modules plus focused tests. Expect a small explicit-option change, roughly fewer than 30 production lines, with tests sized to the behavior matrix. This is a planning bound, not a completed diff.
- **Routing:** Target team TD; Create proposed; Backlog; Bug; Fenic; High (2); standalone. The recommendation protects an existing public contract. Tracker access: Available/current. Verification outcome: not attempted. Action owner: human.
- **Dependencies:** No dependency on DuckDB replacement. N2 is blocked by N1.
- **Rollback:** Revert this compatibility PR only if its tests show that it changes pinned-version behavior; do not compensate by weakening expected outer-explode rows.

### NEW N2: upgrade the Python runtime with boundary contracts and package validation

- **Outcome / Goal:** One independently reviewable dependency PR that changes the Python Polars pin/lock only after compatibility and typing prerequisites hold.
- **Why:** Users can adopt the supported runtime without silent changes at known dataframe boundaries.
- **Scope:** Update Python Polars and its runtime together. Keep DuckDB and the existing Rust crate set fixed unless the target plugin checks establish a concrete incompatibility. Add focused boundary regressions and release notes.
- **Out of scope:** Polars lazy-engine rewrite, automatic SQL rerouting, new logical Map types, DuckDB pin changes, broad ingestion precision changes, or provider behavior.
- **Acceptance criteria:**
  - N1 removes the six new failure deltas.
  - No Null-dtype derived-output regression remains in the prerequisite typing scopes.
  - Public plugin temporal/struct/JSON/Jinja casts pass with the unchanged binary dependency set.
  - Explicit-schema string-to-date/timestamp behavior uses declared parsing contracts, not a removed native cast.
  - Arrow Map inputs preserve the existing list-of-key/value representation at the ingestion boundary, including an explicit decision about null-map values; unsupported Arrow extensions fail explicitly, without being silently retyped.
  - Mixed signed/UInt64 results either normalize to the declared supported logical output before plugin use or raise a clear planning/ingestion error when lossless normalization is impossible.
  - Series/native-frame Struct-cast differences are covered separately; passing one is not treated as proof of the other.
  - Source/Series serde, timezone, embedding shape/dtype, null membership, and empty zero-width boundaries retain documented contracts.
  - The guarded full non-live matrix passes except explicitly identified credential-validation exclusions; no new unexplained delta remains.
- **Verification:** Existing session/type/cast/column, SQL, IO, serde, Jinja/JSON, embedding, decision, and cache suites. Add synthetic Map/Extension, wide-integer, empty-inner-list, temporal-parse, membership-unit, and cross-version serialization fixtures. Validate Python 3.10-3.14 wheels/imports and the oldest supported platform/runtime paths in CI.
- **Grounding:** Research sections 2, 3, 5, and 7; `pyproject.toml:11-36`; `rust/Cargo.toml:13-24`; `src/fenic/api/session/session.py:400-449`; `rust/src/arrow_scalar_extractor.rs:214-276`.
- **Size / review surface:** One dependency/lock PR plus focused ingestion compatibility tests and release notes. Limit compatibility changes to session ingestion, physical-type normalization, and scalar conversion. Do not treat generated lock-line volume as authored implementation size.
- **Split risk:** If boundary changes require a new logical type or a large Rust rebase, split N2a boundary compatibility from N2b pin/wheel validation. Stop for that design decision rather than broaden the dependency PR.
- **Routing:** Target team TD; Create proposed; Backlog; Maintenance; Fenic; Medium (3); standalone. Tracker access: Available/current. Verification outcome: not attempted. Action owner: human. Exact normalization/rejection details require review before execution.
- **Dependencies:** Blocked by N1 and relevant existing typing follow-up. Related to #398, but do not combine its DuckDB update with this experiment.
- **Rollback:** Restore the prior Python pin and regenerate its lock normally. Keep backward-compatible N1 and typing fixes. Do not rewrite or delete persisted catalogs/caches to make rollback work.

### HELD N3: define DuckDB-replacement compatibility and storage boundaries

- **Outcome / Goal:** A separate design report, not an implementation PR or automatic continuation.
- **Why:** Partial moves may reduce conversion/setup overhead, but complete dependency removal requires durable-storage replacement and a public SQL compatibility decision.
- **Scope:** Establish the supported SQL dialect/subset, existing catalog migration/reopen behavior, storage atomicity/concurrency, IO credential/error adapter contracts, and measurable gains. Compare a native-reader-only move with leaving DuckDB unchanged.
- **Out of scope:** Writing a replacement backend before those contracts are approved; declaring TPC benchmarks proof of fenic performance; assuming Lance scans replace vector search.
- **Acceptance criteria:** Every kept/moved use has semantic/error/type parity fixtures; known `USING` and dtype differences are resolved or explicitly excluded behind a new opt-in contract; catalog migration is reversible; packaging/install gains are measured cold; memory/throughput comparisons include representative workloads and null-heavy/empty cases.
- **Verification:** SQL differential corpus, catalog reopen and rollback fixtures, concurrent metadata/data operations, cache/lineage round trips, remote-read mock status fixtures, and isolated performance/install measurements.
- **Grounding:** Research section 4 inventory and local SQL probes. Persistent store anchors: `src/fenic/_backends/local/catalog.py:56-81,555-678`; metadata: `system_table_client.py:63-773`.
- **Size / review surface:** One design/research unit. Implementation sizing follows the chosen storage and public-contract boundaries, not a speculative list of backend PRs.
- **Routing:** Target team TD; Create proposed, held precursor; Backlog; Maintenance; Fenic; Low (4); standalone. Tracker access: Available/current. Verification outcome: not attempted. Action owner: human. Classify as Maintenance for a committed dependency-removal design unit; uncommitted feasibility work belongs in R&D.
- **Dependencies:** Approval to open the separate design problem. This unit does not block the bounded Polars upgrade.

## Sequencing plan

1. **Wave 0:** Existing #403 work and N1 proceed without DuckDB replacement. Deferred typing follows its actual overlapping-stack gates.
2. **Wave 1:** N2 follows N1 and the relevant typing work. Re-read merged main before implementation; re-run the two-version matrix on that exact base.
3. **Held parallel track:** N3 starts only after separate design approval. Any selected reader/SQL/storage implementation gets a fresh scope and reviewable PR boundary.
4. **Independent update:** #398 closed without merge. If a later DuckDB update is approved and lands first, redo the Polars comparison with DuckDB fixed at that base version. Retain #324's `hf://` error checks.

Do not enable process-wide streaming affinity as a fix. The current operators are eager; a future lazy engine requires its own ordering, materialization, semantic-request, and cache contract.

## Tests by risk

| Risk                                                        | Test that detects it                                                                        | Pass/fail distinction                                                      |
| ----------------------------------------------------------- | ------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| Empty-list outer rows disappear                             | Six named `test_explode_with_index.py` regressions                                          | Existing expected rows fail now under 2.0                                  |
| Nested flatten silently loses a null                        | New `[[1], [], [2]]`, outer-null, and empty-outer-list cases                                | Must assert values and output dtype                                        |
| Empty/all-null output infers Null                           | Existing typing-sweep tests plus deferred embedding/generic-model cases                     | Must assert dtype, not only `None` values                                  |
| Row-order changes affect reduction prompts                  | Fake-model reduction order tests and repeat runs with explicit sort keys                    | Compare numbered prompt contents, batching, and cache fingerprints         |
| Independent responses attach to wrong rows                  | Decision/cache dedup and owner-index tests with shuffled/duplicate input                    | Verify content identity and original output association                    |
| Native casts differ from plugin casts                       | Explicit-schema ingestion versus public `Column.cast` cases                                 | Test Series, frame/expression, and plugin routes separately                |
| Arrow Map/Extension or wide integers escape supported types | Synthetic Arrow fields and signed/UInt64 arithmetic followed by rendering/JSON              | Assert declared conversion or clear rejection before opaque plugin failure |
| Membership compares hidden physical mismatches              | Decimal/float and timestamp-unit/awareness fixtures                                         | Logical equality alone must not mask a changed comparison                  |
| Plugin runtime or private import breaks                     | Import every plugin namespace, then invoke each representative plugin                       | Import success alone is insufficient                                       |
| Serialized plans cross versions                             | Local/cloud source/Series round trips and old/new binary fixtures                           | Same-version serialization alone is insufficient                           |
| SQL replacement changes columns/types                       | Differential `USING`, window rank, literals, date parts, nulls, and DuckDB-function cases   | Compare column names, order, dtype, and values                             |
| Reader replacement changes credentials/errors               | Mock no-token/invalid-token/status cases plus separately authorized remote checks           | Same URI support alone is insufficient                                     |
| Persistent replacement loses data/contracts                 | Reopen, atomic table/schema update, rollback, cache/lineage, and concurrent cursor fixtures | Successful SELECT alone is insufficient                                    |

## Empirical time estimate

The completed Mac comparison took 100.81 seconds for pinned tests and 85.58 seconds for target tests. The explicit Rust build took 76 seconds. Reserve about five minutes for a cached two-version guarded suite/probe repeat, plus a separate cold-build/dependency allowance. Keep a 30-minute limit per full run and fall back to Polars-heavy directories if exceeded.

Remote/cloud and Python 3.14 CI have separate scheduling/runtime costs. No inference spend is required for these fake/local tests.

## Filing state

- Existing typing work: REUSE, not refiled or modified.
- N1: NOT filed.
- N2: NOT filed.
- N3: HELD, NOT filed.
- #398: no comment or modification.

## Open Questions

N2's supported-boundary conversion/rejection policy needs review before implementation. N3's storage and dialect design is deliberately unresolved. Cold-install, remote-reader, Python 3.14, and large-memory results are not available from this experiment.

## Decision Log

- **Applied:** Keep compatibility fixes independent from pin changes so they can be reviewed and tested on both runtimes.
- **Applied:** Reuse existing typing work; do not label every schema-less wrapper an inference defect.
- **Applied:** Preserve current DuckDB SQL/storage defaults during the upgrade.
- **Rejected:** Immediate DuckDB removal based on improved SQL benchmarks. Local parity gaps and persistent storage contracts remain.
- **Deferred:** DuckDB replacement to a separately approved design unit. Native-reader moves need their own measurable value; zero wheel savings is not a size-reduction result.
- **Review:** Scoped in-thread value, coherence, and feasibility checks retained the narrow upgrade and rejected immediate storage replacement. The full interactive CEO and multi-persona/cross-model reviews are not claimed. N2 remains a proposal with a boundary-policy gate, not a cold-start implementation authorization.

## Handoff

This planning track ends with the report and draft units. No issue is approved for implementation or filing by this document. When a unit is selected, start a fresh workflow from current main and its reviewed issue scope.
