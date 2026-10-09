---
workflow_id: td-5875-duckdb-replacement-design
phase: design
track: engineering
size_class: full
status: needs_review
portability_level: 1
source_inputs:
  - research.md
  - sql-materialization-measurements.md
last_updated: 2026-10-09
---

# Design: avoid the DuckDB boundary for safe mid-pipeline SQL

Add an opt-in, bounded Polars SQL region inside the existing eager execution engine. Keep DuckDB as the default and explicit compatibility path. This is one SQL execution unit, not a proposal to replace the catalog, readers, storage, or DuckDB dependency.

**Grounding:** [current materialization measurements](sql-materialization-measurements.md), source `fa8761abbf5dac6c17b9eaa56c91ee24236cf95a`, and the [15-case SQL study](research.md#differential-case-list).

## Chosen approach

Use Polars SQL lazily over **today's already materialized child frame**, then keep eligible following filters/projections in the same lazy region. Collect once at the region's output fence. Do not register that frame in DuckDB or export an intermediate SQL-result frame for admitted queries.

The smallest defensible slice is a single-input, scalar SELECT with explicit columns/aliases and simple typed predicates. It admits the final measured query. It excludes arithmetic, aggregates, windows, joins, and unsupported functions until their semantic/error fixtures justify admission.

The pinned selective fixture fell from 323.775 ms / 1578.12 MiB to 36.792 ms / 614.45 MiB with the same full eager child. This supports removing the SQL round trip without rewriting upstream execution. It is a structural measurement, not a promised fenic integration speedup.

The fully lazy 2.911 ms result requires carrying lazy producers through existing eager physical operators. That larger change is not v0. No new reader, general batch protocol, or whole-pipeline lazy executor belongs in this unit.

## Alternatives considered

| Alternative                       | Evidence and tradeoff                                                                                                                     | Decision                                                                                                                                              |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| Leave SQLExec unchanged           | Preserves DuckDB behavior; selective result still exports 3M SQL rows before retaining 1K                                                 | Keep as default/fallback, not the optimized route                                                                                                     |
| Arrow-stream input to DuckDB      | Pinned selective median 163.084 ms / 991.23 MiB; still exports 3M SQL rows; requires producer batches absent from today's executor        | Not v0; does not remove the complete boundary                                                                                                         |
| Arrow input and output            | 155.645 ms / 841.00 MiB; output filtering is batched, but runtime still needs producer/consumer batch lifecycle and may buffer internally | Not v0; added execution protocol for a smaller measured gain                                                                                          |
| Fully lazy Polars plan            | Greatest measured pushdown benefit; needs a new upstream materialization contract                                                         | Explicit later possibility, not authorized by this design                                                                                             |
| Sqlglot-to-fenic logical lowering | Reuses projection/filter nodes and DuckDB parser; AST parsing is cheap after initialization                                               | Real alternative, not implemented; adds SQL binding/semantic lowering and still needs fusion. Prefer the measured existing Polars SQL frontend for v0 |

Arrow input/output is not guaranteed to improve small-input time or to keep all engine buffers bounded. Batch size alone is not that proof.

## Routing contract

### Explicit session policy

Propose a local-only `SessionConfig.sql_execution` field:

- `duckdb` (default): always use today's route.
- `auto`: opt into the fast-path classifier and an observable DuckDB fallback.
- `polars`: require the approved fast subset; unsupported queries fail during planning.

Keep `Session.sql(query, /, **tables)` unchanged. A new `engine=` method keyword would collide with today's valid DataFrame placeholder named `engine` (`src/fenic/api/session/session.py:273-339`). Session policy also covers MCP Analyze, which delegates to the same method (`src/fenic/api/mcp/_tool_generation_utils.py:586-595`). Reject non-default policy for cloud execution in v0.

MCP Analyze continues to advertise DuckDB under `duckdb` and `auto`: auto preserves unsupported query handling through DuckDB. It additionally states that eligible queries may use the observed optimized route. Under strict `polars`, advertise the actual restricted grammar and remove unsupported regex/join examples.

### Deterministic admission before execution

Continue DuckDB-dialect parsing, existing single-statement/DDL-DML validation, and typed-empty DuckDB schema planning. The small schema-planning query remains; the data round trip is removed only when execution is admitted.

Then classify the parsed tree with a positive allowlist:

| Feature                                                                                                    | v0                                                                                     |
| ---------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| One referenced registered DataFrame; SELECT of existing scalar columns with optional unique aliases        | Admit                                                                                  |
| Unambiguous lowercase ASCII identifiers, exact referenced column binding                                   | Admit                                                                                  |
| WHERE with same-type Int64 comparisons, Boolean logic, String equality/inequality, and IS NULL/IS NOT NULL | Admit after per-case parity fixtures                                                   |
| Explicit projected Int64, Boolean, String, Float64 fields                                                  | Admit after typed-empty physical schema equality; no float predicate semantics assumed |
| SELECT star, duplicate output names, quoted/case-ambiguous identifiers                                     | Fallback/refuse initially                                                              |
| JOIN/USING, CTE, UNION, windows, aggregate/SUM/COUNT                                                       | Fallback/refuse initially, even where individual study cases passed                    |
| Arithmetic, casts, temporal/decimal/nested types, functions/UDFs, collations, volatile expressions         | Fallback/refuse initially                                                              |
| `REGEXP_MATCHES`, DuckDB table functions/settings, placeholders for bound values                           | Fallback/refuse                                                                        |
| SQL ORDER BY, LIMIT/OFFSET, or ordering-sensitive operations across the region                             | Fallback/refuse initially                                                              |

“Admit” is a design boundary, not a claim that all its integration fixtures already pass. The measured projection case is demonstrated; nullable predicates, string escaping, and broader schema cases are release gates.

Unknown nodes are not admitted by broad categories such as “expression.” Preserve extra supplied but unused DataFrame arguments without executing them, as today. Never treat a successful parse or equal small values as semantic equivalence.

For `auto`, grammar/binding fallback is selected during planning, with a stable reason such as `sql_feature`, `identifier_semantics`, `dtype_mismatch`, or `cloud_policy`. For `polars`, use `PlanError` with the unsupported feature.

Logical schemas do not always reveal the child's physical integer width. Recheck the materialized child's schema before executing SQL. If this guard fails under `auto`, execute DuckDB using the **same already computed child frame** and report `runtime_input_schema`. Strict mode raises ExecutionError. This is pre-SQL routing, not a retry; the child still executes once. Neither policy retries an executed Polars query through DuckDB.

A guarded region retains its original SQL and following eager operations. Runtime fallback executes that original DuckDB SQL and then the retained eager tail, returning the same final region schema. It does not return the unfiltered SQL result to a parent expecting the fused final result.

### Observable route

Add requested policy, planned route, subset version, and planning fallback reason to the SQL plan's explanation before execution. A candidate fast route is named `polars_guarded`, explicitly awaiting the physical-input schema check. Today's `DataFrame.explain()` prints the logical plan (`src/fenic/api/dataframe/dataframe.py:233-235`), so a physical-stage-only annotation would not satisfy this contract.

The execution details record the final route, any runtime guard reason, and the fused region's member logical nodes. Keep that result per execution rather than mutate a shared logical plan. `auto` is therefore a named compatibility fallback, not a silent engine switch. Default `duckdb` remains available to reproduce old behavior.

## Execution seam and fences

The transpiler currently converts SQL into SQLExec over recursively converted eager children (`src/fenic/_backends/local/transpiler/plan_converter.py:464-471`). The chosen seam is a bounded SQL-region physical operator, not changing every operator's return type.

1. Reuse existing child execution and its metrics. Receive a declared-schema eager Polars frame.
2. Validate its physical schema; route a failed auto guard through DuckDB using that same frame. Otherwise register its lazy wrapper in an operation-local SQLContext.
3. Build the admitted SQL and eligible following projection/filter expressions without collecting.
4. Check the declared output schema and collect once at the output fence.
5. Return an eager frame to the unchanged parent or action.

This is the region's contract, not a phased implementation plan.

Only pure, validated columns/aliases and predicates from the same admission rules may extend the region. Do not fuse through semantic operators, Python/Rust callbacks, sampling, joins, aggregates, sorting, another SQL node, IO, cache writes, sinks, or requests for intermediate materialization.

Cache at the SQL result is a fence: materialize and retain the complete promised SQL output before a downstream filter. Do not replace it with the filtered subset. Existing cache storage is unchanged. SQL lineage remains unsupported as today (`src/fenic/_backends/local/physical_plan/transform.py:544-550`); adding lineage support is not part of this unit.

The v0 child remains eager, including a file reader or semantic producer. Filter pushdown reaches only the already materialized frame. The source-level pruning in the fully lazy measurement is not promised.

No pushed computation may change model request count, prompt bytes, model-response association, cache keys, or call order. Side-effectful/fallible operations remain fences.

## Schema, types, nulls, and ordering

The existing logical schema remains authoritative. Retain three distinct physical-schema witnesses: the declared input schema used to build typed-empty inputs, the typed-empty DuckDB SQL output schema, and the fused region's final output schema. Logical `IntegerType` alone hides physical widths, and Decimal maps to a different logical type than integer SUM (`src/fenic/core/_utils/type_inference.py:154-161`).

Before execution, run Polars `collect_schema()` over typed-empty inputs and compare SQL output names, order, logical types, and physical types with the DuckDB SQL-output witness. Resolve the following admitted operations to the final region-output witness separately. Mismatch selects pre-execution fallback or strict refusal. Do not make decisions from the first value, inference on returned rows, or an empty batch.

The physical child-schema guard checks the actual child against the **input** witness, not against renamed/projected SQL output columns, without inspecting values. An Int32 input hidden behind logical IntegerType is not silently widened into the Int64 fast path. Preserve its existing DuckDB behavior through the observable auto route, or fail strict mode.

At collection, validate the complete output against the planned schema. Use declared schemas/dtypes for constructed values, empty frames, and all-null results. Int64 null output remains Int64, never Null. Retain existing ingestion coercions wherever the accepted boundary requires them; v0 excludes temporal/nested outputs rather than silently changing UTC/array behavior.

Do not cast a mismatch just to make eligibility pass. Widening, overflow, Decimal-to-integer conversion, timezone conversion, and Array/List identity require explicit future admission fixtures.

For the row-preserving single-input v0, preserve observed source order through the SQL/filter/projection region. If the selected Polars runtime cannot prove that fixture, decline admission. Do not infer an order guarantee for arbitrary DuckDB SQL. SQL sorting/window/paging remains on the compatibility path.

## Error phases and lifecycle

| Phase                                  | Contract                                                                                                                  |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| Public arguments                       | Preserve empty-query, missing-placeholder, same-session, and validation errors                                            |
| DuckDB-dialect syntax/read-only checks | Preserve existing PlanError behavior                                                                                      |
| Eligibility/schema planning            | Unknown syntax/type: explicit auto fallback or strict PlanError; evaluate both SQL construction and collect_schema errors |
| Child execution                        | Preserve existing errors and execute the child at most once                                                               |
| Runtime physical-input guard           | Auto chooses DuckDB before SQL using the existing frame; strict mode reports ExecutionError                               |
| Fast execution                         | Wrap through the existing ExecutionError boundary; do not catch and retry through DuckDB                                  |
| Output validation                      | Schema/value contract violation is ExecutionError, not silent conversion or fallback                                      |
| Cancellation/failure                   | Operation-local context is released; no process-wide SQL registry, credentials, or streaming-affinity change              |

Projection pruning can otherwise hide errors in an unused SQL expression. Excluding casts, arithmetic, and functions is intentional: v0 is not allowed to erase an error that today's route would raise.

The Arrow alternative would also need producer stop/reader close and error/cancellation ownership. Since that alternative is not selected, v0 does not add an unproved batch cleanup contract.

## Metrics and serialization

Root QueryMetrics row count, costs, and executed child metrics retain their meaning. A fused region has measured wall time and output rows at its actual fence. Do not fabricate the removed full SQL-frame row count or per-node timing, and do not run extra counts to reconstruct them.

Represent a fused region as one physical stage with its logical members. Explicitly identify this change in execution details; existing persisted root metrics remain unchanged. Today metrics are assembled around physical execution (`src/fenic/_backends/local/physical_plan/base.py:77-111`). Fusion is not required to preserve hypothetical per-logical intermediate counts.

Policy and subset version must round-trip with the SQL logical node. Existing SQL protobuf fields hold inputs, template names, and query text only (`protos/logical_plan/v1/plans.proto:174-178`; `src/fenic/core/_serde/proto/plans/transform.py:299-319`). Add compatible optional policy/subset metadata for future execution; absent fields mean `duckdb`, regardless of the loading session's opt-in.

Persisted old plans therefore keep their old route. Reclassify auto plans on the executing runtime and expose any changed fallback reason. Strict Polars plans require a supported recorded subset version. This is plan-serde compatibility, not catalog migration or storage redesign.

## Verification fixtures and release gate

| Fixture                                                                          | What would fail                                                                    |
| -------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| Admitted projection/alias/predicate corpus, duplicate values and nullable fields | Different names/order/dtypes, dropped/duplicated rows, or changed null filtering   |
| 0 rows and all-null values with declared scalar schemas                          | Null dtype, changed empty result schema, or inference from values                  |
| Same physical input widths; mismatched Float/Decimal/temporal/nested results     | Classifier admits logical-only equality or silently casts a mismatch               |
| Existing 15 SQL cases, plus regex/settings/bound-value/USING examples            | Default DuckDB behavior changes, or an excluded case enters the fast path          |
| Auto and strict policy; parser errors and delayed collect_schema errors          | Auto hides its reason, strict executes unsupported SQL, or failure phase changes   |
| Error in an executed fast path, tracked child execution counter                  | A retry calls the child twice or repeats a semantic producer                       |
| Runtime Int32 input with logical IntegerType under auto/strict                   | Guard widens it silently, misreports the final route, or recomputes the child      |
| Cached SQL output followed by selective work                                     | Cache contains only the downstream subset or changes its key/schema                |
| Fallible/plugin/semantic/sort/sink fence                                         | Fusion removes errors, reorders side effects, or crosses the fence                 |
| Concurrent SQL calls with the same aliases                                       | Operation-local frame registrations leak between calls                             |
| Explain, physical metrics, and old/new plan serde                                | Fallback is invisible; phantom intermediate metrics; old plans inherit a new route |
| Whole-pipeline integration on pinned and target runtime                          | Structural-engine win disappears or correctness differs in the real executor       |

Require actual fenic differential integration before shipping. Compare the complete ordered result and logical/physical schema with explicit DuckDB mode. Repeat the large selective/full-output and null/empty measurements through the real physical operator, with spread and peak RSS. No provider request is needed; a counted fake semantic child proves exactly-once behavior.

The empirical acceptance is removal of the DuckDB data registration/export on admitted queries, one region collect at the fence, and a useful measured integrated gain. No universal speedup threshold is inferred from three local repetitions.

## Scope and open questions

**Not in scope:** catalog/durable storage, readers or IO adapters, DuckDB dependency removal, a general lazy/batch executor, new SQL features, lineage support, provider calls, or implementation in this research branch.

**Already present:** eager child execution, expression conversion, DuckDB-dialect validation, typed-empty schema planning, SQLContext in the installed Polars dependency, and public explanation/metrics surfaces.

**Later only:** source-level lazy pushdown, batch DuckDB fallback, or additional grammar admission. Each needs its own evidence; this design does not commission those changes.

Open verification questions are whether the real executor preserves source order and exactly-once boundaries, and whether integrated gains hold under its metrics/cache overhead. These are explicit release tests, not unresolved architecture choices.

## Decision log and review

- **Chosen:** Opt-in lazy Polars SQL region over current eager children. It directly removes the measured boundary without introducing a new upstream execution protocol.
- **Reduced:** Admit row-preserving scalar projection/predicate SQL only. Preserve the larger DuckDB dialect through explicit fallback.
- **Rejected for v0:** Claiming the fully lazy approximately 3 ms result for a SQLExec-only change.
- **Deferred:** Arithmetic, joins, aggregates, windows, functions, temporal/nested types, and generic streaming.
- **Premise/scope check:** The target is the mid-pipeline barrier, not storage replacement or package size. The eager-child comparison is the relevant bounded evidence.
- **Coherence/mechanism check:** Registration is not claimed to write a persistent table; actual current session reuse is reflected in corrected timing. Schema planning remains metadata-only.
- **Review coverage:** In-thread premise, scope, coherence, feasibility, and error/fence checks. Independent persona/cross-model review has not run. This document is a proposal requiring review, not implementation approval.

## Handoff

Stop at this design for review. No Structure, implementation, additional design units, tracker filing, push, or PR is authorized by this document.
