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

Add an opt-in Polars route to SQLExec inside the existing eager execution engine. Keep DuckDB as the default and explicit compatibility path. This is one SQL execution unit, not a proposal to replace the catalog, readers, storage, or DuckDB dependency.

**Grounding:** [current materialization measurements](sql-materialization-measurements.md), source `fa8761abbf5dac6c17b9eaa56c91ee24236cf95a`, and the [15-case SQL study](research.md#differential-case-list).

## Chosen approach

Use Polars SQL over **today's already materialized child frame**. Collect the SQL result once and return an eager frame from SQLExec. Following filters/projections remain separate existing eager operators. Do not register the child in DuckDB or export the SQL result through DuckDB for admitted queries. Do not add downstream fusion.

The smallest defensible slice is a single-input, scalar SELECT with explicit columns/aliases and simple typed predicates. It admits the final measured query. It excludes arithmetic, aggregates, windows, joins, and unsupported functions until their semantic/error fixtures justify admission.

The paired v4 full-output fixture fell from 227.996 ms / 1571.77 MiB to 34.928 ms / 615.62 MiB on pinned Polars. The fused control was 34.230 ms / 612.70 MiB. On Polars 2, unfused was 37.156 ms against fused 35.722 ms. The unfused route keeps over 99% of the full-output wall-time gain on both runtimes. It also keeps most of the selective gain, despite noisy pinned timings. These structural measurements support the smaller SQL-only seam, not a promised fenic integration speedup.

The earlier selective DuckDB baseline varied widely and exceeded its full-output control. V4 repeats that variability; it does not establish the cause. Use the full-output comparison as the conservative decision evidence, not the earlier approximately 287 ms selective saving.

The fully lazy 2.911 ms result requires carrying lazy producers through existing eager physical operators. That larger change is not v0. No new reader, general batch protocol, or whole-pipeline lazy executor belongs in this unit.

## Alternatives considered

| Alternative                       | Evidence and tradeoff                                                                                                                     | Decision                                                                                                                                               |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Leave SQLExec unchanged           | Preserves DuckDB behavior; selective result still exports 3M SQL rows before retaining 1K                                                 | Keep as default/fallback, not the optimized route                                                                                                      |
| Arrow-stream input to DuckDB      | Pinned selective median 163.084 ms / 991.23 MiB; still exports 3M SQL rows; requires producer batches absent from today's executor        | Not v0; does not remove the complete boundary                                                                                                          |
| Arrow input and output            | 155.645 ms / 841.00 MiB; output filtering is batched, but runtime still needs producer/consumer batch lifecycle and may buffer internally | Not v0; added execution protocol for a smaller measured gain                                                                                           |
| Fully lazy Polars plan            | Greatest measured pushdown benefit; needs a new upstream materialization contract                                                         | Explicit later possibility, not authorized by this design                                                                                              |
| Sqlglot-to-fenic logical lowering | Reuses existing eager projection/filter nodes, their metrics and serde; no downstream fusion is needed                                    | Close alternative, not implemented or benchmarked. Prefer the demonstrated Polars SQL route for v0; do not claim it is faster than unmeasured lowering |

Lowering could avoid a second SQL parser. It still needs tested SQL binding, aliases, literal/null semantics, and output-schema parity. The prior claim that it needs physical fusion was wrong for this v0 subset. The v4 unfused result removes that objection.

Arrow input/output is not guaranteed to improve small-input time or to keep all engine buffers bounded. Batch size alone is not that proof.

## Routing contract

### Explicit session policy

Propose a local-only `SessionConfig.sql_execution` field:

- `duckdb` (default): new SQL nodes use today's route.
- `auto`: opt into the fast-path classifier and an observable DuckDB fallback.
- `polars`: require the approved fast subset; unsupported queries fail during planning.

This is a **node-construction policy**, not an execution-time override. A SQL node captures the session policy when created. A loaded node's recorded policy wins over the executing session's setting. Missing policy metadata means `duckdb`; opting the loading session into `auto` does not upgrade an old node.

Keep `Session.sql(query, /, **tables)` unchanged. A new `engine=` method keyword would collide with today's valid DataFrame placeholder named `engine` (`src/fenic/api/session/session.py:273-339`). Session policy also covers newly created MCP Analyze queries, which delegate to the same method (`src/fenic/api/mcp/_tool_generation_utils.py:586-595`). Reject non-default policy in cloud configuration. Also reject a loaded non-default SQL node before cloud execution; there is no cloud auto-fallback in v0.

To compare a captured fast node with DuckDB, create a fresh SQL node from the original query and tables in a DuckDB-configured session. Merely loading it into that session does not change its recorded policy. This does not override policies of SQL nodes nested in its inputs.

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
| SELECT star, quoted/case-ambiguous identifiers                                                             | Fallback/refuse initially                                                              |
| JOIN/USING, CTE, UNION, windows, aggregate/SUM/COUNT                                                       | Fallback/refuse initially, even where individual study cases passed                    |
| Arithmetic, casts, temporal/decimal/nested types, functions/UDFs, collations, volatile expressions         | Fallback/refuse initially                                                              |
| `REGEXP_MATCHES`, DuckDB table functions/settings, placeholders for bound values                           | Fallback/refuse                                                                        |
| SQL ORDER BY, LIMIT/OFFSET, or ordering-sensitive SQL operations                                           | Fallback/refuse initially                                                              |

“Admit” is a design boundary, not a claim that all its integration fixtures already pass. The measured projection case is demonstrated; nullable predicates, string escaping, and broader schema cases are release gates.

Unknown nodes are not admitted by broad categories such as “expression.” Preserve extra supplied but unused DataFrame arguments without executing them, as today. Never treat a successful parse or equal small values as semantic equivalence.

Duplicate output names retain today's `PlanError` on every route (`src/fenic/core/_logical_plan/plans/base.py:35-49`). They are not a classifier fallback choice.

For `auto`, grammar/binding fallback is selected during planning, with a stable reason such as `sql_feature`, `identifier_semantics`, or `dtype_mismatch`. For `polars`, use `PlanError` with the unsupported feature.

Logical schemas do not always reveal the child's physical integer width. Recheck the materialized child's schema before executing SQL. If this guard fails under `auto`, execute DuckDB using the **same already computed child frame** and report `runtime_input_schema`. Strict mode raises ExecutionError. This is pre-SQL routing, not a retry; the child still executes once. Neither policy retries an executed Polars query through DuckDB.

Runtime fallback executes the original DuckDB SQL over that same child and returns the declared SQL-output frame. Its parent executes the following eager operations normally. No retained tail or tail replay is needed.

### Observable route

Add recorded policy, recorded subset version, classification state, and any available planning route/reason to the SQL plan's explanation. A candidate fast route is named `polars_guarded`, explicitly awaiting the physical-input schema check. A freshly deserialized node instead displays `unclassified (deserialized)`, not a guessed fast route. Today's `DataFrame.explain()` prints the logical plan (`src/fenic/api/dataframe/dataframe.py:233-235`); see the preparation contract below.

Execution preparation classifies the execution-local rebuilt node before child execution or physical cache selection. Execution details record its planned route, executing subset version, final route, and any guard reason. A cache hit remains a cache read, not a reported SQL-engine execution. Keep these details per execution rather than mutate the original shared logical plan. `auto` is therefore a named compatibility fallback, not a silent engine switch.

## SQL-only execution seam

The transpiler currently converts SQL into SQLExec over recursively converted eager children (`src/fenic/_backends/local/transpiler/plan_converter.py:464-471`). Keep that operator boundary and eager return type.

1. Reuse existing child execution and its metrics. Receive a declared-schema eager Polars frame.
2. Validate its physical schema; route a failed auto guard through DuckDB using that same frame. Otherwise register its lazy wrapper in an operation-local SQLContext.
3. Build the admitted SQL only, without collecting.
4. Check the declared SQL-output schema and collect that result once.
5. Return an eager frame to the unchanged parent or action.

This is the SQL operator's contract, not a phased implementation plan.

Following operators keep their existing order, expression conversion, errors, metrics, and cache boundaries. There is no new allowlist for fenic tail expressions. Semantic operators, callbacks, IO and sinks do not enter this SQL operator.

SQLExec returns the complete promised SQL output before any downstream filter. Its existing cache write path therefore caches that output, not a downstream subset. Existing cache storage is unchanged. SQL lineage remains unsupported as today (`src/fenic/_backends/local/physical_plan/transform.py:544-550`); adding lineage support is not part of this unit.

The v0 child remains eager, including a file reader or semantic producer. SQL's own predicates can act on that frame. Downstream predicates do not move into SQL or into its source. The source-level pruning in the fully lazy measurement is not promised.

No pushed computation may change model request count, prompt bytes, model-response association, cache keys, or call order. Side-effectful/fallible operations remain fences.

## Schema, types, nulls, and ordering

The existing logical schema remains authoritative. Retain two distinct physical-schema witnesses: the declared input schema used to build typed-empty inputs and the typed-empty DuckDB SQL-output schema. There is no fused-final-output witness. Logical `IntegerType` alone hides physical widths, and Decimal maps to a different logical type than integer SUM (`src/fenic/core/_utils/type_inference.py:154-161`).

Before execution, run Polars `collect_schema()` over typed-empty inputs and compare SQL output names, order, logical types, and physical types with the DuckDB SQL-output witness. Mismatch selects pre-execution fallback or strict refusal. For a rebuilt or loaded node, also compare the reconstructed SQL schema with its declared logical schema. A changed declared schema is a planning error, not an auto engine fallback. Do not infer a schema from values or an empty batch.

The physical child-schema guard checks the actual child against the **input** witness, not against renamed/projected SQL output columns, without inspecting values. An Int32 input hidden behind logical IntegerType is not silently widened into the Int64 fast path. Preserve its existing DuckDB behavior through the observable auto route, or fail strict mode.

At collection, validate the complete output against the planned schema. Use declared schemas/dtypes for constructed values, empty frames, and all-null results. Int64 null output remains Int64, never Null. Retain existing ingestion coercions wherever the accepted boundary requires them; v0 excludes temporal/nested outputs rather than silently changing UTC/array behavior.

Physical-output equality with the fast-path witness applies to fast execution. DuckDB fallback retains its existing ingestion and declared-logical-schema contract; do not force its Int32 result into the fast route's Int64 witness.

Do not cast a mismatch just to make eligibility pass. Widening, overflow, Decimal-to-integer conversion, timezone conversion, and Array/List identity require explicit future admission fixtures.

For the single-input v0, preserve observed source order through SQL. If the selected Polars runtime cannot prove that fixture, decline admission. Do not infer an order guarantee for arbitrary DuckDB SQL. SQL sorting/window/paging remains on the compatibility path.

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

Keep the existing SQLExec `OperatorMetrics`, including its measured wall time and actual SQL-output row count. The child and following operators retain their own metrics. Root QueryMetrics remains unchanged. Do not introduce a fused-stage metric or extra counting action. This uses today's per-operator execution boundary (`src/fenic/_backends/local/physical_plan/base.py:77-111`).

### Recorded-policy precedence and node lifecycle

Policy and subset version must round-trip with the SQL logical node. Existing SQL protobuf fields hold inputs, template names, and query text only (`protos/logical_plan/v1/plans.proto:174-178`; `src/fenic/core/_serde/proto/plans/transform.py:299-319`). Add compatible optional metadata:

| Lifecycle                    | Required behavior                                                                                                         |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| New SQL node                 | Capture `sql_execution` once from its creating session; record the current subset version for non-default policy          |
| Loaded node with no metadata | Set node policy to `duckdb`, subset absent; ignore the executing session's opt-in                                         |
| Loaded node with metadata    | Recorded node policy wins; never replace it with the executing session policy                                             |
| Auto preparation             | Reclassify against the executing runtime's supported subset; report both recorded and executing versions and any fallback |
| Strict preparation           | Require a supported recorded subset version; otherwise raise `PlanError` before executing children                        |

Today `SQL.with_children` calls `from_session_state` and copies only query, template names and cache info (`src/fenic/core/_logical_plan/plans/transform.py:709-716`). All three active optimizer rules call `with_children` (`src/fenic/_backends/local/transpiler/plan_converter.py:88-95`; optimizer `not_filter_pushdown_rule.py:44`, `merge_filters_rule.py:31`, `semantic_filter_rewrite_rule.py:54`). The new rebuild contract must explicitly pass the original node's policy and recorded subset version. It may refresh derived classification/schema witnesses, but must not recapture session policy. Preserve cache info and compare the refreshed declared schema.

Extend `_eq_specific`, which today compares query and template names only (`src/fenic/core/_logical_plan/plans/transform.py:718-722`), to compare recorded policy and subset version too. Derived routes, runtime guards, and per-execution metrics are not node identity.

Serde must restore those fields through `from_schema`, which today skips `_build_schema` (`src/fenic/core/_logical_plan/plans/transform.py:636-646`; `src/fenic/core/_logical_plan/plans/base.py:29-32`; serde `transform.py:311-319`). A deserialized node retains its declared schema and reports an unclassified route in `explain()`. Execution preparation rebuilds/classifies it before child execution, regenerates typed-empty witnesses, and checks the stored schema. Invalid policy metadata fails preparation with `PlanError`. Unknown auto subset versions permit current-runtime reclassification; strict unsupported or absent versions fail.

These rules apply independently to every SQL node in a view/tool graph. A default executing session is not a global override of persisted strict nodes. This is plan-serde compatibility, not catalog migration or storage redesign.

## Verification fixtures and release gate

| Fixture                                                                            | What would fail                                                                     |
| ---------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| Admitted projection/alias/predicate corpus, duplicate values and nullable fields   | Different names/order/dtypes, dropped/duplicated rows, or changed null filtering    |
| 0 rows and all-null values with declared scalar schemas                            | Null dtype, changed empty result schema, or inference from values                   |
| Same physical input widths; mismatched Float/Decimal/temporal/nested results       | Classifier admits logical-only equality or silently casts a mismatch                |
| Existing 15 SQL cases, plus regex/settings/bound-value/USING examples              | Default DuckDB behavior changes, or an excluded case enters the fast path           |
| Auto and strict policy; parser errors and delayed collect_schema errors            | Auto hides its reason, strict executes unsupported SQL, or failure phase changes    |
| Error in an executed fast path, tracked child execution counter                    | A retry calls the child twice or repeats a semantic producer                        |
| Runtime Int32 input with logical IntegerType under auto/strict                     | Guard widens it silently, misreports the final route, or recomputes the child       |
| Cached SQL output followed by selective work                                       | Cache contains only the downstream subset or changes its key/schema                 |
| SQL followed by fallible/plugin/semantic/sort/sink operations                      | SQL-only route changes eager operator order, errors, metrics, or side effects       |
| Concurrent SQL calls with the same aliases                                         | Operation-local frame registrations leak between calls                              |
| Explain and SQLExec metrics before/after execution                                 | Fallback is invisible or SQL/child/parent metrics lose their existing boundary      |
| Old/default/auto/strict serde under differently configured sessions                | Session policy overrides recorded policy; old plans acquire a fast route            |
| Three optimizer rebuilds, equality, cached views/tools and unsupported versions    | Rebuild drops policy; distinct policies compare equal; strict version is ignored    |
| Deserialized explain before preparation, schema recheck and exactly-once execution | A guessed route is displayed; stored schema drifts; a child runs during preparation |
| Whole-pipeline integration on pinned and target runtime                            | Structural-engine win disappears or correctness differs in the real executor        |

Require actual fenic differential integration before shipping. Compare the complete ordered result and logical/physical schema with explicit DuckDB mode. Repeat the large selective/full-output and null/empty measurements through the real physical operator, with spread and peak RSS. No provider request is needed; a counted fake semantic child proves exactly-once behavior.

The empirical acceptance is removal of DuckDB data registration/export on admitted queries, one SQL-result collect, unchanged eager operator boundaries, and a useful measured integrated gain. No universal speedup threshold is inferred from three local repetitions.

## Scope and open questions

**Not in scope:** catalog/durable storage, readers or IO adapters, DuckDB dependency removal, a general lazy/batch executor, new SQL features, lineage support, provider calls, or implementation in this research branch.

**Already present:** eager child execution, expression conversion, DuckDB-dialect validation, typed-empty schema planning, SQLContext in the installed Polars dependency, and public explanation/metrics surfaces.

**Later only:** source-level lazy pushdown, batch DuckDB fallback, or additional grammar admission. Each needs its own evidence; this design does not commission those changes.

Open verification questions are whether the real executor preserves source order and exactly-once boundaries, and whether integrated gains hold under its metrics/cache overhead. These are explicit release tests, not unresolved architecture choices.

**Open parser-agreement requirement (review Minor 3):** execute a canonical rendering of the bound admitted AST, not unchecked original text. Its identifier, literal escaping, Boolean precedence and operator spellings must agree between DuckDB/sqlglot admission and Polars execution. Require differential fixtures before admitting each form; until then it falls back or strict-refuses. No renderer or parser-agreement corpus was implemented in this docs-only round.

**Open usefulness requirement (review Minor 4):** assess representative mid-pipeline queries before broadening adoption. The narrow subset excludes every current MCP Analyze example (`src/fenic/api/mcp/_tool_generation_utils.py:611-613`), so those remain DuckDB in auto. Single-input star is an intentional initial exclusion pending column-order/schema fixtures, not evidence of a single-input incompatibility. Projection/predicate workloads can use the DataFrame API; their real SQL frequency is unmeasured. No representative query capture or broader grammar is commissioned here.

## Decision log and review

- **Chosen after the authorized fix round:** Opt-in SQL-only Polars execution over current eager children. The unfused arm retains most of the measured gain; remove downstream fusion, retained tail, fused metrics, and tail-admission rules.
- **Reduced:** Admit row-preserving scalar projection/predicate SQL only. Preserve the larger DuckDB dialect through explicit fallback.
- **Rejected for v0:** Claiming the fully lazy approximately 3 ms result for a SQLExec-only change.
- **Deferred:** Arithmetic, joins, aggregates, windows, functions, temporal/nested types, and generic streaming.
- **Premise/scope check:** The target is the mid-pipeline barrier, not storage replacement or package size. The eager-child comparison is the relevant bounded evidence.
- **Coherence/mechanism check:** Registration is not claimed to write a persistent table; actual current session reuse is reflected in corrected timing. Schema planning remains metadata-only.
- **Review coverage:** Opus 5.5 returned Not ready at `64e1bc3b` for missing unfused evidence and incomplete policy precedence. This revision addresses that authorized delta. Its new review is pending; this proposal is not implementation approval.

## Handoff

Stop at this design for review. No Structure, implementation, additional design units, tracker filing, push, or PR is authorized by this document.
