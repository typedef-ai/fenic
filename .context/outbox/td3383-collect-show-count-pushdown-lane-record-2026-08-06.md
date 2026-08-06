# TD-3383 — local collect/show/count pushdown — design freeze

**Status:** EVIDENCE-BACKED NO-OP FROZEN — request no-op review FROM HERD
COMMAND. Do not implement or retry the benchmark.
**Branch:** `herd/fenic-exec-engine-td3383-pushdown`, stacked on mixed-workload
directed closure `b2de73a`.
**Budget:** $0.00; local-only. The provider-free selector rule at
`.context/PROVIDER-FREE-TEST-SELECTION.md` applies.

## Framing and freshness

TD-3383 is the fenic-side investigation for shrinking typedef pass gaps. The
high-leverage external shape is intentional today: typedef executes each pass as
`pass_instance.execute(df).cache(); df.count()` for lazy-evaluation error
visibility. TD-4372 has already removed intra-stage schema-distrust collects;
this node considers cheaper *action/checkpoint* behavior without silently
changing failure, resume, progress, cache, or metric semantics.

Freshness check: fetched `origin` on 2026-08-06; the stack parent and its remote
both resolve to `b2de73a6729ec6dc60e2cb3105874c1867d7d283`. No upstream stack
commit was absent from this checkout. The preceding workload evidence is
therefore current. The `td-design` framework informed this decision record, but
the experiment's canonical internal artifact is this outbox lane record rather
than public `docs/`.

## Findings that constrain the design

1. The public actions do not have distinct local execution strategies:
   `DataFrame.show()` calls `execution.show(plan, n)`;
   `DataFrame.collect()` calls `execution.collect(plan)`; and
   `DataFrame.count()` calls `execution.count(plan)` (`src/fenic/api/dataframe/dataframe.py:237-360`).
   Locally, all three call `_execute_query(plan)`, which transpiles and executes
   the full physical plan; `show` only controls formatting after execution,
   `collect(n)` only applies `df.limit(n)` after execution, and `count` only
   returns `df.shape[0]` (`src/fenic/_backends/local/execution.py:38-66,197-208`).
2. `PhysicalPlan.execute()` recursively obtains every child DataFrame, executes
   the node eagerly, collects per-operator model/resource metrics, and writes a
   cache when `cache_info` is present (`src/fenic/_backends/local/physical_plan/base.py:38-111`).
   An action-local terminal limit cannot therefore reduce upstream work today.
3. The normal `LimitExec` itself runs only after its child has already produced a
   DataFrame (`src/fenic/_backends/local/physical_plan/transform.py:469-492`).
   Existing logical optimization covers filter rewrites/merging, not action-aware
   limit or count pushdown (`src/fenic/_backends/local/transpiler/plan_converter.py:78-100`).
4. Caches are a correctness boundary: transpilation swaps a repeated/cached
   logical subtree for `CacheReadExec`, while physical execution writes uncached
   node output after successful execution (`plan_converter.py:102-118`; `base.py:85-90`).
   A count action that skipped the materializing node would defeat the typedef
   checkpoint it is intended to create.
5. The mixed-workload matrix makes the product context measurable. Under its
   12-step cache/count baseline, barrier relaxation was lane-dependent; the
   governor-bound B1 overlay was actively engaged but changed wall by at most
   0.09%. At 192 rows, deterministic token expansion 415,002 against a 150,000
   TPM bucket produced 107.5–113.1s matching arms. This supports treating
   checkpoint semantics and rate/governor pressure as separate variables, not
   promising that broadening fusion or deleting counts is a fenic-only win.

## Chosen approach — action-aware proof before production rewrite

Do **not** add a generic pushdown or a count shortcut. First add only a
test-visible action-planning seam, then admit a production rewrite only when it
passes the boundary classifier and proves a material local win on a large-source,
expensive-projection fixture.

The smallest defensible candidate is `show(n)`/internal bounded collection where
the entire path from the action root to a source is a cache-free, order-preserving
projection of direct columns or direct column aliases. In that narrow shape,
moving a limit below the projection preserves the displayed prefix and avoids a
wide output projection. It is expressly *not* a claim that an arbitrary
expression projection is safe: a cast, regex, UDF-like expression, Series
literal, or semantic expression can alter errors, work, metrics, or output.

The v0 implementation decision is gated, not pre-approved:

- implement the narrow `show` path only if proof tests demonstrate identical
  output/metrics contract and the benchmark clears a predeclared material win;
- otherwise land an evidence-backed no-op with the benchmarks and retain the
  seam only if it is test-only and harmless;
- `count`-without-full-materialization is a separate future architecture change
  requiring an action-mode physical interface, not a low-risk optimization.

## Boundary classifier and contracts

| Plan segment / action | TD-3383 position | Reason and required proof |
| --- | --- | --- |
| Cache marker or `CacheReadExec` | Hard stop | A checkpoint must execute/write before the next typedef pass can rely on it. Assert cache hit/write behavior and query metrics unchanged. |
| Semantic map/extract/predicate, clustering, semantic join/sim-join | Hard stop | Provider/model request count, lifecycle order, LMMetrics, retry/error timing, B0 bounded streaming, and B1 eligibility are observable contracts. Never elide or move a limit across them. |
| Native join, aggregate, sort, distinct, explode/unnest | Hard stop for v0 | Their output prefix/count depends on all input rows or changes cardinality/order. Require a separate algebraic proof before any future rule. |
| SQL and file/table sinks | Hard stop | SQL registers full child frames and cleans temporary views; sinks have write/mode side effects. No action pushdown crosses them. |
| Filter | Hard stop for v0 | `limit(filter(x))` cannot become `filter(limit(x))`; a lazy scan is a separate design. |
| Cache-free projection of `ColumnExpr` or alias-of-`ColumnExpr` only | Candidate for `show(n)` prefix path | Preserve column order/names and the first n row IDs/data exactly. Reject computed expressions, casts, selectors, literals, semantic expressions, and any cache. |
| `collect()` / `count()` at root | No production shortcut in v0 | Existing action contract fully evaluates and records metrics; count only reads height after full execution. A new physical action protocol needs its own design. |

The classifier must be explicit and fail closed. It must not infer safety from an
operator name alone. The test seam observes (a) transpiled plan shape, (b) whether
the candidate path was used, (c) returned rows/schema/order, (d) QueryMetrics and
LMMetrics, and (e) cache/model-client counters. No public API signature changes
are permitted in v0; cloud execution is out of scope.

## Benchmarks and decision rule

All benchmarking is local and provider-free. The predeclared material-win
threshold is **both at least 15% lower median wall time and at least 2.0 ms
absolute median saving** versus the same-stack baseline across 15+ warm rounds;
otherwise no production fast path lands. This threshold was recorded before the
first TD-3383 proof or benchmark execution.

Use a fixed seeded wide Polars/InMemory fixture (at least 100k rows and 128
columns), a narrow `show(10)` projection, and a baseline action shape that is
identical except for candidate eligibility. TD-3384's already-landed ingestion
identity fast path remains enabled in **both** arms; construct/coerce the fixture
before the timed interval, so no reported gain can be attributed to ingestion.
Measure 15+ warm rounds with fixture construction excluded: wall median/p95,
output frame bytes/rows, and operator metrics. Add a deliberately expensive
computed projection only as a **negative control**: it must not select the path.
Add cache, filter, sort, aggregate, join, SQL, and semantic fake-client negative
proofs; they must not select it and must preserve their existing side
effects/metrics.

Promotion requires all of: exact data/schema/order parity, equal applicable
metrics/cache behavior, zero selection for every excluded boundary, and a clear
representative improvement beyond round noise. A null/slow result is a complete,
preferred no-op outcome. No live provider validation is warranted.

## Alternatives considered

- **Generic action limit/count pushdown** — rejected. It would change cache
  materialization, semantic/model dispatch, metrics, error timing, and results
  across order/cardinality-changing operators.
- **Remove typedef `cache()+count()` barriers directly** — rejected. That is a
  typedef product decision about failure/resume/progress semantics. The workload
  arms show its cost separately from fenic fusion; TD-3383 must not relabel it as
  an engine defect.
- **No-op immediately** — not chosen yet. The direct-column `show(n)` case has a
  bounded algebraic basis and deserves local proof/measurement, but it remains
  conditional on the gate above.

## Gate resolution — no-op

The only v0 candidate failed its hard output-parity gate. On the predeclared
100,000-row × 128-column direct-projection fixture, incumbent `show(10)`
formats a full result as first/last rows, while source-limiting formats only rows
0–9. The representative actual-versus-candidate output comparison is preserved
at `.context/validation/td3383-pushdown/show-prefix-parity-failure-2026-08-06.md`.
This is a user-visible semantic divergence, so the prototype was removed rather
than repaired by changing display semantics or adding pagination/sampling scope.

The output assertion ran after 15 alternating pairs but before the prototype
serialized its timing arrays. The arrays are not invented and the run is not
retried: the 15%/2.0ms promotion threshold is irrelevant once output parity
fails. That missing failure receipt is recorded in the evidence artifact as a
harness limitation. The direct-column `show(n)` path is therefore **not** a
safe low-risk pushdown in the present API.

TD-3383's primary deliverable is complete independent of this null: the boundary
classifier above answers where action pushdown is safe today (nowhere in a
behavior-preserving v0) and why. `count` without full materialization requires a
new action-mode physical execution contract that separately specifies operator
count semantics, QueryMetrics/LMMetrics ownership, cache-write/checkpoint
behavior, error timing, and source/SQL/sink capability; it cannot be an
optimization hidden behind the existing `count()` API.

## Open questions

1. Does the narrow source/projection `show(n)` path produce a material gain once
   source ingestion/coercion dominates? The benchmark decides; no assumption is
   made.
2. If a future count action is needed, can an action-mode physical API preserve
   QueryMetrics and cache checkpoints without producing a full DataFrame? This is
   explicitly deferred until production evidence demands it.
3. TD-4372 removed intra-stage collects in typedef; TD-3383 must re-check the
   executor shape at implementation time, because the first new unmaterialized
   production chain is also TD-3334's concrete N-op-fusion revisit trigger.

## No-op review request — FROM HERD COMMAND

Freeze this no-op at the commit below. **Request no-op review FROM HERD COMMAND**
for the standalone boundary classifier/design note, the exact `show(n)` parity
failure, the removed prototype, and the deferred action-mode physical API
contract. **HOLD:** do not implement or rerun TD-3383 until Herd Command disposes
of this gate.
