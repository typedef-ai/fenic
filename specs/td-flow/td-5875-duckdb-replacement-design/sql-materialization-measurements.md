---
workflow_id: td-5875-duckdb-replacement-design
phase: research
research_stage: findings_ready
track: engineering
size_class: full
status: needs_review
portability_level: 1
source_inputs:
  - research.md
last_updated: 2026-10-09
---

# Measurements: SQL in the middle of a pipeline

**Source base:** `fa8761abbf5dac6c17b9eaa56c91ee24236cf95a`. **Date:** 2026-10-09 UTC.

## Question and current mechanism

How much does the SQL boundary cost when a later filter or projection could reduce its result?

Fenic recursively executes children to eager Polars frames before executing each physical node (`src/fenic/_backends/local/physical_plan/base.py:65-80`). SQL registers those frames on an intermediate-database cursor, calls `execute(...).pl()`, applies ingestion coercions, and drops the views (`src/fenic/_backends/local/physical_plan/transform.py:518-531`).

Registration is a view over an external frame, not a demonstrated write of every input row into a persistent DuckDB table. The proven barriers are the eager child frame and full SQL-result export before the next physical operator. This report does not equate registration with a full input copy or measure catalog storage.

Planning already derives SQL schemas from typed empty frames (`src/fenic/core/_logical_plan/plans/transform.py:671-681`). Execution reuses a session connection. The final timed region therefore excludes connection creation, typed-empty planning, imports, and result verification.

## Comparison arms

All arms use the same fixture and upstream `x * 2` transformation. “Materialized rows” below means rows in user-visible full Polars frames, not every internal engine allocation.

| Arm                       | Producer and SQL                                                              | Work after SQL                                             |
| ------------------------- | ----------------------------------------------------------------------------- | ---------------------------------------------------------- |
| Current-style DuckDB      | Collect the full child; register it; export the full SQL frame with `.pl()`   | Eager filter/projection                                    |
| Arrow-input DuckDB        | Send child batches through Arrow RecordBatchReader; export the full SQL frame | Eager filter/projection                                    |
| Fully lazy Polars SQL     | Keep producer, SQL, and following operations in one lazy plan                 | Collect only at the final boundary                         |
| Eager-child Polars SQL    | Collect the same full child as today; register its lazy wrapper in SQLContext | Keep SQL and following operations lazy until final collect |
| Arrow input/output DuckDB | Read child batches and fetch result batches; apply following work per batch   | Concatenate surviving typed batches                        |

The eager-child Polars arm is the bounded replacement opportunity. The fully lazy arm measures a larger execution opportunity, not an existing fenic capability or a reader-migration recommendation.

DuckDB documents RecordBatchReader input and batch output [D1, D2]. Polars documents lazy SQL execution [P1]. Its batch producer is unstable and warns about cost relative to native sinks [P2]. A 65,536-row batch limit does not prove bounded internal memory.

## Fixtures and measurement method

The large fixture has 3,000,000 rows and three declared columns: Int64 `id`, Int64 `x`, and String `payload`. IDs and `x` start at zero and ascend together. Each payload is the ID as a 128-character decimal string. A 100,000-row fixture has null `x` and null payload with the same declared schema. An empty fixture retains that schema.

The reusable Parquet fixtures are experimental input only. Their reader is identical across the arms that can retain lazy input. No production reader or reader contract changes.

The final v3 SQL is `SELECT id, x AS y, payload FROM child`. Following work selects `id,y`, either for all rows or after `id < 1000`. One case places the same predicate inside SQL as well. The admitted query adds no arithmetic or fallible casts inside SQL.

An earlier v2 uses `x + 1 AS y` to inspect a derived expression. Its results remain, but arithmetic is outside the proposed v0 grammar. The exploratory v1 also includes connection setup; it is not used for the final recommendation.

Both tested runtimes use DuckDB 1.4.5, PyArrow 23.0.1, CPython 3.11.11, and four threads on macOS arm64. Polars versions are pinned 1.43.2 and target 2.0.0. Each arm runs in a fresh process three times. Arm order reverses in the middle repetition. OS caches are warm; neither cold IO nor the whole fenic runtime is measured.

Peak RSS is the process high-water mark after execution and before verification. It includes imports and metadata planning, which every arm performs, but not the later equality oracle's Python lists. Wall time covers child production, SQL, and following work; setup and connection teardown are excluded.

Every result must have exactly the declared Int64 `id,y` schema. The oracle checks row count, every ascending unique ID, and the exact expected `y`, or all-null `y`. Each runtime's 75 final runs compare equal signatures across all five arms. These are structural engine fixtures, not 150 fenic integration tests.

## Final v3 results

### Selective work after SQL

Final output is 1,000 rows. Times are milliseconds: median and minimum–maximum across three processes. Memory is median process peak MiB.

| Arm                       | Pinned time               | Pinned peak MiB | Polars 2 time             | Polars 2 peak MiB |
| ------------------------- | ------------------------- | --------------- | ------------------------- | ----------------- |
| Current-style DuckDB      | 323.775 (245.340–396.904) | 1578.12         | 236.803 (189.602–272.446) | 1609.05           |
| Arrow-input DuckDB        | 163.084 (160.210–164.308) | 991.23          | 145.752 (140.322–148.332) | 1020.84           |
| Fully lazy Polars SQL     | 2.911 (2.730–10.462)      | 117.50          | 3.136 (3.042–9.121)       | 120.19            |
| Eager-child Polars SQL    | 36.792 (36.288–50.095)    | 614.45          | 35.979 (35.844–40.119)    | 623.41            |
| Arrow input/output DuckDB | 155.645 (151.182–325.947) | 841.00          | 162.938 (156.344–173.220) | 884.47            |

The pinned eager-child arm removes about 287 ms and 964 MiB from this structural comparison. It still materializes the entire child. The approximately 3 ms fully lazy result cannot be promised by merely replacing SQLExec.

### Controls on pinned Polars

Each cell is median milliseconds / median peak MiB.

| Case                                  | Current DuckDB    | Arrow-input DuckDB | Fully lazy Polars | Eager-child Polars | Arrow input/output DuckDB |
| ------------------------------------- | ----------------- | ------------------ | ----------------- | ------------------ | ------------------------- |
| Full final output, 3M rows            | 201.891 / 1572.41 | 154.178 / 980.62   | 14.470 / 184.66   | 33.714 / 613.22    | 148.460 / 893.88          |
| Predicate inside SQL, 1K output       | 82.883 / 987.12   | 67.459 / 586.50    | 2.661 / 117.64    | 34.757 / 614.89    | 67.619 / 601.95           |
| 100K rows with null values, 1K output | 4.809 / 124.36    | 5.055 / 125.86     | 2.005 / 115.78    | 2.203 / 117.86     | 4.824 / 125.16            |
| Empty input/output                    | 1.713 / 112.08    | 1.729 / 113.00     | 0.892 / 111.28    | 1.045 / 111.78     | 1.590 / 111.89            |

All controls also passed on Polars 2. For its full-output case, current DuckDB measured 186.451 ms / 1603.41 MiB; eager-child Polars measured 34.772 ms / 623.45 MiB. Its all-null and empty cases retained the declared Int64 schema. Complete per-run values and spread remain in the retained results.

Arrow input is not automatically faster on small inputs. Its all-null pinned median slightly exceeds the current path. Arrow output further lowers some peaks, but does not consistently improve time. These alternatives require a batch execution seam that fenic does not currently expose.

### Visible materialization for the selective case

| Arm                       | Full child frame rows | Full SQL frame rows | Arrow rows consumed/emitted | Final frame rows |
| ------------------------- | --------------------: | ------------------: | --------------------------- | ---------------: |
| Current-style DuckDB      |             3,000,000 |           3,000,000 | None                        |            1,000 |
| Arrow-input DuckDB        |                     0 |           3,000,000 | 3M input, 46 batches        |            1,000 |
| Fully lazy Polars         |                     0 |                   0 | Not instrumented            |            1,000 |
| Eager-child Polars        |             3,000,000 |                   0 | None                        |            1,000 |
| Arrow input/output DuckDB |                     0 |                   0 | 3M input and 3M output      |            1,000 |

Input batches never exceeded 65,536 rows. These counts omit transient internal Polars/DuckDB buffers, row-group decoding, and optimizer allocations. Peak RSS remains the relevant memory observation.

The fully lazy optimized plans show `PROJECT 2/3 COLUMNS` and `SELECTION: id < 1000` at the scan on both runtimes. Thus the unused payload and downstream predicate cross the SQL boundary in that plan. The eager-child arm pushes them into the in-memory lazy region, not into the already completed source read.

## SQL-to-fenic alternative probe

Sqlglot 30.14.0 parsed five sample queries. Simple projection used Select/From/Table/Column/Alias/Identifier nodes. Adding the predicate adds Where/LT/Literal. Join, SUM, and regex introduce separate Join/Star, Sum, and RegexpLike nodes.

The first parse, including dialect initialization, took 26.77 ms. Subsequent parses took roughly 0.08–0.11 ms. These are AST observations only. No translator into fenic logical plans was written or benchmarked.

Current fenic already has projection/filter logical nodes and serializers (`src/fenic/core/_serde/proto/plans/transform.py:43-91`). A translator could reuse them. It would still need SQL binding, null/type semantics, and physical fusion; producing today's separate eager operators would not preserve the fully lazy gains.

## Reproduction and retained evidence

The scripts assert all eight provider variables are absent and refuse socket connections. The parent starts each child sequentially with a 300-second timeout. The following final run preserves the same inherited unset-key environment:

```sh
R="$(git rev-parse --show-toplevel)"
env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY -u GOOGLE_API_KEY \
  -u GEMINI_API_KEY -u TYPESAFE_API_KEY -u COHERE_API_KEY \
  -u OPENROUTER_API_KEY -u HF_TOKEN \
  POLARS_MAX_THREADS=4 TD5875_SQL_PASSTHROUGH=1 \
  "$R/.context/probe-venv/bin/python" \
  "$R/.context/sql-materialization/measure.py"
```

The final script SHA-256 is `caa7ff6fbd1ba19ca260d9c6a96e108515b6d2874aa5fd2ad3f5ec24b22b34e6`. Its corrected arithmetic snapshot, `measure-v2.py`, is `72e5f0f832248846b655a3a952aa9f5439dcd5fc5fefee16c31acf2511778b0a`. The original `measure-v1.py` is retained too. To rerun, use a new output/run label rather than overwrite these records.

Fixtures were generated in typed Arrow batches of 100,000 rows and written with Zstandard compression. The large file is 22,780,861 bytes, SHA-256 `ed55eb29ffae9ca55fb986e3a581da892d10b41bb96de7ef6d7ce1338a87dd30`. Empty and all-null files are 632 and 298,614 bytes. The summary contains their hashes.

Final results are `.context/sql-materialization/summary-v3.json` and 150 `*-v3-*.json` records under `runs/`. The records retain exact commands, versions, schemas, signatures, gate readings, visible row counters, and optimized plans. Earlier `summary.json` and `summary-v2.json` remain distinct. `sqlglot-alternative.json` retains the AST observations.

Every heavy run checked five-minute load below 16 and free disk at least 70,000,000,000 bytes before starting. Fixture generation checked those gates between batches. Final v3 maximum five-minute load was 4.175; minimum free disk was 95.959 GB. The initial gate was 71.606 GB, so no GiB/GB conversion was hidden.

The SQL probe directory occupied about 29.1 MB at final summary. The complete retained `.context/`, including prior environments/caches and historical probes, had about 3.01 GB of logical file bytes before the small AST/source additions. This is below the 5 GB limit. Nothing was deleted.

## Limitations

- Structural engine paths mirror SQLExec, but no fenic session, semantic operator, cache, or metrics instrumentation ran.
- Full lazy source execution is a measured alternative, not the selected v0 or an available current physical-plan capability.
- Projection/arithmetic cases do not establish join, decimal, timezone, aggregate, overflow, or regex equivalence.
- Warm caches, three repetitions, and background activity do not support a universal speedup claim.
- Source rows are known fixture cardinalities, not measured decoded rows after lazy scan pruning.
- No provider, remote IO, catalog/storage, cold-install, or dependency-removal measurement belongs to this unit.

## Primary sources

Retrieved **2026-10-09 UTC**:

- **D1:** [DuckDB SQL on Arrow](https://duckdb.org/docs/current/guides/python/sql_on_arrow). Distinguishes datasets/scanners with pushdown from generic RecordBatchReader input.
- **D2:** [DuckDB Arrow output](https://duckdb.org/docs/current/guides/python/export_arrow). Current docs prefer `to_arrow_reader`; the pinned 1.4.5 probe uses its tested `fetch_record_batch` API. This difference is not an upgrade recommendation.
- **P1:** [Polars SQLContext.execute](https://docs.pola.rs/api/python/stable/reference/sql/api/polars.SQLContext.execute.html). Lazy frame execution; exact versions are recorded above.
- **P2:** [Polars collect_batches](https://docs.pola.rs/api/python/stable/reference/lazyframe/api/polars.LazyFrame.collect_batches.html). Unstable status, cost warning, and producer lifetime/stop behavior.
- **P3:** [Polars 2 upgrade guide](https://docs.pola.rs/releases/upgrade/2/). Streaming/SQL behavior differs by runtime; this comparison also tests the current pin.
