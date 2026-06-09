# Semantic operator memory benchmark protocol

`tools/benchmark_semantic_operator_memory.py` is the opt-in, local-only peak RSS
harness for representative semantic operators. Use this protocol when a PR claims
that semantic operator memory usage improved or regressed.

The harness reports child-process peak RSS in bytes from
`resource.getrusage(RUSAGE_SELF).ru_maxrss`. Polars allocator peak bytes are not
available through the existing fenic benchmark/test tooling, so PR evidence must
compare peak RSS instead of inventing allocator precision.

## Default benchmark matrix

The default CLI matrix is intentionally small enough for local repeatability and
large enough to exercise the full operator path for each case. These are the
starting parameters used when no size flags are supplied:

- Similarity join
  - Selector: `--cases sim_join`
  - Size: `--rows 64 --right-rows 32 --embedding-dimensions 8 --k 2`
  - Exercises embedding column construction, vector casting, and top-k similarity
    join materialization.
- Semantic reduce
  - Selector: `--cases semantic_reduce`
  - Size: `--rows 64 --groups 8`
  - Exercises grouped semantic reduce with deterministic local completions and
    ordered aggregation.
- Semantic join
  - Selector: `--cases semantic_join`
  - Size: `--rows 64 --right-rows 32`
  - Exercises predicate-based semantic join using deterministic local completions.
- Map/extract chain
  - Selector: `--cases map_extract_chain`
  - Size: `--rows 64`
  - Exercises chained `semantic.map` -> `semantic.extract` materialization.

`--cases all` runs the full matrix. Unless a PR has a narrower reason to target a
single operator, before/after evidence should include the full matrix.

## Smoke tests versus evidence-grade runs

Pytest-sized cases such as `--rows 2` are smoke tests only. They prove that the
harness executes, emits JSON/Markdown, and avoids external provider calls. They do
not provide stable evidence for memory claims because fixed process/session
startup overhead can dominate the operator work.

For evidence-grade runs, start with the default matrix. Increase `--rows`,
`--right-rows`, `--groups`, or `--embedding-dimensions` when before/after numbers
are close enough that process startup noise could hide the operator signal. Larger
sizes are especially useful when the reported delta is within a small multiple of
the baseline idle/session RSS or when a change affects per-row, per-candidate, or
per-group memory behavior.

## Evidence-grade commands

Run commands from the repo root. Prefer `--json` for artifacts checked by agents
or CI-adjacent review tooling; omit `--json` when a reviewer wants copyable
Markdown with the embedded JSON block.

Full default matrix:

```bash
uv run python tools/benchmark_semantic_operator_memory.py \
  --json \
  --label TD-XXXX-before-default

uv run python tools/benchmark_semantic_operator_memory.py \
  --json \
  --label TD-XXXX-after-default
```

Per-operator commands:

```bash
uv run python tools/benchmark_semantic_operator_memory.py \
  --cases sim_join \
  --json \
  --label TD-XXXX-before-sim-join

uv run python tools/benchmark_semantic_operator_memory.py \
  --cases semantic_reduce \
  --json \
  --label TD-XXXX-before-semantic-reduce

uv run python tools/benchmark_semantic_operator_memory.py \
  --cases semantic_join \
  --json \
  --label TD-XXXX-before-semantic-join

uv run python tools/benchmark_semantic_operator_memory.py \
  --cases map_extract_chain \
  --json \
  --label TD-XXXX-before-map-extract-chain
```

Repeat the same command set after the implementation change with `after` labels.
Keep all non-code review artifacts under `.hermes/evidence/` or another ignored
local path; do not commit raw benchmark outputs unless the PR explicitly needs a
curated fixture.

## Labeling convention

Use labels in this shape:

```text
<TD issue>-<before|after>-<scope>[-<size>]
```

Examples:

- `TD-3381-before-default`
- `TD-3381-after-default`
- `TD-3381-before-semantic-join-rows-512-right-256`
- `TD-3381-after-semantic-join-rows-512-right-256`

The before and after labels must differ only by the `before`/`after` segment when
they are meant to compare the same matrix and size.

## Reviewer checklist for memory-improvement PRs

A PR that claims semantic operator peak-memory improvement should cite:

1. Before and after benchmark JSON or Markdown output produced by this harness.
2. The exact command line for each cited artifact, including case selectors and
   size flags.
3. The git commit and branch embedded in each benchmark payload.
4. Confirmation that pytest-sized runs such as `--rows 2` were used only as smoke
   checks, not as memory evidence.
5. Rationale for any non-default size, including why a larger matrix was needed
   to escape process/session startup noise.
6. A short interpretation of the peak RSS delta by case, without introducing hard
   pass/fail memory thresholds unless a separate benchmark policy explicitly adds
   them.

Do not add large benchmark runs to default CI as part of this protocol. Keep the
harness opt-in until a separate CI policy defines runtime budget, hardware shape,
and acceptable noise bounds.
