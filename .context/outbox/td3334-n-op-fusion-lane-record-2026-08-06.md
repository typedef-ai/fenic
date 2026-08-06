# TD-3334 generalized N-op fusion — lane record

**Status:** pre-design evidence in progress.  
**Branch:** `herd/fenic-exec-engine-td3334-n-op-fusion`, stacked at accepted
validation head `90987341f0af9e490e4e68862ca2f3f1ef4533da`.

## Mandatory $0 LMMetrics pre-check — PASS

`tests/_inference/test_language_model_metrics_paths.py` uses a fake provider
whose successful calls record deterministic LMMetrics. It proves:

1. `LanguageModel.get_completions` (legacy list) records two calls' metrics.
2. `LanguageModel.iter_completions` (B0 iterator, batch size one) records the
   same two calls' metrics.
3. Reading a distinct, unused model client yields the expected all-zero metrics,
   while the active iterator client is non-zero.

The focused fake/local gate passed: **2 passed**. Code trace confirms the
iterator delegates every bounded block to the same `make_batch_requests` path,
and both the harness and physical-plan conversion retrieve the default model
from the same session registry. Therefore B0's iterator does **not** bypass
LMMetrics accounting and no TD-3334 metrics-fix card is required on that basis.
The Amendment A zero-valued LMMetrics remains an external/provider-or-telemetry
limitation to disclose, not a permission to infer `$0` billing or to run another
live probe.

## Next evidence before design freeze

Read-only survey typedef's data-intelligence sync pipelines for real Fenic
semantic-operator chains. The design will target observed shapes while retaining
general open-source semantics. If production contains no chain longer than two
semantic operators, record the evidence-backed conclusion that B1 already
covers production and stop before speculative N-op implementation.

## Read-only production survey — COMPLETE

Survey scope was the canonical checkout only:
`/Users/brandoncallender/dev/data-intelligence/main/libs/python/ingest/src` and
`services/data-intelligence/src`; disposable worktrees and tools were excluded.
The primary sync pipeline is
`ingest/static_loaders/semantic/pipeline/`.

| Evidence | Finding |
|---|---|
| `pipeline/dependencies.py` | 12 logical semantic passes form a dependency chain from `relation_analysis` through `analysis_summary`. |
| `pipeline/executor.py:161-164` | Every pass executes as `pass_instance.execute(df).cache()` followed immediately by `df.count()`: an intentional cache/materialization barrier per pass. |
| `passes/pass_01_relations.py` through `pass_11_summary.py` | Each pass adds exactly one `fc.semantic.map` or `fc.semantic.extract` expression to its DataFrame. |
| Other production ingest uses (`ontology/function_interpretation.py`, `overview/compute*.py`, `clustering/enrichment.py`, static/unified Salesforce stitchers, targeted classifiers) | Each surveyed call builds one semantic operation then materializes it with `to_pylist`, `count`, or an equivalent handoff; no same-plan semantic chain longer than one was found. |

The long *pipeline* chain is deliberately a sequence of materialized semantic
queries, not a single physical plan. Removing those checkpoints would change
failure, resume, cache, and progress semantics and is outside TD-3334. There is
therefore no observed production N-op fusion candidate; production chains are
at most one operator per physical plan. B1 already covers the only relevant
two-operator shape without needing a generalized implementation.

## TD-3334 design freeze — NOT NEEDED, evidence-backed

No generalized N-op fusion code will be written. This is the charter-authorized
"B1 already covers production" result, stronger than the stated `≤2` threshold:
the surveyed sync pipeline has no unmaterialized two-op chain either. The
open-source design is preserved by not introducing speculative machinery for a
shape absent from production. The conditional validation hypothesis remains
recorded: fusion can remove idle handoff time, while provider governor/TPM
pressure can dominate wall time at larger inputs.

## Frozen review request — FROM HERD COMMAND

The fake metrics pre-check, production survey, and no-op design decision are
frozen. **Request review FROM HERD COMMAND** for TD-3334 at the current branch
head; scope is the metrics regression test and this evidence record. **HOLD:**
do not start TD-3384 or TD-3385 until this disposition.
