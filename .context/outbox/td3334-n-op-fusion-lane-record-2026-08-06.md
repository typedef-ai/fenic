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
