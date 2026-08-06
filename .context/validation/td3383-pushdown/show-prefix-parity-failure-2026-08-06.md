# TD-3383 `show(n)` prefix candidate — hard-gate failure

**Command:**
`RUN_TD3383_SHOW_BENCHMARK=1 TD3383_BENCHMARK_OUTPUT_DIR=.context/validation/td3383-pushdown uv run --no-sync pytest tests/_backends/local/test_show_pushdown.py::test_td3383_show_prefix_benchmark -q`

**Result:** failed after the 15 alternating baseline/candidate action pairs, at
the required `candidate_output == baseline_output` assertion. No provider call
occurred; the test-local synthetic key was fixture construction only.

## Reproduced mismatch

Fixture: 100,000 rows × 128 Int64 columns; direct projection of `column_0` and
`column_127`; `show(10)`.

The incumbent action executes the full result, then Polars formats ten display
rows as a first/last sample:

```text
column_0 / column_127
0 / 127
1 / 128
2 / 129
3 / 130
4 / 131
… / …
99995 / 100122
99996 / 100123
99997 / 100124
99998 / 100125
99999 / 100126
```

The proposed source-limited action instead materializes only its first ten rows,
so Polars displays:

```text
column_0 / column_127
0 / 127
1 / 128
2 / 129
3 / 130
4 / 131
5 / 132
6 / 133
7 / 134
8 / 135
9 / 136
```

This is a user-visible output divergence, not a timing-only difference. The
candidate is therefore disqualified; preserving incumbent first/last display
semantics would require a broader sampling/pagination contract, beyond TD-3383's
low-risk boundary.

## Receipt limitation

The prototype benchmark serialized timing samples only after the output-parity
assertion. The assertion raised first, so its in-memory 15-round timing arrays
were not persisted. They are not reconstructed or inferred here, and the
benchmark is not retried: output parity is the load-bearing gate. This is a
harness-evidence limitation recorded for any future action-mode experiment, not
a reason to weaken the no-op conclusion.
