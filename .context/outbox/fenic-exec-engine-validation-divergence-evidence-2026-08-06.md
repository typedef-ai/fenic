# Validation divergence evidence availability

**Status:** raw actual-vs-expected values were **not captured** before the
unfused/64 parity assertion raised. No additional keyed/provider call or retry
was issued while checking this fact.

The retained process output establishes only this:

- the real unfused arm completed its 64 map and 64 extract request batches;
- 64 output rows reached the parity check; and
- the extracted `record_id → category` mapping differed from the deterministic
  expected mapping.

The actual extracted values lived only in the process-local `observed` dict. The
harness raised a count-only `AssertionError`, did not serialize that dict, and
the temporary session/database was removed in the normal `TemporaryDirectory`
cleanup. There is therefore no honest representative mismatch to paste.

The deterministic expected corpus values for the first rows were:

| record_id | expected category | actual extracted value |
|---:|---|---|
| 0 | `ALPHA` | unavailable — not captured |
| 1 | `BETA` | unavailable — not captured |
| 2 | `GAMMA` | unavailable — not captured |
| 3 | `DELTA` | unavailable — not captured |

This is the **second validation-harness deficiency**, after the missing
post-execution LMMetrics serialization: it preserves only the existence of
semantic divergence, not its value-level shape. The report must not suggest
otherwise. A future authorized probe would need to persist a redacted synthetic
comparison and LMMetrics before asserting, but this HOLD does not diagnose,
retry, or change the completed run.
