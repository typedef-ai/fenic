# TD-3385 reproducible embedding-math evidence

`benchmark_embedding_math.py` is a seeded, local-only reproduction of the
NumPy callback paths and native Polars candidates considered in TD-3385. It
does not import Fenic, construct a session, read `.env`, or make a network call.

Run from the repository root:

```text
PATH=/Users/brandoncallender/.rustup/toolchains/1.94.1-aarch64-apple-darwin/bin:$PATH \
uv run --no-sync python .context/validation/td3385-evidence/benchmark_embedding_math.py
```

It writes the raw per-round measurement arrays to `benchmark-results.json` and
the dot-product numerical parity receipt to `parity-boundaries.json`. Timings
are machine-local and should be used to reproduce the direction, not as a
cross-machine throughput claim.
