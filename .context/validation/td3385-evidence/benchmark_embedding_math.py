"""Reproduce TD-3385's local NumPy-versus-Polars embedding-math evidence.

This script intentionally reimplements the current dense NumPy callback paths
and their considered native Polars alternatives. It never constructs a Fenic
session or provider client. Run it from the repository root with:

PATH=/Users/brandoncallender/.rustup/toolchains/1.94.1-aarch64-apple-darwin/bin:$PATH \
uv run --no-sync python .context/validation/td3385-evidence/benchmark_embedding_math.py
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Callable

import numpy as np
import polars as pl
import pyarrow as pa

OUTPUT_DIR = Path(__file__).parent
BENCHMARK_OUTPUT = OUTPUT_DIR / "benchmark-results.json"
BOUNDARY_OUTPUT = OUTPUT_DIR / "parity-boundaries.json"
SEED = 3385
RTOL = ATOL = 2e-6


def norm(expr: pl.Expr) -> pl.Expr:
    """Return the native Polars L2 norm considered for the replacement."""
    return expr.arr.eval(pl.element() * pl.element()).arr.sum().sqrt()


def numpy_normalize(batch: pl.Series) -> pl.Series:
    """Current dense branch of EmbeddingNormalizeExpr's map_batches callback."""
    embeddings = batch.to_numpy()
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = np.divide(embeddings, norms, where=norms != 0)
    return pl.Series(normalized)


def calculate_similarity(embeddings: np.ndarray, other: np.ndarray, metric: str) -> np.ndarray:
    """Current _calculate_similarity_numpy implementation."""
    if metric == "dot":
        return np.sum(embeddings * other, axis=1)
    if metric == "cosine":
        dots = np.sum(embeddings * other, axis=1)
        denominator = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(other, axis=-1)
        return np.divide(dots, denominator, out=np.full_like(dots, np.nan), where=denominator != 0)
    if metric == "l2":
        return np.linalg.norm(embeddings - other, axis=1)
    raise ValueError(f"unsupported metric: {metric}")


def numpy_pair_similarity(metric: str) -> Callable[[pl.Series], pl.Series]:
    """Return the current pairwise map-batches callback for one metric."""
    def callback(batch: pl.Series) -> pl.Series:
        fields = batch.struct.fields
        return pl.Series(
            calculate_similarity(
                batch.struct.field(fields[0]).to_numpy(),
                batch.struct.field(fields[1]).to_numpy(),
                metric,
            )
        )

    return callback


def numpy_query_similarity(metric: str, query: np.ndarray) -> Callable[[pl.Series], pl.Series]:
    """Return the current vector-query map-batches callback for one metric."""
    def callback(batch: pl.Series) -> pl.Series:
        return pl.Series(calculate_similarity(batch.to_numpy(), query, metric))

    return callback


def numpy_embedding_avg(dimensions: int) -> Callable[[pl.Series], pl.Series]:
    """Current embedding AvgExpr callback, including null-vector handling."""
    def callback(series: pl.Series) -> pl.Series:
        result = []
        for embedding_list in series.to_list():
            filtered = [embedding for embedding in embedding_list if embedding is not None]
            result.append(np.mean(filtered, axis=0).astype(np.float32) if filtered else None)
        return pl.from_arrow(pa.array(result, type=pa.list_(pa.float32(), dimensions)))

    return callback


def native_avg(dimensions: int) -> pl.Expr:
    """The dimension-expanded native candidate investigated by TD-3385."""
    struct = pl.col("embedding").arr.to_struct()
    means = [struct.struct.field(f"field_{index}").mean() for index in range(dimensions)]
    assembled = pl.concat_arr(means)
    return pl.when(pl.col("embedding").count() > 0).then(assembled).otherwise(
        pl.lit(None, dtype=pl.Array(pl.Float32, dimensions))
    )


def frame(rows: int, dimensions: int, groups: int | None = None) -> tuple[pl.DataFrame, np.ndarray]:
    """Build the seeded dense Float32 fixture and query vector."""
    rng = np.random.default_rng(SEED)
    df = pl.DataFrame(
        {
            "embedding": pl.Series(rng.normal(size=(rows, dimensions)).astype(np.float32), dtype=pl.Array(pl.Float32, dimensions)),
            "other": pl.Series(rng.normal(size=(rows, dimensions)).astype(np.float32), dtype=pl.Array(pl.Float32, dimensions)),
        }
    )
    if groups is not None:
        df = df.with_columns(pl.Series("group", np.arange(rows) % groups))
    return df, rng.normal(size=dimensions).astype(np.float32)


def benchmark(before: Callable[[], pl.DataFrame], after: Callable[[], pl.DataFrame], rounds: int) -> tuple[list[float], list[float]]:
    """Warm each expression, then return raw execution-only timings in milliseconds."""
    for _ in range(3):
        before()
        after()
    before_ms = []
    after_ms = []
    for _ in range(rounds):
        started = perf_counter()
        before()
        before_ms.append((perf_counter() - started) * 1000)
        started = perf_counter()
        after()
        after_ms.append((perf_counter() - started) * 1000)
    return before_ms, after_ms


def arrays_allclose(before: pl.DataFrame, after: pl.DataFrame) -> bool:
    """Check schema, shape, and numerical parity at the recorded tolerance."""
    if before.schema != after.schema or before.shape != after.shape:
        return False
    return all(
        np.allclose(left.to_numpy(), right.to_numpy(), rtol=RTOL, atol=ATOL, equal_nan=True)
        for left, right in zip(before.get_columns(), after.get_columns(), strict=True)
    )


def case_matrix(rows: int, dimensions: int, rounds: int, include_avg: bool) -> list[dict[str, object]]:
    """Run the native candidate matrix for one dense embedding shape."""
    df, query = frame(rows, dimensions, groups=128 if include_avg else None)
    query_expr = pl.lit(query.tolist(), dtype=pl.Array(pl.Float32, dimensions))
    cases: list[tuple[str, Callable[[], pl.DataFrame], Callable[[], pl.DataFrame]]] = [
        (
            "normalize",
            lambda: df.select(pl.col("embedding").map_batches(numpy_normalize, return_dtype=pl.Array(pl.Float32, dimensions))),
            lambda: df.select((pl.col("embedding") / norm(pl.col("embedding"))).alias("embedding")),
        ),
    ]
    for metric in ("dot", "cosine", "l2"):
        if metric == "dot":
            pair_native = (pl.col("embedding") * pl.col("other")).arr.sum()
            query_native = (pl.col("embedding") * query_expr).arr.sum()
        elif metric == "cosine":
            pair_native = (pl.col("embedding") * pl.col("other")).arr.sum() / (norm(pl.col("embedding")) * norm(pl.col("other")))
            query_native = (pl.col("embedding") * query_expr).arr.sum() / (norm(pl.col("embedding")) * float(np.linalg.norm(query)))
        else:
            pair_native = ((pl.col("embedding") - pl.col("other")) * (pl.col("embedding") - pl.col("other"))).arr.sum().sqrt()
            query_native = ((pl.col("embedding") - query_expr) * (pl.col("embedding") - query_expr)).arr.sum().sqrt()
        cases.extend(
            [
                (
                    f"pair-{metric}",
                    lambda metric=metric: df.select(pl.struct("embedding", "other").map_batches(numpy_pair_similarity(metric), return_dtype=pl.Float32)),
                    lambda native=pair_native: df.select(native.alias("embedding")),
                ),
                (
                    f"query-{metric}",
                    lambda metric=metric: df.select(pl.col("embedding").map_batches(numpy_query_similarity(metric, query), return_dtype=pl.Float32)),
                    lambda native=query_native: df.select(native.alias("embedding")),
                ),
            ]
        )
    if include_avg:
        cases.append(
            (
                "avg",
                lambda: df.group_by("group", maintain_order=True).agg(
                    pl.col("embedding").implode().map_batches(
                        numpy_embedding_avg(dimensions),
                        return_dtype=pl.Array(pl.Float32, dimensions),
                        returns_scalar=True,
                    ).alias("embedding")
                ),
                lambda: df.group_by("group", maintain_order=True).agg(native_avg(dimensions).alias("embedding")),
            )
        )

    results = []
    for name, before, after in cases:
        before_output, after_output = before(), after()
        before_ms, after_ms = benchmark(before, after, rounds)
        before_median = median(before_ms)
        after_median = median(after_ms)
        results.append(
            {
                "operation": name,
                "numpy_ms": before_ms,
                "native_ms": after_ms,
                "numpy_median_ms": before_median,
                "native_median_ms": after_median,
                "native_gain_percent": (1 - after_median / before_median) * 100,
                "schema": str(after_output.schema),
                "allclose_rtol_atol_2e-6": arrays_allclose(before_output, after_output),
            }
        )
    return results


def boundary_receipt() -> dict[str, object]:
    """Capture the dot-product reduction-order counterexample independently."""
    rows, dimensions = 32_768, 384
    df, query = frame(rows, dimensions)
    pair_numpy = df.select(
        pl.struct("embedding", "other").map_batches(numpy_pair_similarity("dot"), return_dtype=pl.Float32)
    ).to_series().to_numpy()
    pair_native = df.select((pl.col("embedding") * pl.col("other")).arr.sum()).to_series().to_numpy()
    query_numpy = df.select(
        pl.col("embedding").map_batches(numpy_query_similarity("dot", query), return_dtype=pl.Float32)
    ).to_series().to_numpy()
    query_native = df.select(
        (pl.col("embedding") * pl.lit(query.tolist(), dtype=pl.Array(pl.Float32, dimensions))).arr.sum()
    ).to_series().to_numpy()
    return {
        "fixture": {"rows": rows, "dimensions": dimensions, "seed": SEED, "dtype": "Float32"},
        "tolerance": {"rtol": RTOL, "atol": ATOL},
        "pair_dot": {
            "allclose": bool(np.allclose(pair_numpy, pair_native, rtol=RTOL, atol=ATOL)),
            "different_rows": int(np.count_nonzero(pair_numpy != pair_native)),
            "max_absolute_delta": float(np.max(np.abs(pair_numpy - pair_native))),
        },
        "query_dot": {
            "allclose": bool(np.allclose(query_numpy, query_native, rtol=RTOL, atol=ATOL)),
            "different_rows": int(np.count_nonzero(query_numpy != query_native)),
            "max_absolute_delta": float(np.max(np.abs(query_numpy - query_native))),
        },
    }


def main() -> None:
    """Write benchmark and numerical-parity receipts beside this script."""
    benchmark_receipt = {
        "environment": {"polars": pl.__version__, "numpy": np.__version__},
        "method": {"warmups": 3, "fixture_construction_timed": False},
        "matrices": [
            {"rows": 32_768, "dimensions": 384, "rounds": 9, "groups": 128, "results": case_matrix(32_768, 384, 9, include_avg=True)},
            {"rows": 8_192, "dimensions": 1_536, "rounds": 7, "groups": None, "results": case_matrix(8_192, 1_536, 7, include_avg=False)},
        ],
    }
    BENCHMARK_OUTPUT.write_text(json.dumps(benchmark_receipt, indent=2) + "\n")
    BOUNDARY_OUTPUT.write_text(json.dumps(boundary_receipt(), indent=2) + "\n")
    print(f"wrote {BENCHMARK_OUTPUT}")
    print(f"wrote {BOUNDARY_OUTPUT}")


if __name__ == "__main__":
    main()
