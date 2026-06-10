"""Live-API stress benchmark for fenic's rate-limiting strategies.

Demonstrates the real-world impact of adaptive output-token estimation +
settlement under a deliberately tight TPM budget with a naively high
``max_output_tokens``, on a task that actually emits very few tokens
(with per-row output variance).

Three modes, run sequentially against a real provider:

- **pre-feature**: adaptive estimation disabled AND ``settle()`` no-op'd on the
  session's rate-limit strategy — reproduces the old ceiling-reserve-no-refund
  behavior.
- **settle-only**: ``AdaptiveTokenEstimationConfig(enabled=False)`` — reservations
  use the static ceiling, but settlement (always-on) refunds over-reservation.
- **adaptive**: default config — learned reservations + settlement. Runs a
  warm-up phase first (same session) so the estimator has >= ``min_samples``
  observations before the measured phase.

This script makes REAL API calls and costs (a little) money. It is deliberately
NOT a pytest test and lives outside ``tests/`` and ``examples/`` so CI never
runs it. Run manually:

    uv run --env-file .env python benchmarks/rate_limit_stress.py
    uv run --env-file .env python benchmarks/rate_limit_stress.py --rows 20 --tpm 10000

Notes on interpretation:

- pre-feature vs the others is the headline wall-clock difference: the TPM
  bucket reserves the full ceiling per row and only time-refill restores it.
- settle-only vs adaptive can be close on wall-clock at fast model latencies
  (settlement refunds cascade quickly); the honest differentiator is
  reservation efficiency (actual/reserved output tokens), reported per mode.
"""

import argparse
import logging
import random
import time
import uuid
from dataclasses import dataclass
from typing import Optional

import fenic as fc

COLORS = [
    "red", "blue", "green", "yellow", "purple", "orange", "teal", "maroon",
    "olive", "navy", "coral", "crimson", "indigo", "violet", "amber", "beige",
]

FILLER = [
    "Inventory was checked at the loading dock before the morning shift.",
    "The warehouse manager signed off on the manifest after a brief delay.",
    "Several pallets arrived with minor water damage on the outer wrap.",
    "A forklift was out of service, so totes were moved by hand cart.",
    "The receiving team flagged two cartons for a recount next week.",
    "Labels on the lower shelf had faded and were reprinted on Tuesday.",
    "An auditor requested photos of the staging area for the records.",
    "The afternoon truck was rescheduled due to weather on the pass.",
]

PROMPT = (
    "Read the following inventory note and list each DISTINCT color mentioned, "
    "as a comma-separated list with no other text. If no colors are mentioned, "
    "reply 'none'.\n\nNote: {{ note }}"
)


def build_rows(n_rows: int, seed: int) -> list:
    """Generate unique inventory notes with 1..12 embedded colors and varied filler."""
    # trunk-ignore(bandit/B311): non-cryptographic randomness for workload generation
    rng = random.Random(seed)
    rows = []
    for i in range(n_rows):
        n_colors = rng.randint(1, 12)
        colors = rng.sample(COLORS, n_colors)
        n_filler = rng.randint(1, 4)
        filler = rng.sample(FILLER, n_filler)
        items = [f"{rng.randint(2, 40)} {c} bins" for c in colors]
        # A unique token per row prevents request deduplication.
        note = f"(ref #{i}-{seed}) " + " ".join(filler) + " Counted: " + ", ".join(items) + "."
        rows.append(note)
    return rows


class BackoffCounter(logging.Handler):
    """Counts 'Backing off' warnings from the model client (429/backoff proxy)."""

    def __init__(self):
        """Initialize the counter handler at WARNING level."""
        super().__init__(level=logging.WARNING)
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        """Count records whose message indicates a rate-limit backoff."""
        if "Backing off" in record.getMessage():
            self.count += 1


@dataclass
class PhaseResult:
    """Metrics captured for one measured benchmark phase."""

    wall_s: float
    rows: int
    actual_output_tokens: int
    reserved_output_tokens: int
    input_tokens: int
    cost_usd: float
    backoffs: int

    @property
    def reservation_efficiency(self) -> Optional[float]:
        """Actual / reserved output tokens (None when nothing was reserved)."""
        if self.reserved_output_tokens <= 0:
            return None
        return self.actual_output_tokens / self.reserved_output_tokens


def make_session(args, mode: str) -> fc.Session:
    """Create a fresh fenic Session configured for the given benchmark mode."""
    adaptive = (
        fc.AdaptiveTokenEstimationConfig(enabled=True, safety_margin=args.safety_margin)
        if mode == "adaptive"
        else fc.AdaptiveTokenEstimationConfig(enabled=False)
    )
    config = fc.SessionConfig(
        app_name=f"rate_limit_stress_{mode}_{uuid.uuid4().hex[:6]}",
        semantic=fc.SemanticConfig(
            language_models={
                "bench": fc.OpenAILanguageModel(
                    model_name=args.model, rpm=args.rpm, tpm=args.tpm
                )
            },
            default_language_model="bench",
            adaptive_token_estimation=adaptive,
        ),
    )
    return fc.Session.get_or_create(config)


def run_phase(session: fc.Session, rows: list, args, backoffs: BackoffCounter) -> PhaseResult:
    """Run one semantic.map batch and capture its per-query metrics."""
    backoffs_before = backoffs.count

    df = session.create_dataframe({"note": rows})
    df = df.select(
        fc.semantic.map(
            PROMPT, max_output_tokens=args.max_output_tokens, note=fc.col("note")
        ).alias("colors")
    )
    start = time.time()
    # collect() returns a QueryResult whose .metrics carries the per-query
    # LMMetrics (execution attributes + resets the registry counters per query).
    result = df.collect("polars")
    wall = time.time() - start

    lm_metrics = result.metrics.total_lm_metrics
    return PhaseResult(
        wall_s=wall,
        rows=len(result.data),
        actual_output_tokens=lm_metrics.num_output_tokens,
        reserved_output_tokens=lm_metrics.num_reserved_output_tokens,
        input_tokens=lm_metrics.num_uncached_input_tokens + lm_metrics.num_cached_input_tokens,
        cost_usd=lm_metrics.cost,
        backoffs=backoffs.count - backoffs_before,
    )


def run_mode(mode: str, args, backoffs: BackoffCounter) -> PhaseResult:
    """Run one mode end-to-end (warm-up phase for adaptive) and return the measured phase."""
    session = make_session(args, mode)
    try:
        if mode == "pre-feature":
            # Reproduce old behavior: no settlement refunds. Patch the INSTANCE so
            # nothing leaks to other sessions/modes.
            client = session._session_state._model_registry.get_language_model().client
            client.rate_limit_strategy.settle = lambda reserved, actual: None

        if mode == "adaptive":
            # Warm-up phase: same session => same client => estimator accumulates
            # actuals (rows > min_samples=30) before the measured phase.
            warm_rows = build_rows(args.rows, seed=args.seed + 1)
            print(f"  [{mode}] warm-up phase ({len(warm_rows)} rows)...")
            warm = run_phase(session, warm_rows, args, backoffs)
            print(f"  [{mode}] warm-up done in {warm.wall_s:.1f}s")
            # Let the token bucket refill so the measured phase starts ~full,
            # comparable to the fresh-session modes.
            time.sleep(args.settle_pause)

        rows = build_rows(args.rows, seed=args.seed)
        print(f"  [{mode}] measured phase ({len(rows)} rows)...")
        return run_phase(session, rows, args, backoffs)
    finally:
        session.stop()


def main():
    """Parse args, run the selected modes sequentially, and print the comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=40, help="rows per phase")
    parser.add_argument("--tpm", type=int, default=20_000, help="configured client TPM")
    parser.add_argument("--rpm", type=int, default=300, help="configured client RPM")
    parser.add_argument("--max-output-tokens", type=int, default=2048,
                        help="naive per-request output cap (the static ceiling)")
    parser.add_argument("--model", default="gpt-4.1-nano")
    parser.add_argument("--safety-margin", type=float, default=1.15)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--settle-pause", type=float, default=15.0,
                        help="seconds to let the bucket refill between phases")
    parser.add_argument("--modes", nargs="+", default=["pre-feature", "settle-only", "adaptive"],
                        choices=["pre-feature", "settle-only", "adaptive"])
    args = parser.parse_args()

    n_requests = sum(2 * args.rows if m == "adaptive" else args.rows for m in args.modes)
    print(f"Benchmark: {args.model} | configured tpm={args.tpm} rpm={args.rpm} | "
          f"max_output_tokens={args.max_output_tokens} | rows/phase={args.rows}")
    print(f"Total live requests: ~{n_requests} (warm-up + measured phases)\n")

    logging.getLogger("fenic._inference.model_client").setLevel(logging.WARNING)
    backoffs = BackoffCounter()
    logging.getLogger("fenic._inference.model_client").addHandler(backoffs)

    results: dict = {}
    for mode in args.modes:
        print(f"== mode: {mode} ==")
        results[mode] = run_mode(mode, args, backoffs)
        r = results[mode]
        print(f"  [{mode}] measured: {r.wall_s:.1f}s, {r.rows} rows, "
              f"out={r.actual_output_tokens} reserved={r.reserved_output_tokens}\n")

    print("=" * 88)
    header = (f"{'mode':<14} {'wall (s)':>9} {'rows':>5} {'out tok':>8} "
              f"{'reserved':>9} {'efficiency':>10} {'backoffs':>8} {'cost ($)':>9}")
    print(header)
    print("-" * 88)
    for mode, r in results.items():
        eff = f"{r.reservation_efficiency:.3f}" if r.reservation_efficiency is not None else "n/a"
        print(f"{mode:<14} {r.wall_s:>9.1f} {r.rows:>5} {r.actual_output_tokens:>8} "
              f"{r.reserved_output_tokens:>9} {eff:>10} {r.backoffs:>8} {r.cost_usd:>9.4f}")
    print("=" * 88)
    if "pre-feature" in results and "adaptive" in results:
        speedup = results["pre-feature"].wall_s / max(0.001, results["adaptive"].wall_s)
        print(f"\nadaptive vs pre-feature wall-clock: {speedup:.1f}x faster")


if __name__ == "__main__":
    main()
