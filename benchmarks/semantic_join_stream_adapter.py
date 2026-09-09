"""Provider-free, real-clock throughput evidence for the bounded join adapter.

This is deliberately a benchmark rather than a production test.  It uses the
real ``ModelClient`` queue, scheduler, rate-limit gate, and ``iter_completions``
path with ``SimulatedCompletionsClient`` at the provider boundary.  No provider
credential or network request is used.

The default workload is shaped so the measurement can fail meaningfully:

* 512 left rows x 2 right rows = 1,024 unique predicate requests;
* pair blocks are capped at 256 (below the 1,024 implementation default), so
  there are four physical join blocks;
* each block is split by a token budget into two 128-request predicate calls;
* the effective streaming watermark is ``max(batch_size=100, rpm=100) = 100``.

Thus every arm processes multiple blocks, every block has a token-budget split,
and each split is larger than the streaming watermark. The harness deliberately
decouples the request bucket's capacity from configured RPM. This manufactured
condition makes the admission binding matter for elapsed time by letting the
standard arm exceed the configured burst. A coherent limiter sets its burst to
RPM, and ``W = max(batch_size, rpm)`` is never below that burst, so the limiter
binds first and the ordinary short-latency configuration would not bind the
window. The recorded high-water mark counts outstanding admitted requests, not
dispatch concurrency. No saturation or rate-limit throughput claim is made.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fenic._backends.local.semantic_operators.base import BaseOperator  # noqa: E402
from fenic._backends.local.semantic_operators.join import Join  # noqa: E402
from fenic._inference import model_client as model_client_module  # noqa: E402
from fenic._inference.language_model import LanguageModel  # noqa: E402
from fenic._inference.types import FenicCompletionsResponse  # noqa: E402
from tests._inference.rate_limit_harness.harness import (  # noqa: E402
    RateLimitScenario,
    SimulatedCompletionsClient,
    constant,
)

_tqdm = model_client_module.tqdm


def _silent_tqdm(*args, **kwargs):
    """Remove terminal rendering overhead from both timing arms."""
    kwargs["disable"] = True
    return _tqdm(*args, **kwargs)


DEFAULT_LEFT_ROWS = 512
DEFAULT_RIGHT_ROWS = 2
DEFAULT_PAIR_BLOCK_SIZE = 256
DEFAULT_BLOCK_TOKEN_BUDGET = 18_000
DEFAULT_RPM = 100
DEFAULT_BATCH_SIZE = 100
DEFAULT_REPETITIONS = 7
DEFAULT_LATENCY_SECONDS = 0.01
DEFAULT_INPUT_TOKENS = 16
DEFAULT_OUTPUT_TOKENS = 2
SIMULATED_JOIN_STEP = {
    "operator": "join",
    "input_columns": ["left_on", "right_on"],
    "output_column": "matched",
    "prompt_template": "{{ left_on }} -- {{ right_on }}",
    "input_profile": "deterministic",
    "output_profile": "deterministic-true",
    "output_schema": None,
}


class PredicateSimulatedClient(SimulatedCompletionsClient):
    """Return a deterministic true predicate result for every simulated call."""

    _left_row = re.compile(r"left-(\d+)")

    @staticmethod
    def _row_index(request) -> int:
        """Map the rendered join prompt back to a deterministic output draw."""
        match = PredicateSimulatedClient._left_row.search(request.messages.user or "")
        if match is None:
            raise AssertionError("simulated join prompt omitted its left-row marker")
        return int(match.group(1))

    async def make_single_request(self, request):
        """Return the harness's deterministic predicate response."""
        response = await super().make_single_request(request)
        if isinstance(response, FenicCompletionsResponse):
            return FenicCompletionsResponse(
                completion='{"output": true}',
                logprobs=None,
                usage=response.usage,
            )
        return response


@dataclass(frozen=True)
class Workload:
    """Geometry and simulator settings for one interleaved comparison."""

    left_rows: int = DEFAULT_LEFT_ROWS
    right_rows: int = DEFAULT_RIGHT_ROWS
    pair_block_size: int = DEFAULT_PAIR_BLOCK_SIZE
    block_token_budget: int = DEFAULT_BLOCK_TOKEN_BUDGET
    rpm: int = DEFAULT_RPM
    batch_size: int = DEFAULT_BATCH_SIZE
    repetitions: int = DEFAULT_REPETITIONS
    latency_seconds: float = DEFAULT_LATENCY_SECONDS
    input_seed: int = 0

    @property
    def expected_requests(self) -> int:
        """Return the number of unique left/right predicate pairs."""
        return self.left_rows * self.right_rows

    @property
    def watermark(self) -> int:
        """Return the streaming admission watermark for this workload."""
        return max(self.batch_size, self.rpm)


def _scenario(workload: Workload, seed: int) -> RateLimitScenario:
    return RateLimitScenario(
        # ``rpm`` is the configured admission watermark.  The request bucket is
        # accelerated in ``_new_client`` below so this evidence does not spend
        # minutes refilling a simulated provider bucket.
        rpm=workload.rpm,
        tpm=10_000_000,
        true_rpm=10_000_000,
        true_tpm=10_000_000,
        n_rows=workload.expected_requests,
        static_ceiling=DEFAULT_OUTPUT_TOKENS,
        input_tokens=DEFAULT_INPUT_TOKENS,
        output_spec=constant(DEFAULT_OUTPUT_TOKENS),
        seed=seed,
        latency_s=workload.latency_seconds,
    )


def _new_client(workload: Workload, seed: int) -> PredicateSimulatedClient:
    client = PredicateSimulatedClient(_scenario(workload, seed))
    client.model = "gpt-4.1-nano"
    # This mutation manufactures the binding regime by decoupling bucket
    # capacity from configured RPM. It is not a faithful limiter configuration.
    # The streaming W calculation still reads strategy.rpm (100), while the
    # standard arm can admit all 128 requests in a token-bounded block.
    bucket = client.rate_limit_strategy.requests_bucket
    bucket.max_capacity = 1_000_000
    bucket.current_capacity_ = 1_000_000
    bucket.last_update_time_ = time.time()
    return client


def _dataframes(workload: Workload) -> tuple[pl.DataFrame, pl.DataFrame]:
    left_payload = " ".join(["alpha"] * 80)
    right_payload = " ".join(["beta"] * 20)
    left_start = workload.input_seed * workload.left_rows
    right_start = workload.input_seed * workload.right_rows
    return (
        pl.DataFrame(
            {
                # Keep ``left-{i}`` as a local output-draw index; the added
                # marker makes seeded benchmark inputs observably distinct
                # without indexing outside the simulator's per-row draws.
                "left_on": [
                    f"left-{i:04d} seed-{workload.input_seed} {left_payload}"
                    for i in range(workload.left_rows)
                ],
                "record_id": list(range(left_start, left_start + workload.left_rows)),
            }
        ),
        pl.DataFrame(
            {
                "right_on": [
                    f"right-{i:02d} seed-{workload.input_seed} {right_payload}"
                    for i in range(workload.right_rows)
                ],
                "right_id": list(
                    range(right_start, right_start + workload.right_rows)
                ),
            }
        ),
    )


def _join(model: LanguageModel, workload: Workload) -> Join:
    left, right = _dataframes(workload)
    return Join(
        left_df=left,
        right_df=right,
        jinja_template=SIMULATED_JOIN_STEP["prompt_template"],
        strict=True,
        model=model,
        temperature=0,
        pair_block_size=workload.pair_block_size,
        block_token_budget=workload.block_token_budget,
    )


def workload_geometry(workload: Workload) -> dict[str, Any]:
    """Return the structural proof that this workload exercises the adapter."""
    client = _new_client(workload, seed=0)
    try:
        model = LanguageModel(client)
        join = _join(model, workload)
        documents = join._join_documents()
        if documents is None:
            raise AssertionError("join did not produce predicate documents")
        left_documents, right_documents = documents
        pair_blocks = list(join._iter_join_pair_blocks(left_documents, right_documents))
        token_blocks = [
            chunk
            for pair_block in pair_blocks
            for chunk in join._split_block_by_token_budget(pair_block)
        ]
        sizes = [len(block) for block in token_blocks]
        return {
            "left_rows": workload.left_rows,
            "right_rows": workload.right_rows,
            "expected_requests": workload.expected_requests,
            "pair_block_size": workload.pair_block_size,
            "pair_block_count": len(pair_blocks),
            "token_bounded_block_count": len(token_blocks),
            "token_bounded_block_sizes": sizes,
            "token_budget": workload.block_token_budget,
            "batch_size": workload.batch_size,
            "configured_rpm": workload.rpm,
            "effective_watermark": workload.watermark,
            "window_binds": workload.expected_requests > workload.watermark
            and all(size > workload.watermark for size in sizes),
            "multiple_pair_blocks": len(pair_blocks) > 1,
            "token_budget_splits": len(token_blocks) > len(pair_blocks),
            "all_token_blocks_within_pair_cap": all(
                size <= workload.pair_block_size for size in sizes
            ),
        }
    finally:
        client.shutdown()


def assert_workload_geometry(workload: Workload) -> dict[str, Any]:
    """Require a window-binding, multi-block workload before measuring it."""
    geometry = workload_geometry(workload)
    if not geometry["window_binds"]:
        raise AssertionError(f"benchmark workload does not bind W: {geometry}")
    if not geometry["multiple_pair_blocks"]:
        raise AssertionError(f"benchmark workload has only one pair block: {geometry}")
    if not geometry["token_budget_splits"]:
        raise AssertionError(
            f"benchmark workload has no token-budget split: {geometry}"
        )
    return geometry


def run_arm(workload: Workload, streaming: bool, repetition: int) -> dict[str, Any]:
    """Run one arm and return its correctness and timing receipt."""
    client = _new_client(workload, seed=repetition)
    previous_stream_requests = Join.stream_requests
    previous_batch_size = BaseOperator.request_batch_size
    previous_tqdm = model_client_module.tqdm
    events: list[Any] = []
    client.set_request_lifecycle_collector(
        events.append,
        execution_id=f"{'streaming' if streaming else 'standard'}-{repetition}",
    )
    try:
        # The adapter propagates this class opt-in to each constructed Predicate.
        Join.stream_requests = streaming
        # Keep the benchmark's explicit W contract visible: the adapter's
        # Predicate inherits this batch size when it enters iter_completions.
        BaseOperator.request_batch_size = workload.batch_size
        model_client_module.tqdm = _silent_tqdm
        join = _join(LanguageModel(client), workload)
        started = time.monotonic()
        result = join.execute()
        wall = time.monotonic() - started
        metrics = client.get_metrics()
        counts = Counter(event.event for event in events)
        live_requests = 0
        max_live_requests = 0
        for event in sorted(events, key=lambda item: item.timestamp_ns):
            if event.event == "queued":
                live_requests += 1
                max_live_requests = max(max_live_requests, live_requests)
            elif event.event in {"settled", "failed"}:
                live_requests -= 1
        if len(result) != workload.expected_requests:
            raise AssertionError(
                "join result count diverged from pair geometry: "
                f"expected={workload.expected_requests}, actual={len(result)}"
            )
        if metrics.num_requests != workload.expected_requests:
            raise AssertionError(
                "simulated request count diverged from pair geometry: "
                f"expected={workload.expected_requests}, "
                f"actual={metrics.num_requests}"
            )
        if live_requests != 0:
            raise AssertionError(
                "lifecycle accounting left requests unsettled: "
                f"live_requests={live_requests}"
            )
        return {
            "arm": "streaming" if streaming else "standard",
            "repetition": repetition,
            "wall_seconds": wall,
            "result_rows": len(result),
            "result_hash": hashlib.sha256(
                json.dumps(
                    result.to_dicts(),
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                ).encode()
            ).hexdigest(),
            "request_count": metrics.num_requests,
            "output_tokens": metrics.num_output_tokens,
            "lifecycle_counts": dict(sorted(counts.items())),
            "simulated_429": sum(
                1 for event in client.trace if event[0] == "server_429"
            ),
            "dispatch_count": sum(
                1 for event in client.trace if event[0] == "dispatch"
            ),
            "max_live_requests": max_live_requests,
        }
    finally:
        Join.stream_requests = previous_stream_requests
        BaseOperator.request_batch_size = previous_batch_size
        model_client_module.tqdm = previous_tqdm
        client.shutdown()


def _summary(samples: list[float]) -> dict[str, Any]:
    median = statistics.median(samples)
    mad = statistics.median(abs(sample - median) for sample in samples)
    return {
        "median_seconds": median,
        "mad_seconds": mad,
        "mad_band_seconds": [median - mad, median + mad],
        "min_seconds": min(samples),
        "max_seconds": max(samples),
        "spread_seconds": max(samples) - min(samples),
    }


def run(workload: Workload) -> dict[str, Any]:
    """Run all interleaved repetitions and summarize spread-aware timings."""
    geometry = assert_workload_geometry(workload)

    receipts: list[dict[str, Any]] = []
    for repetition in range(1, workload.repetitions + 1):
        arms = (False, True) if repetition % 2 else (True, False)
        for streaming in arms:
            receipts.append(run_arm(workload, streaming, repetition))

    by_arm = {
        arm: _summary(
            [
                float(receipt["wall_seconds"])
                for receipt in receipts
                if receipt["arm"] == arm
            ]
        )
        for arm in ("standard", "streaming")
    }
    standard_high_water = [
        int(receipt["max_live_requests"])
        for receipt in receipts
        if receipt["arm"] == "standard"
    ]
    streaming_high_water = [
        int(receipt["max_live_requests"])
        for receipt in receipts
        if receipt["arm"] == "streaming"
    ]
    admission_behavior_differs = all(
        value > workload.watermark for value in standard_high_water
    ) and all(value <= workload.watermark for value in streaming_high_water)
    if not admission_behavior_differs:
        raise AssertionError(
            "benchmark arms did not exhibit distinct admission bounds: "
            f"standard={standard_high_water}, streaming={streaming_high_water}, "
            f"W={workload.watermark}"
        )
    delta = (
        by_arm["streaming"]["median_seconds"] / by_arm["standard"]["median_seconds"] - 1
    ) * 100
    standard_band = by_arm["standard"]["mad_band_seconds"]
    streaming_band = by_arm["streaming"]["mad_band_seconds"]
    bands_overlap = max(standard_band[0], streaming_band[0]) <= min(
        standard_band[1], streaming_band[1]
    )
    if delta <= 20:
        verdict = "PASS"
    elif bands_overlap:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "FAIL"
    return {
        "provider_calls": 0,
        "clock": "real",
        "arms": "interleaved and alternated within each repetition",
        "workload": geometry,
        "admission_high_water": {
            "standard": standard_high_water,
            "streaming": streaming_high_water,
        },
        "admission_behavior_differs": admission_behavior_differs,
        "summaries": by_arm,
        "streaming_delta_percent": delta,
        "mad_bands_overlap": bands_overlap,
        "evidence_verdict": verdict,
        "receipts": receipts,
    }


def main() -> None:
    """Parse benchmark options and print a JSON evidence receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--left-rows", type=int, default=DEFAULT_LEFT_ROWS)
    parser.add_argument("--right-rows", type=int, default=DEFAULT_RIGHT_ROWS)
    args = parser.parse_args()
    if args.repetitions < 3:
        parser.error("--repetitions must be at least 3 for dispersion")
    result = run(
        Workload(
            left_rows=args.left_rows,
            right_rows=args.right_rows,
            repetitions=args.repetitions,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["evidence_verdict"] not in {"PASS", "OBSERVATIONAL"}:
        raise SystemExit(f"benchmark verdict: {result['evidence_verdict']}")


if __name__ == "__main__":
    main()
