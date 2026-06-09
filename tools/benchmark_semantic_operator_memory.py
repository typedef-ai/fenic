#!/usr/bin/env python3
"""Peak-memory benchmark harness for representative semantic operators.

This harness is intentionally opt-in and local-only. It runs each requested case
in an isolated child process, records the child process peak RSS via
``resource.getrusage(RUSAGE_SELF).ru_maxrss``, and emits copyable JSON evidence
that can be compared before/after a change. See
``tools/benchmark_semantic_operator_memory_protocol.md`` for the full benchmark
matrix, PR evidence protocol, and reviewer checklist.

Default benchmark matrix:
    ``--cases all`` runs ``sim_join``, ``semantic_reduce``, ``semantic_join``, and
    ``map_extract_chain`` using ``--rows 64 --right-rows 32 --groups 8
    --embedding-dimensions 8 --k 2``. These defaults are the starting point for
    evidence-grade local before/after runs.

Smoke-test boundary:
    Pytest-sized cases such as ``--rows 2`` prove that the harness works and does
    not call external providers. They are smoke tests only, not evidence for
    memory-improvement claims. Increase sizes when process/session startup RSS
    dominates the operator signal.

Polars allocation note:
    Polars does not currently expose a stable process peak allocator counter
    through the existing fenic test/benchmark tooling. This harness therefore
    treats peak RSS as the authoritative memory signal and reports Polars
    allocation as unavailable instead of inventing allocator precision.

Evidence examples (wrap as needed):
    uv run python tools/benchmark_semantic_operator_memory.py
        --json --label TD-XXXX-before-default
    uv run python tools/benchmark_semantic_operator_memory.py
        --json --label TD-XXXX-after-default
    uv run python tools/benchmark_semantic_operator_memory.py
        --cases sim_join --json --label TD-XXXX-before-sim-join
    uv run python tools/benchmark_semantic_operator_memory.py
        --cases semantic_reduce --json --label TD-XXXX-before-semantic-reduce
    uv run python tools/benchmark_semantic_operator_memory.py
        --cases semantic_join --json --label TD-XXXX-before-semantic-join
    uv run python tools/benchmark_semantic_operator_memory.py
        --cases map_extract_chain --json --label TD-XXXX-before-map-extract-chain
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import multiprocessing as mp
import os
import platform
import resource
import subprocess  # nosec B404 - used only to invoke the local git executable for metadata.
import sys
import tempfile
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

CaseName = Literal["sim_join", "semantic_reduce", "semantic_join", "map_extract_chain"]
ALL_CASES: tuple[CaseName, ...] = (
    "sim_join",
    "semantic_reduce",
    "semantic_join",
    "map_extract_chain",
)
RSS_SOURCE = "resource.getrusage(RUSAGE_SELF).ru_maxrss"
POLARS_ALLOCATION_NOTE = (
    "Polars does not expose a process peak allocator counter through the existing "
    "fenic benchmark/test tooling; peak RSS is reported as the stable comparable "
    "memory signal."
)


@dataclasses.dataclass(frozen=True)
class BenchmarkConfig:
    rows: int
    right_rows: int
    groups: int
    embedding_dimensions: int
    k: int
    label: str


def _ru_maxrss_to_bytes(ru_maxrss: int) -> int:
    # Linux reports KiB; macOS reports bytes. The worker host for this task is Linux.
    return ru_maxrss if sys.platform == "darwin" else ru_maxrss * 1024


def _peak_rss_bytes() -> int:
    return _ru_maxrss_to_bytes(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _git_value(args: list[str]) -> str | None:
    try:
        result = subprocess.run(  # nosec
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _patch_provider_validation() -> None:
    import fenic._backends.local.model_registry as model_registry

    async def _no_validate_provider_api_keys(providers: set[object]) -> None:
        return None

    model_registry._validate_provider_api_keys = _no_validate_provider_api_keys


def _configure_local_language_model(session: Any) -> None:
    from fenic._inference.types import FenicCompletionsResponse

    model = session._session_state.get_language_model()

    def count_tokens(value: object) -> int:
        if value is None:
            return 1
        if isinstance(value, str):
            return max(1, len(value.split()))
        return max(1, len(str(value).split()))

    def fake_get_completions(
        messages: list[object | None],
        max_tokens: int | None = None,
        temperature: float = 0,
        response_format: object | None = None,
        top_logprobs: int | None = None,
        model_profile: str | None = None,
        operation_name: str | None = None,
        request_timeout: float | None = None,
    ) -> list[FenicCompletionsResponse | None]:
        responses: list[FenicCompletionsResponse | None] = []
        for idx, message in enumerate(messages):
            if message is None:
                responses.append(None)
                continue
            user = getattr(message, "user", "") or ""
            if operation_name == "semantic.predicate":
                completion = json.dumps({"output": _deterministic_predicate(user)})
            elif operation_name and operation_name.startswith("semantic.reduce"):
                completion = f"summary-{idx}-{_stable_bucket(user)}"
            elif operation_name == "semantic.extract":
                completion = json.dumps(
                    {
                        "category": _stable_bucket(user),
                        "priority": (idx % 5) + 1,
                    }
                )
            else:
                completion = f"mapped {_stable_bucket(user)} :: {user[:80]}"
            responses.append(FenicCompletionsResponse(completion=completion, logprobs=None))
        return responses

    model.count_tokens = count_tokens
    model.get_completions = fake_get_completions


def _deterministic_predicate(rendered_prompt: str) -> bool:
    # Keep the semantic.join result bounded but non-empty without relying on a provider.
    return _stable_bucket(rendered_prompt) in {"alpha", "gamma"}


def _stable_bucket(value: str) -> str:
    buckets = ("alpha", "beta", "gamma", "delta")
    return buckets[sum(value.encode("utf-8")) % len(buckets)]


def _new_session(tmpdir: str, *, with_language_model: bool) -> Any:
    from fenic import OpenAILanguageModel, SemanticConfig, Session, SessionConfig

    _patch_provider_validation()
    # The configured client is patched before any semantic operation executes; this
    # dummy value only lets provider client construction run without external I/O.
    os.environ.setdefault("OPENAI_API_KEY", "local-semantic-memory-harness-no-network")

    semantic_config = None
    if with_language_model:
        semantic_config = SemanticConfig(
            language_models={
                "local": OpenAILanguageModel(
                    model_name="gpt-4.1-nano",
                    rpm=1_000_000,
                    tpm=1_000_000,
                )
            },
            default_language_model="local",
        )

    return Session.get_or_create(
        SessionConfig(
            app_name=f"semantic-memory-{os.getpid()}-{time.time_ns()}",
            db_path=Path(tmpdir),
            semantic=semantic_config,
        )
    )


def _embedding_vector(seed: int, dimensions: int) -> list[float]:
    return [float(((seed + 1) * (idx + 3)) % 17) / 17.0 for idx in range(dimensions)]


def _run_sim_join(config: BenchmarkConfig) -> dict[str, Any]:
    import polars as pl

    from fenic import EmbeddingType, col

    with tempfile.TemporaryDirectory() as tmpdir:
        session = _new_session(tmpdir, with_language_model=False)
        try:
            left = session.create_dataframe(
                pl.DataFrame(
                    {
                        "left_id": list(range(config.rows)),
                        "left_label": [f"left-{idx % 8}" for idx in range(config.rows)],
                        "left_vec": [
                            _embedding_vector(idx, config.embedding_dimensions)
                            for idx in range(config.rows)
                        ],
                    },
                    schema={
                        "left_id": pl.Int64,
                        "left_label": pl.String,
                        "left_vec": pl.List(pl.Float32),
                    },
                )
            ).with_column(
                "left_vec",
                col("left_vec").cast(
                    EmbeddingType(
                        dimensions=config.embedding_dimensions,
                        embedding_model="local-memory-harness",
                    )
                ),
            )
            right = session.create_dataframe(
                pl.DataFrame(
                    {
                        "right_id": list(range(config.right_rows)),
                        "right_label": [f"right-{idx % 8}" for idx in range(config.right_rows)],
                        "right_vec": [
                            _embedding_vector(idx, config.embedding_dimensions)
                            for idx in range(config.right_rows)
                        ],
                    },
                    schema={
                        "right_id": pl.Int64,
                        "right_label": pl.String,
                        "right_vec": pl.List(pl.Float32),
                    },
                )
            ).with_column(
                "right_vec",
                col("right_vec").cast(
                    EmbeddingType(
                        dimensions=config.embedding_dimensions,
                        embedding_model="local-memory-harness",
                    )
                ),
            )
            result = left.semantic.sim_join(
                right,
                left_on="left_vec",
                right_on="right_vec",
                k=config.k,
                similarity_metric="l2",
                similarity_score_column="distance",
            ).to_polars()
            return {
                "result_rows": len(result),
                "result_columns": result.columns,
            }
        finally:
            session.stop(skip_usage_summary=True)


def _run_semantic_reduce(config: BenchmarkConfig) -> dict[str, Any]:
    from fenic import col, semantic
    from fenic import sum as fenic_sum

    with tempfile.TemporaryDirectory() as tmpdir:
        session = _new_session(tmpdir, with_language_model=True)
        try:
            _configure_local_language_model(session)
            rows = max(config.rows, config.groups)
            source = session.create_dataframe(
                {
                    "bucket": [f"bucket-{idx % config.groups}" for idx in range(rows)],
                    "sort_key": list(range(rows)),
                    "notes": [
                        f"doc-{idx}: customer {idx % 5} reported memory signal {idx % 7}."
                        for idx in range(rows)
                    ],
                    "weight": [1 for _ in range(rows)],
                }
            )
            result = (
                source.group_by("bucket")
                .agg(
                    semantic.reduce(
                        "Summarize memory signals in order.",
                        col("notes"),
                        order_by=[col("sort_key")],
                    ).alias("summary"),
                    fenic_sum("weight").alias("weight_sum"),
                )
                .to_polars()
            )
            return {
                "result_rows": len(result),
                "result_columns": result.columns,
            }
        finally:
            session.stop(skip_usage_summary=True)


def _run_semantic_join(config: BenchmarkConfig) -> dict[str, Any]:
    from fenic import col

    with tempfile.TemporaryDirectory() as tmpdir:
        session = _new_session(tmpdir, with_language_model=True)
        try:
            _configure_local_language_model(session)
            left = session.create_dataframe(
                {
                    "left_id": list(range(config.rows)),
                    "course_name": [f"course-{idx % 6}" for idx in range(config.rows)],
                    "left_payload": [f"payload-left-{idx}" for idx in range(config.rows)],
                }
            )
            right = session.create_dataframe(
                {
                    "right_id": list(range(config.right_rows)),
                    "skill": [f"skill-{idx % 6}" for idx in range(config.right_rows)],
                    "right_payload": [f"payload-right-{idx}" for idx in range(config.right_rows)],
                }
            )
            result = left.semantic.join(
                right,
                "Does {{left_on}} correspond to {{right_on}}?",
                left_on=col("course_name"),
                right_on=col("skill"),
            ).to_polars()
            return {
                "result_rows": len(result),
                "result_columns": result.columns,
            }
        finally:
            session.stop(skip_usage_summary=True)


def _run_map_extract_chain(config: BenchmarkConfig) -> dict[str, Any]:
    from pydantic import BaseModel, Field

    from fenic import col, semantic

    class ExtractedSignal(BaseModel):
        category: str = Field(description="Stable synthetic category")
        priority: int = Field(description="Stable synthetic priority")

    with tempfile.TemporaryDirectory() as tmpdir:
        session = _new_session(tmpdir, with_language_model=True)
        try:
            _configure_local_language_model(session)
            source = session.create_dataframe(
                {
                    "record_id": list(range(config.rows)),
                    "description": [
                        f"record-{idx}: operator memory behavior for category {idx % 4}"
                        for idx in range(config.rows)
                    ],
                }
            )
            mapped = source.select(
                col("record_id"),
                semantic.map(
                    "Normalize this benchmark record: {{description}}",
                    description=col("description"),
                ).alias("normalized"),
            )
            result = mapped.select(
                col("record_id"),
                semantic.extract(col("normalized"), ExtractedSignal).alias("signal"),
            ).to_polars()
            return {
                "result_rows": len(result),
                "result_columns": result.columns,
            }
        finally:
            session.stop(skip_usage_summary=True)


CASE_RUNNERS: dict[CaseName, Callable[[BenchmarkConfig], dict[str, Any]]] = {
    "sim_join": _run_sim_join,
    "semantic_reduce": _run_semantic_reduce,
    "semantic_join": _run_semantic_join,
    "map_extract_chain": _run_map_extract_chain,
}


def _run_case_child(case: CaseName, config: BenchmarkConfig, queue: mp.Queue) -> None:
    started = time.perf_counter()
    try:
        details = CASE_RUNNERS[case](config)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        queue.put(
            {
                "ok": True,
                "case": case,
                "elapsed_ms": elapsed_ms,
                "peak_rss_bytes": _peak_rss_bytes(),
                "result_rows": details["result_rows"],
                "result_columns": details["result_columns"],
                "parameters": {
                    "rows": config.rows,
                    "right_rows": config.right_rows,
                    "groups": config.groups,
                    "embedding_dimensions": config.embedding_dimensions,
                    "k": config.k,
                    "network_calls": "disabled",
                },
            }
        )
    except Exception as exc:
        queue.put(
            {
                "ok": False,
                "case": case,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "peak_rss_bytes": _peak_rss_bytes(),
            }
        )


def _run_case(case: CaseName, config: BenchmarkConfig) -> dict[str, Any]:
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    process = ctx.Process(target=_run_case_child, args=(case, config, queue))
    process.start()
    process.join()
    if queue.empty():
        return {
            "ok": False,
            "case": case,
            "error": f"child exited with code {process.exitcode} before reporting",
            "peak_rss_bytes": 0,
        }
    result = queue.get()
    result["exit_code"] = process.exitcode
    return result


def _parse_cases(value: str) -> list[CaseName]:
    if value == "all":
        return list(ALL_CASES)
    selected: list[CaseName] = []
    for raw_case in value.split(","):
        case = raw_case.strip()
        if case not in ALL_CASES:
            raise argparse.ArgumentTypeError(
                f"unknown case {case!r}; choose from {', '.join(ALL_CASES)} or all"
            )
        selected.append(case)  # type: ignore[arg-type]
    if not selected:
        raise argparse.ArgumentTypeError("at least one case is required")
    return selected


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    config = BenchmarkConfig(
        rows=args.rows,
        right_rows=args.right_rows,
        groups=args.groups,
        embedding_dimensions=args.embedding_dimensions,
        k=args.k,
        label=args.label,
    )
    case_results = [_run_case(case, config) for case in args.cases]
    failures = [case for case in case_results if not case.get("ok")]
    payload = {
        "schema_version": 1,
        "label": args.label,
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "git": {
            "branch": _git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
            "commit": _git_value(["rev-parse", "HEAD"]),
            "merge_base_origin_main": _git_value(["merge-base", "HEAD", "origin/main"]),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "measurement": {
            "rss_source": RSS_SOURCE,
            "peak_rss_unit": "bytes",
            "polars_allocation_bytes": None,
            "polars_allocation_note": POLARS_ALLOCATION_NOTE,
            "process_isolation": "one spawned child process per case",
        },
        "cases": case_results,
    }
    if failures:
        payload["failures"] = failures
    return payload


def _format_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Semantic operator memory benchmark",
        "",
        f"Label: `{payload['label']}`",
        f"Generated: `{payload['generated_at']}`",
        f"Git commit: `{payload['git']['commit']}`",
        "",
        "Peak memory is reported as child-process peak RSS in bytes. Polars "
        "allocator peak bytes are unavailable in this harness and intentionally "
        "reported as null.",
        "",
        "| case | peak_rss_bytes | elapsed_ms | result_rows |",
        "| --- | ---: | ---: | ---: |",
    ]
    for case in payload["cases"]:
        lines.append(
            f"| {case['case']} | {case.get('peak_rss_bytes', 0)} | "
            f"{case.get('elapsed_ms', 'n/a')} | {case.get('result_rows', 'n/a')} |"
        )
    lines.extend(
        [
            "",
            "```json",
            json.dumps(payload, indent=2, sort_keys=True),
            "```",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--cases", type=_parse_cases, default=list(ALL_CASES))
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--right-rows", type=int, default=32)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--embedding-dimensions", type=int, default=8)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--label", default="local")
    parser.add_argument("--json", action="store_true", help="Emit raw JSON only.")
    args = parser.parse_args()

    if args.rows <= 0 or args.right_rows <= 0 or args.groups <= 0:
        raise SystemExit("--rows, --right-rows, and --groups must be positive")
    if args.embedding_dimensions <= 0 or args.k <= 0:
        raise SystemExit("--embedding-dimensions and --k must be positive")

    payload = run_benchmark(args)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(_format_markdown(payload))
    if payload.get("failures"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
