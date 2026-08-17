# ruff: noqa: D101,D102,D103
"""Provider-independent models and calculations for the streaming gate.

The benchmark deliberately keeps its input contract independent of fenic.  This
lets ``plan`` validate and price a run without importing a provider SDK or
creating a session.
"""

from __future__ import annotations

import hashlib
import json
import random
import socket
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

SCHEMA_VERSION = "streaming-benchmark.v1"
GATE_THRESHOLD = 0.20


@dataclass(frozen=True)
class Pricing:
    input_per_million_usd: float
    cached_input_per_million_usd: float
    output_per_million_usd: float
    input_tokens_per_request: int = 250
    output_tokens_per_request: int = 32
    reserve_fraction: float = 0.15

    def estimate(self, physical_requests: int) -> float:
        direct = (
            physical_requests
            * (
                self.input_tokens_per_request * self.input_per_million_usd
                + self.output_tokens_per_request * self.output_per_million_usd
            )
            / 1_000_000
        )
        return direct * (1 + self.reserve_fraction)


@dataclass(frozen=True)
class ScenarioStep:
    operator: str
    input_columns: tuple[str, ...]
    output_column: str
    prompt_template: str
    max_output_tokens: int = 32
    input_profile: str = "deterministic"
    output_profile: str = "deterministic"
    output_schema: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class Scenario:
    id: str
    kind: str
    steps: tuple[ScenarioStep, ...]
    provider_execution: bool
    execution_shape: str = "single"

    @property
    def operation(self) -> str:
        return self.steps[0].operator


@dataclass(frozen=True)
class Matrix:
    schema_version: str
    scenario_version: str
    model_alias: str
    model_name: str
    client_rpm: int
    client_tpm: int
    arms: tuple[str, ...]
    repetitions: int
    interleaving_seed: int
    row_counts: tuple[int, ...]
    unique_input_counts: tuple[int, ...]
    batch_sizes: tuple[int, ...]
    input_seed: int
    cell_timeout_seconds: float
    max_cost_usd: float
    pricing: Pricing
    scenarios: tuple[Scenario, ...]
    workload_shapes: tuple[tuple[int, int], ...] = ()

    @property
    def cell_count(self) -> int:
        return len(expand_cells(self))


@dataclass(frozen=True)
class Cell:
    id: str
    scenario_id: str
    scenario_kind: str
    execution_shape: str
    operation: str
    prompt_template: str
    max_output_tokens: int
    arm: str
    rows: int
    unique_inputs: int
    batch_size: int
    repetition: int
    input_seed: int
    provider_execution: bool
    checkout: str = "candidate"

    @property
    def cache_heavy(self) -> bool:
        return self.unique_inputs < self.rows

    @property
    def physical_requests(self) -> int:
        return self.unique_inputs if self.cache_heavy else self.rows

    @property
    def comparison_key(self) -> tuple[Any, ...]:
        return (
            self.scenario_id,
            self.rows,
            self.unique_inputs,
            self.batch_size,
            self.repetition,
        )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _step_from_dict(raw: Mapping[str, Any]) -> ScenarioStep:
    return ScenarioStep(
        operator=str(raw["operator"]),
        input_columns=tuple(raw["input_columns"]),
        output_column=str(raw["output_column"]),
        prompt_template=str(raw["prompt_template"]),
        max_output_tokens=int(raw.get("max_output_tokens", 32)),
        input_profile=str(raw.get("input_profile", "deterministic")),
        output_profile=str(raw.get("output_profile", "deterministic")),
        output_schema=raw.get("output_schema"),
    )


def parse_matrix(raw: Mapping[str, Any]) -> Matrix:
    """Parse a validated matrix and apply semantic constraints."""
    _require(raw.get("schema_version") == SCHEMA_VERSION, "unsupported schema_version")
    _require(
        raw.get("arms") == ["standard", "streaming"],
        "arms must be standard then streaming",
    )
    _require(
        int(raw.get("repetitions", 0)) >= 3,
        "the gate requires at least three repetitions",
    )
    model = raw.get("model") or {}
    limits = raw.get("limits") or {}
    workload = raw.get("workload") or {}
    pricing_raw = raw.get("pricing") or {}
    scenarios = []
    for item in raw.get("scenarios", []):
        steps = tuple(_step_from_dict(step) for step in item["steps"])
        _require(item["kind"] in {"operator", "chain"}, "unknown scenario kind")
        if item["kind"] == "operator":
            _require(len(steps) == 1, "operator scenarios have one step")
            if item.get("provider_execution", False):
                _require(
                    steps[0].operator in {"map", "predicate"},
                    "provider-enabled operator scenarios support map or predicate",
                )
        else:
            _require(len(steps) in {2, 3}, "chains must contain two or three steps")
            _require(
                item.get("execution_shape")
                in {"barriered", "unfused_unbarriered", "fused_unbarriered"},
                "invalid chain shape",
            )
        scenarios.append(
            Scenario(
                id=str(item["id"]),
                kind=str(item["kind"]),
                steps=steps,
                provider_execution=bool(item.get("provider_execution", False)),
                execution_shape=str(item.get("execution_shape", "single")),
            )
        )
    _require(scenarios, "at least one scenario is required")
    row_counts = tuple(int(value) for value in workload["row_counts"])
    unique_counts = tuple(int(value) for value in workload["unique_input_counts"])
    _require(all(value > 0 for value in row_counts), "row counts must be positive")
    _require(
        all(0 < value <= max(row_counts) for value in unique_counts),
        "unique input counts are invalid",
    )
    _require(
        all(int(value) > 0 for value in workload["batch_sizes"]),
        "batch sizes must be positive",
    )
    shapes = tuple(
        (int(shape["rows"]), int(shape["unique_inputs"]))
        for shape in workload.get("shapes", [])
    )
    _require(
        all(rows > 0 and 0 < unique <= rows for rows, unique in shapes),
        "workload shapes are invalid",
    )
    return Matrix(
        schema_version=str(raw["schema_version"]),
        scenario_version=str(raw["scenario_version"]),
        model_alias=str(model["alias"]),
        model_name=str(model["name"]),
        client_rpm=int(limits["client_rpm"]),
        client_tpm=int(limits["client_tpm"]),
        arms=tuple(raw["arms"]),
        repetitions=int(raw["repetitions"]),
        interleaving_seed=int(raw["interleaving_seed"]),
        row_counts=row_counts,
        unique_input_counts=unique_counts,
        batch_sizes=tuple(int(value) for value in workload["batch_sizes"]),
        input_seed=int(workload["input_seed"]),
        cell_timeout_seconds=float(limits["cell_timeout_seconds"]),
        max_cost_usd=float(limits["max_cost_usd"]),
        pricing=Pricing(
            input_per_million_usd=float(pricing_raw["input_per_million_usd"]),
            cached_input_per_million_usd=float(
                pricing_raw["cached_input_per_million_usd"]
            ),
            output_per_million_usd=float(pricing_raw["output_per_million_usd"]),
            input_tokens_per_request=int(
                pricing_raw.get("input_tokens_per_request", 250)
            ),
            output_tokens_per_request=int(
                pricing_raw.get("output_tokens_per_request", 32)
            ),
            reserve_fraction=float(pricing_raw.get("reserve_fraction", 0.15)),
        ),
        scenarios=tuple(scenarios),
        workload_shapes=shapes,
    )


def validate_matrix_document(
    document: Mapping[str, Any], schema_path: Path | None = None
) -> None:
    """Validate JSON shape and semantic version using the repository schema."""
    if schema_path is None:
        schema_path = Path(__file__).with_name("matrix.schema.json")
    try:
        import jsonschema
    except (
        ImportError
    ):  # pragma: no cover - minimal environments still get semantic checks below
        required = {
            "schema_version",
            "scenario_version",
            "model",
            "limits",
            "arms",
            "repetitions",
            "interleaving_seed",
            "workload",
            "pricing",
            "scenarios",
        }
        missing = required.difference(document)
        if missing:
            raise ValueError(
                f"matrix is missing required fields: {sorted(missing)}"
            ) from None
    else:
        schema = json.loads(schema_path.read_text())
        jsonschema.Draft202012Validator(schema).validate(document)
    parse_matrix(document)


def load_matrix(path: Path) -> Matrix:
    document = json.loads(path.read_text())
    schema_path = path.with_name("matrix.schema.json")
    if not schema_path.exists():
        schema_path = Path(__file__).with_name("matrix.schema.json")
    validate_matrix_document(document, schema_path)
    return parse_matrix(document)


def expand_cells(matrix: Matrix, *, checkout: str = "candidate") -> list[Cell]:
    cells = []
    for scenario in matrix.scenarios:
        if not scenario.provider_execution:
            continue
        shapes = matrix.workload_shapes or tuple(
            (rows, unique_inputs)
            for rows in matrix.row_counts
            for unique_inputs in matrix.unique_input_counts
        )
        for rows, unique_inputs in shapes:
            if unique_inputs > rows:
                continue
            for batch_size in matrix.batch_sizes:
                for repetition in range(1, matrix.repetitions + 1):
                    for arm in matrix.arms:
                        key = (
                            scenario.id,
                            checkout,
                            rows,
                            unique_inputs,
                            batch_size,
                            repetition,
                            arm,
                        )
                        cell_id = "-".join(str(part) for part in key)
                        cells.append(
                            Cell(
                                id=cell_id,
                                scenario_id=scenario.id,
                                scenario_kind=scenario.kind,
                                execution_shape=scenario.execution_shape,
                                operation=scenario.operation,
                                prompt_template=scenario.steps[0].prompt_template,
                                max_output_tokens=scenario.steps[0].max_output_tokens,
                                arm=arm,
                                rows=rows,
                                unique_inputs=unique_inputs,
                                batch_size=batch_size,
                                repetition=repetition,
                                input_seed=matrix.input_seed,
                                provider_execution=scenario.provider_execution,
                                checkout=checkout,
                            )
                        )
    return cells


def interleave_cells(cells: Iterable[Cell], seed: int = 0) -> list[Cell]:
    """Interleave comparison arms while retaining deterministic cell groups."""
    grouped: dict[tuple[Any, ...], list[Cell]] = {}
    for cell in cells:
        grouped.setdefault(cell.comparison_key, []).append(cell)
    output: list[Cell] = []
    for key in sorted(grouped, key=repr):
        group = grouped[key]
        rng = random.Random(f"{seed}:{key!r}")
        rng.shuffle(group)
        output.extend(group)
    return output


def assert_interleaved_same_run(cells: Iterable[Cell]) -> None:
    """Require both arms of every comparison to be present in one run."""
    groups: dict[tuple[Any, ...], dict[str, set[str]]] = {}
    for cell in cells:
        checkout_arms = groups.setdefault(cell.comparison_key, {})
        checkout_arms.setdefault(cell.checkout, set()).add(cell.arm)
    for key, checkout_arms in groups.items():
        for checkout, arms in checkout_arms.items():
            if arms != {"standard", "streaming"}:
                raise ValueError(
                    f"comparison group {key!r} checkout {checkout!r} does not contain both arms"
                )


def projected_cost(matrix: Matrix, cells: Iterable[Cell]) -> float:
    # Treat every input token as uncached for a conservative cap estimate. A
    # cache hit may lower actual spend, but must never enlarge the run budget.
    return sum(matrix.pricing.estimate(cell.physical_requests) for cell in cells)


def projected_requests(cells: Iterable[Cell]) -> int:
    return sum(cell.physical_requests for cell in cells)


def cost_within_cap(actual_spend: float, remaining_estimate: float, cap: float) -> bool:
    return actual_spend + remaining_estimate <= cap


def require_metrics(metrics: Mapping[str, Any] | None) -> None:
    """Reject absent/zero provider metrics; zero is never treated as free."""
    if not metrics:
        raise RuntimeError("provider-backed cell returned no LMMetrics")
    if "cost" not in metrics or metrics.get("cost") in (None, 0):
        raise RuntimeError("provider-backed cell returned empty/zero LMMetrics")
    if int(metrics.get("num_requests", 0) or 0) <= 0:
        raise RuntimeError("provider-backed cell returned no request count")
    token_count = sum(
        int(metrics.get(name, 0) or 0)
        for name in (
            "num_uncached_input_tokens",
            "num_cached_input_tokens",
            "num_output_tokens",
        )
    )
    if token_count <= 0:
        raise RuntimeError("provider-backed cell returned no token counts")


def stamp_receipt(
    receipt: Mapping[str, Any],
    *,
    tested_commit: str,
    cumulative_actual_spend_usd: float,
    physical_requests: int,
) -> dict[str, Any]:
    """Add immutable run identity and spend fields to a child-process receipt."""
    stamped = dict(receipt)
    stamped["tested_commit"] = tested_commit
    stamped["cumulative_actual_spend_usd"] = cumulative_actual_spend_usd
    stamped["physical_requests"] = physical_requests
    return stamped


def median_absolute_deviation(values: Iterable[float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("at least one value is required")
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2
    )
    deviations = sorted(abs(value - median) for value in ordered)
    middle = len(deviations) // 2
    return (
        deviations[middle]
        if len(deviations) % 2
        else (deviations[middle - 1] + deviations[middle]) / 2
    )


def classify_comparison(
    candidate: Iterable[float],
    baseline: Iterable[float],
    *,
    cache_heavy: bool = False,
    correctness_ok: bool = True,
    rate_limit_events: int | None = None,
) -> str:
    """Return PASS, FAIL, INCONCLUSIVE, or observational regime states."""
    candidate_values = list(candidate)
    baseline_values = list(baseline)
    if not correctness_ok:
        return "FAIL"
    if cache_heavy:
        return "OBSERVATIONAL"
    if rate_limit_events is None:
        return "REGIME_UNVERIFIED"
    if rate_limit_events:
        return "OUTSIDE_REGIME"
    if len(candidate_values) < 3 or len(baseline_values) < 3:
        return "INCONCLUSIVE"
    candidate_median = sorted(candidate_values)[len(candidate_values) // 2]
    baseline_median = sorted(baseline_values)[len(baseline_values) // 2]
    if baseline_median <= 0:
        return "INCONCLUSIVE"
    candidate_mad = median_absolute_deviation(candidate_values)
    baseline_mad = median_absolute_deviation(baseline_values)
    candidate_band = (
        candidate_median - candidate_mad,
        candidate_median + candidate_mad,
    )
    baseline_band = (baseline_median - baseline_mad, baseline_median + baseline_mad)
    overlap = max(candidate_band[0], baseline_band[0]) <= min(
        candidate_band[1], baseline_band[1]
    )
    slower = candidate_median / baseline_median - 1
    if slower > GATE_THRESHOLD and not overlap:
        return "FAIL"
    if slower > GATE_THRESHOLD:
        return "INCONCLUSIVE"
    return "PASS"


def deterministic_inputs(rows: int, unique_inputs: int, seed: int) -> list[str]:
    _require(rows > 0, "rows must be positive")
    _require(0 < unique_inputs <= rows, "unique_inputs must be between one and rows")
    return [
        f"benchmark record {index % unique_inputs:04d} group {(index % unique_inputs) % 7} seed {seed}"
        for index in range(rows)
    ]


def result_hash(values: Iterable[Any]) -> str:
    encoded = json.dumps(
        list(values), sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def environment_metadata() -> dict[str, Any]:
    return {
        "host": socket.gethostname(),
        "platform": __import__("platform").platform(),
        "python": __import__("platform").python_version(),
        "captured_at": datetime.now(UTC).isoformat(),
    }


def lifecycle_unavailable(
    reason: str = "idle-gap collector is not available in this landing",
) -> dict[str, Any]:
    return {
        "availability": {
            "event_counts": {"available": False, "reason": reason},
            "idle_gap": {"available": False, "reason": reason},
            "max_queue_depth": {"available": False, "reason": reason},
            "rate_limit_events": {"available": False, "reason": reason},
        },
        "event_counts": None,
        "max_queue_depth": None,
        "idle_gap": None,
        "rate_limit_events": None,
    }


def as_jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {key: as_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): as_jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [as_jsonable(item) for item in value]
    return value
