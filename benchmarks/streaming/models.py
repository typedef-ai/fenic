# ruff: noqa: D101,D102,D103
"""Validated models and calculations for the streaming benchmark."""

from __future__ import annotations

import hashlib
import json
import random
import socket
import statistics
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
    reserve_fraction: float

    def estimate(
        self,
        physical_requests: int,
        input_tokens_per_request: int,
        output_tokens_per_request: int,
    ) -> float:
        direct = (
            physical_requests
            * (
                input_tokens_per_request * self.input_per_million_usd
                + output_tokens_per_request * self.output_per_million_usd
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
    estimated_input_tokens: int
    max_output_tokens: int
    input_profile: str
    output_profile: str
    output_schema: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class Scenario:
    id: str
    kind: str
    execution_mode: str
    execution_shape: str
    steps: tuple[ScenarioStep, ...]

    @property
    def operation(self) -> str:
        return self.steps[0].operator


@dataclass(frozen=True)
class MatrixLimits:
    client_rpm: int
    client_tpm: int
    cell_timeout_seconds: float
    max_cost_usd: float
    max_rows_per_input: int
    max_physical_requests: int
    max_input_tokens_per_request: int
    max_output_tokens_per_request: int
    max_batch_size: int
    max_repetitions: int


@dataclass(frozen=True)
class Matrix:
    schema_version: str
    scenario_version: str
    model_alias: str
    model_name: str
    limits: MatrixLimits
    arms: tuple[str, ...]
    repetitions: int
    interleaving_seed: int
    workload_shapes: tuple[tuple[int, int, int], ...]
    batch_sizes: tuple[int, ...]
    input_seed: int
    pricing: Pricing
    scenarios: tuple[Scenario, ...]

    @property
    def cell_count(self) -> int:
        return len(expand_cells(self))


@dataclass(frozen=True)
class Cell:
    id: str
    scenario_id: str
    scenario_kind: str
    execution_mode: str
    execution_shape: str
    operation: str
    prompt_template: str
    estimated_input_tokens: int
    max_output_tokens: int
    arm: str
    rows: int
    right_rows: int
    unique_inputs: int
    batch_size: int
    repetition: int
    input_seed: int
    checkout: str = "candidate"

    @property
    def cache_heavy(self) -> bool:
        return self.operation != "join" and self.unique_inputs < self.rows

    @property
    def physical_requests(self) -> int:
        if self.operation == "join":
            return self.rows * self.right_rows
        return self.unique_inputs if self.cache_heavy else self.rows

    @property
    def expected_result_count(self) -> int:
        return self.physical_requests if self.operation == "join" else self.rows

    @property
    def comparison_key(self) -> tuple[Any, ...]:
        return (
            self.scenario_id,
            self.rows,
            self.right_rows,
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
        estimated_input_tokens=int(raw["estimated_input_tokens"]),
        max_output_tokens=int(raw["max_output_tokens"]),
        input_profile=str(raw["input_profile"]),
        output_profile=str(raw["output_profile"]),
        output_schema=raw.get("output_schema"),
    )


def parse_matrix(raw: Mapping[str, Any]) -> Matrix:
    """Parse a schema-valid matrix and enforce cross-field limits."""
    _require(raw.get("schema_version") == SCHEMA_VERSION, "unsupported schema_version")
    _require(
        raw.get("arms") == ["standard", "streaming"],
        "arms must be standard then streaming",
    )
    limits_raw = raw["limits"]
    limits = MatrixLimits(
        client_rpm=int(limits_raw["client_rpm"]),
        client_tpm=int(limits_raw["client_tpm"]),
        cell_timeout_seconds=float(limits_raw["cell_timeout_seconds"]),
        max_cost_usd=float(limits_raw["max_cost_usd"]),
        max_rows_per_input=int(limits_raw["max_rows_per_input"]),
        max_physical_requests=int(limits_raw["max_physical_requests"]),
        max_input_tokens_per_request=int(limits_raw["max_input_tokens_per_request"]),
        max_output_tokens_per_request=int(limits_raw["max_output_tokens_per_request"]),
        max_batch_size=int(limits_raw["max_batch_size"]),
        max_repetitions=int(limits_raw["max_repetitions"]),
    )
    _require(1 <= limits.client_rpm <= 10_000, "client_rpm is outside the schema bound")
    _require(
        1 <= limits.client_tpm <= 10_000_000, "client_tpm is outside the schema bound"
    )
    _require(
        0 < limits.cell_timeout_seconds <= 1_800,
        "cell timeout is outside the schema bound",
    )
    _require(0 < limits.max_cost_usd <= 50, "cost cap is outside the schema bound")
    _require(
        1 <= limits.max_rows_per_input <= 10_000,
        "row limit is outside the schema bound",
    )
    _require(
        1 <= limits.max_physical_requests <= 100_000,
        "request limit is outside the schema bound",
    )
    _require(
        1 <= limits.max_input_tokens_per_request <= 8_192,
        "input-token limit is outside the schema bound",
    )
    _require(
        1 <= limits.max_output_tokens_per_request <= 4_096,
        "output-token limit is outside the schema bound",
    )
    _require(
        1 <= limits.max_batch_size <= 1_000, "batch limit is outside the schema bound"
    )
    _require(
        3 <= limits.max_repetitions <= 9, "repetition limit is outside the schema bound"
    )
    repetitions = int(raw["repetitions"])
    _require(
        3 <= repetitions <= limits.max_repetitions, "repetitions exceed declared limits"
    )

    workload = raw["workload"]
    shapes = tuple(
        (
            int(shape["rows"]),
            int(shape.get("right_rows", 1)),
            int(shape["unique_inputs"]),
        )
        for shape in workload["shapes"]
    )
    for rows, right_rows, unique_inputs in shapes:
        _require(rows <= limits.max_rows_per_input, "rows exceed declared limits")
        _require(
            right_rows <= limits.max_rows_per_input, "right_rows exceed declared limits"
        )
        _require(
            0 < unique_inputs <= rows, "unique_inputs must be between one and rows"
        )
        _require(
            rows * right_rows <= limits.max_physical_requests,
            "shape exceeds physical-request limit",
        )
    batch_sizes = tuple(int(value) for value in workload["batch_sizes"])
    _require(
        all(value <= limits.max_batch_size for value in batch_sizes),
        "batch size exceeds declared limits",
    )

    scenarios = []
    for item in raw["scenarios"]:
        steps = tuple(_step_from_dict(step) for step in item["steps"])
        mode = str(item["execution_mode"])
        kind = str(item["kind"])
        _require(kind in {"operator", "chain"}, "unknown scenario kind")
        _require(
            mode in {"disabled", "simulated", "provider"}, "unknown execution mode"
        )
        if kind == "operator":
            _require(len(steps) == 1, "operator scenarios have one step")
        else:
            _require(len(steps) in {2, 3}, "chains must contain two or three steps")
        if mode != "disabled":
            _require(
                kind == "operator" and steps[0].operator == "join",
                "only semantic.join is executable",
            )
        for step in steps:
            _require(
                step.estimated_input_tokens <= limits.max_input_tokens_per_request,
                "estimated input tokens exceed declared limits",
            )
            _require(
                step.max_output_tokens <= limits.max_output_tokens_per_request,
                "max output tokens exceed declared limits",
            )
        scenarios.append(
            Scenario(
                id=str(item["id"]),
                kind=kind,
                execution_mode=mode,
                execution_shape=str(item.get("execution_shape", "single")),
                steps=steps,
            )
        )
    _require(
        any(s.execution_mode != "disabled" for s in scenarios),
        "matrix has no executable scenario",
    )

    pricing_raw = raw["pricing"]
    _require(
        float(pricing_raw["input_per_million_usd"]) > 0,
        "input pricing must be positive",
    )
    _require(
        float(pricing_raw["cached_input_per_million_usd"]) > 0,
        "cached-input pricing must be positive",
    )
    _require(
        float(pricing_raw["output_per_million_usd"]) > 0,
        "output pricing must be positive",
    )
    _require(
        0 <= float(pricing_raw["reserve_fraction"]) <= 1,
        "reserve fraction is outside the schema bound",
    )
    matrix = Matrix(
        schema_version=str(raw["schema_version"]),
        scenario_version=str(raw["scenario_version"]),
        model_alias=str(raw["model"]["alias"]),
        model_name=str(raw["model"]["name"]),
        limits=limits,
        arms=tuple(raw["arms"]),
        repetitions=repetitions,
        interleaving_seed=int(raw["interleaving_seed"]),
        workload_shapes=shapes,
        batch_sizes=batch_sizes,
        input_seed=int(workload["input_seed"]),
        pricing=Pricing(
            input_per_million_usd=float(pricing_raw["input_per_million_usd"]),
            cached_input_per_million_usd=float(
                pricing_raw["cached_input_per_million_usd"]
            ),
            output_per_million_usd=float(pricing_raw["output_per_million_usd"]),
            reserve_fraction=float(pricing_raw["reserve_fraction"]),
        ),
        scenarios=tuple(scenarios),
    )
    for cell in expand_cells(matrix):
        _require(
            cell.physical_requests <= limits.max_physical_requests,
            "cell exceeds physical-request limit",
        )
    return matrix


def validate_matrix_document(
    document: Mapping[str, Any], schema_path: Path | None = None
) -> None:
    """Validate JSON shape and semantic constraints."""
    schema_path = schema_path or Path(__file__).with_name("matrix.schema.json")
    try:
        import jsonschema
    except ImportError:  # pragma: no cover
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
        jsonschema.Draft202012Validator(json.loads(schema_path.read_text())).validate(
            document
        )
    parse_matrix(document)


def load_matrix(path: Path) -> Matrix:
    document = json.loads(path.read_text())
    validate_matrix_document(document, Path(__file__).with_name("matrix.schema.json"))
    return parse_matrix(document)


def expand_cells(matrix: Matrix, *, checkout: str = "candidate") -> list[Cell]:
    cells = []
    for scenario in matrix.scenarios:
        if scenario.execution_mode == "disabled":
            continue
        step = scenario.steps[0]
        for rows, right_rows, unique_inputs in matrix.workload_shapes:
            for batch_size in matrix.batch_sizes:
                for repetition in range(1, matrix.repetitions + 1):
                    for arm in matrix.arms:
                        key = (
                            scenario.id,
                            checkout,
                            rows,
                            right_rows,
                            unique_inputs,
                            batch_size,
                            repetition,
                            arm,
                        )
                        cells.append(
                            Cell(
                                id="-".join(str(part) for part in key),
                                scenario_id=scenario.id,
                                scenario_kind=scenario.kind,
                                execution_mode=scenario.execution_mode,
                                execution_shape=scenario.execution_shape,
                                operation=scenario.operation,
                                prompt_template=step.prompt_template,
                                estimated_input_tokens=step.estimated_input_tokens,
                                max_output_tokens=step.max_output_tokens,
                                arm=arm,
                                rows=rows,
                                right_rows=right_rows,
                                unique_inputs=unique_inputs,
                                batch_size=batch_size,
                                repetition=repetition,
                                input_seed=matrix.input_seed,
                                checkout=checkout,
                            )
                        )
    return cells


def interleave_cells(cells: Iterable[Cell], seed: int = 0) -> list[Cell]:
    grouped: dict[tuple[Any, ...], list[Cell]] = {}
    for cell in cells:
        grouped.setdefault(cell.comparison_key, []).append(cell)
    output = []
    for key in sorted(grouped, key=repr):
        group = grouped[key]
        random.Random(f"{seed}:{key!r}").shuffle(  # nosec B311 - deterministic arm ordering, not security
            group
        )
        output.extend(group)
    return output


def assert_interleaved_same_run(cells: Iterable[Cell]) -> None:
    groups: dict[tuple[Any, ...], dict[str, set[str]]] = {}
    for cell in cells:
        groups.setdefault(cell.comparison_key, {}).setdefault(cell.checkout, set()).add(
            cell.arm
        )
    for key, checkout_arms in groups.items():
        for checkout, arms in checkout_arms.items():
            if arms != {"standard", "streaming"}:
                raise ValueError(
                    f"comparison group {key!r} checkout {checkout!r} does not contain both arms"
                )


def cell_estimated_cost(matrix: Matrix, cell: Cell) -> float:
    if cell.execution_mode != "provider":
        return 0.0
    return matrix.pricing.estimate(
        cell.physical_requests,
        cell.estimated_input_tokens,
        cell.max_output_tokens,
    )


def projected_cost(matrix: Matrix, cells: Iterable[Cell]) -> float:
    return sum(cell_estimated_cost(matrix, cell) for cell in cells)


def projected_requests(cells: Iterable[Cell], *, provider_only: bool = False) -> int:
    return sum(
        cell.physical_requests
        for cell in cells
        if not provider_only or cell.execution_mode == "provider"
    )


def cost_within_cap(
    accounted_spend: float, remaining_estimate: float, cap: float
) -> bool:
    return accounted_spend + remaining_estimate <= cap


def require_metrics(metrics: Mapping[str, Any] | None) -> None:
    if not metrics:
        raise RuntimeError("provider-backed cell returned no LMMetrics")
    if float(metrics.get("cost", 0) or 0) <= 0:
        raise RuntimeError("provider-backed cell returned empty or zero cost")
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


def median(values: Iterable[float]) -> float:
    samples = [float(value) for value in values]
    if not samples:
        raise ValueError("at least one value is required")
    return float(statistics.median(samples))


def median_absolute_deviation(values: Iterable[float]) -> float:
    samples = [float(value) for value in values]
    center = median(samples)
    return median(abs(value - center) for value in samples)


def classify_comparison(
    candidate: Iterable[float],
    baseline: Iterable[float],
    *,
    cache_heavy: bool = False,
    correctness_ok: bool = True,
    rate_limit_events: int | None = None,
    path_engaged: bool = True,
) -> str:
    candidate_values = list(candidate)
    baseline_values = list(baseline)
    if not correctness_ok:
        return "FAIL"
    if cache_heavy:
        return "OBSERVATIONAL"
    if rate_limit_events is None or not path_engaged:
        return "REGIME_UNVERIFIED"
    if rate_limit_events:
        return "OUTSIDE_REGIME"
    if len(candidate_values) < 3 or len(baseline_values) < 3:
        return "INCONCLUSIVE"
    candidate_median = median(candidate_values)
    baseline_median = median(baseline_values)
    if baseline_median <= 0:
        return "INCONCLUSIVE"
    candidate_mad = median_absolute_deviation(candidate_values)
    baseline_mad = median_absolute_deviation(baseline_values)
    overlap = max(
        candidate_median - candidate_mad, baseline_median - baseline_mad
    ) <= min(candidate_median + candidate_mad, baseline_median + baseline_mad)
    slower = candidate_median / baseline_median - 1
    if slower > GATE_THRESHOLD and not overlap:
        return "FAIL"
    if slower > GATE_THRESHOLD:
        return "INCONCLUSIVE"
    return "PASS"


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
    reason: str = "request-lifecycle instrumentation is unavailable",
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
