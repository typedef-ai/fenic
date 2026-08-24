# ruff: noqa: D101,D102,D103
"""Validated models and verdict rules for the provider-free benchmark."""

from __future__ import annotations

import hashlib
import json
import socket
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import jsonschema

SCHEMA_VERSION = "streaming-benchmark.v1"
GATE_THRESHOLD = 0.20


@dataclass(frozen=True)
class ScenarioStep:
    operator: str
    input_columns: tuple[str, ...]
    output_column: str
    prompt_template: str
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
class WorkloadShape:
    rows: int
    right_rows: int
    unique_inputs: int
    pair_block_size: int
    block_token_budget: int
    rpm: int
    latency_seconds: float


@dataclass(frozen=True)
class MatrixLimits:
    cell_timeout_seconds: float
    max_rows_per_input: int
    max_physical_requests: int
    max_batch_size: int
    max_pair_block_size: int
    max_block_token_budget: int
    max_repetitions: int


@dataclass(frozen=True)
class Matrix:
    schema_version: str
    scenario_version: str
    limits: MatrixLimits
    arms: tuple[str, ...]
    repetitions: int
    interleaving_seed: int
    workload_shapes: tuple[WorkloadShape, ...]
    batch_sizes: tuple[int, ...]
    input_seed: int
    scenarios: tuple[Scenario, ...]

@dataclass(frozen=True)
class Cell:
    id: str
    scenario_id: str
    scenario_kind: str
    execution_mode: str
    execution_shape: str
    operation: str
    step: Mapping[str, Any]
    arm: str
    rows: int
    right_rows: int
    unique_inputs: int
    pair_block_size: int
    block_token_budget: int
    rpm: int
    latency_seconds: float
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
            self.pair_block_size,
            self.block_token_budget,
            self.rpm,
            self.latency_seconds,
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
        input_profile=str(raw["input_profile"]),
        output_profile=str(raw["output_profile"]),
        output_schema=raw.get("output_schema"),
    )


def parse_matrix(raw: Mapping[str, Any]) -> Matrix:
    """Parse a schema-valid provider-free matrix and enforce cross-field limits."""
    _require(raw.get("schema_version") == SCHEMA_VERSION, "unsupported schema_version")
    _require(
        raw.get("arms") == ["standard", "streaming"],
        "arms must be standard then streaming",
    )
    limits_raw = raw["limits"]
    limits = MatrixLimits(
        cell_timeout_seconds=float(limits_raw["cell_timeout_seconds"]),
        max_rows_per_input=int(limits_raw["max_rows_per_input"]),
        max_physical_requests=int(limits_raw["max_physical_requests"]),
        max_batch_size=int(limits_raw["max_batch_size"]),
        max_pair_block_size=int(limits_raw["max_pair_block_size"]),
        max_block_token_budget=int(limits_raw["max_block_token_budget"]),
        max_repetitions=int(limits_raw["max_repetitions"]),
    )
    repetitions = int(raw["repetitions"])
    _require(
        3 <= repetitions <= limits.max_repetitions, "repetitions exceed declared limits"
    )

    workload = raw["workload"]
    shapes = tuple(
        WorkloadShape(
            rows=int(shape["rows"]),
            right_rows=int(shape["right_rows"]),
            unique_inputs=int(shape["unique_inputs"]),
            pair_block_size=int(shape["pair_block_size"]),
            block_token_budget=int(shape["block_token_budget"]),
            rpm=int(shape["rpm"]),
            latency_seconds=float(shape["latency_seconds"]),
        )
        for shape in workload["shapes"]
    )
    for shape in shapes:
        _require(shape.rows <= limits.max_rows_per_input, "rows exceed declared limits")
        _require(
            shape.right_rows <= limits.max_rows_per_input,
            "right_rows exceed declared limits",
        )
        _require(
            0 < shape.unique_inputs <= shape.rows,
            "unique_inputs must be between one and rows",
        )
        _require(
            shape.rows * shape.right_rows <= limits.max_physical_requests,
            "shape exceeds physical-request limit",
        )
        _require(
            shape.pair_block_size <= limits.max_pair_block_size,
            "pair block exceeds declared limit",
        )
        _require(
            shape.block_token_budget <= limits.max_block_token_budget,
            "token budget exceeds declared limit",
        )
    batch_sizes = tuple(int(value) for value in workload["batch_sizes"])
    _require(
        all(value <= limits.max_batch_size for value in batch_sizes),
        "batch size exceeds declared limit",
    )

    scenarios = []
    for item in raw["scenarios"]:
        steps = tuple(_step_from_dict(step) for step in item["steps"])
        mode = str(item["execution_mode"])
        kind = str(item["kind"])
        _require(mode in {"disabled", "simulated"}, "unknown execution mode")
        _require(kind in {"operator", "chain"}, "unknown scenario kind")
        _require(
            len(steps) == 1 if kind == "operator" else len(steps) in {2, 3},
            "scenario step count does not match its kind",
        )
        if mode == "simulated":
            _require(
                kind == "operator" and steps[0].operator == "join",
                "only semantic.join is executable",
            )
        scenarios.append(
            Scenario(
                id=str(item["id"]),
                kind=kind,
                execution_mode=mode,
                execution_shape=str(item["execution_shape"]),
                steps=steps,
            )
        )
    _require(
        any(item.execution_mode == "simulated" for item in scenarios),
        "matrix has no executable scenario",
    )
    return Matrix(
        schema_version=str(raw["schema_version"]),
        scenario_version=str(raw["scenario_version"]),
        limits=limits,
        arms=tuple(raw["arms"]),
        repetitions=repetitions,
        interleaving_seed=int(raw["interleaving_seed"]),
        workload_shapes=shapes,
        batch_sizes=batch_sizes,
        input_seed=int(workload["input_seed"]),
        scenarios=tuple(scenarios),
    )


def validate_matrix_document(
    document: Mapping[str, Any], schema_path: Path | None = None
) -> Matrix:
    """Require full JSON Schema validation before semantic parsing."""
    schema_path = schema_path or Path(__file__).with_name("matrix.schema.json")
    jsonschema.Draft202012Validator(json.loads(schema_path.read_text())).validate(
        document
    )
    return parse_matrix(document)


def load_matrix(path: Path) -> Matrix:
    document = json.loads(path.read_text())
    return validate_matrix_document(document)


def expand_cells(matrix: Matrix, *, checkout: str = "candidate") -> list[Cell]:
    cells = []
    for scenario in matrix.scenarios:
        if scenario.execution_mode == "disabled":
            continue
        for shape in matrix.workload_shapes:
            for batch_size in matrix.batch_sizes:
                for repetition in range(1, matrix.repetitions + 1):
                    for arm in matrix.arms:
                        key = (
                            scenario.id,
                            checkout,
                            shape.rows,
                            shape.right_rows,
                            shape.unique_inputs,
                            shape.pair_block_size,
                            shape.block_token_budget,
                            shape.rpm,
                            shape.latency_seconds,
                            batch_size,
                            repetition,
                            matrix.input_seed,
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
                                step=as_jsonable(scenario.steps[0]),
                                arm=arm,
                                rows=shape.rows,
                                right_rows=shape.right_rows,
                                unique_inputs=shape.unique_inputs,
                                pair_block_size=shape.pair_block_size,
                                block_token_budget=shape.block_token_budget,
                                rpm=shape.rpm,
                                latency_seconds=shape.latency_seconds,
                                batch_size=batch_size,
                                repetition=repetition,
                                input_seed=matrix.input_seed,
                                checkout=checkout,
                            )
                        )
    return cells


def interleave_cells(cells: Iterable[Cell], seed: int = 0) -> list[Cell]:
    """Interleave arms within each repetition and alternate their first position.

    A single random arm order can bias every repetition toward the same arm.
    The deterministic seed chooses only the first repetition's order; subsequent
    repetitions alternate, keeping the first-position count balanced to one.
    """
    grouped: dict[tuple[Any, ...], dict[int, list[Cell]]] = {}
    for cell in cells:
        group_key = (*cell.comparison_key[:-1], cell.checkout)
        grouped.setdefault(group_key, {}).setdefault(cell.repetition, []).append(cell)
    output: list[Cell] = []
    for key in sorted(grouped, key=repr):
        repetitions = grouped[key]
        first_is_standard = (
            int(hashlib.sha256(f"{seed}:{key!r}".encode()).hexdigest(), 16) & 1
        ) == 0
        for offset, repetition in enumerate(sorted(repetitions)):
            arms = {cell.arm: cell for cell in repetitions[repetition]}
            if set(arms) != {"standard", "streaming"}:
                raise ValueError(
                    f"interleaving group {key!r} repetition {repetition} lacks an arm"
                )
            standard_first = first_is_standard == (offset % 2 == 0)
            arm_order = ("standard", "streaming") if standard_first else (
                "streaming",
                "standard",
            )
            output.extend(arms[arm] for arm in arm_order)
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
    slower = candidate_median / baseline_median - 1
    if slower <= GATE_THRESHOLD:
        return "PASS"
    candidate_mad = median_absolute_deviation(candidate_values)
    baseline_mad = median_absolute_deviation(baseline_values)
    bands_overlap = max(
        candidate_median - candidate_mad, baseline_median - baseline_mad
    ) <= min(candidate_median + candidate_mad, baseline_median + baseline_mad)
    return "INCONCLUSIVE" if bands_overlap else "FAIL"


def environment_metadata() -> dict[str, Any]:
    return {
        "host": socket.gethostname(),
        "platform": __import__("platform").platform(),
        "python": __import__("platform").python_version(),
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }


def as_jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {key: as_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): as_jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [as_jsonable(item) for item in value]
    return value
