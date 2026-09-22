"""Reproducible FE endpoint banks for the revision sweeps.

This module deliberately contains *experiment orchestration*, not a second
implementation of the acoustic objective.  A solver adapter receives an
``EndpointRequest`` and returns an ``EndpointSolution`` (or an equivalent
mapping).  The request carries an explicit scientific contract, so caches made
with the historically reversed density term cannot be consumed by this code.

The only accepted Gor'kov convention is

    U = K1 |p|^2 - K2 |grad p|^2,
    K2 = 3 V (rho_p-rho_0) /
         [4 omega^2 rho_0 (rho_0 + 2 rho_p)].

Thus ``K2 > 0`` for the denser particles considered in the manuscript.  Field
evaluation must also be global: directivity may depend on every evaluation
point, but must not be frozen around a separately supplied expansion centre.

Two public runners create matched-cold-start banks for the array and frequency
experiments.  Their output is independent of plotting.  The final
section turns any bank into fixed-frame endpoint, coverage, node, and edge
tables suitable for Figure 6 and Appendices C/E.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from rineng_content_id import content_identity as content_id
from itertools import permutations
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

import numpy as np


SPEED_OF_SOUND_M_S = 343.0
SWEEP_SCHEMA_VERSION = 1
STANDARD_GORKOV_CONVENTION_ID = "gorkov-standard-density-contrast-v1"
STANDARD_GORKOV_FORMULA = (
    "U=K1*|p|^2-K2*|grad(p)|^2;"
    "K2=3*V*(rho_p-rho_0)/(4*omega^2*rho_0*(rho_0+2*rho_p))"
)
GLOBAL_FIELD_PROTOCOL_ID = "pressure(points)-global-directivity-v1"


class ScientificConventionError(ValueError):
    """Raised before a run/cache load when its physics contract is unsafe."""


class EndpointCacheMissError(RuntimeError):
    """A required endpoint record is absent or invalid in cache-only mode."""

    def __init__(self, request: "EndpointRequest", path: Path):
        self.request = request
        self.path = Path(path)
        super().__init__(
            "Endpoint sweep cache-only miss: "
            f"sweep={request.sweep_name!r}, condition={request.condition_id!r}, "
            f"restart={request.restart}; expected a valid standard-Gor'kov/global-field "
            f"record at {self.path}. Run the matching quick/full mode first, then "
            "retry plot-only."
        )


def _jsonable(value: Any) -> Any:
    """Return a deterministic JSON-compatible representation."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0.0 else "-Infinity"
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return repr(value)


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(payload: Mapping[str, Any]) -> str:
    return content_id(_canonical_json(payload).encode("utf-8")).hexdigest()


def _array_digest(array: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
    return content_id(value.tobytes()).hexdigest()


@dataclass(frozen=True)
class SolverContract:
    """Immutable provenance required for every endpoint solve.

    ``implementation_id`` should change whenever objective or differentiation
    code changes.  It is intentionally included in every cache key.
    """

    solver_id: str
    implementation_id: str
    gorkov_convention_id: str = STANDARD_GORKOV_CONVENTION_ID
    gorkov_formula: str = STANDARD_GORKOV_FORMULA
    field_protocol_id: str = GLOBAL_FIELD_PROTOCOL_ID
    independent_cold_starts: bool = True

    def validate(self) -> None:
        if self.gorkov_convention_id != STANDARD_GORKOV_CONVENTION_ID:
            raise ScientificConventionError(
                "The endpoint solver does not declare the standard Gor'kov density term."
            )
        if self.gorkov_formula != STANDARD_GORKOV_FORMULA:
            raise ScientificConventionError("The declared Gor'kov formula does not match the revision contract.")
        if self.field_protocol_id != GLOBAL_FIELD_PROTOCOL_ID:
            raise ScientificConventionError(
                "Sweep endpoints require one global pressure(points) field; locally frozen directivity is rejected."
            )
        if not self.independent_cold_starts:
            raise ScientificConventionError("Endpoint-bank coverage requires independent cold starts.")
        if not self.solver_id or not self.implementation_id:
            raise ScientificConventionError("solver_id and implementation_id must be non-empty.")

    def as_payload(self) -> dict[str, Any]:
        self.validate()
        return {
            "solver_id": self.solver_id,
            "implementation_id": self.implementation_id,
            "gorkov_convention_id": self.gorkov_convention_id,
            "gorkov_formula": self.gorkov_formula,
            "field_protocol_id": self.field_protocol_id,
            "independent_cold_starts": self.independent_cold_starts,
        }


@dataclass(frozen=True)
class TargetPoint:
    """One physical target and its optional two-dimensional chart index."""

    target_id: str
    position_m: tuple[float, float, float]
    grid_index: tuple[int, int] | None = None
    chart_coordinate: tuple[float, float] | None = None

    def validate(self) -> None:
        position = np.asarray(self.position_m, dtype=float)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError(f"Invalid target position for {self.target_id!r}.")
        if not self.target_id:
            raise ValueError("target_id must be non-empty.")


@dataclass(frozen=True)
class EndpointRequest:
    """Complete, cache-addressable input to one independent FE solve."""

    sweep_name: str
    axis_label: str
    axis_value: float | str
    condition_id: str
    array_id: str
    positions_m: np.ndarray
    target: TargetPoint
    frequency_hz: float
    alpha: float
    restart: int
    seed: int
    solver_options: Mapping[str, Any]
    contract: SolverContract
    schema_version: int = SWEEP_SCHEMA_VERSION

    def validate(self) -> None:
        self.contract.validate()
        self.target.validate()
        positions = np.asarray(self.positions_m, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) < 2:
            raise ValueError("positions_m must have shape (n_actuators, 3).")
        if not np.all(np.isfinite(positions)):
            raise ValueError("positions_m contains non-finite values.")
        if not math.isfinite(self.frequency_hz) or self.frequency_hz <= 0.0:
            raise ValueError("frequency_hz must be positive and finite.")
        if not math.isfinite(self.alpha) or self.alpha < 0.0:
            raise ValueError("alpha must be finite and non-negative.")
        if self.restart < 0 or self.seed < 0:
            raise ValueError("restart and seed must be non-negative.")
        if self.schema_version != SWEEP_SCHEMA_VERSION:
            raise ValueError("Unsupported sweep schema version.")

    def payload(self) -> dict[str, Any]:
        self.validate()
        from .fe_solver_policy import FE_SOLVER_REVISION
        return {
            "fe_solver_policy": FE_SOLVER_REVISION,
            "schema_version": self.schema_version,
            "sweep_name": self.sweep_name,
            "axis_label": self.axis_label,
            "axis_value": self.axis_value,
            "condition_id": self.condition_id,
            "array_id": self.array_id,
            "positions_shape": list(np.asarray(self.positions_m).shape),
            "positions_content_id": _array_digest(self.positions_m),
            "target_id": self.target.target_id,
            "target_position_m": list(self.target.position_m),
            "target_grid_index": None if self.target.grid_index is None else list(self.target.grid_index),
            "target_chart_coordinate": (
                None if self.target.chart_coordinate is None else list(self.target.chart_coordinate)
            ),
            "frequency_hz": self.frequency_hz,
            "alpha": self.alpha,
            "restart": self.restart,
            "seed": self.seed,
            "solver_options": dict(self.solver_options),
            "contract": self.contract.as_payload(),
        }

    @property
    def request_id(self) -> str:
        return _digest(self.payload())


@dataclass
class EndpointSolution:
    """Minimum solver output; all other diagnostics are optional."""

    phase_radians: np.ndarray
    success: bool = True
    objective_value: float = math.nan
    gradient_norm: float = math.nan
    iterations: int = 0
    wall_time_s: float = math.nan
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    physics_convention_id: str = STANDARD_GORKOV_CONVENTION_ID
    field_protocol_id: str = GLOBAL_FIELD_PROTOCOL_ID
    carrier_removed_state: np.ndarray | None = None


@dataclass
class EndpointRecord:
    request: EndpointRequest
    solution: EndpointSolution
    state: np.ndarray
    cache_status: str

    @property
    def record_id(self) -> str:
        return self.request.request_id

    def as_row(self) -> dict[str, Any]:
        target = self.request.target
        row: dict[str, Any] = {
            "record_id": self.record_id,
            "sweep_name": self.request.sweep_name,
            "axis_label": self.request.axis_label,
            "axis_value": self.request.axis_value,
            "condition_id": self.request.condition_id,
            "array_id": self.request.array_id,
            "target_id": target.target_id,
            "target_x_m": target.position_m[0],
            "target_y_m": target.position_m[1],
            "target_z_m": target.position_m[2],
            "grid_i": None if target.grid_index is None else target.grid_index[0],
            "grid_j": None if target.grid_index is None else target.grid_index[1],
            "chart_u": None if target.chart_coordinate is None else target.chart_coordinate[0],
            "chart_v": None if target.chart_coordinate is None else target.chart_coordinate[1],
            "frequency_hz": self.request.frequency_hz,
            "alpha": self.request.alpha,
            "restart": self.request.restart,
            "seed": self.request.seed,
            "success": bool(self.solution.success),
            "objective_value": float(self.solution.objective_value),
            "gradient_norm": float(self.solution.gradient_norm),
            "iterations": int(self.solution.iterations),
            "wall_time_s": float(self.solution.wall_time_s),
            "cache_status": self.cache_status,
            "gorkov_convention_id": self.request.contract.gorkov_convention_id,
            "field_protocol_id": self.request.contract.field_protocol_id,
        }
        for key, value in sorted(self.solution.diagnostics.items()):
            if key not in row and isinstance(value, (str, int, float, bool, np.generic)):
                row[key] = _jsonable(value)
        return row


@dataclass
class EndpointBank:
    """In-memory endpoint archive returned by every sweep runner."""

    sweep_name: str
    mode: str
    contract: SolverContract
    records: list[EndpointRecord]
    design: Mapping[str, Any]

    def rows(self) -> list[dict[str, Any]]:
        return [record.as_row() for record in self.records]

    def successful(self) -> list[EndpointRecord]:
        return [record for record in self.records if record.solution.success]

    def states(self, successful_only: bool = True) -> np.ndarray:
        records = self.successful() if successful_only else self.records
        if not records:
            return np.empty((0, 0), dtype=np.complex128)
        array_ids = {record.request.array_id for record in records}
        if len(array_ids) != 1:
            raise ValueError(
                "Elementwise projective states from unrelated arrays must be split before comparison."
            )
        dimensions = {record.state.size for record in records}
        if len(dimensions) != 1:
            raise ValueError("States from different actuator counts cannot share one projective frame.")
        return np.stack([record.state for record in records])


def summarize_endpoint_timing(bank: EndpointBank) -> list[dict[str, Any]]:
    """Median/IQR solve time for the compact appendix speed panels."""
    groups: dict[tuple[Any, ...], list[float]] = {}
    for record in bank.records:
        value = float(record.solution.wall_time_s)
        if not math.isfinite(value):
            continue
        key = (
            record.request.axis_value,
            record.request.array_id,
            record.request.frequency_hz,
            record.request.alpha,
        )
        groups.setdefault(key, []).append(value)
    rows: list[dict[str, Any]] = []
    for key, values in groups.items():
        sample = np.asarray(values, dtype=float)
        rows.append(
            {
                "axis_value": key[0],
                "array_id": key[1],
                "frequency_hz": key[2],
                "alpha": key[3],
                "n": len(sample),
                "time_median_s": float(np.median(sample)),
                "time_q25_s": float(np.quantile(sample, 0.25)),
                "time_q75_s": float(np.quantile(sample, 0.75)),
            }
        )
    return sorted(rows, key=lambda row: (str(row["array_id"]), str(row["axis_value"])))


class EndpointSolver(Protocol):
    def __call__(self, request: EndpointRequest) -> EndpointSolution | Mapping[str, Any]: ...


def _coerce_solution(value: EndpointSolution | Mapping[str, Any], elapsed_s: float) -> EndpointSolution:
    if isinstance(value, EndpointSolution):
        solution = value
    elif isinstance(value, Mapping):
        phase = value.get("phase_radians", value.get("phase"))
        if phase is None:
            raise KeyError("Solver mapping must contain 'phase_radians' or 'phase'.")
        solution = EndpointSolution(
            phase_radians=np.asarray(phase, dtype=float),
            success=bool(value.get("success", True)),
            objective_value=float(value.get("objective_value", value.get("objective", math.nan))),
            gradient_norm=float(value.get("gradient_norm", value.get("phase_gradient_norm", math.nan))),
            iterations=int(value.get("iterations", value.get("nit", 0))),
            wall_time_s=float(value.get("wall_time_s", value.get("runtime_s", elapsed_s))),
            diagnostics=value.get("diagnostics", {}),
            physics_convention_id=str(
                value.get("physics_convention_id", STANDARD_GORKOV_CONVENTION_ID)
            ),
            field_protocol_id=str(value.get("field_protocol_id", GLOBAL_FIELD_PROTOCOL_ID)),
            carrier_removed_state=value.get("carrier_removed_state"),
        )
    else:
        raise TypeError("Solver must return EndpointSolution or a mapping.")
    if not math.isfinite(solution.wall_time_s):
        solution.wall_time_s = float(elapsed_s)
    return solution


def wrap_phase(phase: np.ndarray) -> np.ndarray:
    return (np.asarray(phase, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def propagation_carrier(request: EndpointRequest) -> np.ndarray:
    distance = np.linalg.norm(
        np.asarray(request.target.position_m, dtype=float)[None, :]
        - np.asarray(request.positions_m, dtype=float),
        axis=1,
    )
    wavelength = SPEED_OF_SOUND_M_S / request.frequency_hz
    return -(2.0 * np.pi / wavelength) * distance


def carrier_removed_projective_state(request: EndpointRequest, phase_radians: np.ndarray) -> np.ndarray:
    phase = np.asarray(phase_radians, dtype=float).reshape(-1)
    if phase.size != len(request.positions_m):
        raise ValueError(
            f"Phase length {phase.size} does not match {len(request.positions_m)} actuators."
        )
    if not np.all(np.isfinite(phase)):
        raise ValueError("Solver returned non-finite phases.")
    residual = wrap_phase(phase - propagation_carrier(request))
    state = np.exp(1j * residual) / math.sqrt(residual.size)
    return np.asarray(state, dtype=np.complex128)


def projective_similarity(states_a: np.ndarray, states_b: np.ndarray | None = None) -> np.ndarray:
    a = np.asarray(states_a, dtype=np.complex128)
    b = a if states_b is None else np.asarray(states_b, dtype=np.complex128)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("Projective states must be two matrices with the same state dimension.")
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1.0e-30)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1.0e-30)
    return np.clip(np.abs(a.conj() @ b.T), 0.0, 1.0)


def projector_distance(states_a: np.ndarray, states_b: np.ndarray | None = None) -> np.ndarray:
    similarity = projective_similarity(states_a, states_b)
    return np.sqrt(np.maximum(0.0, 1.0 - similarity * similarity))


def _validate_solution(request: EndpointRequest, solution: EndpointSolution) -> np.ndarray:
    if solution.physics_convention_id != STANDARD_GORKOV_CONVENTION_ID:
        raise ScientificConventionError("Solver output reports a stale/non-standard Gor'kov convention.")
    if solution.field_protocol_id != GLOBAL_FIELD_PROTOCOL_ID:
        raise ScientificConventionError("Solver output reports a locally frozen/non-global field protocol.")
    state = carrier_removed_projective_state(request, solution.phase_radians)
    if solution.carrier_removed_state is not None:
        supplied = np.asarray(solution.carrier_removed_state, dtype=np.complex128).reshape(1, -1)
        if supplied.shape[1] != state.size:
            raise ValueError("Solver-supplied projective state has the wrong size.")
        overlap = float(projective_similarity(state[None, :], supplied)[0, 0])
        if overlap < 1.0 - 1.0e-9:
            raise ValueError("Solver-supplied projective state is inconsistent with its phase endpoint.")
    solution.phase_radians = wrap_phase(np.asarray(solution.phase_radians, dtype=float).reshape(-1))
    solution.carrier_removed_state = state
    return state


def _cache_path(cache_dir: Path, request: EndpointRequest) -> Path:
    return cache_dir / request.sweep_name / f"{request.request_id}.npz"


def _save_record_cache(path: Path, record: EndpointRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = record.as_row() | {
        "request_payload": record.request.payload(),
        "diagnostics": _jsonable(record.solution.diagnostics),
        "physics_convention_id": record.solution.physics_convention_id,
        "solution_field_protocol_id": record.solution.field_protocol_id,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            phase_radians=np.asarray(record.solution.phase_radians, dtype=float),
            state=np.asarray(record.state, dtype=np.complex128),
            metadata_json=np.asarray(_canonical_json(metadata)),
        )
    temporary.replace(path)


def _load_record_cache(path: Path, request: EndpointRequest) -> EndpointRecord | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            phase = np.asarray(archive["phase_radians"], dtype=float)
            stored_state = np.asarray(archive["state"], dtype=np.complex128)
        if metadata.get("record_id") != request.request_id:
            return None
        if _canonical_json(metadata.get("request_payload", {})) != _canonical_json(request.payload()):
            return None
        if metadata.get("gorkov_convention_id") != STANDARD_GORKOV_CONVENTION_ID:
            return None
        if metadata.get("field_protocol_id") != GLOBAL_FIELD_PROTOCOL_ID:
            return None
        if metadata.get("physics_convention_id") != STANDARD_GORKOV_CONVENTION_ID:
            return None
        if metadata.get("solution_field_protocol_id") != GLOBAL_FIELD_PROTOCOL_ID:
            return None
        solution = EndpointSolution(
            phase_radians=phase,
            success=bool(metadata["success"]),
            objective_value=float(metadata["objective_value"]),
            gradient_norm=float(metadata["gradient_norm"]),
            iterations=int(metadata["iterations"]),
            wall_time_s=float(metadata["wall_time_s"]),
            diagnostics=metadata.get("diagnostics", {}),
            physics_convention_id=metadata["physics_convention_id"],
            field_protocol_id=metadata["solution_field_protocol_id"],
            carrier_removed_state=stored_state,
        )
        state = _validate_solution(request, solution)
        if float(projective_similarity(state[None, :], stored_state[None, :])[0, 0]) < 1.0 - 1.0e-10:
            return None
        return EndpointRecord(request=request, solution=solution, state=state, cache_status="hit")
    except (KeyError, ValueError, TypeError, OSError, json.JSONDecodeError):
        return None


def _run_requests(
    requests: Sequence[EndpointRequest],
    solver: EndpointSolver,
    cache_dir: Path | None,
    *,
    cache_only: bool = False,
) -> list[EndpointRecord]:
    if cache_only and cache_dir is None:
        raise ValueError("cache_only=True requires cache_dir")
    records: list[EndpointRecord] = []
    for request in requests:
        request.validate()
        cache_path = None if cache_dir is None else _cache_path(cache_dir, request)
        cached = None if cache_path is None else _load_record_cache(cache_path, request)
        if cached is not None:
            records.append(cached)
            continue
        if cache_only:
            assert cache_path is not None
            raise EndpointCacheMissError(request, cache_path)
        started = time.perf_counter()
        raw = solver(request)
        elapsed = time.perf_counter() - started
        solution = _coerce_solution(raw, elapsed)
        state = _validate_solution(request, solution)
        record = EndpointRecord(request=request, solution=solution, state=state, cache_status="computed")
        if cache_dir is not None:
            _save_record_cache(_cache_path(cache_dir, request), record)
        records.append(record)
    return records


def default_frequencies_hz(mode: str = "quick") -> tuple[float, ...]:
    if mode == "quick":
        return (36_000.0, 40_000.0, 44_000.0)
    if mode == "full":
        return (32_000.0, 36_000.0, 40_000.0, 44_000.0, 48_000.0)
    raise ValueError("mode must be 'quick' or 'full'.")


def default_restarts(mode: str = "quick") -> int:
    if mode == "quick":
        return 4
    if mode == "full":
        return 12
    raise ValueError("mode must be 'quick' or 'full'.")


def square_array(side: int, pitch_m: float = 0.010) -> np.ndarray:
    axis = np.arange(int(side), dtype=float) - 0.5 * (int(side) - 1)
    xy = np.array([(pitch_m * x, pitch_m * y) for x in axis for y in axis], dtype=float)
    return np.column_stack([xy, np.zeros(len(xy), dtype=float)])


def _match_rms_aperture(xy: np.ndarray, reference: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=float) - np.mean(xy, axis=0, keepdims=True)
    reference_xy = np.asarray(reference, dtype=float)[:, :2]
    reference_rms = float(np.sqrt(np.mean(np.sum(reference_xy * reference_xy, axis=1))))
    current_rms = float(np.sqrt(np.mean(np.sum(xy * xy, axis=1))))
    if current_rms <= 0.0:
        raise ValueError("Degenerate array geometry.")
    xy = xy * (reference_rms / current_rms)
    return np.column_stack([xy, np.zeros(len(xy), dtype=float)])


def _farthest_point_disk(count: int, seed: int, candidates: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    theta = rng.uniform(0.0, 2.0 * np.pi, size=int(candidates))
    radius = np.sqrt(rng.uniform(0.0, 1.0, size=int(candidates)))
    pool = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
    result = np.empty((int(count), 2), dtype=float)
    selected = int(rng.integers(len(pool)))
    result[0] = pool[selected]
    nearest_sq = np.sum((pool - result[0]) ** 2, axis=1)
    nearest_sq[selected] = -np.inf
    for index in range(1, count):
        selected = int(np.argmax(nearest_sq))
        result[index] = pool[selected]
        nearest_sq = np.minimum(nearest_sq, np.sum((pool - result[index]) ** 2, axis=1))
        nearest_sq[selected] = -np.inf
    return result


def standard_array_ensemble(mode: str = "quick", pitch_m: float = 0.010) -> dict[str, np.ndarray]:
    """Deterministic equal-RMS-aperture layouts used by Appendix C.

    Both modes use the manuscript's 16 x 16 / 256-actuator hardware.  Quick
    mode reduces only the number of layouts to three; it never changes the
    physical inverse problem.  Full mode uses all six predeclared layouts.
    Equal RMS aperture avoids silently crediting a geometry merely for being
    larger.
    """
    if mode not in {"quick", "full"}:
        raise ValueError("mode must be 'quick' or 'full'.")
    side = 16
    reference = square_array(side, pitch_m)
    axis = np.arange(side, dtype=float) - 0.5 * (side - 1)
    rectangular = np.array([(1.24 * pitch_m * x, pitch_m * y / 1.24) for x in axis for y in axis])
    count = side * side
    index = np.arange(count, dtype=float)
    radius = np.sqrt((index + 0.5) / count)
    theta = index * (np.pi * (3.0 - np.sqrt(5.0)))
    fermat = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
    ensemble = {
        "square": reference,
        "rectangular": _match_rms_aperture(rectangular, reference),
        "fermat-disk": _match_rms_aperture(fermat, reference),
    }
    if mode == "full":
        rng = np.random.default_rng(271828)
        jittered = reference[:, :2] + rng.uniform(-0.22 * pitch_m, 0.22 * pitch_m, size=(count, 2))
        candidates = max(20_000, 180 * count)
        ensemble.update(
            {
                "jittered": _match_rms_aperture(jittered, reference),
                "blue-noise-314159": _match_rms_aperture(
                    _farthest_point_disk(count, 314159, candidates), reference
                ),
                "blue-noise-271828": _match_rms_aperture(
                    _farthest_point_disk(count, 271828, candidates), reference
                ),
            }
        )
    return ensemble


def default_target_grid(mode: str = "quick") -> tuple[TargetPoint, ...]:
    """Tilted core-volume chart used only when the caller supplies no grid."""
    if mode not in {"quick", "full"}:
        raise ValueError("mode must be 'quick' or 'full'.")
    count = 3 if mode == "quick" else 5
    coordinates = np.linspace(-1.0, 1.0, count)
    center = np.array([0.0, 0.0, 0.050], dtype=float)
    e_u = np.array([0.010, 0.0, 0.0075], dtype=float)
    e_v = np.array([0.0, 0.010, 0.0], dtype=float)
    targets: list[TargetPoint] = []
    for j, v in enumerate(coordinates):
        for i, u in enumerate(coordinates):
            position = center + u * e_u + v * e_v
            targets.append(
                TargetPoint(
                    target_id=f"u{u:+.3f}_v{v:+.3f}",
                    position_m=tuple(float(x) for x in position),
                    grid_index=(i, j),
                    chart_coordinate=(float(u), float(v)),
                )
            )
    return tuple(targets)


def _coerce_targets(targets: Sequence[TargetPoint | Sequence[float]] | None, mode: str) -> tuple[TargetPoint, ...]:
    if targets is None:
        return default_target_grid(mode)
    result: list[TargetPoint] = []
    for index, target in enumerate(targets):
        if isinstance(target, TargetPoint):
            item = target
        else:
            position = tuple(float(x) for x in target)
            item = TargetPoint(f"target-{index:03d}", position)  # type: ignore[arg-type]
        item.validate()
        result.append(item)
    if not result:
        raise ValueError("At least one target is required.")
    if len({item.target_id for item in result}) != len(result):
        raise ValueError("target_id values must be unique.")
    return tuple(result)


def _matched_seed(base_seed: int, target_id: str, restart: int) -> int:
    from rineng_protocol import matched_seed
    return matched_seed(base_seed, target_id, restart)

def _make_request(
    *,
    sweep_name: str,
    axis_label: str,
    axis_value: float | str,
    array_id: str,
    positions_m: np.ndarray,
    target: TargetPoint,
    frequency_hz: float,
    alpha: float,
    restart: int,
    base_seed: int,
    solver_options: Mapping[str, Any],
    contract: SolverContract,
) -> EndpointRequest:
    # The alpha/frequency/array coordinate is deliberately absent from the
    # seed key: corresponding conditions see the same random cold starts.
    seed = _matched_seed(base_seed, target.target_id, restart)
    condition_id = f"{axis_label}={axis_value}|array={array_id}|target={target.target_id}"
    frozen_positions = np.array(positions_m, dtype=float, copy=True)
    frozen_positions.setflags(write=False)
    return EndpointRequest(
        sweep_name=sweep_name,
        axis_label=axis_label,
        axis_value=axis_value,
        condition_id=condition_id,
        array_id=array_id,
        positions_m=frozen_positions,
        target=target,
        frequency_hz=float(frequency_hz),
        alpha=float(alpha),
        restart=int(restart),
        seed=seed,
        solver_options=dict(solver_options),
        contract=contract,
    )


def run_array_endpoint_bank(
    solver: EndpointSolver,
    contract: SolverContract,
    *,
    arrays: Mapping[str, np.ndarray] | None = None,
    targets: Sequence[TargetPoint | Sequence[float]] | None = None,
    alpha: float = 1000.0,
    frequency_hz: float = 40_000.0,
    mode: str = "quick",
    restarts: int | None = None,
    base_seed: int = 821_026,
    solver_options: Mapping[str, Any] | None = None,
    cache_dir: str | Path | None = None,
    cache_only: bool = False,
) -> EndpointBank:
    """Run independent FE endpoint webs on equal-aperture array layouts."""
    contract.validate()
    layouts = standard_array_ensemble(mode) if arrays is None else {
        str(name): np.asarray(value, dtype=float) for name, value in arrays.items()
    }
    if not layouts:
        raise ValueError("At least one array geometry is required.")
    element_counts = {len(value) for value in layouts.values()}
    if len(element_counts) != 1:
        raise ValueError("The common array benchmark requires equal actuator counts.")
    target_points = _coerce_targets(targets, mode)
    count = default_restarts(mode) if restarts is None else int(restarts)
    if count < 1:
        raise ValueError("restarts must be positive.")
    options = {} if solver_options is None else dict(solver_options)
    requests = [
        _make_request(
            sweep_name="array",
            axis_label="array_id",
            axis_value=array_id,
            array_id=array_id,
            positions_m=positions,
            target=target,
            frequency_hz=frequency_hz,
            alpha=alpha,
            restart=restart,
            base_seed=base_seed,
            solver_options=options,
            contract=contract,
        )
        for array_id, positions in layouts.items()
        for target in target_points
        for restart in range(count)
    ]
    records = _run_requests(
        requests,
        solver,
        None if cache_dir is None else Path(cache_dir),
        cache_only=cache_only,
    )
    return EndpointBank(
        "array",
        mode,
        contract,
        records,
        {"array_ids": tuple(layouts), "restarts": count, "frequency_hz": frequency_hz, "alpha": alpha},
    )


def run_frequency_endpoint_bank(
    solver: EndpointSolver,
    contract: SolverContract,
    *,
    positions_m: np.ndarray | None = None,
    targets: Sequence[TargetPoint | Sequence[float]] | None = None,
    frequencies_hz: Sequence[float] | None = None,
    alpha: float = 1000.0,
    array_id: str = "square",
    mode: str = "quick",
    restarts: int | None = None,
    base_seed: int = 821_026,
    solver_options: Mapping[str, Any] | None = None,
    cache_dir: str | Path | None = None,
    cache_only: bool = False,
) -> EndpointBank:
    """Run the same physical array/target chart at several frequencies.

    Geometry and target coordinates remain fixed in metres.  Only frequency
    changes; the carrier removal uses the corresponding wavelength.  This is
    recorded explicitly to avoid conflating frequency with geometric scaling.
    """
    contract.validate()
    values = tuple(
        float(x) for x in (default_frequencies_hz(mode) if frequencies_hz is None else frequencies_hz)
    )
    if not values or any((not math.isfinite(x) or x <= 0.0) for x in values):
        raise ValueError("frequencies_hz must be positive and finite.")
    positions = square_array(16) if positions_m is None else np.asarray(positions_m)
    target_points = _coerce_targets(targets, mode)
    count = default_restarts(mode) if restarts is None else int(restarts)
    if count < 1:
        raise ValueError("restarts must be positive.")
    options = {} if solver_options is None else dict(solver_options)
    requests = [
        _make_request(
            sweep_name="frequency",
            axis_label="frequency_hz",
            axis_value=frequency,
            array_id=array_id,
            positions_m=positions,
            target=target,
            frequency_hz=frequency,
            alpha=alpha,
            restart=restart,
            base_seed=base_seed,
            solver_options=options,
            contract=contract,
        )
        for frequency in values
        for target in target_points
        for restart in range(count)
    ]
    records = _run_requests(
        requests,
        solver,
        None if cache_dir is None else Path(cache_dir),
        cache_only=cache_only,
    )
    return EndpointBank(
        "frequency",
        mode,
        contract,
        records,
        {
            "frequencies_hz": values,
            "restarts": count,
            "alpha": alpha,
            "array_id": array_id,
            "geometry_policy": "fixed-physical-metres",
        },
    )


class FixedFrame(Protocol):
    """Structural protocol shared with ``branch_figures.FrozenBranchFrame2D``."""

    def project(self, states: np.ndarray) -> np.ndarray: ...


@dataclass
class FixedFrameProjection:
    endpoint_rows: list[dict[str, Any]]
    coverage_rows: list[dict[str, Any]]
    node_rows: list[dict[str, Any]]
    edge_rows: list[dict[str, Any]]
    component_labels: tuple[str, ...]


def summarize_projection_coverage(projection: FixedFrameProjection) -> list[dict[str, Any]]:
    """Aggregate cellwise coverage without discarding the underlying map."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in projection.coverage_rows:
        key = (row["axis_value"], row["array_id"], row["frequency_hz"], row["alpha"])
        groups.setdefault(key, []).append(row)
    summaries: list[dict[str, Any]] = []
    for key, rows in groups.items():
        endpoints = sum(int(row["n_endpoints"]) for row in rows)
        assigned = sum(int(row["n_assigned"]) for row in rows)
        summaries.append(
            {
                "axis_value": key[0],
                "array_id": key[1],
                "frequency_hz": key[2],
                "alpha": key[3],
                "target_cells": len(rows),
                "complete_cells": sum(bool(row["complete_reference_coverage"]) for row in rows),
                "complete_cell_fraction": (
                    sum(bool(row["complete_reference_coverage"]) for row in rows) / len(rows)
                ),
                "mean_component_coverage": float(
                    np.mean([float(row["component_coverage_fraction"]) for row in rows])
                ),
                "assigned_endpoint_fraction": assigned / endpoints if endpoints else math.nan,
                "solver_success_fraction": (
                    sum(int(row["n_successful"]) for row in rows) / endpoints if endpoints else math.nan
                ),
            }
        )
    return sorted(summaries, key=lambda row: (str(row["array_id"]), str(row["axis_value"])))


def _intrinsic_matrix_for_nodes(
    bank: EndpointBank,
    projection: FixedFrameProjection,
    component: str,
) -> tuple[list[str], np.ndarray]:
    records = {record.record_id: record for record in bank.records}
    nodes = sorted(
        (row for row in projection.node_rows if row["component"] == component),
        key=lambda row: (str(row["target_id"]), int(row["grid_j"] or 0), int(row["grid_i"] or 0)),
    )
    target_ids = [str(row["target_id"]) for row in nodes]
    if len(target_ids) != len(set(target_ids)):
        raise ValueError("Intrinsic comparison expects one sweep slice per array projection.")
    states = np.stack([records[str(row["record_id"])].state for row in nodes])
    similarity = projective_similarity(states)
    return target_ids, np.arccos(np.clip(similarity, 0.0, 1.0))


def _distance_matrix_comparison(
    reference_ids: Sequence[str],
    reference: np.ndarray,
    query_ids: Sequence[str],
    query: np.ndarray,
) -> dict[str, float | int]:
    common = [target_id for target_id in reference_ids if target_id in set(query_ids)]
    if len(common) < 2:
        return {
            "common_targets": len(common),
            "distance_correlation": math.nan,
            "relative_frobenius_error": math.nan,
            "scale": math.nan,
            "scale_fitted_nrmse": math.nan,
        }
    ri = [reference_ids.index(target_id) for target_id in common]
    qi = [query_ids.index(target_id) for target_id in common]
    x_matrix = np.asarray(reference)[np.ix_(ri, ri)]
    y_matrix = np.asarray(query)[np.ix_(qi, qi)]
    upper = np.triu_indices(len(common), k=1)
    x, y = x_matrix[upper], y_matrix[upper]
    correlation = float(np.corrcoef(x, y)[0, 1]) if len(x) >= 2 and np.std(x) > 0 and np.std(y) > 0 else math.nan
    scale = float(np.dot(x, y) / max(float(np.dot(y, y)), 1.0e-30))
    return {
        "common_targets": len(common),
        "distance_correlation": correlation,
        "relative_frobenius_error": float(
            np.linalg.norm(y_matrix - x_matrix) / max(float(np.linalg.norm(x_matrix)), 1.0e-30)
        ),
        "scale": scale,
        "scale_fitted_nrmse": float(
            np.linalg.norm(scale * y - x) / max(float(np.linalg.norm(x)), 1.0e-30)
        ),
    }


def compare_array_intrinsic_geometry(
    banks: Mapping[str, EndpointBank],
    projections: Mapping[str, FixedFrameProjection],
    *,
    reference_array: str | None = None,
) -> list[dict[str, Any]]:
    """Compare target-indexed branch geometry without cross-array phase matching.

    Each layout supplies its own within-array Fubini--Study distance matrix.
    The two data-driven component names are then matched to the reference by
    the permutation with the smallest scale-fitted matrix error.  This is the
    appropriate quantitative companion to the separate array-web panels.
    """
    if set(banks) != set(projections) or not banks:
        raise ValueError("banks and projections must contain the same non-empty array keys.")
    reference_name = reference_array or ("square" if "square" in banks else sorted(banks)[0])
    if reference_name not in banks:
        raise KeyError(reference_name)
    components = tuple(projections[reference_name].component_labels)
    if not components:
        raise ValueError("Reference projection has no components.")
    matrices: dict[tuple[str, str], tuple[list[str], np.ndarray]] = {}
    for array_id in banks:
        if tuple(projections[array_id].component_labels) != components:
            raise ValueError("All array projections must recover the same component count.")
        for component in components:
            matrices[(array_id, component)] = _intrinsic_matrix_for_nodes(
                banks[array_id], projections[array_id], component
            )

    rows: list[dict[str, Any]] = []
    for array_id in sorted(banks):
        candidates: list[tuple[float, tuple[str, ...], list[dict[str, Any]]]] = []
        for query_order in permutations(components):
            comparisons: list[dict[str, Any]] = []
            score = 0.0
            for reference_component, query_component in zip(components, query_order):
                reference_ids, reference_matrix = matrices[(reference_name, reference_component)]
                query_ids, query_matrix = matrices[(array_id, query_component)]
                metrics = _distance_matrix_comparison(
                    reference_ids, reference_matrix, query_ids, query_matrix
                )
                error = float(metrics["scale_fitted_nrmse"])
                score += error if math.isfinite(error) else 1.0e6
                comparisons.append(
                    {
                        "reference_array": reference_name,
                        "array_id": array_id,
                        "reference_component": reference_component,
                        "array_component": query_component,
                    }
                    | metrics
                )
            candidates.append((score, query_order, comparisons))
        _, _, chosen = min(candidates, key=lambda item: (item[0], item[1]))
        rows.extend(chosen)
    return rows


def _component_label(index: int) -> str:
    return f"B{index + 1}"


def _connected_components(distance: np.ndarray, threshold: float) -> np.ndarray:
    n = len(distance)
    parent = np.arange(n)

    def root(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for i in range(n):
        for j in range(i):
            if distance[i, j] <= threshold:
                union(i, j)
    roots = np.array([root(i) for i in range(n)], dtype=int)
    unique = sorted(set(roots), key=lambda r: int(np.min(np.where(roots == r)[0])))
    lookup = {root_id: index for index, root_id in enumerate(unique)}
    return np.array([lookup[value] for value in roots], dtype=int)


def discover_reference_components(
    anchor_states: np.ndarray,
    *,
    distance_threshold: float = 0.32,
    minimum_size: int = 2,
) -> tuple[np.ndarray, tuple[str, ...], dict[str, np.ndarray]]:
    """Discover data-driven reference components without winding labels.

    Components smaller than ``minimum_size`` receive ``-1`` and are not used
    as branch anchors.  Remaining components are ordered deterministically by
    their medoid state hash, not by winding or embedding orientation.
    """
    states = np.asarray(anchor_states, dtype=np.complex128)
    if states.ndim != 2 or len(states) == 0:
        raise ValueError("anchor_states must be a non-empty matrix.")
    distance = projector_distance(states)
    raw = _connected_components(distance, float(distance_threshold))
    groups: list[tuple[str, np.ndarray, int]] = []
    for raw_label in sorted(set(raw)):
        members = np.where(raw == raw_label)[0]
        if len(members) < int(minimum_size):
            continue
        local = distance[np.ix_(members, members)]
        medoid = int(members[int(np.argmin(np.mean(local, axis=1)))])
        state = states[medoid]
        canonical_phase = np.angle(state * np.exp(-1j * np.angle(state[0])))
        key = np.ascontiguousarray(canonical_phase, dtype="<f8").tobytes()
        groups.append((key, members, medoid))
    groups.sort(key=lambda item: item[0])
    labels = np.full(len(states), -1, dtype=int)
    anchors: dict[str, np.ndarray] = {}
    names: list[str] = []
    for index, (_, members, _) in enumerate(groups):
        labels[members] = index
        name = _component_label(index)
        names.append(name)
        anchors[name] = states[members]
    return labels, tuple(names), anchors


def _assign_components(
    states: np.ndarray,
    component_anchors: Mapping[str, np.ndarray],
    assignment_threshold: float,
) -> tuple[list[str | None], np.ndarray, np.ndarray]:
    names = tuple(component_anchors)
    if not names:
        return [None] * len(states), np.full(len(states), np.inf), np.full(len(states), np.nan)
    distances = np.column_stack(
        [np.min(projector_distance(states, component_anchors[name]), axis=1) for name in names]
    )
    best = np.argmin(distances, axis=1)
    ordered = np.sort(distances, axis=1)
    nearest = ordered[:, 0]
    margin = ordered[:, 1] - ordered[:, 0] if len(names) > 1 else np.full(len(states), np.nan)
    labels: list[str | None] = [
        names[int(index)] if nearest[row] <= assignment_threshold else None
        for row, index in enumerate(best)
    ]
    return labels, nearest, margin


def _as_coordinates(frame: FixedFrame | Callable[[np.ndarray], np.ndarray], states: np.ndarray) -> np.ndarray:
    if hasattr(frame, "project"):
        coordinates = frame.project(states)  # type: ignore[union-attr]
    else:
        coordinates = frame(states)  # type: ignore[operator]
    # Permit branch_figures to return a result object with ``coordinates``.
    if hasattr(coordinates, "coordinates"):
        coordinates = coordinates.coordinates
    result = np.asarray(coordinates, dtype=float)
    if result.ndim != 2 or len(result) != len(states) or result.shape[1] < 2:
        raise ValueError("Fixed-frame projector must return shape (n, >=2).")
    return result[:, :2]


def _medoid_index(indices: Sequence[int], states: np.ndarray) -> int:
    values = np.asarray(indices, dtype=int)
    if len(values) == 1:
        return int(values[0])
    distance = projector_distance(states[values])
    return int(values[int(np.argmin(np.mean(distance, axis=1)))])


def project_endpoint_bank_fixed_frame(
    bank: EndpointBank,
    frame: FixedFrame | Callable[[np.ndarray], np.ndarray],
    *,
    anchor_selector: Callable[[EndpointRecord], bool] | None = None,
    component_bank: Any | None = None,
    component_distance_threshold: float = 0.32,
    assignment_threshold: float = 0.42,
    minimum_anchor_component_size: int = 2,
    minimum_replicates_per_component: int = 1,
) -> FixedFrameProjection:
    """Project a bank and build plotting/coverage tables in one fixed frame.

    Either ``anchor_selector`` identifies the reference alpha/frequency/array
    slice on which components are discovered, or ``component_bank`` supplies
    the ``labels`` and ``anchors`` properties exposed by
    :mod:`hat_revision_pipeline.branch_figures`.  The same frame and component
    anchors are then used for every other slice; no per-panel refit is allowed.
    """
    # A SciPy ``success=False`` commonly means max-iteration termination while
    # still providing a finite terminal phase.  It is therefore recorded, not
    # silently erased from branch-recovery coverage.  The solver status remains
    # an explicit table column and can be filtered by a caller-defined anchor
    # selector when appropriate.
    records = list(bank.records)
    if not records:
        raise ValueError("The endpoint bank contains no endpoint records.")
    array_ids = {record.request.array_id for record in records}
    if len(array_ids) != 1:
        raise ValueError(
            "Project each array in its own fixed frame; call split_bank_by_array first."
        )
    dimensions = {record.state.size for record in records}
    if len(dimensions) != 1:
        raise ValueError("One fixed frame requires a common actuator dimension.")
    states = np.stack([record.state for record in records])
    if component_bank is not None:
        try:
            component_names = tuple(str(value) for value in component_bank.labels)
            anchors = {
                name: np.asarray(component_bank.anchors[name], dtype=np.complex128)
                for name in component_names
            }
        except (AttributeError, KeyError, TypeError) as error:
            raise TypeError("component_bank must expose labels and anchors.") from error
    else:
        if anchor_selector is None:
            raise ValueError("Provide anchor_selector or component_bank.")
        anchor_indices = np.array(
            [index for index, record in enumerate(records) if anchor_selector(record)], dtype=int
        )
        if len(anchor_indices) < 2:
            raise ValueError("anchor_selector must choose at least two successful endpoints.")
        _, component_names, anchors = discover_reference_components(
            states[anchor_indices],
            distance_threshold=component_distance_threshold,
            minimum_size=minimum_anchor_component_size,
        )
    if not component_names:
        raise ValueError("No repeatable reference component was recovered.")
    labels, nearest, margin = _assign_components(states, anchors, assignment_threshold)
    coordinates = _as_coordinates(frame, states)
    endpoint_rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        endpoint_rows.append(
            record.as_row()
            | {
                "state_index": index,
                "embedding_1": float(coordinates[index, 0]),
                "embedding_2": float(coordinates[index, 1]),
                "component": labels[index],
                "component_distance": float(nearest[index]),
                "component_margin": float(margin[index]),
                "assigned": labels[index] is not None,
            }
        )

    def cell_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            row["axis_value"],
            row["array_id"],
            row["frequency_hz"],
            row["alpha"],
            row["target_id"],
        )

    grouped: dict[tuple[Any, ...], list[int]] = {}
    for index, row in enumerate(endpoint_rows):
        grouped.setdefault(cell_key(row), []).append(index)

    coverage_rows: list[dict[str, Any]] = []
    node_rows: list[dict[str, Any]] = []
    for key, indices in grouped.items():
        template = endpoint_rows[indices[0]]
        counts = {name: sum(endpoint_rows[i]["component"] == name for i in indices) for name in component_names}
        recovered = [name for name in component_names if counts[name] >= minimum_replicates_per_component]
        coverage_rows.append(
            {
                "axis_value": key[0],
                "array_id": key[1],
                "frequency_hz": key[2],
                "alpha": key[3],
                "target_id": key[4],
                "grid_i": template["grid_i"],
                "grid_j": template["grid_j"],
                "chart_u": template["chart_u"],
                "chart_v": template["chart_v"],
                "n_endpoints": len(indices),
                "n_successful": sum(bool(endpoint_rows[i]["success"]) for i in indices),
                "n_solver_failures": sum(not bool(endpoint_rows[i]["success"]) for i in indices),
                "n_assigned": sum(endpoint_rows[i]["assigned"] for i in indices),
                "n_unassigned": sum(not endpoint_rows[i]["assigned"] for i in indices),
                "recovered_components": ",".join(recovered),
                "component_coverage_fraction": len(recovered) / len(component_names),
                "complete_reference_coverage": len(recovered) == len(component_names),
            }
            | {f"n_{name}": counts[name] for name in component_names}
        )
        for name in component_names:
            member_indices = [i for i in indices if endpoint_rows[i]["component"] == name]
            if len(member_indices) < minimum_replicates_per_component:
                continue
            chosen = _medoid_index(member_indices, states)
            node_rows.append(
                {
                    **{key_name: endpoint_rows[chosen][key_name] for key_name in (
                        "record_id", "axis_value", "array_id", "frequency_hz", "alpha",
                        "target_id", "grid_i", "grid_j", "chart_u", "chart_v",
                    )},
                    "component": name,
                    "embedding_1": endpoint_rows[chosen]["embedding_1"],
                    "embedding_2": endpoint_rows[chosen]["embedding_2"],
                    "replicate_count": len(member_indices),
                }
            )

    node_lookup: dict[tuple[Any, ...], dict[tuple[int, int], dict[str, Any]]] = {}
    for node in node_rows:
        if node["grid_i"] is None or node["grid_j"] is None:
            continue
        slice_key = (
            node["axis_value"], node["array_id"], node["frequency_hz"], node["alpha"], node["component"]
        )
        node_lookup.setdefault(slice_key, {})[(int(node["grid_i"]), int(node["grid_j"]))] = node
    edge_rows: list[dict[str, Any]] = []
    for slice_key, grid in node_lookup.items():
        for (i, j), source in sorted(grid.items()):
            for neighbor in ((i + 1, j), (i, j + 1)):
                if neighbor not in grid:
                    continue
                target = grid[neighbor]
                edge_rows.append(
                    {
                        "axis_value": slice_key[0],
                        "array_id": slice_key[1],
                        "frequency_hz": slice_key[2],
                        "alpha": slice_key[3],
                        "component": slice_key[4],
                        "source_record_id": source["record_id"],
                        "target_record_id": target["record_id"],
                        "x0": source["embedding_1"],
                        "y0": source["embedding_2"],
                        "x1": target["embedding_1"],
                        "y1": target["embedding_2"],
                    }
                )
    return FixedFrameProjection(endpoint_rows, coverage_rows, node_rows, edge_rows, component_names)


def split_bank_by_array(bank: EndpointBank) -> dict[str, EndpointBank]:
    """Return one bank per layout so unrelated actuator charts are never mixed."""
    result: dict[str, EndpointBank] = {}
    for array_id in sorted({record.request.array_id for record in bank.records}):
        records = [record for record in bank.records if record.request.array_id == array_id]
        result[array_id] = EndpointBank(
            sweep_name=bank.sweep_name,
            mode=bank.mode,
            contract=bank.contract,
            records=records,
            design=dict(bank.design) | {"selected_array_id": array_id},
        )
    return result
