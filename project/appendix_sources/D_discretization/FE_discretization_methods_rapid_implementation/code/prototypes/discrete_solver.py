#!/usr/bin/env python3
"""Finite-alphabet phase transition algorithms for FE-designed HAT fields.

The acoustic field is linear in ``a = exp(1j * phase)``.  Consequently, all
local Gor'kov quantities used by the corrected FE objective can be written as
Hermitian quadratic forms ``a.H @ Q_r @ a``.  This module exploits that
structure to provide:

* fixed-gauge nearest-neighbour quantization;
* exact best-global-offset rounding by a sorted quantizer-breakpoint sweep;
* cached, all-alphabet coordinate descent (not merely +/- one phase bin);
* exact nested-alphabet continuation, e.g. 32 -> 128 -> 2048 levels; and
* a deterministic random control with the same coordinate order, tolerance,
  alphabet scans, and sweep budget at every continuation stage.

No acoustic approximation is introduced here: the module only changes the
phase alphabet used to evaluate a supplied quadratic FE model.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ObjectiveSpec:
    """Weights matching the corrected objective in the benchmark backend."""

    method: str = "branch_hybrid"
    pressure_weight: float = 25.0
    force_weight: float = 0.1
    force_squared_weight: float = 1.0
    branch_pressure_squared_weight: float = 2.0
    branch_force_squared_weight: float = 0.1
    branch_stiffness_weight: float = 1.0
    branch_softmin_tau: float = 0.25
    regularization_relative: float = 1.0e-3


@dataclass
class QuantizationResult:
    levels: int
    indices: np.ndarray
    phase: np.ndarray
    objective: float
    evaluations: int
    rule: str
    offset_rad: float = 0.0
    distinct_candidates: int = 1
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class DescentResult:
    levels: int
    indices: np.ndarray
    phase: np.ndarray
    objective: float
    evaluations: int
    accepted_moves: int
    sweeps_completed: int
    converged: bool
    history: list[dict[str, float | int | bool]]


@dataclass
class ContinuationResult:
    levels: tuple[int, ...]
    stages: list[DescentResult]
    initialization: QuantizationResult | None = None

    @property
    def final(self) -> DescentResult:
        return self.stages[-1]

    @property
    def evaluations(self) -> int:
        return int(sum(stage.evaluations for stage in self.stages))


@dataclass
class MatchedControlResult:
    warm: ContinuationResult
    random: ContinuationResult
    coordinate_orders: dict[int, np.ndarray]
    random_seed: int


class TransitionProblem:
    """Stacked Hermitian forms and frozen scales for discrete phase search.

    Matrix order is
    ``curvature, Fx, Fy, Fz, pressure^2, Hxx, Hyy, Hzz, Hxy, Hxz, Hyz``.
    ``from_model`` accepts the corrected backend's ``QuadraticModel``.
    """

    def __init__(
        self,
        matrices: np.ndarray,
        scales: Mapping[str, float],
        spec: ObjectiveSpec | None = None,
        hermitian_tolerance: float = 2.0e-10,
    ) -> None:
        matrices = np.asarray(matrices, dtype=np.complex128)
        if matrices.ndim != 3 or matrices.shape[0] != 11:
            raise ValueError("matrices must have shape (11, N, N)")
        if matrices.shape[1] != matrices.shape[2]:
            raise ValueError("quadratic matrices must be square")
        scale = max(float(np.max(np.abs(matrices))), np.finfo(float).tiny)
        defect = float(np.max(np.abs(matrices - matrices.conj().transpose(0, 2, 1))))
        if defect > hermitian_tolerance * scale:
            raise ValueError(f"quadratic matrices are not Hermitian (relative defect {defect / scale:.3e})")
        required = ("curvature", "force", "pressure", "stiffness")
        missing = [name for name in required if name not in scales]
        if missing:
            raise ValueError(f"missing objective scales: {missing}")
        checked_scales = {name: float(scales[name]) for name in required}
        if any(not np.isfinite(value) or value <= 0.0 for value in checked_scales.values()):
            raise ValueError("all objective scales must be finite and positive")
        self.matrices = (matrices + matrices.conj().transpose(0, 2, 1)) / 2.0
        self.diagonal = np.real(np.diagonal(self.matrices, axis1=1, axis2=2))
        self.scales = checked_scales
        self.spec = spec or ObjectiveSpec()

    @classmethod
    def from_model(
        cls,
        model: object,
        spec: ObjectiveSpec | None = None,
        hermitian_tolerance: float = 2.0e-10,
    ) -> "TransitionProblem":
        matrices = np.stack(
            [
                model.q_curv,
                *model.q_force,
                model.q_pressure,
                *model.q_hessian,
            ]
        )
        return cls(matrices, model.scales, spec, hermitian_tolerance)

    @property
    def n(self) -> int:
        return int(self.matrices.shape[1])

    def state_from_phase(self, phase: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        phase = np.asarray(phase, dtype=float)
        if phase.shape != (self.n,):
            raise ValueError(f"phase must have shape ({self.n},)")
        amplitude = np.exp(1j * phase)
        products = np.einsum("rij,j->ri", self.matrices, amplitude, optimize=True)
        values = np.real(np.einsum("i,ri->r", np.conj(amplitude), products, optimize=True))
        return products, values, float(self.score_qvalues(values))

    def state_from_indices(
        self, indices: np.ndarray, levels: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        indices = gauge_fix_indices(indices, levels)
        amplitude = np.exp(2j * np.pi * indices / levels)
        products = np.einsum("rij,j->ri", self.matrices, amplitude, optimize=True)
        values = np.real(np.einsum("i,ri->r", np.conj(amplitude), products, optimize=True))
        return amplitude, products, values, float(self.score_qvalues(values))

    def score_phase(self, phase: np.ndarray) -> float:
        return self.state_from_phase(phase)[2]

    def score_qvalues(self, qvalues: np.ndarray) -> np.ndarray | float:
        """Evaluate one or a batch of cached quadratic-value vectors."""

        qvalues = np.asarray(qvalues, dtype=float)
        if qvalues.shape[-1] != 11:
            raise ValueError("the final qvalues dimension must have length 11")
        s = self.scales
        w = self.spec
        curv = qvalues[..., 0]
        force = qvalues[..., 1:4]
        p2 = np.maximum(qvalues[..., 4], 0.0)
        force_norm = np.linalg.norm(force, axis=-1)

        if w.method == "conventional":
            result = -curv / s["curvature"] + w.pressure_weight * np.sqrt(p2) / s["pressure"]
        elif w.method == "regularized":
            floor = w.regularization_relative * s["pressure"]
            result = -curv / s["curvature"] + w.pressure_weight * np.sqrt(p2 + floor**2) / s["pressure"]
        elif w.method == "force_equilibrium":
            result = -curv / s["curvature"] + w.force_weight * force_norm / s["force"]
        elif w.method == "force_squared":
            scaled_force = force_norm / s["force"]
            result = -curv / s["curvature"] + 0.5 * w.force_squared_weight * scaled_force**2
        elif w.method == "branch_hybrid":
            h = qvalues[..., 5:11]
            hessian = np.empty(qvalues.shape[:-1] + (3, 3), dtype=float)
            hessian[..., 0, 0] = h[..., 0]
            hessian[..., 1, 1] = h[..., 1]
            hessian[..., 2, 2] = h[..., 2]
            hessian[..., 0, 1] = hessian[..., 1, 0] = h[..., 3]
            hessian[..., 0, 2] = hessian[..., 2, 0] = h[..., 4]
            hessian[..., 1, 2] = hessian[..., 2, 1] = h[..., 5]
            eigenvalues = np.linalg.eigvalsh(hessian) / s["stiffness"]
            logits = -eigenvalues / w.branch_softmin_tau
            peak = np.max(logits, axis=-1, keepdims=True)
            logsumexp = np.squeeze(peak, axis=-1) + np.log(
                np.sum(np.exp(logits - peak), axis=-1)
            )
            softmin = -w.branch_softmin_tau * logsumexp
            result = (
                -w.branch_stiffness_weight * softmin
                + 0.5 * w.branch_pressure_squared_weight * p2 / s["pressure"] ** 2
                + 0.5
                * w.branch_force_squared_weight
                * (force_norm / s["force"]) ** 2
            )
        else:
            raise ValueError(f"unknown objective method: {w.method}")
        if np.ndim(result) == 0:
            return float(result)
        return np.asarray(result, dtype=float)


def wrap_phase(phase: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * np.asarray(phase, dtype=float)))


def gauge_fix_phase(phase: np.ndarray, gauge: int = 0) -> np.ndarray:
    phase = np.asarray(phase, dtype=float)
    if phase.ndim != 1:
        raise ValueError("phase must be one-dimensional")
    if not 0 <= gauge < phase.size:
        raise ValueError("gauge index is out of range")
    return wrap_phase(phase - phase[gauge])


def gauge_fix_indices(indices: np.ndarray, levels: int, gauge: int = 0) -> np.ndarray:
    if int(levels) != levels or levels < 2:
        raise ValueError("levels must be an integer >= 2")
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError("indices must be one-dimensional")
    if not 0 <= gauge < indices.size:
        raise ValueError("gauge index is out of range")
    return np.mod(indices - indices[gauge], int(levels)).astype(np.int64, copy=False)


def indices_to_phase(indices: np.ndarray, levels: int, gauge: int = 0) -> np.ndarray:
    indices = gauge_fix_indices(indices, levels, gauge)
    return wrap_phase((2.0 * np.pi / levels) * indices)


def nearest_indices(phase: np.ndarray, levels: int, gauge: int = 0) -> np.ndarray:
    """Half-up nearest quantization after removing the continuous gauge."""

    fixed = gauge_fix_phase(phase, gauge)
    step = 2.0 * np.pi / levels
    indices = np.floor(fixed / step + 0.5).astype(np.int64)
    return gauge_fix_indices(indices, levels, gauge)


def fixed_gauge_nearest(
    problem: TransitionProblem, phase: np.ndarray, levels: int, gauge: int = 0
) -> QuantizationResult:
    indices = nearest_indices(phase, levels, gauge)
    qphase = indices_to_phase(indices, levels, gauge)
    objective = problem.score_phase(qphase)
    return QuantizationResult(
        levels=int(levels),
        indices=indices,
        phase=qphase,
        objective=objective,
        evaluations=1,
        rule="fixed_gauge_nearest",
    )


def _breakpoint_groups(phase: np.ndarray, step: float) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return circularly unique breakpoints and the coordinates crossing there."""

    raw = np.mod(0.5 * step - np.asarray(phase, dtype=float), step)
    tolerance = 128.0 * np.finfo(float).eps * max(step, 1.0)
    raw[np.abs(raw - step) <= tolerance] = 0.0
    order = np.argsort(raw, kind="stable")
    groups: list[list[int]] = []
    values: list[float] = []
    for coordinate in order:
        value = float(raw[coordinate])
        if not values or value - values[-1] > tolerance:
            values.append(value)
            groups.append([int(coordinate)])
        else:
            groups[-1].append(int(coordinate))
    # Zero and step are the same point on the offset circle.  The normalization
    # above catches normal cases; this cyclic merge protects round-off outliers.
    if len(values) > 1 and values[0] + step - values[-1] <= tolerance:
        groups[0] = groups[-1] + groups[0]
        values[0] = 0.0
        groups.pop()
        values.pop()
    return np.asarray(values, dtype=float), [np.asarray(group, dtype=np.int64) for group in groups]


def _cached_coordinate_update(
    problem: TransitionProblem,
    amplitude: np.ndarray,
    products: np.ndarray,
    qvalues: np.ndarray,
    coordinate: int,
    new_amplitude: complex,
) -> None:
    delta = new_amplitude - amplitude[coordinate]
    if delta == 0.0:
        return
    qvalues += 2.0 * np.real(np.conj(delta) * products[:, coordinate])
    qvalues += abs(delta) ** 2 * problem.diagonal[:, coordinate]
    products += delta * problem.matrices[:, :, coordinate]
    amplitude[coordinate] = new_amplitude


def best_global_offset_rounding(
    problem: TransitionProblem,
    phase: np.ndarray,
    levels: int,
    gauge: int = 0,
    tie_tolerance: float = 1.0e-11,
) -> QuantizationResult:
    """Exactly minimize over all global offsets before nearest rounding.

    For offset ``alpha`` in one quantizer step, each element has one breakpoint.
    Thus there are at most N projectively distinct commands.  Sorting those
    breakpoints and updating the changed coordinates with the Hermitian cache
    enumerates every open interval exactly.  Adjacent intervals cover both
    half-step tie choices, so no special tie evaluation is needed.
    """

    phase = np.asarray(phase, dtype=float)
    if phase.shape != (problem.n,):
        raise ValueError(f"phase must have shape ({problem.n},)")
    levels = int(levels)
    step = 2.0 * np.pi / levels
    breakpoints, groups = _breakpoint_groups(phase, step)
    if breakpoints.size == 0:
        raise RuntimeError("quantizer breakpoint enumeration is empty")

    next_breakpoint = breakpoints[1] if breakpoints.size > 1 else breakpoints[0] + step
    alpha = float(np.mod(breakpoints[0] + 0.5 * (next_breakpoint - breakpoints[0]), step))
    raw_indices = np.mod(np.floor((phase + alpha) / step + 0.5).astype(np.int64), levels)
    amplitude = np.exp(2j * np.pi * raw_indices / levels)
    products = np.einsum("rij,j->ri", problem.matrices, amplitude, optimize=True)
    qvalues = np.real(np.einsum("i,ri->r", np.conj(amplitude), products, optimize=True))
    roots = np.exp(2j * np.pi * np.arange(levels) / levels)

    best_indices: np.ndarray | None = None
    best_objective = math.inf
    best_offset = alpha
    candidates_seen: set[bytes] = set()

    def consider(candidate_offset: float) -> None:
        nonlocal best_indices, best_objective, best_offset
        fixed = gauge_fix_indices(raw_indices, levels, gauge)
        key = np.ascontiguousarray(fixed).tobytes()
        if key in candidates_seen:
            return
        candidates_seen.add(key)
        value = float(problem.score_qvalues(qvalues))
        lexicographic = tuple(int(x) for x in fixed)
        best_lexicographic = (
            tuple(int(x) for x in best_indices) if best_indices is not None else None
        )
        if (
            value < best_objective - tie_tolerance
            or (
                abs(value - best_objective) <= tie_tolerance
                and (best_lexicographic is None or lexicographic < best_lexicographic)
            )
        ):
            best_indices = fixed.copy()
            best_objective = value
            best_offset = float(candidate_offset)

    consider(alpha)
    for group_index in range(1, len(groups)):
        for coordinate in groups[group_index]:
            coordinate = int(coordinate)
            raw_indices[coordinate] = (raw_indices[coordinate] + 1) % levels
            _cached_coordinate_update(
                problem,
                amplitude,
                products,
                qvalues,
                coordinate,
                roots[raw_indices[coordinate]],
            )
        following = (
            breakpoints[group_index + 1]
            if group_index + 1 < breakpoints.size
            else breakpoints[0] + step
        )
        representative = float(
            np.mod(
                breakpoints[group_index]
                + 0.5 * (following - breakpoints[group_index]),
                step,
            )
        )
        consider(representative)

    assert best_indices is not None
    best_phase = indices_to_phase(best_indices, levels, gauge)
    # Re-evaluate once from scratch to remove event-sweep accumulation error.
    exact_objective = problem.score_phase(best_phase)
    return QuantizationResult(
        levels=levels,
        indices=best_indices,
        phase=best_phase,
        objective=exact_objective,
        evaluations=len(candidates_seen),
        rule="exact_best_global_offset",
        offset_rad=best_offset,
        distinct_candidates=len(candidates_seen),
        metadata={
            "breakpoint_groups": int(len(groups)),
            "maximum_candidates": int(problem.n),
            "event_sweep_objective": float(best_objective),
        },
    )


def make_coordinate_orders(
    n: int, sweeps: int, seed: int, gauge: int = 0
) -> np.ndarray:
    if sweeps < 0:
        raise ValueError("sweeps must be non-negative")
    coordinates = np.delete(np.arange(n, dtype=np.int64), gauge)
    rng = np.random.default_rng(seed)
    return np.stack([rng.permutation(coordinates) for _ in range(sweeps)]) if sweeps else np.empty((0, n - 1), dtype=np.int64)


def all_level_coordinate_descent(
    problem: TransitionProblem,
    initial_indices: np.ndarray,
    levels: int,
    max_sweeps: int,
    *,
    coordinate_orders: np.ndarray | None = None,
    order_seed: int = 0,
    gauge: int = 0,
    improvement_tolerance: float = 1.0e-12,
    selection_tolerance: float = 1.0e-13,
    stop_when_converged: bool = True,
) -> DescentResult:
    """Coordinate descent that scores every one of the Q hardware phases.

    Each candidate is evaluated using the exact rank-one Hermitian update

    ``g'_r = g_r + 2 Re(conj(d) y[r,n]) + |d|^2 Q_r[n,n]``.

    Accepted moves are strict decreases.  On a finite alphabet this guarantees
    finite termination at a one-coordinate local optimum when convergence
    stopping is enabled.
    """

    levels = int(levels)
    indices = gauge_fix_indices(initial_indices, levels, gauge).copy()
    if indices.shape != (problem.n,):
        raise ValueError(f"initial_indices must have shape ({problem.n},)")
    if coordinate_orders is None:
        coordinate_orders = make_coordinate_orders(problem.n, max_sweeps, order_seed, gauge)
    else:
        coordinate_orders = np.asarray(coordinate_orders, dtype=np.int64)
        if coordinate_orders.shape != (max_sweeps, problem.n - 1):
            raise ValueError(
                f"coordinate_orders must have shape ({max_sweeps}, {problem.n - 1})"
            )
        expected = set(np.delete(np.arange(problem.n), gauge).tolist())
        if any(set(row.tolist()) != expected for row in coordinate_orders):
            raise ValueError("each coordinate-order row must permute every non-gauge coordinate")

    amplitude, products, qvalues, current = problem.state_from_indices(indices, levels)
    roots = np.exp(2j * np.pi * np.arange(levels) / levels)
    history: list[dict[str, float | int | bool]] = [
        {
            "sweep": 0,
            "objective": current,
            "accepted_moves": 0,
            "evaluations": 0,
            "improved": False,
        }
    ]
    evaluations = 0
    accepted_total = 0
    converged = max_sweeps == 0

    for sweep, order in enumerate(coordinate_orders, start=1):
        sweep_start = current
        accepted_sweep = 0
        for coordinate in order:
            coordinate = int(coordinate)
            old_index = int(indices[coordinate])
            delta = roots - amplitude[coordinate]
            candidate_qvalues = (
                qvalues[None, :]
                + 2.0 * np.real(np.conj(delta)[:, None] * products[:, coordinate][None, :])
                + np.abs(delta)[:, None] ** 2 * problem.diagonal[:, coordinate][None, :]
            )
            candidate_objectives = np.asarray(problem.score_qvalues(candidate_qvalues))
            candidate_objectives[old_index] = current
            minimum = float(np.min(candidate_objectives))
            tied = np.flatnonzero(candidate_objectives <= minimum + selection_tolerance)
            best_index = int(tied[0])
            best_objective = float(candidate_objectives[best_index])
            evaluations += levels
            if best_objective < current - improvement_tolerance:
                _cached_coordinate_update(
                    problem,
                    amplitude,
                    products,
                    qvalues,
                    coordinate,
                    roots[best_index],
                )
                indices[coordinate] = best_index
                current = best_objective
                accepted_sweep += 1
                accepted_total += 1

        # Rebuild once per sweep.  This bounds round-off accumulation without
        # changing which candidate objective is used for any accepted move.
        amplitude, products, qvalues, rebuilt = problem.state_from_indices(indices, levels)
        drift = abs(rebuilt - current)
        current = rebuilt
        improved = accepted_sweep > 0
        history.append(
            {
                "sweep": sweep,
                "objective": current,
                "accepted_moves": accepted_sweep,
                "evaluations": evaluations,
                "improved": improved,
                "cache_rebuild_drift": float(drift),
                "objective_decrease": float(sweep_start - current),
            }
        )
        if not improved:
            converged = True
            if stop_when_converged:
                break

    return DescentResult(
        levels=levels,
        indices=gauge_fix_indices(indices, levels, gauge),
        phase=indices_to_phase(indices, levels, gauge),
        objective=float(current),
        evaluations=evaluations,
        accepted_moves=accepted_total,
        sweeps_completed=len(history) - 1,
        converged=converged,
        history=history,
    )


def lift_indices(
    indices: np.ndarray, levels_from: int, levels_to: int, gauge: int = 0
) -> np.ndarray:
    """Embed a nested alphabet exactly, preserving every complex command."""

    if levels_to % levels_from:
        raise ValueError("levels_to must be an integer multiple of levels_from")
    fixed = gauge_fix_indices(indices, levels_from, gauge)
    return gauge_fix_indices(fixed * (levels_to // levels_from), levels_to, gauge)


def _normalize_sweeps(
    levels: Sequence[int], sweeps: int | Mapping[int, int]
) -> dict[int, int]:
    if isinstance(sweeps, Mapping):
        missing = [int(level) for level in levels if int(level) not in sweeps]
        if missing:
            raise ValueError(f"missing sweep counts for levels {missing}")
        return {int(level): int(sweeps[int(level)]) for level in levels}
    return {int(level): int(sweeps) for level in levels}


def make_nested_coordinate_orders(
    n: int,
    levels: Sequence[int],
    sweeps: int | Mapping[int, int],
    seed: int,
    gauge: int = 0,
) -> dict[int, np.ndarray]:
    sweep_map = _normalize_sweeps(levels, sweeps)
    output: dict[int, np.ndarray] = {}
    for level in levels:
        level_seed = int(np.random.SeedSequence([int(seed), int(level)]).generate_state(1)[0])
        output[int(level)] = make_coordinate_orders(n, sweep_map[int(level)], level_seed, gauge)
    return output


def continuation_from_indices(
    problem: TransitionProblem,
    initial_indices: np.ndarray,
    levels: Sequence[int] = (32, 128, 2048),
    sweeps: int | Mapping[int, int] = 3,
    *,
    coordinate_orders: Mapping[int, np.ndarray] | None = None,
    order_seed: int = 0,
    gauge: int = 0,
    improvement_tolerance: float = 1.0e-12,
    selection_tolerance: float = 1.0e-13,
    stop_when_converged: bool = True,
    initialization: QuantizationResult | None = None,
) -> ContinuationResult:
    levels = tuple(int(level) for level in levels)
    if not levels:
        raise ValueError("levels must not be empty")
    if any(b % a for a, b in zip(levels, levels[1:])):
        raise ValueError("each continuation alphabet must contain the preceding alphabet")
    sweep_map = _normalize_sweeps(levels, sweeps)
    if coordinate_orders is None:
        coordinate_orders = make_nested_coordinate_orders(
            problem.n, levels, sweep_map, order_seed, gauge
        )

    stages: list[DescentResult] = []
    indices = gauge_fix_indices(initial_indices, levels[0], gauge)
    previous_level = levels[0]
    for stage_index, level in enumerate(levels):
        if stage_index:
            indices = lift_indices(indices, previous_level, level, gauge)
        result = all_level_coordinate_descent(
            problem,
            indices,
            level,
            sweep_map[level],
            coordinate_orders=coordinate_orders[level],
            gauge=gauge,
            improvement_tolerance=improvement_tolerance,
            selection_tolerance=selection_tolerance,
            stop_when_converged=stop_when_converged,
        )
        stages.append(result)
        indices = result.indices
        previous_level = level
    return ContinuationResult(levels=levels, stages=stages, initialization=initialization)


def nested_continuation(
    problem: TransitionProblem,
    continuous_phase: np.ndarray,
    levels: Sequence[int] = (32, 128, 2048),
    sweeps: int | Mapping[int, int] = 3,
    *,
    rounding: str = "best_offset",
    coordinate_orders: Mapping[int, np.ndarray] | None = None,
    order_seed: int = 0,
    gauge: int = 0,
    improvement_tolerance: float = 1.0e-12,
    selection_tolerance: float = 1.0e-13,
    stop_when_converged: bool = True,
) -> ContinuationResult:
    levels = tuple(int(level) for level in levels)
    if rounding == "best_offset":
        initial = best_global_offset_rounding(problem, continuous_phase, levels[0], gauge)
    elif rounding == "fixed_gauge":
        initial = fixed_gauge_nearest(problem, continuous_phase, levels[0], gauge)
    else:
        raise ValueError("rounding must be 'best_offset' or 'fixed_gauge'")
    return continuation_from_indices(
        problem,
        initial.indices,
        levels,
        sweeps,
        coordinate_orders=coordinate_orders,
        order_seed=order_seed,
        gauge=gauge,
        improvement_tolerance=improvement_tolerance,
        selection_tolerance=selection_tolerance,
        stop_when_converged=stop_when_converged,
        initialization=initial,
    )


def matched_nested_random_control(
    problem: TransitionProblem,
    continuous_phase: np.ndarray,
    levels: Sequence[int] = (32, 128, 2048),
    sweeps: int | Mapping[int, int] = 3,
    *,
    seed: int = 20260901,
    gauge: int = 0,
    improvement_tolerance: float = 1.0e-12,
    selection_tolerance: float = 1.0e-13,
) -> MatchedControlResult:
    """Compare FE initialization with random under an identical fixed budget.

    Both arms use the same nested alphabets, every-state coordinate scans,
    coordinate order, tolerance, and number of sweeps.  Convergence stopping is
    disabled so the reported coordinate-search evaluation budgets are exactly
    equal.  The warm arm's offset-enumeration cost is reported separately in its
    ``initialization`` record.
    """

    levels = tuple(int(level) for level in levels)
    sweep_map = _normalize_sweeps(levels, sweeps)
    orders = make_nested_coordinate_orders(problem.n, levels, sweep_map, seed, gauge)
    warm_initial = best_global_offset_rounding(problem, continuous_phase, levels[0], gauge)
    warm = continuation_from_indices(
        problem,
        warm_initial.indices,
        levels,
        sweep_map,
        coordinate_orders=orders,
        gauge=gauge,
        improvement_tolerance=improvement_tolerance,
        selection_tolerance=selection_tolerance,
        stop_when_converged=False,
        initialization=warm_initial,
    )
    rng = np.random.default_rng(seed)
    random_indices = rng.integers(0, levels[0], size=problem.n, dtype=np.int64)
    random_indices = gauge_fix_indices(random_indices, levels[0], gauge)
    _, _, _, random_initial_objective = problem.state_from_indices(random_indices, levels[0])
    random_initial = QuantizationResult(
        levels=levels[0],
        indices=random_indices.copy(),
        phase=indices_to_phase(random_indices, levels[0], gauge),
        objective=random_initial_objective,
        evaluations=1,
        rule="deterministic_uniform_random",
        metadata={"seed": int(seed)},
    )
    random = continuation_from_indices(
        problem,
        random_indices,
        levels,
        sweep_map,
        coordinate_orders=orders,
        gauge=gauge,
        improvement_tolerance=improvement_tolerance,
        selection_tolerance=selection_tolerance,
        stop_when_converged=False,
        initialization=random_initial,
    )
    if warm.evaluations != random.evaluations:
        raise AssertionError("matched-control coordinate-search budgets differ")
    return MatchedControlResult(
        warm=warm,
        random=random,
        coordinate_orders=orders,
        random_seed=int(seed),
    )


def projective_distance(phase_a: np.ndarray, phase_b: np.ndarray) -> float:
    a = np.exp(1j * np.asarray(phase_a, dtype=float))
    b = np.exp(1j * np.asarray(phase_b, dtype=float))
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("phase vectors must be one-dimensional with equal shape")
    similarity = float(abs(np.vdot(a, b)) / a.size)
    return float(np.sqrt(max(0.0, 2.0 - 2.0 * min(similarity, 1.0))))


def assert_transition_invariants(
    problem: TransitionProblem,
    phase: np.ndarray,
    *,
    test_levels: int = 16,
    gauge: int = 0,
    seed: int = 20260901,
) -> dict[str, float | int | bool]:
    """Run inexpensive invariants against a supplied physical FE model."""

    rng = np.random.default_rng(seed)
    phase = np.asarray(phase, dtype=float)
    base = problem.score_phase(phase)
    shifted = problem.score_phase(phase + 0.731)
    gauge_error = abs(base - shifted)
    if gauge_error > 5.0e-10 * max(1.0, abs(base)):
        raise AssertionError("objective is not invariant to global phase")

    nearest = fixed_gauge_nearest(problem, phase, test_levels, gauge)
    best = best_global_offset_rounding(problem, phase, test_levels, gauge)
    if best.objective > nearest.objective + 2.0e-10 * max(1.0, abs(nearest.objective)):
        raise AssertionError("best-offset rounding is worse than fixed-gauge nearest")
    if best.distinct_candidates > problem.n:
        raise AssertionError("offset enumeration exceeded the N-candidate bound")
    if best.indices[gauge] != 0 or np.any((best.indices < 0) | (best.indices >= test_levels)):
        raise AssertionError("quantized command violates gauge or alphabet membership")

    # Independent brute-force interval representatives: unlike the production
    # event sweep, this oracle performs a fresh matrix evaluation per interval.
    step = 2.0 * np.pi / test_levels
    brute_breakpoints, _ = _breakpoint_groups(phase, step)
    brute_best = math.inf
    brute_indices: np.ndarray | None = None
    for index, left in enumerate(brute_breakpoints):
        right = (
            brute_breakpoints[index + 1]
            if index + 1 < brute_breakpoints.size
            else brute_breakpoints[0] + step
        )
        representative = float(np.mod(left + 0.5 * (right - left), step))
        raw = np.mod(
            np.floor((phase + representative) / step + 0.5).astype(np.int64),
            test_levels,
        )
        candidate_indices = gauge_fix_indices(raw, test_levels, gauge)
        candidate = problem.score_phase(indices_to_phase(candidate_indices, test_levels, gauge))
        if (
            candidate < brute_best - 1.0e-11
            or (
                abs(candidate - brute_best) <= 1.0e-11
                and (
                    brute_indices is None
                    or tuple(candidate_indices.tolist()) < tuple(brute_indices.tolist())
                )
            )
        ):
            brute_best = candidate
            brute_indices = candidate_indices.copy()
    if brute_indices is None or not np.array_equal(best.indices, brute_indices):
        raise AssertionError("cached offset sweep disagrees with the brute-force interval oracle")
    if not np.isclose(best.objective, brute_best, rtol=2.0e-11, atol=2.0e-11):
        raise AssertionError("cached offset objective disagrees with the brute-force interval oracle")

    indices = best.indices.copy()
    amplitude, products, qvalues, _ = problem.state_from_indices(indices, test_levels)
    coordinate = int(rng.choice(np.delete(np.arange(problem.n), gauge)))
    candidate_index = int(rng.integers(0, test_levels))
    roots = np.exp(2j * np.pi * np.arange(test_levels) / test_levels)
    delta = roots[candidate_index] - amplitude[coordinate]
    cached_qvalues = (
        qvalues
        + 2.0 * np.real(np.conj(delta) * products[:, coordinate])
        + abs(delta) ** 2 * problem.diagonal[:, coordinate]
    )
    direct_indices = indices.copy()
    direct_indices[coordinate] = candidate_index
    _, _, direct_qvalues, _ = problem.state_from_indices(direct_indices, test_levels)
    cache_error = float(np.max(np.abs(cached_qvalues - direct_qvalues)))
    cache_scale = max(float(np.max(np.abs(direct_qvalues))), 1.0)
    if cache_error > 2.0e-10 * cache_scale:
        raise AssertionError("cached Hermitian candidate update disagrees with direct evaluation")

    orders = make_coordinate_orders(problem.n, 2, seed, gauge)
    descent = all_level_coordinate_descent(
        problem,
        best.indices,
        test_levels,
        2,
        coordinate_orders=orders,
        gauge=gauge,
        stop_when_converged=False,
    )
    trajectory = np.array([float(row["objective"]) for row in descent.history])
    if np.any(np.diff(trajectory) > 2.0e-10 * np.maximum(1.0, np.abs(trajectory[:-1]))):
        raise AssertionError("coordinate-descent objective is not monotone")

    lifted = lift_indices(best.indices, test_levels, 2 * test_levels, gauge)
    lift_error = projective_distance(
        indices_to_phase(best.indices, test_levels, gauge),
        indices_to_phase(lifted, 2 * test_levels, gauge),
    )
    if lift_error > 2.0e-8:
        raise AssertionError("nested-alphabet lift changed the command")

    return {
        "global_phase_objective_error": float(gauge_error),
        "best_offset_not_worse": True,
        "offset_candidates": int(best.distinct_candidates),
        "offset_candidate_bound": int(problem.n),
        "offset_bruteforce_match": True,
        "cache_max_abs_error": cache_error,
        "descent_monotone": True,
        "alphabet_membership": True,
        "lift_projective_distance": float(lift_error),
    }


def _synthetic_problem(seed: int = 7, n: int = 9) -> tuple[TransitionProblem, np.ndarray]:
    rng = np.random.default_rng(seed)
    matrices = []
    for _ in range(11):
        raw = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        matrices.append((raw + raw.conj().T) / (2.0 * np.sqrt(n)))
    pressure_row = rng.normal(size=n) + 1j * rng.normal(size=n)
    matrices[4] = np.outer(np.conj(pressure_row), pressure_row)
    problem = TransitionProblem(
        np.stack(matrices),
        {"curvature": 2.0, "force": 2.5, "pressure": 3.0, "stiffness": 2.2},
    )
    phase = rng.uniform(-np.pi, np.pi, n)
    return problem, phase


def run_self_tests() -> dict[str, float | int | bool]:
    problem, phase = _synthetic_problem()
    results = assert_transition_invariants(problem, phase, test_levels=8, seed=19)
    matched = matched_nested_random_control(
        problem,
        phase,
        levels=(8, 16, 32),
        sweeps=1,
        seed=23,
    )
    if matched.warm.evaluations != matched.random.evaluations:
        raise AssertionError("matched budgets differ")
    results["matched_evaluations"] = int(matched.warm.evaluations)
    results["matched_budget_equal"] = True
    # Exercise the canonical unsquared FE objective independently of BFE.
    fe_problem = TransitionProblem(
        problem.matrices,
        problem.scales,
        ObjectiveSpec(method="force_equilibrium"),
    )
    fe_results = assert_transition_invariants(fe_problem, phase, test_levels=8, seed=29)
    results["force_equilibrium_offset_bruteforce_match"] = bool(
        fe_results["offset_bruteforce_match"]
    )
    results["self_test_passed"] = True
    return results


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run deterministic synthetic invariants")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.self_test:
        print(json.dumps(run_self_tests(), indent=2, sort_keys=True))
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
